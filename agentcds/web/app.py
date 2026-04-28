from __future__ import annotations

import asyncio
import contextlib
import io
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from agentcds.graph import diagnose
from agentcds.mcp.fhir import MOCK_PATIENTS
from agentcds.schemas import DiagnosticResult, LabResult, Patient


class DiagnoseRequest(BaseModel):
    patient_id: str | None = None
    patient: dict[str, Any] | None = None


class _QueueWriter(io.TextIOBase):
    def __init__(self, loop: asyncio.AbstractEventLoop, queue: asyncio.Queue[dict[str, Any]]):
        self._loop = loop
        self._queue = queue
        self._buf = ""

    def write(self, s: str) -> int:
        if not s:
            return 0
        self._buf += s
        while "\n" in self._buf:
            line, self._buf = self._buf.split("\n", 1)
            cleaned = line.strip()
            if cleaned:
                self._loop.call_soon_threadsafe(
                    self._queue.put_nowait,
                    {"type": "log", "message": cleaned},
                )
        return len(s)

    def flush(self) -> None:
        if self._buf.strip():
            self._loop.call_soon_threadsafe(
                self._queue.put_nowait,
                {"type": "log", "message": self._buf.strip()},
            )
        self._buf = ""


def _patient_to_json(patient: Patient) -> dict[str, Any]:
    return {
        "patient_id": patient.patient_id,
        "age": patient.age,
        "sex": patient.sex,
        "complaint": patient.complaint,
        "hpi": patient.hpi,
        "pmh": patient.pmh,
        "medications": patient.medications,
        "allergies": patient.allergies,
        "vitals": patient.vitals,
        "labs": [asdict(l) for l in patient.labs],
        "findings": patient.findings,
        "absent": patient.absent,
        "imaging": patient.imaging,
    }


def _result_to_json(result: DiagnosticResult) -> dict[str, Any]:
    return {
        "patient_id": result.patient_id,
        "rag_iterations": result.rag_iterations,
        "confidence_band": result.confidence_band,
        "uncertainty_factors": result.uncertainty_factors,
        "reasoning_trace": result.reasoning_trace,
        "next_steps": result.next_steps,
        "clarifications": result.clarifications,
        "drug_warnings": result.drug_warnings,
        "disclaimer": result.disclaimer,
        "differential": [
            {
                "label": h.label,
                "confidence": h.confidence,
                "icd11": h.icd11,
                "urgency": h.urgency,
                "workup": h.workup,
                "evidence": [asdict(e) for e in h.evidence],
                "supporting_factors": h.supporting_factors,
                "opposing_factors": h.opposing_factors,
                "missing_data": h.missing_data,
                "confidence_components": h.confidence_components,
            }
            for h in result.differential
        ],
    }


def _coerce_patient(data: dict[str, Any]) -> Patient:
    try:
        labs = [
            LabResult(
                name=str(l["name"]),
                value=float(l["value"]),
                unit=str(l.get("unit", "")),
                abnormal=bool(l.get("abnormal", False)),
                ref_low=float(l["ref_low"]) if l.get("ref_low") is not None else None,
                ref_high=float(l["ref_high"]) if l.get("ref_high") is not None else None,
            )
            for l in data.get("labs", [])
        ]
        return Patient(
            patient_id=str(data["patient_id"]),
            age=int(data["age"]),
            sex=str(data["sex"]),
            complaint=str(data["complaint"]),
            hpi=str(data.get("hpi", "")),
            pmh=list(data.get("pmh", [])),
            medications=list(data.get("medications", [])),
            allergies=list(data.get("allergies", [])),
            vitals=dict(data.get("vitals", {})),
            labs=labs,
            findings=list(data.get("findings", [])),
            absent=list(data.get("absent", [])),
            imaging=list(data.get("imaging", [])),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=f"Invalid patient payload: {exc}") from exc


def _resolve_patient(payload: DiagnoseRequest) -> Patient:
    if payload.patient_id:
        patient = MOCK_PATIENTS.get(payload.patient_id)
        if not patient:
            raise HTTPException(status_code=404, detail=f"Unknown patient_id: {payload.patient_id}")
        return patient
    if payload.patient:
        return _coerce_patient(payload.patient)
    raise HTTPException(status_code=400, detail="Provide patient_id or patient")


def _progress_from_log(log_line: str) -> tuple[int, str] | None:
    mapping = [
           ("Initial differential formed", (40, "Initial differential formed")),
        ("Lab Agent", (20, "Lab agent complete")),
        ("Radiology", (28, "Radiology agent complete")),
        ("Pharma", (36, "Pharmacology agent complete")),
        ("Seeding knowledge", (50, "Knowledge seeding")),
        ("RAG iteration 1", (65, "RAG iteration 1")),
        ("RAG iteration 2", (78, "RAG iteration 2")),
        ("RAG iteration 3", (88, "RAG iteration 3")),
        ("Synthesizing", (94, "Synthesizing result")),
    ]
    lower = log_line.lower()
    for key, progress in mapping:
        if key.lower() in lower:
            return progress
    return None


app = FastAPI(title="AgentCDS Web Demo", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_STATIC_DIR = Path(__file__).resolve().parent / "static"
app.mount("/assets", StaticFiles(directory=str(_STATIC_DIR)), name="assets")


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(_STATIC_DIR / "index.html")


@app.get("/favicon.ico", include_in_schema=False)
async def favicon() -> Response:
    return Response(status_code=204)


@app.get("/api/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/patients")
async def list_patients() -> list[dict[str, Any]]:
    return [
        {
            "patient_id": pid,
            "age": p.age,
            "sex": p.sex,
            "complaint": p.complaint,
        }
        for pid, p in MOCK_PATIENTS.items()
    ]


@app.get("/api/patients/{patient_id}")
async def get_patient(patient_id: str) -> dict[str, Any]:
    patient = MOCK_PATIENTS.get(patient_id)
    if not patient:
        raise HTTPException(status_code=404, detail=f"Unknown patient_id: {patient_id}")
    return _patient_to_json(patient)


@app.post("/api/diagnose")
async def diagnose_rest(payload: DiagnoseRequest) -> dict[str, Any]:
    patient = _resolve_patient(payload)
    result = await asyncio.to_thread(diagnose, patient)
    return _result_to_json(result)


@app.websocket("/ws/diagnose")
async def diagnose_stream(websocket: WebSocket) -> None:
    await websocket.accept()
    try:
        raw = await websocket.receive_text()
        request = DiagnoseRequest.model_validate_json(raw)
        patient = _resolve_patient(request)

        await websocket.send_json({
            "type": "status",
            "phase": "started",
            "progress": 5,
            "message": f"Running diagnosis for {patient.patient_id}",
        })

        loop = asyncio.get_running_loop()
        queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()

        def _run() -> None:
            writer = _QueueWriter(loop, queue)
            try:
                with contextlib.redirect_stdout(writer):
                    result = diagnose(patient)
                writer.flush()
                loop.call_soon_threadsafe(
                    queue.put_nowait,
                    {"type": "result", "data": _result_to_json(result)},
                )
            except Exception as exc:  # pragma: no cover
                loop.call_soon_threadsafe(
                    queue.put_nowait,
                    {"type": "error", "message": str(exc)},
                )
            finally:
                loop.call_soon_threadsafe(queue.put_nowait, {"type": "done"})

        task = asyncio.create_task(asyncio.to_thread(_run))

        while True:
            event = await queue.get()
            if event["type"] == "log":
                progress_update = _progress_from_log(event["message"])
                if progress_update:
                    progress, phase = progress_update
                    await websocket.send_json({
                        "type": "status",
                        "phase": phase,
                        "progress": progress,
                        "message": event["message"],
                    })
            await websocket.send_json(event)
            if event["type"] == "done":
                break

        await task
    except WebSocketDisconnect:
        return
    except Exception as exc:
        await websocket.send_json({"type": "error", "message": str(exc)})


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("agentcds.web.app:app", host="0.0.0.0", port=8000, reload=True)
