"""
AgentCDS — LangGraph demo runner
==================================
Runs the full 5-layer diagnostic pipeline using the LangGraph graph.

Usage:
    python examples/run_graph_demo.py [PATIENT_ID]

    PATIENT_ID defaults to DEMO-001 (pancytopenia workup).
    Other built-in patients: DEMO-002 … DEMO-005

Environment:
    Copy .env.example → .env and fill in at minimum:
        LLM_PROVIDER=openai
        OPENAI_API_KEY=<your key>
        LLM_MODEL=gpt-4o-mini       # or any chat-completions model
        EMBEDDING_MODEL=text-embedding-3-small
"""
import sys
import asyncio

from fastmcp import Client
from agentcds.mcp.fhir import mcp as fhir_mcp
from agentcds.graph import diagnose          # ← LangGraph pipeline


async def _get_patient(patient_id: str):
    async with Client(fhir_mcp) as c:
        result = await c.call_tool("get_patient", {"patient_id": patient_id})
    return result


def main():
    patient_id = sys.argv[1] if len(sys.argv) > 1 else "DEMO-001"
    print(f"AgentCDS (LangGraph)  —  loading patient {patient_id} …")

    # Fetch patient via FHIR MCP (in-process)
    raw_result = asyncio.run(_get_patient(patient_id))

    # Unwrap FastMCP result into a Patient object
    import json, re
    from agentcds.schemas import Patient, LabResult

    content = getattr(raw_result, "content", None)
    if content:
        text = "".join(ch.text if hasattr(ch, "text") else str(ch) for ch in content)
    else:
        text = raw_result.text if hasattr(raw_result, "text") else str(raw_result)

    try:
        data = json.loads(text)
        if isinstance(data, dict) and "patient_id" in data:
            labs = [
                LabResult(
                    name=l["name"], value=l["value"], unit=l["unit"],
                    abnormal=l.get("abnormal", False),
                    ref_low=l.get("ref_low"), ref_high=l.get("ref_high"),
                )
                for l in data.get("labs", [])
            ]
            patient = Patient(
                patient_id=data["patient_id"],
                age=data["age"], sex=data["sex"],
                complaint=data["complaint"],
                hpi=data.get("hpi", ""),
                pmh=data.get("pmh", []),
                medications=data.get("medications", []),
                allergies=data.get("allergies", []),
                vitals=data.get("vitals", {}),
                labs=labs,
                findings=data.get("findings", []),
                absent=data.get("absent", []),
                imaging=data.get("imaging", []),
            )
        else:
            raise ValueError("FHIR response did not contain a patient object")
    except (json.JSONDecodeError, KeyError, ValueError) as exc:
        # Fall back to in-memory mock patients
        from agentcds.mcp.fhir import MOCK_PATIENTS
        if patient_id not in MOCK_PATIENTS:
            print(f"[Error] Unknown patient '{patient_id}'. "
                  f"Available: {list(MOCK_PATIENTS.keys())}")
            sys.exit(1)
        print(f"[Fallback] Using in-memory mock patient ({exc})")
        patient = MOCK_PATIENTS[patient_id]

    # ── Run the LangGraph pipeline ─────────────────────────────────
    result = diagnose(patient)

    # Summary already printed inside build_result node;
    # return the result object for scripting use.
    return result


if __name__ == "__main__":
    main()
