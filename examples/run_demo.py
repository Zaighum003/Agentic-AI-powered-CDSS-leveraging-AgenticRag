"""
AgentCDS Demo
==============
Run a full diagnostic session on a mock patient.

Setup:
  1. pip install -e .
  2. cp .env.example .env  # add your HF_TOKEN
  3. python examples/run_demo.py [PATIENT_ID]

Available patients:
  DEMO-001  54yo M — Pancytopenia / fatigue / bruising  (MDS workup)
  DEMO-002  32yo F — Pleuritic chest pain / leg swelling (post-partum PE)
  DEMO-003  62yo M — Tearing chest pain radiating to back (dissection vs MI)
  DEMO-004  19yo F — Vomiting / confusion / fruity breath (DKA / HHS)
  DEMO-009  58yo M — Crushing chest pain + left arm + sweating  ★ LAYMAN-FRIENDLY (classic heart attack)

Examples:
  python examples/run_demo.py           # defaults to DEMO-001
  python examples/run_demo.py DEMO-002
"""
import argparse
import asyncio
import json
import sys

# Force UTF-8 output on Windows so Unicode arrows/symbols print correctly.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

from fastmcp import Client
from agentcds.mcp.fhir import mcp as fhir_mcp
from agentcds.agents.orchestrator import diagnose
from agentcds.schemas import Patient, LabResult


def _tool_result_text(result) -> str:
    """Extract text payload from FastMCP call_tool result across API versions."""
    if result is None:
        return ""

    # FastMCP/MCP result object shape: CallToolResult(content=[TextContent(...)])
    content = getattr(result, "content", None)
    if content:
        first = content[0]
        return first.text if hasattr(first, "text") else str(first)

    # Older list-like shape used by earlier examples.
    if isinstance(result, (list, tuple)) and result:
        first = result[0]
        return first.text if hasattr(first, "text") else str(first)

    return result.text if hasattr(result, "text") else str(result)


async def load_patient_from_mcp(patient_id: str) -> dict:
    """Load patient data via FHIR MCP tool."""
    async with Client(fhir_mcp) as client:
        result = await client.call_tool("get_patient", {"patient_id": patient_id})
    return json.loads(_tool_result_text(result))


def dict_to_patient(data: dict) -> Patient:
    """Convert raw FHIR MCP dict back into a Patient object."""
    labs = [
        LabResult(
            name=l["name"], value=l["value"], unit=l["unit"],
            abnormal=l.get("abnormal", False),
        )
        for l in data.get("labs", [])
    ]
    return Patient(
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AgentCDS — Clinical Decision Support Demo")
    parser.add_argument(
        "patient_id",
        nargs="?",
        default="DEMO-001",
        help="Patient ID to run (DEMO-001 … DEMO-004). Default: DEMO-001",
    )
    args = parser.parse_args()
    patient_id = args.patient_id.upper()

    print(f"Loading patient {patient_id} via FHIR MCP...")
    patient_data = asyncio.run(load_patient_from_mcp(patient_id))

    if "error" in patient_data:
        print(patient_data["error"])
        exit(1)

    patient = dict_to_patient(patient_data)
    print(f"Loaded: {patient.age}yo {patient.sex} — {patient.complaint}")
    print()

    print("Running AgentCDS diagnostic session...")
    result = diagnose(patient)

    print(result.summary())
