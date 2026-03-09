"""
AgentCDS Demo
==============
Run a full diagnostic session on a mock patient.

Setup:
  1. pip install -e .
  2. cp .env.example .env  # add your HF_TOKEN
  3. python examples/run_demo.py

To try the post-partum PE case: change patient_id to "DEMO-002"
"""
import asyncio
import json
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
    print("Loading patient via FHIR MCP...")
    patient_data = asyncio.run(load_patient_from_mcp("DEMO-001"))

    if "error" in patient_data:
        print(patient_data["error"])
        exit(1)

    patient = dict_to_patient(patient_data)
    print(f"Loaded: {patient.age}yo {patient.sex} — {patient.complaint}")
    print()

    print("Running AgentCDS diagnostic session...")
    result = diagnose(patient)

    print(result.summary())
