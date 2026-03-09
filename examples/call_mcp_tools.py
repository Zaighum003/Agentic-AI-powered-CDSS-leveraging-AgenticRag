"""
Directly call FastMCP 2.0 tools in-process.
Shows how each MCP server works independently.
"""
import asyncio
import json
from fastmcp import Client
from agentcds.mcp.pubmed  import mcp as pubmed_mcp
from agentcds.mcp.rxnorm  import mcp as rxnorm_mcp
from agentcds.mcp.fhir    import mcp as fhir_mcp


def _tool_result_text(result) -> str:
    """Extract text payload from FastMCP call_tool result across API versions."""
    if result is None:
        return ""

    content = getattr(result, "content", None)
    if content:
        first = content[0]
        return first.text if hasattr(first, "text") else str(first)

    if isinstance(result, (list, tuple)) and result:
        first = result[0]
        return first.text if hasattr(first, "text") else str(first)

    return result.text if hasattr(result, "text") else str(result)


async def main():
    # --- FHIR: list and load patients ---
    print("=== FHIR MCP ===")
    async with Client(fhir_mcp) as client:
        r = await client.call_tool("list_patients", {})
        patients = json.loads(_tool_result_text(r))
        for p in patients:
            print(f"  {p['patient_id']}: {p['age']}yo {p['sex']} — {p['complaint']}")

    # --- PubMed: search ---
    print("\n=== PubMed MCP ===")
    async with Client(pubmed_mcp) as client:
        r = await client.call_tool("pubmed_pico", {
            "population": "pancytopenia adult male",
            "intervention": "bone marrow biopsy",
            "outcome":      "myelodysplastic syndrome diagnosis",
            "n": 2,
        })
        articles = json.loads(_tool_result_text(r))
        for art in articles:
            print(f"  [{art.get('study_type','?')}] {art.get('title','')[:500]}")

    # --- RxNorm: drug interactions ---
    print("\n=== RxNorm MCP ===")
    async with Client(rxnorm_mcp) as client:
        r = await client.call_tool("interaction_summary", {
            "drug_names": ["metformin", "lisinopril", "aspirin"]
        })
        print(f"  {_tool_result_text(r)}")


if __name__ == "__main__":
    asyncio.run(main())
