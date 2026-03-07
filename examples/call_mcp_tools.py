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


async def main():
    # --- FHIR: list and load patients ---
    print("=== FHIR MCP ===")
    async with Client(fhir_mcp) as client:
        r = await client.call_tool("list_patients", {})
        patients = json.loads(r[0].text)
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
        articles = json.loads(r[0].text)
        for art in articles:
            print(f"  [{art.get('study_type','?')}] {art.get('title','')[:80]}")

    # --- RxNorm: drug interactions ---
    print("\n=== RxNorm MCP ===")
    async with Client(rxnorm_mcp) as client:
        r = await client.call_tool("interaction_summary", {
            "drug_names": ["metformin", "lisinopril", "aspirin"]
        })
        print(f"  {r[0].text}")


if __name__ == "__main__":
    asyncio.run(main())
