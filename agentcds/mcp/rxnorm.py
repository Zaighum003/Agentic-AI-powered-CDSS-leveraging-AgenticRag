"""
RxNorm MCP server (FastMCP 2.0).
Drug interaction checking via NLM RxNorm REST API — free, no key needed.
"""
import httpx
from fastmcp import FastMCP

mcp = FastMCP("rxnorm-mcp")

BASE        = "https://rxnav.nlm.nih.gov/REST"
INTERACTION = "https://rxnav.nlm.nih.gov/REST/interaction"


def _rxcui(drug_name: str) -> str | None:
    """Resolve a drug name to its RxCUI."""
    try:
        r = httpx.get(f"{BASE}/rxcui.json", params={"name": drug_name, "search": 1}, timeout=8)
        ids = r.json().get("idGroup", {}).get("rxnormId", [])
        return ids[0] if ids else None
    except Exception:
        return None


@mcp.tool()
def check_interactions(drug_names: list[str]) -> list[dict]:
    """
    Check for drug-drug interactions among a list of drug names.
    Returns list of interaction dicts; empty list = no significant interactions.
    """
    rxcuis = [cui for name in drug_names if (cui := _rxcui(name))]
    if len(rxcuis) < 2:
        return []

    try:
        r = httpx.get(f"{INTERACTION}/list.json", params={"rxcuis": " ".join(rxcuis)}, timeout=10)
        r.raise_for_status()
        data = r.json()
    except Exception:
        return []

    SEVERITY_MAP = {"high": "major", "moderate": "moderate", "low": "minor", "n/a": "moderate"}
    interactions = []

    for group in data.get("fullInteractionTypeGroup", []):
        for itype in group.get("fullInteractionType", []):
            for pair in itype.get("interactionPair", []):
                concepts = pair.get("interactionConcept", [])
                d1 = concepts[0].get("minConceptItem", {}).get("name", "?") if len(concepts) > 0 else "?"
                d2 = concepts[1].get("minConceptItem", {}).get("name", "?") if len(concepts) > 1 else "?"
                sev = SEVERITY_MAP.get(pair.get("severity", "").lower(), "moderate")
                interactions.append({
                    "drug_1": d1, "drug_2": d2,
                    "severity": sev,
                    "description": pair.get("description", "")[:200],
                })

    return interactions


@mcp.tool()
def interaction_summary(drug_names: list[str]) -> str:
    """Human-readable drug interaction summary for a patient's medication list."""
    ixs = check_interactions(drug_names)
    if not ixs:
        return f"No significant interactions among: {', '.join(drug_names)}"
    lines = [f"Found {len(ixs)} interaction(s):"]
    for ix in sorted(ixs, key=lambda x: {"major": 0, "moderate": 1, "minor": 2}.get(x["severity"], 3)):
        lines.append(f"  [{ix['severity'].upper()}] {ix['drug_1']} + {ix['drug_2']}: {ix['description'][:100]}")
    return "\n".join(lines)


if __name__ == "__main__":
    mcp.run(transport="streamable-http", host="0.0.0.0", port=8002, path="/mcp")
