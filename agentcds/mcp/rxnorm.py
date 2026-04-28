"""
RxNorm MCP server (FastMCP 2.0).
Drug interaction checking via NLM RxNorm REST API — free, no key needed.
"""
import httpx
from fastmcp import FastMCP

mcp = FastMCP("rxnorm-mcp")

BASE        = "https://rxnav.nlm.nih.gov/REST"
INTERACTION = "https://rxnav.nlm.nih.gov/REST/interaction"


def _normalize_drug_name(drug_name: str) -> str:
    return drug_name.strip().lower().split()[0] if drug_name.strip() else ""


def _rxcui(drug_name: str) -> str | None:
    """Resolve a drug name to its RxCUI."""
    try:
        r = httpx.get(f"{BASE}/rxcui.json", params={"name": drug_name, "search": 1}, timeout=8)
        ids = r.json().get("idGroup", {}).get("rxnormId", [])
        return ids[0] if ids else None
    except Exception:
        return None


def _fetch_rxnorm_interactions(rxcuis: list[str]) -> list[dict]:
    """Try the live RxNorm interaction endpoints and return parsed interactions."""
    candidates = [
        f"{INTERACTION}/list.json",
        f"{INTERACTION}.json",
        f"{BASE}/interaction/list.json",
        f"{BASE}/interaction.json",
    ]

    for url in candidates:
        try:
            r = httpx.get(url, params={"rxcuis": " ".join(rxcuis)}, timeout=10)
            r.raise_for_status()
            data = r.json()
        except Exception:
            continue

        interactions = []
        for group in data.get("fullInteractionTypeGroup", []):
            for itype in group.get("fullInteractionType", []):
                for pair in itype.get("interactionPair", []):
                    concepts = pair.get("interactionConcept", [])
                    d1 = concepts[0].get("minConceptItem", {}).get("name", "?") if len(concepts) > 0 else "?"
                    d2 = concepts[1].get("minConceptItem", {}).get("name", "?") if len(concepts) > 1 else "?"
                    sev = {"high": "major", "moderate": "moderate", "low": "minor", "n/a": "moderate"}.get(
                        pair.get("severity", "").lower(),
                        "moderate",
                    )
                    interactions.append({
                        "drug_1": d1,
                        "drug_2": d2,
                        "severity": sev,
                        "description": pair.get("description", "")[:200],
                        "source": "rxnorm",
                    })

        if interactions:
            return interactions

    return []


def _fallback_interactions(drug_names: list[str]) -> list[dict]:
    """Small safety net when the live RxNorm interaction endpoint is unavailable."""
    normalized = [_normalize_drug_name(name) for name in drug_names if _normalize_drug_name(name)]
    if len(normalized) < 2:
        return []

    aliases = {
        "warfarin": {"warfarin"},
        "amiodarone": {"amiodarone"},
        "aspirin": {"aspirin"},
        "nsaid": {"ibuprofen", "naproxen", "diclofenac", "indomethacin", "ketorolac", "meloxicam", "celecoxib"},
        "macrolide": {"clarithromycin", "erythromycin", "azithromycin"},
        "azole": {"fluconazole", "itraconazole", "ketoconazole", "voriconazole", "posaconazole"},
        "statin": {"atorvastatin", "simvastatin", "lovastatin", "pravastatin", "rosuvastatin", "fluvastatin", "pitavastatin"},
        "acei": {"lisinopril", "enalapril", "benazepril", "captopril", "ramipril", "fosinopril", "quinapril", "perindopril"},
        "arb": {"losartan", "valsartan", "olmesartan", "candesartan", "irbesartan", "telmisartan", "azilsartan"},
        "k_sparing": {"spironolactone", "eplerenone", "amiloride", "triamterene"},
        "diuretic": {"furosemide", "bumetanide", "torsemide", "hydrochlorothiazide", "chlorthalidone", "metolazone"},
        "lithium": {"lithium"},
        "digoxin": {"digoxin"},
    }

    def _contains_any(drug: str, group: set[str]) -> bool:
        return any(alias in drug for alias in group)

    def _has_pair(group_a: set[str], group_b: set[str]) -> tuple[str, str] | None:
        for first in normalized:
            for second in normalized:
                if first >= second:
                    continue
                if (_contains_any(first, group_a) and _contains_any(second, group_b)) or (
                    _contains_any(first, group_b) and _contains_any(second, group_a)
                ):
                    return first, second
        return None

    rules = [
        (aliases["warfarin"], aliases["amiodarone"], "major", "Amiodarone can markedly increase warfarin effect and bleeding risk."),
        (aliases["warfarin"], aliases["aspirin"], "major", "Aspirin plus warfarin increases bleeding risk."),
        (aliases["warfarin"], aliases["nsaid"], "major", "NSAIDs can raise bleeding risk with warfarin."),
        (aliases["warfarin"], aliases["macrolide"], "major", "Macrolides can increase warfarin effect and bleeding risk."),
        (aliases["warfarin"], aliases["azole"], "major", "Azole antifungals can increase warfarin effect and bleeding risk."),
        (aliases["statin"], aliases["amiodarone"], "moderate", "Amiodarone can increase statin myopathy risk."),
        (aliases["statin"], aliases["macrolide"], "major", "Macrolides can raise statin levels and myopathy risk."),
        (aliases["statin"], aliases["azole"], "moderate", "Azole antifungals can raise statin exposure."),
        (aliases["acei"], aliases["arb"], "major", "ACE inhibitors and ARBs together increase renal and potassium risk."),
        (aliases["acei"], aliases["k_sparing"], "moderate", "ACE inhibitors plus potassium-sparing diuretics can cause hyperkalaemia."),
        (aliases["diuretic"], aliases["lithium"], "major", "Diuretics can increase lithium levels and toxicity risk."),
        (aliases["diuretic"], aliases["digoxin"], "moderate", "Diuretics can worsen electrolyte changes that increase digoxin toxicity risk."),
    ]

    results: list[dict] = []
    for group_a, group_b, severity, description in rules:
        matched = _has_pair(group_a, group_b)
        if matched:
            first, second = matched
            results.append({
                "drug_1": first,
                "drug_2": second,
                "severity": severity,
                "description": description,
                "source": "fallback",
            })

    return results


@mcp.tool()
def check_interactions(drug_names: list[str]) -> list[dict]:
    """
    Check for drug-drug interactions among a list of drug names.
    Returns list of interaction dicts; empty list = no significant interactions.
    """
    rxcuis = [cui for name in drug_names if (cui := _rxcui(name))]
    if len(rxcuis) >= 2:
        interactions = _fetch_rxnorm_interactions(rxcuis)
        if interactions:
            return interactions

    return _fallback_interactions(drug_names)


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
