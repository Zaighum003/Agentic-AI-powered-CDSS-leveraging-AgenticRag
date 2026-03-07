"""
PubMed MCP server (FastMCP 2.0).
Wraps NCBI E-utilities as callable MCP tools.

Run standalone:  python -m agentcds.mcp.pubmed
Call in-process: from fastmcp import Client
                 async with Client(mcp) as c: await c.call_tool(...)
"""
import time
import xml.etree.ElementTree as ET
import httpx
from fastmcp import FastMCP
from agentcds import config

mcp = FastMCP("pubmed-mcp")

SEARCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
FETCH_URL  = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"


def _base_params():
    p = {"retmode": "json", "retmax": 5}
    if config.PUBMED_KEY:
        p["api_key"] = config.PUBMED_KEY
    return p


def _fetch(query: str, n: int = 5) -> list[dict]:
    """Search PubMed and fetch abstracts. Returns list of article dicts."""
    time.sleep(0.35)  # respect rate limit

    with httpx.Client(timeout=15) as client:
        # Step 1: get PMIDs
        r = client.get(SEARCH_URL, params={**_base_params(), "db": "pubmed", "term": query, "retmax": n})
        r.raise_for_status()
        pmids = r.json().get("esearchresult", {}).get("idlist", [])
        if not pmids:
            return []

        time.sleep(0.35)

        # Step 2: fetch abstracts as XML
        r2 = client.get(FETCH_URL, params={
            **_base_params(), "db": "pubmed",
            "id": ",".join(pmids), "rettype": "abstract", "retmode": "xml",
        })
        r2.raise_for_status()

    articles = []
    try:
        root = ET.fromstring(r2.text)
        for art in root.findall(".//PubmedArticle"):
            pmid     = art.findtext(".//PMID") or ""
            title    = art.findtext(".//ArticleTitle") or ""
            abstract = " ".join(
                (ab.text or "") for ab in art.findall(".//AbstractText")
            ) or "No abstract."
            journal  = art.findtext(".//Journal/Title")
            year_str = art.findtext(".//PubDate/Year")
            pub_types = [e.text for e in art.findall(".//PublicationType") if e.text]

            study_type = "Unknown"
            for pt in pub_types:
                if "randomized" in pt.lower():      study_type = "RCT"
                elif "meta-analysis" in pt.lower(): study_type = "MetaAnalysis"
                elif "guideline" in pt.lower():     study_type = "Guideline"
                elif "cohort" in pt.lower():        study_type = "Cohort"
                elif "case report" in pt.lower():   study_type = "CaseReport"

            articles.append({
                "pmid": pmid, "title": title,
                "abstract": abstract[:600],
                "journal": journal,
                "year": int(year_str) if year_str and year_str.isdigit() else None,
                "study_type": study_type,
                "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
            })
    except ET.ParseError:
        pass

    return articles


@mcp.tool()
def pubmed_search(query: str, n: int = 5) -> list[dict]:
    """Search PubMed by keyword. Returns article list with title, abstract, study type."""
    return _fetch(query, n)


@mcp.tool()
def pubmed_pico(population: str, intervention: str, outcome: str, n: int = 5) -> list[dict]:
    """PICO-structured PubMed search — more precise for clinical questions."""
    query = f"{population} AND {intervention} AND {outcome}"
    return _fetch(query, n)


@mcp.tool()
def pubmed_guidelines(condition: str) -> list[dict]:
    """Retrieve clinical practice guidelines for a condition."""
    return _fetch(f"{condition} AND Practice Guideline[pt]", 5)


if __name__ == "__main__":
    mcp.run(transport="streamable-http", host="0.0.0.0", port=8001, path="/mcp")
