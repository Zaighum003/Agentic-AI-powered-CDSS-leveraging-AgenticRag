"""
Web Search MCP server (FastMCP 2.0).
General and medical-focused web search via DuckDuckGo — free, no API key needed.
"""
from ddgs import DDGS
from fastmcp import FastMCP

mcp = FastMCP("websearch-mcp")

# Medical-authority site filter for focused clinical queries
_MEDICAL_SITES = (
    "site:pubmed.ncbi.nlm.nih.gov OR site:nejm.org OR site:thelancet.com "
    "OR site:jamanetwork.com OR site:bmj.com OR site:uptodate.com "
    "OR site:medscape.com OR site:mayoclinic.org"
)


def _search(query: str, n: int) -> list[dict]:
    """Run a DuckDuckGo text search and return normalised results."""
    results = []
    try:
        with DDGS() as ddgs:
            for r in ddgs.text(query, max_results=n):
                results.append({
                    "title":   r.get("title", ""),
                    "url":     r.get("href", ""),
                    "snippet": r.get("body", ""),
                })
    except Exception as exc:
        results.append({"title": "error", "url": "", "snippet": str(exc)})
    return results


@mcp.tool()
def web_search(query: str, n: int = 5) -> list[dict]:
    """
    General web search.
    Returns up to *n* results, each with title, url, and snippet.
    """
    return _search(query, n)


@mcp.tool()
def web_search_medical(query: str, n: int = 5) -> list[dict]:
    """
    Medical-focused web search restricted to high-quality clinical sources
    (PubMed, NEJM, Lancet, JAMA, BMJ, UpToDate, Medscape, Mayo Clinic).
    Returns up to *n* results, each with title, url, and snippet.
    """
    focused = f"({query}) {_MEDICAL_SITES}"
    return _search(focused, n)


@mcp.tool()
def web_search_news(query: str, n: int = 5) -> list[dict]:
    """
    Search recent news/publications about a medical topic.
    Returns up to *n* results sorted by recency.
    """
    results = []
    try:
        with DDGS() as ddgs:
            for r in ddgs.news(query, max_results=n):
                results.append({
                    "title":   r.get("title", ""),
                    "url":     r.get("url", ""),
                    "snippet": r.get("body", ""),
                    "date":    r.get("date", ""),
                    "source":  r.get("source", ""),
                })
    except Exception as exc:
        results.append({"title": "error", "url": "", "snippet": str(exc)})
    return results


if __name__ == "__main__":
    mcp.run(transport="streamable-http", host="0.0.0.0", port=8005, path="/mcp")
