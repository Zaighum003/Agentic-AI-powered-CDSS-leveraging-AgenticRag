<p align="center">
  <img src="logo_cdss.jpg" alt="Project Logo" width="200">
</p>

# AgentCDS 

Agentic Clinical Decision Support using HyDE + Self-RAG + CRAG
with FastMCP 2.0 clinical knowledge integrations.

## Setup

```bash
pip install -e .
cp .env.example .env
```

Choose an LLM backend in `.env`:

```env
# Option 1: local transformers model (default)
LLM_PROVIDER=local
LLM_MODEL=TinyLlama/TinyLlama-1.1B-Chat-v1.0

# Option 2: OpenAI API
LLM_PROVIDER=openai
OPENAI_API_KEY=<your-openai-key>
LLM_MODEL=gpt-4o-mini
# Optional: OPENAI_BASE_URL=https://<your-compatible-endpoint>/v1
```

## Technologies Used

The project is built with these main tools and technologies:

- Python 3.10+
- LangGraph for the agent workflow
- FastAPI for the web API and WebSocket endpoint
- Uvicorn as the ASGI server
- Pydantic for data models and validation
- FastMCP for MCP tool servers and local tool calls
- ChromaDB for the vector store and retrieval cache
- Transformers and Torch for local LLM execution
- OpenAI API support as an optional LLM backend
- httpx for HTTP-based integrations
- python-dotenv for environment and config loading
- ddgs for web search
- Mermaid for graph visualization
- unittest and pytest-style tests for validation

Domain-specific integrations:

- PubMed
- RxNorm
- FHIR/mock patient data
- Web search

## Run

```bash
python examples/run_graph_demo.py            # full LangGraph diagnostic session
python examples/run_graph_demo.py DEMO-003   # run a specific demo patient
python examples/call_mcp_tools.py            # call MCP tools directly
python examples/visualise_graph.py --mermaid # print graph topology (Mermaid)
```

## Web Demo (FastAPI + SPA)

```bash
python examples/run_web_demo.py
```

Open `http://127.0.0.1:8000` in your browser.

Web API endpoints:

- `GET /api/health`
- `GET /api/patients`
- `GET /api/patients/{patient_id}`
- `POST /api/diagnose`
- `WS /ws/diagnose` (live status + logs + final result)

## Project Structure

```
agentcds/
├── config.py           # settings from .env
├── schemas.py          # Patient, Hypothesis, DiagnosticResult dataclasses
├── llm.py              # LLM backend wrapper (local transformers or OpenAI)
├── vector_store.py     # ChromaDB (in-memory)
├── graph/
│   ├── state.py        # LangGraph state schema
│   ├── agents.py       # specialist clinical agents
│   ├── nodes.py        # graph node functions
│   └── graph.py        # compiled LangGraph + diagnose() entrypoint
├── rag/
│   ├── hyde.py         # Hypothetical Document Embeddings retrieval
│   ├── self_rag.py     # evidence grading (relevance, support, utility)
│   ├── crag.py         # Corrective RAG on contradictions
│   └── __init__.py
├── mcp/
│   ├── pubmed.py       # FastMCP 2.0 server — NCBI PubMed
│   ├── rxnorm.py       # FastMCP 2.0 server — NLM RxNorm drug interactions
│   ├── web_search.py   # FastMCP 2.0 server — web evidence search
│   └── fhir.py         # FastMCP 2.0 server — patient data (mock + live)
├── web/
│   ├── app.py          # FastAPI server + REST/WebSocket endpoints
│   └── static/         # Single-page frontend (Stitch-based)
└── __init__.py
```

## Demo Patients

| ID | Case |
|----|------|
| `DEMO-001` | 54yo M — pancytopenia (MDS vs aplastic anemia vs AML) |
| `DEMO-002` | 32yo F — post-partum DVT/PE |

## MCP Servers (optional standalone mode)

Each server can also be run over HTTP independently:
```bash
python -m agentcds.mcp.pubmed    # port 8001
python -m agentcds.mcp.rxnorm    # port 8002
python -m agentcds.mcp.fhir      # port 8003
```

## Key Config (`.env`)

| Variable | Default | Description |
|----------|---------|-------------|
| `LLM_PROVIDER` | `local` | LLM backend (`local` or `openai`) |
| `HF_TOKEN` | required | HuggingFace API token |
| `LLM_MODEL` | `BioMistral/BioMistral-7B` | Medical LLM |
| `OPENAI_API_KEY` | empty | OpenAI API key |
| `OPENAI_BASE_URL` | empty | Optional OpenAI-compatible API base URL |
| `PUBMED_KEY` | optional | NCBI key (raises rate limit) |

<p align="center">
  <img src="logo_cdss.jpg" alt="Project Logo" width="200">
</p>
