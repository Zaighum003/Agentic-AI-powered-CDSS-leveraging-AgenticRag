"""
AgentCDS LangGraph Graph
=========================
Constructs and compiles the full 5-layer diagnostic pipeline as a
LangGraph StateGraph.

Graph topology
--------------
                        ┌─────────────────────────────────┐
  START                 │         Layer 2 (parallel)       │
    │                   │  lab_agent_node      ──────────┐ │
    ▼                   │  radiology_agent_node ─────────┤ │
  form_differential ────┤  pharmacology_agent_node ──────┘ │
                        └─────────────┬───────────────────┘
                                      │ (all three nodes run in
                                      │  their own branch then join)
                                      ▼
                             merge_agent_outputs   (not needed: form_differential
                                      │            already runs agents inline and
                                      │            returns enriched_context)
                                      ▼
                               seed_knowledge       (Layer 3)
                                      │
                                      ▼
                      ┌────────── rag_node ◄──┐   (Layer 4 loop)
                      │               │       │
                      │    should_continue_rag?│
                      │       ├── "rag_node" ──┘
                      │       └── "synthesize"
                      │               │
                      └───────────────┼──────────────────────────────
                                      ▼
                                 synthesize          (Layer 5)
                                      │
                                      ▼
                                   clarify
                                      │
                                      ▼
                                 build_result
                                      │
                                     END

NOTE: Because form_differential now runs the three specialist agents
internally (before LLM differential generation), we keep the graph
linear and avoid the complexity of a parallel fan-out.  The agents
still run concurrently in spirit — you can trivially switch to true
parallel LangGraph branches by following the PARALLEL VARIANT comment
below once you need it.
"""
from __future__ import annotations

from langgraph.graph import StateGraph, START, END

from agentcds.graph.state import AgentCDSState
from agentcds.graph.nodes import (
    form_differential,
    seed_knowledge,
    rag_node,
    synthesize,
    clarify,
    build_result,
    should_continue_rag,
)
from agentcds.schemas import Patient, DiagnosticResult


# ──────────────────────────────────────────────────────────────────────────────
# Graph construction
# ──────────────────────────────────────────────────────────────────────────────

def build_graph():
    """
    Construct and compile the AgentCDS LangGraph StateGraph.

    Returns a compiled graph that accepts an AgentCDSState-compatible
    dict as input (minimum: {"patient": <Patient>}).
    """
    sg = StateGraph(AgentCDSState)

    # ── Layer 2: Specialist agents + initial differential ─────────
    sg.add_node("form_differential", form_differential)

    # ── Layer 3: Knowledge seeding ────────────────────────────────
    sg.add_node("seed_knowledge", seed_knowledge)

    # ── Layer 4: Agentic RAG loop ─────────────────────────────────
    sg.add_node("rag_node", rag_node)

    # ── Layer 5: Output ───────────────────────────────────────────
    sg.add_node("synthesize", synthesize)
    sg.add_node("clarify", clarify)
    sg.add_node("build_result", build_result)

    # ── Edges ─────────────────────────────────────────────────────
    sg.add_edge(START, "form_differential")
    sg.add_edge("form_differential", "seed_knowledge")
    sg.add_edge("seed_knowledge", "rag_node")

    # RAG loop: conditional self-edge
    sg.add_conditional_edges(
        "rag_node",
        should_continue_rag,
        {
            "rag_node":   "rag_node",    # loop back
            "synthesize": "synthesize",  # exit loop
        },
    )

    sg.add_edge("synthesize", "clarify")
    sg.add_edge("clarify", "build_result")
    sg.add_edge("build_result", END)

    return sg.compile()


# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────

# Module-level compiled graph — import this from other modules
graph = build_graph()


def diagnose(patient: Patient) -> DiagnosticResult:
    """
    Run the full AgentCDS diagnostic pipeline for a patient.

    This is the LangGraph-backed equivalent of the old
    orchestrator.diagnose() function and has the same signature.

    Args:
        patient: Patient object (from schemas.py or FHIR MCP)

    Returns:
        DiagnosticResult with differential, next steps, warnings, etc.
    """
    initial_state: AgentCDSState = {
        "patient": patient,
        # All other fields are initialised by the first node
        "hypotheses": [],
        "lab_signals": "",
        "radiology_signals": "",
        "pharma_signals": "",
        "enriched_context": "",
        "knowledge_seeded": False,
        "rag_iteration": 0,
        "rag_done": False,
        "drug_warnings": [],
        "next_steps": [],
        "clarifications": [],
        "rag_iterations_run": 0,
        "result": None,
    }

    final_state = graph.invoke(initial_state)
    result = final_state.get("result")

    if result is None:
        # Fallback — should not normally happen
        from agentcds.schemas import DiagnosticResult as DR
        result = DR(
            patient_id=patient.patient_id,
            differential=final_state.get("hypotheses", []),
            next_steps=final_state.get("next_steps", []),
            clarifications=final_state.get("clarifications", []),
            drug_warnings=final_state.get("drug_warnings", []),
            rag_iterations=final_state.get("rag_iterations_run", 0),
        )

    return result


# ──────────────────────────────────────────────────────────────────────────────
# PARALLEL VARIANT (reference — not wired by default)
# ──────────────────────────────────────────────────────────────────────────────
# If you want true concurrent LangGraph fan-out for the three specialist
# agents, replace form_differential with the pattern below.
#
#   from agentcds.graph.nodes import (
#       bootstrap_differential,   # only does the LLM differential call
#       lab_agent_node,           # wraps run_lab_agent
#       radiology_agent_node,     # wraps run_radiology_agent
#       pharmacology_agent_node,  # wraps run_pharmacology_agent
#       merge_agent_outputs,      # merges the three signals
#   )
#
#   sg.add_node("bootstrap_differential", bootstrap_differential)
#   sg.add_node("lab_agent_node", lab_agent_node)
#   sg.add_node("radiology_agent_node", radiology_agent_node)
#   sg.add_node("pharmacology_agent_node", pharmacology_agent_node)
#   sg.add_node("merge_agent_outputs", merge_agent_outputs)
#
#   sg.add_edge(START, "bootstrap_differential")
#   sg.add_edge("bootstrap_differential", "lab_agent_node")
#   sg.add_edge("bootstrap_differential", "radiology_agent_node")
#   sg.add_edge("bootstrap_differential", "pharmacology_agent_node")
#   # All three join at merge_agent_outputs (LangGraph waits for all)
#   sg.add_edge("lab_agent_node", "merge_agent_outputs")
#   sg.add_edge("radiology_agent_node", "merge_agent_outputs")
#   sg.add_edge("pharmacology_agent_node", "merge_agent_outputs")
#   sg.add_edge("merge_agent_outputs", "seed_knowledge")
#   ...
