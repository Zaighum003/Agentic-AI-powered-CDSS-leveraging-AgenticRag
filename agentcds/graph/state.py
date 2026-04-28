"""
AgentCDS LangGraph State
=========================
Single TypedDict that flows through every node in the graph.

Fields are annotated with operator.add for list channels that multiple
parallel nodes write to (e.g. drug_warnings written by both the
pharmacology agent AND the synthesis node).
"""
from __future__ import annotations

import operator
from typing import Annotated, Optional
from typing_extensions import TypedDict

from agentcds.schemas import Patient, Hypothesis, DiagnosticResult


class AgentCDSState(TypedDict):
    # ── Layer 1: Data ──────────────────────────────────────────────
    patient: Patient

    # ── Layer 2: Agent outputs (set by specialist nodes) ───────────
    hypotheses: list[Hypothesis]          # initialised by form_differential
    lab_signals: str                       # Lab Agent clinical pattern text
    radiology_signals: str                 # Radiology Agent structured findings
    pharma_signals: str                    # Pharmacology Agent drug context
    enriched_context: str                  # merged by merge_agent_outputs

    # ── Layer 3: Knowledge seeding ──────────────────────────────────
    knowledge_seeded: bool

    # ── Layer 4: Agentic RAG loop ───────────────────────────────────
    rag_iteration: int
    rag_done: bool

    # ── Layer 5: Output ─────────────────────────────────────────────
    # Annotated with operator.add so pharmacology_agent and synthesis_node
    # can both append without clobbering each other.
    drug_warnings: Annotated[list[str], operator.add]
    reasoning_trace: Annotated[list[str], operator.add]
    uncertainty_factors: list[str]
    next_steps: list[str]
    clarifications: list[str]
    rag_iterations_run: int
    result: Optional[DiagnosticResult]
