import unittest
import importlib
from unittest.mock import patch

graph_module = importlib.import_module("agentcds.graph.graph")
from agentcds.mcp.fhir import MOCK_PATIENTS
from agentcds.schemas import DiagnosticResult, Hypothesis


def _fake_form_differential(state):
    return {
        "hypotheses": [
            Hypothesis(label="Myelodysplastic syndrome", confidence=0.58, urgency="urgent"),
            Hypothesis(label="Aplastic anemia", confidence=0.31, urgency="urgent"),
            Hypothesis(label="Acute myeloid leukemia", confidence=0.22, urgency="emergent"),
        ],
        "lab_signals": "Lab agent stub output",
        "radiology_signals": "Radiology agent stub output",
        "pharma_signals": "Pharma agent stub output",
        "enriched_context": "Stubbed specialist context",
        "drug_warnings": ["[INTERACTION] example warning"],
        "rag_iteration": 0,
        "rag_done": False,
        "knowledge_seeded": False,
    }


def _fake_seed_knowledge(state):
    return {"knowledge_seeded": True}


def _fake_rag_node(state):
    hypotheses = state["hypotheses"]
    hypotheses[0].confidence = 0.81
    return {
        "hypotheses": hypotheses,
        "rag_iteration": state.get("rag_iteration", 0) + 1,
        "rag_done": True,
        "rag_iterations_run": 1,
    }


def _fake_synthesize(state):
    return {
        "next_steps": [
            "Order peripheral smear and bone marrow biopsy",
            "Repeat CBC with differential within 24 hours",
        ],
        "drug_warnings": ["[SAFETY] monitor bleeding risk"],
    }


def _fake_clarify(state):
    return {"clarifications": []}


def _fake_build_result(state):
    result = DiagnosticResult(
        patient_id=state["patient"].patient_id,
        differential=sorted(state["hypotheses"], key=lambda h: h.confidence, reverse=True),
        next_steps=state.get("next_steps", []),
        clarifications=state.get("clarifications", []),
        drug_warnings=state.get("drug_warnings", []),
        rag_iterations=state.get("rag_iterations_run", 0),
    )
    return {"result": result}


class GraphSmokeTest(unittest.TestCase):
    def test_diagnose_returns_result_for_mock_patient(self):
        patient = MOCK_PATIENTS["DEMO-001"]

        with patch.multiple(
            graph_module,
            form_differential=_fake_form_differential,
            seed_knowledge=_fake_seed_knowledge,
            rag_node=_fake_rag_node,
            synthesize=_fake_synthesize,
            clarify=_fake_clarify,
            build_result=_fake_build_result,
        ):
            test_graph = graph_module.build_graph()
            original_graph = graph_module.graph
            graph_module.graph = test_graph
            try:
                result = graph_module.diagnose(patient)
            finally:
                graph_module.graph = original_graph

        self.assertIsInstance(result, DiagnosticResult)
        self.assertEqual(result.patient_id, "DEMO-001")
        self.assertGreaterEqual(result.rag_iterations, 1)
        self.assertGreaterEqual(len(result.differential), 1)
        self.assertEqual(result.differential[0].label, "Myelodysplastic syndrome")
        self.assertTrue(any("SAFETY" in w for w in result.drug_warnings))


if __name__ == "__main__":
    unittest.main()
