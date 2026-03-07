"""
FHIR MCP server (FastMCP 2.0).
Returns structured patient data. Uses built-in mock patients by default.
Set FHIR_URL in .env to connect to a real FHIR R4 server.
"""
from fastmcp import FastMCP
from agentcds.schemas import Patient, LabResult

mcp = FastMCP("fhir-mcp")

# Two demo patients covering common academic test cases
MOCK_PATIENTS = {
    "DEMO-001": Patient(
        patient_id="DEMO-001", age=54, sex="M",
        complaint="Fatigue, easy bruising, and shortness of breath for 3 months",
        hpi="3-month progressive fatigue, exertional dyspnea, easy bruising, two nosebleeds. No fever, weight loss, or lymphadenopathy.",
        pmh=["Hypertension", "Type 2 Diabetes"],
        medications=["Metformin 1000mg BID", "Lisinopril 10mg QD"],
        allergies=["Penicillin (rash)"],
        vitals={"HR": 98, "BP": "128/78", "SpO2": "95%", "Temp": "37.2C"},
        labs=[
            LabResult("WBC",        2.1,  "x10³/µL", abnormal=True,  ref_low=4.5,  ref_high=11.0),
            LabResult("Hemoglobin", 8.2,  "g/dL",    abnormal=True,  ref_low=13.5, ref_high=17.5),
            LabResult("Platelets",  62,   "x10³/µL", abnormal=True,  ref_low=150,  ref_high=400),
            LabResult("MCV",        104,  "fL",       abnormal=True,  ref_low=80,   ref_high=100),
            LabResult("LDH",        320,  "U/L",      abnormal=True,  ref_low=120,  ref_high=250),
            LabResult("Reticulocytes", 0.4, "%",      abnormal=True,  ref_low=0.5,  ref_high=2.5),
            LabResult("Creatinine", 1.1,  "mg/dL",   abnormal=False, ref_low=0.7,  ref_high=1.3),
        ],
        findings=["Pallor", "Petechiae bilateral lower extremities"],
        absent=["Splenomegaly", "Lymphadenopathy", "Sternal tenderness"],
        imaging=["CXR: No acute cardiopulmonary pathology"],
    ),
    "DEMO-002": Patient(
        patient_id="DEMO-002", age=32, sex="F",
        complaint="Pleuritic chest pain, right leg swelling, shortness of breath",
        hpi="32yo female, 3 weeks post-partum. 2-day right pleuritic chest pain, right calf swelling and tenderness, progressive dyspnea.",
        pmh=["Recent vaginal delivery (3 weeks ago)"],
        medications=["Prenatal vitamins"],
        allergies=[],
        vitals={"HR": 112, "BP": "118/74", "SpO2": "93%", "RR": 22},
        labs=[
            LabResult("D-dimer",  4.2,  "µg/mL FEU", abnormal=True,  ref_low=0, ref_high=0.5),
            LabResult("Troponin", 0.04, "ng/mL",      abnormal=False, ref_low=0, ref_high=0.04),
            LabResult("BNP",      180,  "pg/mL",      abnormal=True,  ref_low=0, ref_high=100),
            LabResult("pO2",      68,   "mmHg",       abnormal=True,  ref_low=80, ref_high=100),
        ],
        findings=["Right calf tenderness and swelling", "Tachycardia", "Decreased right breath sounds"],
        absent=["Hemoptysis", "Fever"],
        imaging=["CXR: Subsegmental atelectasis right lower lobe"],
    ),
}


@mcp.tool()
def get_patient(patient_id: str) -> dict:
    """Load a patient's clinical data by ID. Returns all fields as a dict."""
    p = MOCK_PATIENTS.get(patient_id)
    if not p:
        return {"error": f"Patient {patient_id!r} not found", "available": list(MOCK_PATIENTS)}
    return {
        "patient_id": p.patient_id, "age": p.age, "sex": p.sex,
        "complaint": p.complaint, "hpi": p.hpi, "pmh": p.pmh,
        "medications": p.medications, "allergies": p.allergies,
        "vitals": p.vitals,
        "labs": [{"name": l.name, "value": l.value, "unit": l.unit,
                  "abnormal": l.abnormal} for l in p.labs],
        "findings": p.findings, "absent": p.absent, "imaging": p.imaging,
        "clinical_text": p.to_text(),
    }


@mcp.tool()
def list_patients() -> list[dict]:
    """List all available patient IDs and their chief complaints."""
    return [
        {"patient_id": pid, "age": p.age, "sex": p.sex, "complaint": p.complaint}
        for pid, p in MOCK_PATIENTS.items()
    ]


if __name__ == "__main__":
    mcp.run(transport="streamable-http", host="0.0.0.0", port=8003, path="/mcp")
