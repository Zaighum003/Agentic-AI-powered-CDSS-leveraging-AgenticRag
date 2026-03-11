"""
FHIR MCP server (FastMCP 2.0).
Returns structured patient data. Uses built-in mock patients by default.
Set FHIR_URL in .env to connect to a real FHIR R4 server.
"""
from fastmcp import FastMCP
from agentcds.schemas import Patient, LabResult

mcp = FastMCP("fhir-mcp")

# Demo patients covering a range of academic test cases
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

    # DEMO-003: Acute MI vs. aortic dissection vs. pericarditis
    "DEMO-003": Patient(
        patient_id="DEMO-003", age=62, sex="M",
        complaint="Sudden-onset tearing chest pain radiating to the back",
        hpi="62yo male with hypertension presents with sudden severe 10/10 tearing substernal chest pain radiating to the interscapular back, onset 1 hour ago. Diaphoresis and nausea. No relief with nitrates.",
        pmh=["Hypertension", "Hyperlipidemia", "Former smoker (40 pack-years)"],
        medications=["Amlodipine 10mg QD", "Atorvastatin 40mg QD", "Aspirin 81mg QD"],
        allergies=[],
        vitals={"HR": 102, "BP_right": "168/94", "BP_left": "134/82", "SpO2": "97%", "RR": 20},
        labs=[
            LabResult("Troponin-I",  0.06, "ng/mL",  abnormal=True,  ref_low=0,   ref_high=0.04),
            LabResult("D-dimer",     3.8,  "µg/mL",  abnormal=True,  ref_low=0,   ref_high=0.5),
            LabResult("CK-MB",       18,   "ng/mL",  abnormal=True,  ref_low=0,   ref_high=6.3),
            LabResult("Creatinine",  1.2,  "mg/dL",  abnormal=False, ref_low=0.7, ref_high=1.3),
            LabResult("Hemoglobin",  14.1, "g/dL",   abnormal=False, ref_low=13.5,ref_high=17.5),
            LabResult("WBC",         11.8, "x10³/µL",abnormal=True,  ref_low=4.5, ref_high=11.0),
        ],
        findings=["Diaphoresis", "Blood pressure differential >30mmHg between arms", "Aortic regurgitation murmur"],
        absent=["ST elevation on ECG", "Pulse deficits in extremities"],
        imaging=["CXR: Widened mediastinum (8.2 cm)", "ECG: Sinus tachycardia, no ST changes"],
    ),

    # DEMO-004: Diabetic ketoacidosis vs. hyperosmolar hyperglycemic state vs. sepsis
    "DEMO-004": Patient(
        patient_id="DEMO-004", age=19, sex="F",
        complaint="Vomiting, abdominal pain, and confusion for 12 hours",
        hpi="19yo female with Type 1 DM. 12 hours of nausea, vomiting, diffuse abdominal pain, fruity breath odor, and progressive confusion. Missed insulin doses for 2 days due to viral illness.",
        pmh=["Type 1 Diabetes Mellitus (since age 8)"],
        medications=["Insulin glargine 20 units QHS", "Insulin lispro sliding scale"],
        allergies=["Sulfa drugs"],
        vitals={"HR": 118, "BP": "102/64", "SpO2": "99%", "RR": 28, "Temp": "37.8C"},
        labs=[
            LabResult("Glucose",     482,  "mg/dL",  abnormal=True,  ref_low=70,  ref_high=100),
            LabResult("pH",          7.14, "",        abnormal=True,  ref_low=7.35,ref_high=7.45),
            LabResult("Bicarbonate", 8,    "mEq/L",  abnormal=True,  ref_low=22,  ref_high=29),
            LabResult("Anion gap",   24,   "mEq/L",  abnormal=True,  ref_low=8,   ref_high=12),
            LabResult("Potassium",   3.2,  "mEq/L",  abnormal=True,  ref_low=3.5, ref_high=5.0),
            LabResult("Sodium",      132,  "mEq/L",  abnormal=True,  ref_low=136, ref_high=145),
            LabResult("Ketones",     4.2,  "mmol/L", abnormal=True,  ref_low=0,   ref_high=0.6),
            LabResult("Creatinine",  1.4,  "mg/dL",  abnormal=True,  ref_low=0.5, ref_high=1.1),
        ],
        findings=["Kussmaul respirations", "Fruity breath", "Dry mucous membranes", "Diffuse abdominal tenderness"],
        absent=["Focal neurological deficits", "Meningismus"],
        imaging=["CXR: Clear lung fields"],
    ),

    # DEMO-005: Community-acquired pneumonia vs. pulmonary TB vs. lung cancer
    "DEMO-005": Patient(
        patient_id="DEMO-005", age=47, sex="M",
        complaint="3-week cough with hemoptysis, weight loss, and night sweats",
        hpi="47yo male immigrant from Vietnam. 3-week productive cough with occasional blood-streaked sputum, 6kg weight loss over 2 months, drenching night sweats, low-grade fevers. Former smoker (15 pack-years).",
        pmh=["No significant PMH", "No prior TB diagnosis"],
        medications=[],
        allergies=[],
        vitals={"HR": 88, "BP": "122/76", "SpO2": "96%", "Temp": "37.9C"},
        labs=[
            LabResult("WBC",          11.2, "x10³/µL", abnormal=True,  ref_low=4.5,  ref_high=11.0),
            LabResult("Hemoglobin",   11.4, "g/dL",    abnormal=True,  ref_low=13.5, ref_high=17.5),
            LabResult("ESR",          88,   "mm/hr",   abnormal=True,  ref_low=0,    ref_high=15),
            LabResult("CRP",          42,   "mg/L",    abnormal=True,  ref_low=0,    ref_high=8),
            LabResult("LDH",          310,  "U/L",     abnormal=True,  ref_low=120,  ref_high=250),
            LabResult("Albumin",      3.0,  "g/dL",    abnormal=True,  ref_low=3.5,  ref_high=5.0),
        ],
        findings=["Dullness to percussion right upper lobe", "Post-tussive crackles RUL", "Cachexia"],
        absent=["Clubbing", "Cervical lymphadenopathy"],
        imaging=["CXR: Right upper lobe cavitary lesion with surrounding infiltrate and hilar adenopathy"],
    ),

    # DEMO-006: Meningitis vs. subarachnoid hemorrhage vs. hypertensive encephalopathy
    "DEMO-006": Patient(
        patient_id="DEMO-006", age=28, sex="F",
        complaint="Worst headache of my life, neck stiffness, and photophobia",
        hpi="28yo female with sudden-onset severe headache described as 'thunderclap', reaching maximum intensity within seconds. Associated with neck stiffness, photophobia, and one episode of vomiting. No preceding trauma.",
        pmh=["Oral contraceptive use"],
        medications=["Ethinyl estradiol / levonorgestrel"],
        allergies=[],
        vitals={"HR": 96, "BP": "148/92", "SpO2": "99%", "Temp": "38.1C", "GCS": 14},
        labs=[
            LabResult("WBC",          16.4, "x10³/µL", abnormal=True,  ref_low=4.5,  ref_high=11.0),
            LabResult("Neutrophils",  88,   "%",        abnormal=True,  ref_low=50,   ref_high=70),
            LabResult("CRP",          38,   "mg/L",     abnormal=True,  ref_low=0,    ref_high=8),
            LabResult("Glucose",      118,  "mg/dL",    abnormal=True,  ref_low=70,   ref_high=100),
            LabResult("Platelets",    210,  "x10³/µL",  abnormal=False, ref_low=150,  ref_high=400),
        ],
        findings=["Nuchal rigidity", "Kernig sign positive", "Brudzinski sign positive", "Photophobia"],
        absent=["Papilledema", "Focal neurological deficits", "Petechial rash"],
        imaging=["CT head (non-contrast): No hemorrhage, no mass effect, no midline shift"],
    ),

    # DEMO-007: Heart failure vs. COPD exacerbation vs. cardiac tamponade
    "DEMO-007": Patient(
        patient_id="DEMO-007", age=71, sex="M",
        complaint="Progressive dyspnea, orthopnea, and bilateral leg edema for 2 weeks",
        hpi="71yo male with known ischemic cardiomyopathy (EF 30%). 2-week worsening dyspnea, 3-pillow orthopnea, PND, bilateral ankle edema. Gained 4kg this week. Non-compliant with furosemide for 5 days.",
        pmh=["Ischemic cardiomyopathy", "COPD (GOLD II)", "Hypertension", "Previous MI (2019)"],
        medications=["Furosemide 40mg QD", "Carvedilol 12.5mg BID", "Lisinopril 10mg QD", "Spironolactone 25mg QD", "Tiotropium inhaler"],
        allergies=["NSAIDs (fluid retention)"],
        vitals={"HR": 94, "BP": "158/98", "SpO2": "88% on room air", "RR": 24},
        labs=[
            LabResult("BNP",          1840, "pg/mL",   abnormal=True,  ref_low=0,   ref_high=100),
            LabResult("Troponin",     0.03, "ng/mL",   abnormal=False, ref_low=0,   ref_high=0.04),
            LabResult("Creatinine",   1.8,  "mg/dL",   abnormal=True,  ref_low=0.7, ref_high=1.3),
            LabResult("Sodium",       131,  "mEq/L",   abnormal=True,  ref_low=136, ref_high=145),
            LabResult("Potassium",    5.2,  "mEq/L",   abnormal=True,  ref_low=3.5, ref_high=5.0),
            LabResult("pO2",          56,   "mmHg",    abnormal=True,  ref_low=80,  ref_high=100),
        ],
        findings=["Bilateral basal crackles", "JVD at 45 degrees", "Pitting edema to knees", "S3 gallop"],
        absent=["Fever", "Pleuritic chest pain", "Pulsus paradoxus"],
        imaging=["CXR: Cardiomegaly, bilateral pleural effusions, pulmonary vascular congestion", "ECG: Sinus rhythm, LBBB (known)"],
    ),

    # DEMO-009 (LAYMAN-FRIENDLY): Classic heart attack presentation
    # Everyone knows heart attacks. Symptoms are familiar from everyday life.
    # Ideal for non-medical audiences — crushing chest pain, left arm, sweating.
    "DEMO-009": Patient(
        patient_id="DEMO-009", age=58, sex="M",
        complaint="Crushing chest pain spreading to the left arm with sweating — started 40 minutes ago",
        hpi="58yo male with high blood pressure, diabetes, and high cholesterol. While sitting at his desk at work this morning he felt a sudden crushing pressure in the center of his chest, like 'an elephant sitting on my chest'. The pain spreads down his left arm and up to his jaw. He is drenched in cold sweat and feels extremely nauseous. He took two of his wife's aspirin tablets on the way to the ER. Pain is 9/10, not relieved by rest.",
        pmh=["High blood pressure (hypertension)", "Type 2 Diabetes", "High cholesterol (hyperlipidemia)", "Smoker — 1 pack per day for 30 years"],
        medications=["Metformin 1000mg twice daily", "Lisinopril 10mg once daily", "Atorvastatin 40mg at bedtime"],
        allergies=[],
        vitals={"HR": 104, "BP": "158/96", "SpO2": "96%", "RR": 20, "Temp": "37.1C"},
        labs=[
            LabResult("Troponin-I",  1.8,  "ng/mL",  abnormal=True,  ref_low=0,    ref_high=0.04),
            LabResult("CK-MB",       28,   "ng/mL",  abnormal=True,  ref_low=0,    ref_high=6.3),
            LabResult("D-dimer",     0.4,  "µg/mL",  abnormal=False, ref_low=0,    ref_high=0.5),
            LabResult("Glucose",     218,  "mg/dL",  abnormal=True,  ref_low=70,   ref_high=100),
            LabResult("Creatinine",  1.1,  "mg/dL",  abnormal=False, ref_low=0.7,  ref_high=1.3),
            LabResult("Hemoglobin",  14.8, "g/dL",   abnormal=False, ref_low=13.5, ref_high=17.5),
        ],
        findings=["Diaphoresis (cold sweating)", "Pallor", "Clutching chest", "Nausea"],
        absent=["Fever", "Cough", "Leg swelling", "Tearing quality to pain"],
        imaging=["ECG: ST-segment elevation in leads II, III, aVF and V4–V6 (inferior + lateral STEMI)",
                 "CXR: Normal heart size, clear lung fields, no pneumothorax"],
    ),

    # DEMO-008: Septic arthritis vs. gout vs. reactive arthritis
    "DEMO-008": Patient(
        patient_id="DEMO-008", age=55, sex="M",
        complaint="Acute swollen, hot, extremely painful right knee — cannot bear weight",
        hpi="55yo male with gout history. Acute onset right knee swelling, warmth, and excruciating pain over 18 hours. Unable to bear weight. No recent trauma. Drinks 6–8 beers/week. On diuretic for hypertension.",
        pmh=["Gout (2 prior attacks, left first MTP)", "Hypertension"],
        medications=["Hydrochlorothiazide 25mg QD", "Allopurinol 100mg QD"],
        allergies=[],
        vitals={"HR": 100, "BP": "144/88", "SpO2": "98%", "Temp": "38.6C"},
        labs=[
            LabResult("WBC",          14.8, "x10³/µL", abnormal=True,  ref_low=4.5, ref_high=11.0),
            LabResult("CRP",          112,  "mg/L",    abnormal=True,  ref_low=0,   ref_high=8),
            LabResult("ESR",          74,   "mm/hr",   abnormal=True,  ref_low=0,   ref_high=15),
            LabResult("Uric acid",    9.8,  "mg/dL",   abnormal=True,  ref_low=3.5, ref_high=7.2),
            LabResult("Creatinine",   1.0,  "mg/dL",   abnormal=False, ref_low=0.7, ref_high=1.3),
            LabResult("Blood cultures", 0,  "pending", abnormal=False, ref_low=0,   ref_high=0),
        ],
        findings=["Right knee: erythema, warmth, effusion, severe tenderness to palpation", "ROM severely limited by pain"],
        absent=["Skin lesions", "Urethral discharge", "Eye inflammation", "Tophi"],
        imaging=["X-ray right knee: No fracture, moderate joint effusion, no chondrocalcinosis"],
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
