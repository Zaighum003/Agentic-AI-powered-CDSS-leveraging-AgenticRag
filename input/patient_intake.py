import streamlit as st
import pdfplumber
import json
from openai import OpenAI
from dataclasses import dataclass, field, asdict
from typing import Optional
import io

# ─────────────────────────────────────────────
#  Patient Dataclass (matches your constructor)
# ─────────────────────────────────────────────
@dataclass
class Patient:
    patient_id: str
    age: int
    sex: str
    complaint: str
    hpi: str = ""
    pmh: list = field(default_factory=list)
    medications: list = field(default_factory=list)
    allergies: list = field(default_factory=list)
    vitals: dict = field(default_factory=dict)
    labs: dict = field(default_factory=dict)
    findings: list = field(default_factory=list)
    absent: list = field(default_factory=list)
    imaging: list = field(default_factory=list)


# ─────────────────────────────────────────────
#  PDF Text Extractor
# ─────────────────────────────────────────────
def extract_text_from_pdf(uploaded_file) -> str:
    text = ""
    with pdfplumber.open(io.BytesIO(uploaded_file.read())) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text.strip()


# ─────────────────────────────────────────────
#  OpenAI Extractor
# ─────────────────────────────────────────────
def extract_patient_data_with_claude(pdf_text: str, manual_data: dict) -> dict:
     #use your api key here
    prompt = f"""
You are a clinical data extraction assistant. 
Extract structured patient data from the lab report text below.

Here is some manually entered patient information to include:
- Patient ID: {manual_data.get('patient_id', 'UNKNOWN')}
- Age: {manual_data.get('age', 'UNKNOWN')}
- Sex: {manual_data.get('sex', 'UNKNOWN')}
- Chief Complaint: {manual_data.get('complaint', 'UNKNOWN')}

LAB REPORT TEXT:
\"\"\"
{pdf_text}
\"\"\"

Extract ALL available information and return ONLY a valid JSON object with these exact keys:
{{
  "patient_id": "string",
  "age": integer,
  "sex": "string",
  "complaint": "string",
  "hpi": "string - History of Present Illness if available, else empty string",
  "pmh": ["list", "of", "past", "medical", "history"],
  "medications": ["list", "of", "medications"],
  "allergies": ["list", "of", "allergies"],
  "vitals": {{
    "bp": "blood pressure if available",
    "hr": "heart rate if available",
    "temp": "temperature if available",
    "spo2": "oxygen saturation if available",
    "rr": "respiratory rate if available",
    "weight": "weight if available",
    "height": "height if available"
  }},
  "labs": {{
    "use lab test name as key": "value with unit as string"
  }},
  "findings": ["list of positive/present clinical findings"],
  "absent": ["list of absent or negative findings"],
  "imaging": ["list of imaging results if any"]
}}

Rules:
- For any field not found in the report, use empty string "" or empty list [] or empty dict {{}}
- Use the manually entered patient info for patient_id, age, sex, complaint
- Extract ALL lab values you find (CBC, LFT, RFT, lipids, hormones, cultures, etc.)
- Do NOT include any explanation or markdown, return ONLY the JSON object.
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",   # cheap & capable — use "gpt-4o" for best accuracy
        max_tokens=2000,
        messages=[
            {"role": "system", "content": "You are a clinical data extraction assistant. Always return valid JSON only."},
            {"role": "user",   "content": prompt}
        ]
    )

    raw = response.choices[0].message.content.strip()
    # Strip markdown fences if present
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    return json.loads(raw.strip())


# ─────────────────────────────────────────────
#  Streamlit UI
# ─────────────────────────────────────────────
st.set_page_config(page_title="CDSS Patient Intake", page_icon="🏥", layout="wide")

st.title("🏥 CDSS Patient Intake System")
st.markdown("Upload a lab report PDF and fill in basic patient details to auto-generate the Patient object.")

# ── Sidebar: Manual Patient Info ──
st.sidebar.header("📋 Patient Basic Info")
st.sidebar.markdown("*Required — not always in lab reports*")

patient_id  = st.sidebar.text_input("Patient ID", placeholder="e.g. PT-2024-001")
age         = st.sidebar.number_input("Age", min_value=0, max_value=150, value=30)
sex         = st.sidebar.selectbox("Sex", ["Male", "Female", "Other"])
complaint   = st.sidebar.text_area("Chief Complaint", placeholder="e.g. Fever with fatigue for 3 days")
hpi         = st.sidebar.text_area("History of Present Illness (HPI)", placeholder="Optional — describe illness progression")
pmh         = st.sidebar.text_area("Past Medical History (comma-separated)", placeholder="e.g. Diabetes, Hypertension")
medications = st.sidebar.text_area("Current Medications (comma-separated)", placeholder="e.g. Metformin 500mg, Lisinopril 10mg")
allergies   = st.sidebar.text_area("Allergies (comma-separated)", placeholder="e.g. Penicillin, Sulfa drugs")

# ── Main: PDF Upload ──
st.subheader("📄 Upload Lab Report PDF")
uploaded_file = st.file_uploader(
    "Upload any lab report (CBC, LFT, RFT, lipids, hormones, cultures, etc.)",
    type=["pdf"]
)

if uploaded_file:
    st.success(f"✅ File uploaded: **{uploaded_file.name}**")

    if st.button("🔍 Extract & Generate Patient Object", type="primary"):
        if not patient_id:
            st.error("❌ Please enter a Patient ID in the sidebar.")
        elif not complaint:
            st.error("❌ Please enter a Chief Complaint in the sidebar.")
        else:
            with st.spinner("📖 Reading PDF..."):
                pdf_text = extract_text_from_pdf(uploaded_file)

            if not pdf_text:
                st.error("❌ Could not extract text from this PDF. It may be a scanned image-only PDF.")
            else:
                with st.expander("📃 Raw Extracted PDF Text", expanded=False):
                    st.text_area("PDF Content", pdf_text, height=200)

                manual_data = {
                    "patient_id": patient_id,
                    "age": int(age),
                    "sex": sex,
                    "complaint": complaint,
                }

                with st.spinner("🤖 AI is analyzing the lab report..."):
                    try:
                        extracted = extract_patient_data_with_claude(pdf_text, manual_data)

                        # Override with manually entered sidebar data
                        extracted["patient_id"]  = patient_id
                        extracted["age"]         = int(age)
                        extracted["sex"]         = sex
                        extracted["complaint"]   = complaint
                        if hpi:
                            extracted["hpi"] = hpi
                        if pmh:
                            extracted["pmh"] = [x.strip() for x in pmh.split(",") if x.strip()]
                        if medications:
                            extracted["medications"] = [x.strip() for x in medications.split(",") if x.strip()]
                        if allergies:
                            extracted["allergies"] = [x.strip() for x in allergies.split(",") if x.strip()]

                        # Build Patient object
                        patient = Patient(
                            patient_id  = extracted.get("patient_id", patient_id),
                            age         = extracted.get("age", int(age)),
                            sex         = extracted.get("sex", sex),
                            complaint   = extracted.get("complaint", complaint),
                            hpi         = extracted.get("hpi", ""),
                            pmh         = extracted.get("pmh", []),
                            medications = extracted.get("medications", []),
                            allergies   = extracted.get("allergies", []),
                            vitals      = extracted.get("vitals", {}),
                            labs        = extracted.get("labs", {}),
                            findings    = extracted.get("findings", []),
                            absent      = extracted.get("absent", []),
                            imaging     = extracted.get("imaging", []),
                        )

                        st.success("✅ Patient object successfully created!")
                        st.balloons()

                        # ── Display Results ──
                        col1, col2 = st.columns(2)

                        with col1:
                            st.subheader("👤 Patient Demographics")
                            st.table({
                                "Field": ["Patient ID", "Age", "Sex", "Chief Complaint"],
                                "Value": [patient.patient_id, patient.age, patient.sex, patient.complaint]
                            })

                            st.subheader("📊 Vitals")
                            if patient.vitals:
                                vitals_clean = {k: v for k, v in patient.vitals.items() if v}
                                st.json(vitals_clean)
                            else:
                                st.info("No vitals found in report.")

                            st.subheader("🧪 Lab Results")
                            if patient.labs:
                                for test, value in patient.labs.items():
                                    st.metric(label=test, value=str(value))
                            else:
                                st.info("No lab values found.")

                        with col2:
                            st.subheader("📋 Clinical Info")
                            st.markdown(f"**HPI:** {patient.hpi or 'N/A'}")

                            st.markdown("**Past Medical History:**")
                            st.write(patient.pmh if patient.pmh else "None recorded")

                            st.markdown("**Medications:**")
                            st.write(patient.medications if patient.medications else "None recorded")

                            st.markdown("**Allergies:**")
                            st.write(patient.allergies if patient.allergies else "None recorded")

                            st.subheader("🔍 Findings")
                            st.markdown("**Present Findings:**")
                            st.write(patient.findings if patient.findings else "None")
                            st.markdown("**Absent Findings:**")
                            st.write(patient.absent if patient.absent else "None")

                            st.subheader("🩻 Imaging")
                            st.write(patient.imaging if patient.imaging else "No imaging results found.")

                        # ── Raw Patient Object as JSON ──
                        st.subheader("🐍 Patient Object (JSON / Python Dict)")
                        patient_dict = asdict(patient)
                        st.json(patient_dict)

                        # ── Python Constructor Code ──
                        st.subheader("📋 Ready-to-Use Python Code")
                        python_code = f"""patient = Patient(
    patient_id   = {json.dumps(patient.patient_id)},
    age          = {patient.age},
    sex          = {json.dumps(patient.sex)},
    complaint    = {json.dumps(patient.complaint)},
    hpi          = {json.dumps(patient.hpi)},
    pmh          = {json.dumps(patient.pmh)},
    medications  = {json.dumps(patient.medications)},
    allergies    = {json.dumps(patient.allergies)},
    vitals       = {json.dumps(patient.vitals, indent=16)},
    labs         = {json.dumps(patient.labs, indent=16)},
    findings     = {json.dumps(patient.findings)},
    absent       = {json.dumps(patient.absent)},
    imaging      = {json.dumps(patient.imaging)},
)"""
                        st.code(python_code, language="python")

                        # ── Download JSON ──
                        st.download_button(
                            label="⬇️ Download Patient JSON",
                            data=json.dumps(patient_dict, indent=2),
                            file_name=f"patient_{patient.patient_id}.json",
                            mime="application/json"
                        )

                    except json.JSONDecodeError as e:
                        st.error(f"❌ Failed to parse openai's response as JSON: {e}")
                    except Exception as e:
                        st.error(f"❌ Error: {e}")

else:
    st.info("👈 Fill in patient details in the sidebar, then upload a lab report PDF above.")

    st.markdown("---")
    st.subheader("What this app does:")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("### 📄 Step 1\nUpload any lab report PDF — CBC, LFT, RFT, lipids, hormones, cultures, etc.")
    with col2:
        st.markdown("### 🤖 Step 2\nGPT-4o-mini reads the report and extracts all lab values, findings, and vitals automatically.")
    with col3:
        st.markdown("### 🏥 Step 3\nA complete `Patient()` object is generated and ready to pass into your CDSS.")
