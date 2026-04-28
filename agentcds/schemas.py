"""
Data schemas for patients, hypotheses, and diagnostic output.
Kept flat (one file) for academic readability.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class LabResult:
    name: str
    value: float
    unit: str
    abnormal: bool = False
    ref_low:  Optional[float] = None
    ref_high: Optional[float] = None

    def __str__(self):
        flag = " [ABNORMAL]" if self.abnormal else ""
        ref  = f" (ref {self.ref_low}–{self.ref_high})" if self.ref_low else ""
        return f"{self.name}: {self.value} {self.unit}{ref}{flag}"


@dataclass
class Patient:
    """Everything the system knows about a patient at intake."""
    patient_id:    str
    age:           int
    sex:           str                          # M | F
    complaint:     str                          # chief complaint
    hpi:           str = ""                     # history of present illness
    pmh:           list[str] = field(default_factory=list)   # past medical history
    medications:   list[str] = field(default_factory=list)
    allergies:     list[str] = field(default_factory=list)
    vitals:        dict      = field(default_factory=dict)   # {"HR": 98, "SpO2": 95}
    labs:          list[LabResult] = field(default_factory=list)
    findings:      list[str] = field(default_factory=list)   # present symptoms/signs
    absent:        list[str] = field(default_factory=list)   # pertinent negatives
    imaging:       list[str] = field(default_factory=list)

    def to_text(self) -> str:
        """Compact clinical summary for LLM input."""
        lines = [
            f"PATIENT: {self.age}yo {self.sex}",
            f"COMPLAINT: {self.complaint}",
        ]
        if self.hpi:
            lines.append(f"HPI: {self.hpi}")
        if self.pmh:
            lines.append(f"PMH: {', '.join(self.pmh)}")
        if self.vitals:
            v = "  ".join(f"{k}: {v}" for k, v in self.vitals.items())
            lines.append(f"VITALS: {v}")
        abnormal_labs = [str(l) for l in self.labs if l.abnormal]
        normal_labs   = [str(l) for l in self.labs if not l.abnormal]
        if abnormal_labs:
            lines.append("ABNORMAL LABS: " + " | ".join(abnormal_labs))
        if normal_labs:
            lines.append("NORMAL LABS: "   + " | ".join(normal_labs))
        if self.medications:
            lines.append(f"MEDS: {', '.join(self.medications)}")
        if self.allergies:
            lines.append(f"ALLERGIES: {', '.join(self.allergies)}")
        if self.findings:
            lines.append(f"PRESENT: {', '.join(self.findings)}")
        if self.absent:
            lines.append(f"ABSENT: {', '.join(self.absent)}")
        if self.imaging:
            lines.extend(f"IMAGING: {img}" for img in self.imaging)
        return "\n".join(lines)


@dataclass
class Evidence:
    """A single piece of evidence for or against a hypothesis."""
    text:         str
    support:      str    # "supports" | "contradicts" | "neutral"
    weight:       float  # 0–1
    source:       str    # "pubmed:PMID" | "vector_store" | "lab_value"


@dataclass
class Hypothesis:
    """A single candidate diagnosis."""
    label:      str
    confidence: float = 0.2
    icd11:      Optional[str] = None
    urgency:    str = "routine"           # routine | urgent | emergent
    workup:     list[str] = field(default_factory=list)
    evidence:   list[Evidence] = field(default_factory=list)
    supporting_factors: list[str] = field(default_factory=list)
    opposing_factors: list[str] = field(default_factory=list)
    missing_data: list[str] = field(default_factory=list)
    confidence_components: dict[str, float] = field(default_factory=dict)

    def adjust(self, delta: float):
        self.confidence = max(0.0, min(1.0, self.confidence + delta))


@dataclass
class DiagnosticResult:
    """Final output of one diagnostic session."""
    patient_id:       str
    differential:     list[Hypothesis]    # ranked, highest confidence first
    next_steps:       list[str]
    clarifications:   list[str]           # questions for clinician if uncertain
    drug_warnings:    list[str]
    rag_iterations:   int
    confidence_band:  str = "indeterminate"
    uncertainty_factors: list[str] = field(default_factory=list)
    reasoning_trace: list[str] = field(default_factory=list)
    disclaimer:       str = (
        "AI-generated decision support — must be reviewed by a licensed clinician "
        "before informing patient care."
    )

    def summary(self) -> str:
        lines = ["\n" + "="*60, f"AgentCDS \u2014 Patient {self.patient_id}", "="*60]
        lines.append("\nDIFFERENTIAL DIAGNOSIS:")
        for i, h in enumerate(self.differential[:5], 1):
            lines.append(f"  {i}. {h.label:<40} {h.confidence:.0%}  [{h.urgency}]  ICD-11: {h.icd11 or 'N/A'}")
        lines.append(f"\nCONFIDENCE BAND: {self.confidence_band}")
        if self.next_steps:
            lines.append("\nNEXT STEPS:")
            for s in self.next_steps:
                lines.append(f"  \u2022 {s}")
        if self.uncertainty_factors:
            lines.append("\nUNCERTAINTY FACTORS:")
            for u in self.uncertainty_factors[:5]:
                lines.append(f"  ? {u}")
        if self.drug_warnings:
            lines.append("\nDRUG WARNINGS:")
            for w in self.drug_warnings:
                for wline in str(w).splitlines():
                    lines.append(f"  \u26a0 {wline}")
        if self.reasoning_trace:
            lines.append("\nREASONING TRACE:")
            for r in self.reasoning_trace[:8]:
                lines.append(f"  - {r}")
        if self.clarifications:
            lines.append("\nCLARIFICATION NEEDED:")
            for q in self.clarifications:
                lines.append(f"  ? {q}")
        lines.append(f"\n{self.disclaimer}")
        lines.append("="*60)
        return "\n".join(lines)