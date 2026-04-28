const state = {
  patient: null,
  logs: [],
  result: null,
  ws: null,
  checklist: [
    "Patient loaded",
    "Lab agent complete",
    "Radiology agent complete",
    "Pharmacology agent complete",
    "Initial differential formed",
    "Knowledge seeding",
    "RAG iterations",
    "Synthesis and output",
  ],
  checklistStatus: {},
};

const views = {
  intake: document.getElementById("view-intake"),
  running: document.getElementById("view-running"),
  result: document.getElementById("view-result"),
};

function setView(name) {
  Object.entries(views).forEach(([key, element]) => {
    element.classList.toggle("active", key === name);
  });
  document.querySelectorAll(".nav-item").forEach((btn) => {
    btn.classList.toggle("active", btn.dataset.view === name);
  });
}

function setSessionStatus(text) {
  const host = document.getElementById("sessionStatus");
  const icon = text === "Running" ? "neurology" : text === "Completed" ? "verified" : text === "Error" ? "warning" : "monitor_heart";
  host.innerHTML = `<span class="material-symbols-outlined">${icon}</span><span>${escapeHtml(text)}</span>`;
}

function appendLog(message) {
  state.logs.push(message);
  const line = `<p class="logline">${escapeHtml(message)}</p>`;
  document.getElementById("liveLog").insertAdjacentHTML("beforeend", line);
  document.getElementById("resultLog").insertAdjacentHTML("beforeend", line);
  document.getElementById("liveLog").scrollTop = document.getElementById("liveLog").scrollHeight;
  document.getElementById("resultLog").scrollTop = document.getElementById("resultLog").scrollHeight;
}

function escapeHtml(input) {
  return input
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function updateProgress(progress, phase) {
  document.getElementById("progressBar").style.width = `${progress}%`;
  document.getElementById("progressPct").textContent = `${progress}%`;
  if (phase) {
    document.getElementById("phaseLabel").textContent = phase;
  }

  const map = [
    [10, "Patient loaded"],
    [20, "Lab agent complete"],
    [28, "Radiology agent complete"],
    [36, "Pharmacology agent complete"],
    [40, "Initial differential formed"],
    [50, "Knowledge seeding"],
    [65, "RAG iterations"],
    [94, "Synthesis and output"],
  ];
  map.forEach(([threshold, label]) => {
    if (progress >= threshold) {
      state.checklistStatus[label] = "done";
    }
  });
  if (progress < 100 && phase) {
    const active = map.find(([t]) => progress <= t);
    if (active) {
      state.checklistStatus[active[1]] = "active";
    }
  }
  renderChecklist();
}

function renderChecklist() {
  const host = document.getElementById("opsChecklist");
  host.innerHTML = state.checklist
    .map((item) => {
      const status = state.checklistStatus[item] || "pending";
      const label = status === "done" ? "Done" : status === "active" ? "Running" : "Pending";
      return `<li><span>${escapeHtml(item)}</span><span class="status-${status}">${label}</span></li>`;
    })
    .join("");
}

function renderPatientCard() {
  const host = document.getElementById("patientCard");
  if (!state.patient) {
    host.innerHTML = "<h2>Awaiting Patient Data</h2><p class='muted'>Load a demo patient or parse JSON to continue.</p>";
    return;
  }
  const abnormalLabs = (state.patient.labs || []).filter((lab) => lab.abnormal);
  host.innerHTML = `
    <h2>${escapeHtml(state.patient.patient_id)} · ${state.patient.age}yo ${escapeHtml(state.patient.sex)}</h2>
    <p><strong>Chief Complaint:</strong> ${escapeHtml(state.patient.complaint)}</p>
    <p class="muted">${escapeHtml(state.patient.hpi || "No HPI provided")}</p>
    <h3>Abnormal Labs</h3>
    <ul>
      ${abnormalLabs.map((lab) => `<li>${escapeHtml(lab.name)}: ${lab.value} ${escapeHtml(lab.unit || "")}</li>`).join("") || "<li>None</li>"}
    </ul>
    <h3>Medications</h3>
    <p>${(state.patient.medications || []).map(escapeHtml).join(", ") || "None"}</p>
    <h3>Findings</h3>
    <p>${(state.patient.findings || []).map(escapeHtml).join(", ") || "None"}</p>
  `;
}

function renderResult(result) {
  const top = result.differential[0];
  const uncertainty = (result.uncertainty_factors || [])
    .slice(0, 4)
    .map((u) => `<li>${escapeHtml(u)}</li>`)
    .join("");
  const trace = (result.reasoning_trace || [])
    .slice(0, 6)
    .map((r) => `<li>${escapeHtml(r)}</li>`)
    .join("");
  document.getElementById("resultSummary").innerHTML = `
    <h2>Diagnostic Summary</h2>
    <p><strong>Patient:</strong> ${escapeHtml(result.patient_id)}</p>
    <p><strong>Top Diagnosis:</strong> ${escapeHtml(top.label)} (${Math.round(top.confidence * 100)}%)</p>
    <p><strong>Confidence Band:</strong> ${escapeHtml(result.confidence_band || "indeterminate")}</p>
    <p><strong>RAG Iterations:</strong> ${result.rag_iterations}</p>
    <h3>Uncertainty Drivers</h3>
    <ul>${uncertainty || "<li>None</li>"}</ul>
    <h3>Reasoning Timeline</h3>
    <ul>${trace || "<li>No trace available</li>"}</ul>
    <p class="muted">${escapeHtml(result.disclaimer)}</p>
  `;

  document.getElementById("differentialList").innerHTML = result.differential
    .map((dx) => {
      const pct = Math.round(dx.confidence * 100);
      const cls = pct >= 65 ? "high" : pct < 40 ? "low" : "";
      const evidenceRows = (dx.evidence || [])
        .slice(0, 4)
        .map((ev) => `<li>${escapeHtml(ev.support)}: ${escapeHtml(ev.text)} <span class='muted'>(${escapeHtml(ev.source)})</span></li>`)
        .join("");
      const supportRows = (dx.supporting_factors || [])
        .slice(0, 3)
        .map((s) => `<li>${escapeHtml(s)}</li>`)
        .join("");
      const opposeRows = (dx.opposing_factors || [])
        .slice(0, 3)
        .map((s) => `<li>${escapeHtml(s)}</li>`)
        .join("");
      const missingRows = (dx.missing_data || [])
        .slice(0, 3)
        .map((s) => `<li>${escapeHtml(s)}</li>`)
        .join("");
      const componentRows = Object.entries(dx.confidence_components || {})
        .map(([k, v]) => `<li>${escapeHtml(k)}: ${typeof v === "number" ? v.toFixed(2) : escapeHtml(String(v))}</li>`)
        .join("");

      return `
        <article class="dx-card">
          <div class="dx-title">
            <span>${escapeHtml(dx.label)}</span>
            <span>${pct}%</span>
          </div>
          <div class="dx-meta">ICD-11: ${escapeHtml(dx.icd11 || "N/A")} · Urgency: ${escapeHtml(dx.urgency || "routine")}</div>
          <div class="dx-bar"><div class="dx-fill ${cls}" style="width:${pct}%"></div></div>
          ${evidenceRows ? `<details><summary>Evidence</summary><ul>${evidenceRows}</ul></details>` : ""}
          ${supportRows ? `<details><summary>Why it fits</summary><ul>${supportRows}</ul></details>` : ""}
          ${opposeRows ? `<details><summary>Why it may not fit</summary><ul>${opposeRows}</ul></details>` : ""}
          ${missingRows ? `<details><summary>Missing data</summary><ul>${missingRows}</ul></details>` : ""}
          ${componentRows ? `<details><summary>Confidence components</summary><ul>${componentRows}</ul></details>` : ""}
        </article>
      `;
    })
    .join("");

  document.getElementById("nextSteps").innerHTML = result.next_steps
    .map((step) => `<li>${escapeHtml(step)}</li>`)
    .join("") || "<li>None</li>";

  document.getElementById("warnings").innerHTML = result.drug_warnings
    .map((warning) => `<li>${escapeHtml(warning)}</li>`)
    .join("") || "<li>None</li>";

  document.getElementById("clarifications").innerHTML = result.clarifications
    .map((item) => `<li>${escapeHtml(item)}</li>`)
    .join("") || "<li>None</li>";
}

async function fetchPatients() {
  const response = await fetch("/api/patients");
  const items = await response.json();
  const select = document.getElementById("demoSelect");
  select.innerHTML = items
    .map((p) => `<option value="${escapeHtml(p.patient_id)}">${escapeHtml(p.patient_id)} - ${escapeHtml(p.complaint)}</option>`)
    .join("");
}

async function loadPatient(patientId) {
  const response = await fetch(`/api/patients/${encodeURIComponent(patientId)}`);
  if (!response.ok) {
    alert("Could not load patient");
    return;
  }
  state.patient = await response.json();
  document.getElementById("patientJson").value = JSON.stringify(state.patient, null, 2);
  renderPatientCard();
  setSessionStatus(`Loaded ${state.patient.patient_id}`);
}

function parsePatientJson() {
  try {
    const payload = JSON.parse(document.getElementById("patientJson").value.trim());
    state.patient = payload;
    renderPatientCard();
    setSessionStatus(`Loaded ${state.patient.patient_id || "CUSTOM"}`);
  } catch (error) {
    alert(`Invalid JSON: ${error}`);
  }
}

function resetRunUi() {
  state.logs = [];
  state.result = null;
  state.checklistStatus = {};
  document.getElementById("liveLog").innerHTML = "";
  document.getElementById("resultLog").innerHTML = "";
  updateProgress(0, "Starting diagnosis...");
}

function startDiagnosis() {
  if (!state.patient) {
    alert("Load or parse a patient first.");
    return;
  }
  resetRunUi();
  setView("running");
  setSessionStatus("Running");

  const protocol = location.protocol === "https:" ? "wss" : "ws";
  const ws = new WebSocket(`${protocol}://${location.host}/ws/diagnose`);
  state.ws = ws;

  ws.onopen = () => {
    const payload = state.patient.patient_id
      ? { patient_id: state.patient.patient_id }
      : { patient: state.patient };
    ws.send(JSON.stringify(payload));
  };

  ws.onmessage = (event) => {
    const message = JSON.parse(event.data);
    if (message.type === "status") {
      updateProgress(message.progress ?? 0, message.phase || message.message || "Running");
      return;
    }
    if (message.type === "log") {
      appendLog(message.message);
      return;
    }
    if (message.type === "result") {
      state.result = message.data;
      renderResult(state.result);
      updateProgress(100, "Completed");
      setSessionStatus("Completed");
      setView("result");
      return;
    }
    if (message.type === "error") {
      appendLog(`ERROR: ${message.message}`);
      setSessionStatus("Error");
      alert(`Diagnosis failed: ${message.message}`);
      return;
    }
    if (message.type === "done") {
      ws.close();
    }
  };

  ws.onerror = () => {
    setSessionStatus("Error");
    appendLog("WebSocket error");
  };
}

function bindEvents() {
  document.querySelectorAll(".nav-item").forEach((item) => {
    item.addEventListener("click", () => setView(item.dataset.view));
  });
  document.getElementById("newRunBtn").addEventListener("click", () => setView("intake"));
  document.getElementById("loadDemoBtn").addEventListener("click", () => {
    loadPatient(document.getElementById("demoSelect").value);
  });
  document.getElementById("parseJsonBtn").addEventListener("click", parsePatientJson);
  document.getElementById("runBtn").addEventListener("click", startDiagnosis);
}

async function init() {
  bindEvents();
  bindWizardEvents();
  addLabRow();          // seed one empty lab row
  renderChecklist();
  await fetchPatients();
  const select = document.getElementById("demoSelect");
  if (select.value) {
    await loadPatient(select.value);
  }
}

init();

// ══════════════════════════════════════════
//  Manual Patient Entry Wizard
// ══════════════════════════════════════════

let _wizStep = 1;
const _TOTAL_STEPS = 5;

function setWizardStep(step) {
  for (let i = 1; i <= _TOTAL_STEPS; i++) {
    const p = document.getElementById(`wpage-${i}`);
    if (p) p.classList.remove("active");
  }
  const target = document.getElementById(`wpage-${step}`);
  if (target) target.classList.add("active");
  document.querySelectorAll(".wstep").forEach((el) => {
    const s = parseInt(el.dataset.step);
    el.classList.remove("active", "done");
    if (s === step) el.classList.add("active");
    else if (s < step) el.classList.add("done");
  });
  _wizStep = step;
}

function addLabRow() {
  const container = document.getElementById("labRowsContainer");
  const row = document.createElement("div");
  row.className = "lab-row";
  row.innerHTML = `
    <div class="lab-row-inner">
      <label>Test Name *</label>
      <input type="text" class="lab-name" placeholder="e.g. Hemoglobin" />
    </div>
    <div class="lab-row-inner">
      <label>Value *</label>
      <input type="number" step="any" class="lab-value" placeholder="e.g. 8.2" />
    </div>
    <div class="lab-row-inner">
      <label>Unit</label>
      <input type="text" class="lab-unit" placeholder="e.g. g/dL" />
    </div>
    <div class="lab-row-inner">
      <label>Ref Low</label>
      <input type="number" step="any" class="lab-ref-low" placeholder="e.g. 13.5" />
    </div>
    <div class="lab-row-inner">
      <label>Ref High</label>
      <input type="number" step="any" class="lab-ref-high" placeholder="e.g. 17.5" />
    </div>
    <div class="lab-check-wrap">
      <label><input type="checkbox" class="lab-abnormal" /> Abnormal</label>
      <button class="lab-del-btn" title="Remove">
        <span class="material-symbols-outlined">delete</span>
      </button>
    </div>
  `;
  row.querySelector(".lab-del-btn").addEventListener("click", () => row.remove());
  container.appendChild(row);
}

function collectLabs() {
  const labs = [];
  document.querySelectorAll("#labRowsContainer .lab-row").forEach((row) => {
    const name  = row.querySelector(".lab-name").value.trim();
    const value = parseFloat(row.querySelector(".lab-value").value);
    if (!name || isNaN(value)) return;
    const refLow  = row.querySelector(".lab-ref-low").value;
    const refHigh = row.querySelector(".lab-ref-high").value;
    labs.push({
      name,
      value,
      unit:      row.querySelector(".lab-unit").value.trim(),
      abnormal:  row.querySelector(".lab-abnormal").checked,
      ref_low:   refLow  !== "" ? parseFloat(refLow)  : null,
      ref_high:  refHigh !== "" ? parseFloat(refHigh) : null,
    });
  });
  return labs;
}

function splitComma(str) {
  return str.split(",").map((s) => s.trim()).filter(Boolean);
}

function buildVitals() {
  const v = {};
  const map = [
    ["mf_bp",     "BP"],   ["mf_hr",   "HR"],   ["mf_temp",   "Temp"],
    ["mf_spo2",   "SpO2"], ["mf_rr",   "RR"],   ["mf_weight", "Weight"],
    ["mf_height", "Height"],
  ];
  map.forEach(([id, key]) => {
    const val = document.getElementById(id).value.trim();
    if (val) v[key] = val;
  });
  return v;
}

async function saveManualPatient() {
  const statusEl = document.getElementById("manualSaveStatus");
  const btn      = document.getElementById("wSaveBtn");

  const patient_id = document.getElementById("mf_patient_id").value.trim();
  const age        = parseInt(document.getElementById("mf_age").value);
  const sex        = document.getElementById("mf_sex").value;
  const complaint  = document.getElementById("mf_complaint").value.trim();

  if (!patient_id || !complaint || isNaN(age)) {
    statusEl.className   = "save-status err";
    statusEl.textContent = "❌ Patient ID, Age, and Chief Complaint are required.";
    statusEl.style.display = "block";
    return;
  }

  const payload = {
    patient_id, age, sex, complaint,
    hpi:         document.getElementById("mf_hpi").value.trim(),
    pmh:         splitComma(document.getElementById("mf_pmh").value),
    medications: splitComma(document.getElementById("mf_medications").value),
    allergies:   splitComma(document.getElementById("mf_allergies").value),
    vitals:      buildVitals(),
    labs:        collectLabs(),
    findings:    splitComma(document.getElementById("mf_findings").value),
    absent:      splitComma(document.getElementById("mf_absent").value),
    imaging:     splitComma(document.getElementById("mf_imaging").value),
  };

  btn.disabled    = true;
  btn.textContent = "Saving…";
  statusEl.style.display = "none";

  try {
    const res = await fetch("/api/patients", {
      method:  "POST",
      headers: { "Content-Type": "application/json" },
      body:    JSON.stringify(payload),
    });
    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.detail || "Server error");
    }
    const data = await res.json();

    await fetchPatients();          // refresh dropdown so the new patient appears
    await loadPatient(data.patient_id);  // load it into state

    statusEl.className   = "save-status ok";
    statusEl.textContent = `✅ Patient ${data.patient_id} saved and loaded! Click "Run Diagnosis" to continue.`;
    statusEl.style.display = "block";

    // Auto-collapse wizard after 2 s
    setTimeout(() => {
      document.getElementById("manualWizard").style.display = "none";
      document.getElementById("manualChevron").classList.remove("open");
    }, 2000);
  } catch (err) {
    statusEl.className   = "save-status err";
    statusEl.textContent = `❌ ${err.message}`;
    statusEl.style.display = "block";
  } finally {
    btn.disabled = false;
    btn.innerHTML = '<span class="material-symbols-outlined">save</span> Save &amp; Load Patient';
  }
}

function bindWizardEvents() {
  // Toggle wizard open / closed
  document.getElementById("manualToggleBtn").addEventListener("click", () => {
    const wizard  = document.getElementById("manualWizard");
    const chevron = document.getElementById("manualChevron");
    const opening = wizard.style.display === "none";
    wizard.style.display = opening ? "block" : "none";
    chevron.classList.toggle("open", opening);
    if (opening) setWizardStep(1);
  });

  // Step navigation
  document.getElementById("wNext1").addEventListener("click", () => {
    if (!document.getElementById("mf_patient_id").value.trim() ||
        !document.getElementById("mf_complaint").value.trim() ||
        !document.getElementById("mf_age").value) {
      alert("Patient ID, Age, and Chief Complaint are required.");
      return;
    }
    setWizardStep(2);
  });
  document.getElementById("wBack2").addEventListener("click", () => setWizardStep(1));
  document.getElementById("wNext2").addEventListener("click", () => setWizardStep(3));
  document.getElementById("wBack3").addEventListener("click", () => setWizardStep(2));
  document.getElementById("wNext3").addEventListener("click", () => setWizardStep(4));
  document.getElementById("wBack4").addEventListener("click", () => setWizardStep(3));
  document.getElementById("wNext4").addEventListener("click", () => setWizardStep(5));
  document.getElementById("wBack5").addEventListener("click", () => setWizardStep(4));
  document.getElementById("wSaveBtn").addEventListener("click", saveManualPatient);

  // Add lab row
  document.getElementById("addLabRowBtn").addEventListener("click", addLabRow);
}

