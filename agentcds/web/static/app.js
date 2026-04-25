const state = {
  patient: null,
  logs: [],
  result: null,
  ws: null,
  checklist: [
    "Patient loaded",
    "Initial differential formed",
    "Lab agent complete",
    "Radiology agent complete",
    "Pharmacology agent complete",
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
  document.getElementById("sessionStatus").textContent = text;
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
    [15, "Initial differential formed"],
    [20, "Lab agent complete"],
    [28, "Radiology agent complete"],
    [36, "Pharmacology agent complete"],
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
      return `<li><span>${item}</span><span class="status-${status}">${label}</span></li>`;
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
  document.getElementById("resultSummary").innerHTML = `
    <h2>Diagnostic Summary</h2>
    <p><strong>Patient:</strong> ${escapeHtml(result.patient_id)}</p>
    <p><strong>Top Diagnosis:</strong> ${escapeHtml(top.label)} (${Math.round(top.confidence * 100)}%)</p>
    <p><strong>RAG Iterations:</strong> ${result.rag_iterations}</p>
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

      return `
        <article class="dx-card">
          <div class="dx-title">
            <span>${escapeHtml(dx.label)}</span>
            <span>${pct}%</span>
          </div>
          <div class="dx-meta">ICD-11: ${escapeHtml(dx.icd11 || "N/A")} · Urgency: ${escapeHtml(dx.urgency || "routine")}</div>
          <div class="dx-bar"><div class="dx-fill ${cls}" style="width:${pct}%"></div></div>
          ${evidenceRows ? `<details><summary>Evidence</summary><ul>${evidenceRows}</ul></details>` : ""}
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
      if (message.message) {
        appendLog(message.message);
      }
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
  renderChecklist();
  await fetchPatients();
  const select = document.getElementById("demoSelect");
  if (select.value) {
    await loadPatient(select.value);
  }
}

init();
