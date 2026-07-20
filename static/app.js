const state = {
  captureId: null,
  activeSettings: null,
  lastResult: null,
};

const screens = {
  settings: document.getElementById("screen-settings"),
  capture: document.getElementById("screen-capture"),
  result: document.getElementById("screen-result"),
};

function showScreen(name) {
  for (const key of Object.keys(screens)) {
    screens[key].hidden = key !== name;
  }
}

// Where "Save"/"Cancel" on the settings screen should return to.
function postSettingsScreen() {
  return state.lastResult ? "result" : "capture";
}

// ---------------------------------------------------------------------------
// Settings screen
// ---------------------------------------------------------------------------

async function loadModels() {
  const res = await fetch("/api/models");
  const data = await res.json();
  const select = document.getElementById("model-select");
  select.innerHTML = "";
  for (const m of data.models) {
    const opt = document.createElement("option");
    opt.value = m.path;
    opt.textContent = m.label;
    select.appendChild(opt);
  }
}

document.getElementById("settings-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const statusEl = document.getElementById("settings-status");
  statusEl.textContent = "Loading model...";

  const model_path = document.getElementById("model-select").value;
  const suppress_dilation = parseInt(document.getElementById("dilation-input").value || "0", 10);
  const thresholdRaw = document.getElementById("threshold-input").value;
  const score_threshold = thresholdRaw === "" ? null : parseFloat(thresholdRaw);

  const res = await fetch("/api/settings", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ model_path, suppress_dilation, score_threshold }),
  });
  const data = await res.json();
  if (!res.ok) {
    statusEl.textContent = "Error: " + data.error;
    return;
  }

  state.activeSettings = data;
  statusEl.textContent = `Model loaded in ${data.load_time}s.`;
  updateActiveSettingsLabel();
  // Settings just apply to the *next* capture — they don't touch a result
  // that's already on screen.
  showScreen(postSettingsScreen());
});

document.getElementById("settings-cancel-btn").addEventListener("click", () => {
  document.getElementById("settings-status").textContent = "";
  showScreen(postSettingsScreen());
});

function updateActiveSettingsLabel() {
  const s = state.activeSettings;
  if (!s) return;
  const thr = s.score_threshold === null ? "none" : s.score_threshold;
  document.getElementById("active-settings").textContent =
    `Model: ${s.model_path}  |  Dilation: ${s.suppress_dilation}px  |  NG threshold: ${thr}`;
}

function populateSettingsForm() {
  const s = state.activeSettings;
  if (!s) return;
  document.getElementById("model-select").value = s.model_path;
  document.getElementById("dilation-input").value = s.suppress_dilation;
  document.getElementById("threshold-input").value =
    s.score_threshold === null ? "" : s.score_threshold;
}

// First time through (no settings applied yet), this is a "Setup" step, not
// a "Settings" page to revisit -- the heading/button read differently so it
// doesn't look like an optional detour before the real app.
function updateSettingsScreenChrome() {
  const isFirstTime = !state.activeSettings;
  document.getElementById("settings-heading").textContent = isFirstTime ? "Setup" : "Settings";
  document.getElementById("settings-save-btn").textContent = isFirstTime ? "Start" : "Save";
  document.getElementById("settings-cancel-btn").hidden = isFirstTime;
}

document.getElementById("nav-settings").addEventListener("click", () => {
  document.getElementById("settings-status").textContent = "";
  updateSettingsScreenChrome();
  populateSettingsForm();
  showScreen("settings");
});

// ---------------------------------------------------------------------------
// Capture screen
// ---------------------------------------------------------------------------

document.getElementById("capture-btn").addEventListener("click", async () => {
  const statusEl = document.getElementById("capture-status");
  statusEl.textContent = "Capturing...";
  document.getElementById("roi-preview-wrap").hidden = true;

  const res = await fetch("/api/capture", { method: "POST" });
  const data = await res.json();
  if (!res.ok) {
    statusEl.textContent = "Error: " + data.error;
    return;
  }

  state.captureId = data.capture_id;
  document.getElementById("roi-preview-img").src = data.preview_image;
  document.getElementById("roi-preview-wrap").hidden = false;
  document.getElementById("capture-btn").hidden = true;
  statusEl.textContent = "Captured. Confirm the ROI box looks right before running inference.";
});

document.getElementById("retake-btn").addEventListener("click", () => {
  document.getElementById("roi-preview-wrap").hidden = true;
  document.getElementById("capture-btn").hidden = false;
  document.getElementById("capture-status").textContent = "";
  state.captureId = null;
});

function showCaptureControls() {
  document.getElementById("capture-controls").hidden = false;
  document.getElementById("inference-progress").hidden = true;
  document.getElementById("capture-btn").hidden = false;
}

function showInferenceProgress() {
  document.getElementById("capture-controls").hidden = true;
  document.getElementById("inference-progress").hidden = false;
}

let currentInferAbort = null;

document.getElementById("cancel-inference-btn").addEventListener("click", () => {
  if (currentInferAbort) {
    currentInferAbort.abort();
  }
});

document.getElementById("run-inference-btn").addEventListener("click", async () => {
  const statusEl = document.getElementById("capture-status");
  if (!state.captureId) {
    statusEl.textContent = "No pending capture — click Capture first.";
    return;
  }

  showInferenceProgress();
  const timerEl = document.getElementById("infer-timer");
  const startTime = performance.now();
  const tick = () => {
    const elapsed = ((performance.now() - startTime) / 1000).toFixed(1);
    timerEl.textContent = `Running YOLO + PatchCore inference... ${elapsed}s`;
  };
  tick();
  const timerHandle = setInterval(tick, 100);

  currentInferAbort = new AbortController();
  let res, data, cancelled = false;
  try {
    res = await fetch("/api/infer", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ capture_id: state.captureId }),
      signal: currentInferAbort.signal,
    });
    data = await res.json();
  } catch (err) {
    if (err.name === "AbortError") {
      cancelled = true;
    } else {
      throw err;
    }
  } finally {
    clearInterval(timerHandle);
    currentInferAbort = null;
  }

  if (cancelled) {
    // The dev server can't truly interrupt a running inference — the
    // capture is already consumed server-side, and the abandoned request
    // may keep the (single-threaded) server briefly busy in the background.
    showCaptureControls();
    document.getElementById("roi-preview-wrap").hidden = true;
    state.captureId = null;
    statusEl.textContent = "Inference cancelled. Click Capture to try again.";
    return;
  }

  const totalSeconds = (performance.now() - startTime) / 1000;

  if (!res.ok) {
    showCaptureControls();
    document.getElementById("roi-preview-wrap").hidden = true;
    statusEl.textContent = "Error: " + data.error;
    return;
  }

  state.captureId = null;
  data.clientInferSeconds = totalSeconds;
  state.lastResult = data;
  renderResult(data);
  showScreen("result");
});

document.getElementById("capture-again-btn").addEventListener("click", () => {
  showCaptureControls();
  document.getElementById("roi-preview-wrap").hidden = true;
  document.getElementById("capture-status").textContent = "";
  showScreen("capture");
});

// ---------------------------------------------------------------------------
// Result screen
// ---------------------------------------------------------------------------

function renderResult(data) {
  document.getElementById("pcb-img").src = data.pcb_image;
  document.getElementById("yolo-img").src = data.yolo_image;
  document.getElementById("heatmap-img").src = data.heatmap_image;
  document.getElementById("score-value").textContent = data.score.toFixed(4);
  document.getElementById("infer-time-value").textContent =
    (data.clientInferSeconds ?? 0).toFixed(1);

  const t = data.timings || {};
  document.getElementById("timing-breakdown").textContent =
    `Server breakdown — preprocess: ${t.preprocess}s | YOLO: ${t.yolo}s | ` +
    `PatchCore: ${t.patchcore}s | save: ${t.save_and_encode}s | server total: ${t.total}s`;

  const ngThreshold = state.activeSettings ? state.activeSettings.score_threshold : null;
  document.getElementById("ng-threshold-value").textContent =
    ngThreshold === null ? "none set" : ngThreshold;

  const countsEl = document.getElementById("component-counts");
  countsEl.innerHTML = "<h3>Detected components</h3><ul>" +
    Object.entries(data.detected_counts)
      .map(([name, n]) => `<li>${name}: ${n}</li>`)
      .join("") +
    "</ul>";

  const issuesEl = document.getElementById("issues-list");
  issuesEl.innerHTML = data.issues.length
    ? "<h3>Component issues</h3><ul>" + data.issues.map((i) => `<li>${i}</li>`).join("") + "</ul>"
    : "";

  const slider = document.getElementById("threshold-slider");
  slider.min = data.range.min;
  slider.max = data.range.max;
  slider.value = data.suggested_threshold;
  updateThresholdValueLabel(slider.value);

  const pcbImg = document.getElementById("pcb-img");
  const draw = () => renderOverlay(data.grid, parseFloat(slider.value));
  if (pcbImg.complete) {
    draw();
  } else {
    pcbImg.onload = draw;
  }
  // The verdict badge always reflects the real configured NG threshold, not
  // wherever the slider happens to sit -- the slider's own range is clamped
  // to this image's min/max score, so if the NG threshold falls outside
  // that range, slider.value would silently clamp and no longer match the
  // actual threshold being evaluated.
  updateVerdict(data, data.suggested_threshold);
}

document.getElementById("threshold-slider").addEventListener("input", (e) => {
  const threshold = parseFloat(e.target.value);
  updateThresholdValueLabel(threshold);
  if (state.lastResult) {
    renderOverlay(state.lastResult.grid, threshold);
  }
});

function updateThresholdValueLabel(value) {
  document.getElementById("threshold-value").textContent = parseFloat(value).toFixed(2);
}

function updateVerdict(data, threshold) {
  const badge = document.getElementById("verdict-badge");
  const surfaceFail = data.score >= threshold;
  let label, cls;
  if (data.issues.length || surfaceFail) {
    label = "NG";
    cls = "ng";
  } else {
    label = "OK";
    cls = "ok";
  }
  badge.textContent = label;
  badge.className = "badge " + cls;
}

function renderOverlay(grid, threshold) {
  const pcbImg = document.getElementById("pcb-img");
  const canvas = document.getElementById("overlay-canvas");
  const displayW = pcbImg.clientWidth || pcbImg.naturalWidth;
  const displayH = pcbImg.clientHeight || pcbImg.naturalHeight;
  canvas.width = displayW;
  canvas.height = displayH;

  const off = document.createElement("canvas");
  off.width = grid.width;
  off.height = grid.height;
  const offCtx = off.getContext("2d");
  const imgData = offCtx.createImageData(grid.width, grid.height);

  for (let i = 0; i < grid.values.length; i++) {
    const v = grid.values[i];
    const idx = i * 4;
    if (v !== grid.suppress_value && v >= threshold) {
      imgData.data[idx] = 255;
      imgData.data[idx + 1] = 0;
      imgData.data[idx + 2] = 0;
      imgData.data[idx + 3] = 140;
    } else {
      imgData.data[idx + 3] = 0;
    }
  }
  offCtx.putImageData(imgData, 0, 0);

  const ctx = canvas.getContext("2d");
  ctx.imageSmoothingEnabled = false;
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.drawImage(off, 0, 0, canvas.width, canvas.height);
}

window.addEventListener("resize", () => {
  if (state.lastResult) {
    const slider = document.getElementById("threshold-slider");
    renderOverlay(state.lastResult.grid, parseFloat(slider.value));
  }
});

// ---------------------------------------------------------------------------
// Init
// ---------------------------------------------------------------------------

loadModels();
showScreen("settings");
