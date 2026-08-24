const state = {
  captureId: null,
  activeSettings: null,
  lastResult: null,
  devMode: localStorage.getItem("devMode") === "1",
};

const screens = {
  settings: document.getElementById("screen-settings"),
  capture: document.getElementById("screen-capture"),
  result: document.getElementById("screen-result"),
  history: document.getElementById("screen-history"),
};

// "history" is deliberately not part of the step flow -- it's a side view
// reachable any time via the header button, not one of the 3 pipeline steps.
// updateStepIndicator() below no-ops gracefully for a name it doesn't
// recognize (stepOrder.indexOf returns -1, so no step gets .active/.done).
const stepOrder = ["settings", "capture", "result"];

function showScreen(name) {
  for (const key of Object.keys(screens)) {
    screens[key].hidden = key !== name;
  }
  updateStepIndicator(name);
}

function updateStepIndicator(activeName) {
  const activeIdx = stepOrder.indexOf(activeName);
  document.querySelectorAll("#step-indicator .step").forEach((el) => {
    const idx = stepOrder.indexOf(el.dataset.step);
    el.classList.toggle("active", idx === activeIdx);
    el.classList.toggle("done", idx < activeIdx);
  });
}

// Where "Save"/"Cancel" on the settings screen should return to.
function postSettingsScreen() {
  return state.lastResult ? "result" : "capture";
}

function setStatus(el, text, isError = false) {
  el.textContent = text;
  el.classList.toggle("error", isError);
}

// ---------------------------------------------------------------------------
// Settings screen
// ---------------------------------------------------------------------------

let profiles = [];  // cached /api/profiles response, for label lookups + per-profile defaults

async function loadModels() {
  const res = await fetch("/api/profiles");
  const data = await res.json();
  profiles = data.profiles;
  const select = document.getElementById("model-select");
  select.innerHTML = "";
  for (const p of profiles) {
    const opt = document.createElement("option");
    opt.value = p.config_path;
    opt.textContent = p.label;
    select.appendChild(opt);
  }

  // History's filter is keyed by profile *label* (e.g. "CT11_Power/Front"), since
  // that's the field inference_log.jsonl entries are tagged with -- not
  // config_path, which #model-select above uses.
  const historySelect = document.getElementById("history-profile-select");
  historySelect.innerHTML = '<option value="">All modules</option>';
  for (const p of profiles) {
    const opt = document.createElement("option");
    opt.value = p.label;
    opt.textContent = p.label;
    historySelect.appendChild(opt);
  }

  applyProfileDefaults();
}

// Switching modules should switch everything about them, including the
// dilation/NG-threshold starting point -- otherwise picking CT11_Power while the
// form still shows 640C's leftover values silently applies the wrong
// threshold (each profile's own values were calibrated separately).
function applyProfileDefaults() {
  const configPath = document.getElementById("model-select").value;
  const p = profiles.find((p) => p.config_path === configPath);
  if (!p) return;
  document.getElementById("dilation-input").value = p.suppress_dilation;
  document.getElementById("board-dilation-input").value = p.board_dilation;
  document.getElementById("threshold-input").value =
    p.score_threshold === null ? "" : p.score_threshold;
}

document.getElementById("model-select").addEventListener("change", applyProfileDefaults);

const LAST_SETTINGS_KEY = "lastSettings";

// Shared by the manual Start/Save submit and the startup auto-restore --
// posts to /api/settings and, on success, updates state + remembers which
// profile was picked for next time. Doesn't touch #settings-status or
// navigate itself: the two callers want different status text and
// next-screen behavior. Only config_path is persisted, not the numeric
// fields -- those always come fresh from the config file on restore (see
// restoreLastSettings), so a hand-edited live_config.yaml is never shadowed
// by a stale cached number.
async function applySettings(payload) {
  const res = await fetch("/api/settings", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  const data = await res.json();
  if (!res.ok) {
    return { ok: false, error: data.error };
  }
  state.activeSettings = data;
  localStorage.setItem(LAST_SETTINGS_KEY, JSON.stringify({ config_path: payload.config_path }));
  updateActiveSettingsLabel();
  return { ok: true, data };
}

document.getElementById("settings-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const statusEl = document.getElementById("settings-status");
  setStatus(statusEl, "Loading model...");

  const config_path = document.getElementById("model-select").value;
  const suppress_dilation = parseInt(document.getElementById("dilation-input").value || "0", 10);
  const board_dilation = parseInt(document.getElementById("board-dilation-input").value || "0", 10);
  const thresholdRaw = document.getElementById("threshold-input").value;
  const score_threshold = thresholdRaw === "" ? null : parseFloat(thresholdRaw);

  const result = await applySettings({ config_path, suppress_dilation, board_dilation, score_threshold });
  if (!result.ok) {
    setStatus(statusEl, "Error: " + result.error, true);
    return;
  }

  setStatus(statusEl, `Model loaded in ${result.data.load_time}s.`);
  // Settings just apply to the *next* capture — they don't touch a result
  // that's already on screen.
  showScreen(postSettingsScreen());
});

// On startup, pre-fill the Setup form with whatever profile was last
// successfully applied (if any), so a page reload mid-shift doesn't force
// re-picking the module. The dilation/threshold fields are always populated
// from that profile's *current* config_path via applyProfileDefaults() --
// not a cached value -- so hand-editing live_config.yaml and reloading is
// reflected immediately. Deliberately does NOT auto-load the model or jump
// ahead to Capture -- stays on Setup so the operator can change anything (or
// just confirm) before committing to a model load. Falls back to a plain
// empty Setup screen if there's nothing to restore or it no longer matches a
// real profile (renamed/deleted config) -- a corrupted/stale localStorage
// value must never break page load.
function restoreLastSettings() {
  const statusEl = document.getElementById("settings-status");
  let saved = null;
  try {
    saved = JSON.parse(localStorage.getItem(LAST_SETTINGS_KEY));
  } catch {
    saved = null;
  }
  if (!saved || !saved.config_path || !profiles.some((p) => p.config_path === saved.config_path)) {
    showScreen("settings");
    return;
  }

  document.getElementById("model-select").value = saved.config_path;
  // Only the *choice of profile* is restored from the cache -- the numeric
  // fields always come fresh from the config file (same as switching profiles
  // manually), so hand-editing live_config.yaml and reloading picks up the
  // new values immediately instead of reapplying a stale cached number.
  applyProfileDefaults();
  setStatus(statusEl, "Restored last-used settings — review and click Start.");
  showScreen("settings");
}

document.getElementById("settings-cancel-btn").addEventListener("click", () => {
  setStatus(document.getElementById("settings-status"), "");
  showScreen(postSettingsScreen());
});

function updateActiveSettingsLabel() {
  const s = state.activeSettings;
  if (!s) return;
  const p = profiles.find((p) => p.config_path === s.config_path);
  const thr = s.score_threshold === null ? "none" : s.score_threshold;
  document.getElementById("active-settings").textContent =
    `Module: ${p ? p.label : s.config_path}  |  Screw dilation: ${s.suppress_dilation}px  |  ` +
    `Board dilation: ${s.board_dilation}px  |  NG threshold: ${thr}`;
}

function populateSettingsForm() {
  const s = state.activeSettings;
  if (!s) return;
  document.getElementById("model-select").value = s.config_path;
  document.getElementById("dilation-input").value = s.suppress_dilation;
  document.getElementById("board-dilation-input").value = s.board_dilation;
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
  setStatus(document.getElementById("settings-status"), "");
  updateSettingsScreenChrome();
  populateSettingsForm();
  showScreen("settings");
});

// Toggling dev mode should immediately show/hide the dev panel for whatever
// result is already on screen, not just future captures.
document.getElementById("dev-mode-toggle").addEventListener("change", (e) => {
  state.devMode = e.target.checked;
  localStorage.setItem("devMode", state.devMode ? "1" : "0");
  if (state.lastResult) {
    renderDevPanel(state.lastResult);
  }
});

// ---------------------------------------------------------------------------
// Capture screen
// ---------------------------------------------------------------------------

document.getElementById("capture-btn").addEventListener("click", async () => {
  const statusEl = document.getElementById("capture-status");
  setStatus(statusEl, "Capturing...");
  document.getElementById("roi-preview-wrap").hidden = true;

  const res = await fetch("/api/capture", { method: "POST" });
  const data = await res.json();
  if (!res.ok) {
    setStatus(statusEl, "Error: " + data.error, true);
    return;
  }

  state.captureId = data.capture_id;
  document.getElementById("roi-preview-img").src = data.preview_image;
  document.getElementById("roi-preview-wrap").hidden = false;
  document.getElementById("capture-btn").hidden = true;
  document.getElementById("capture-cancel-btn").hidden = true;
  setStatus(statusEl, "Captured. Confirm the ROI box looks right before running inference.");
});

document.getElementById("retake-btn").addEventListener("click", () => {
  document.getElementById("roi-preview-wrap").hidden = true;
  document.getElementById("capture-btn").hidden = false;
  document.getElementById("capture-cancel-btn").hidden = false;
  setStatus(document.getElementById("capture-status"), "");
  state.captureId = null;
});

document.getElementById("capture-cancel-btn").addEventListener("click", () => {
  document.getElementById("roi-preview-wrap").hidden = true;
  document.getElementById("capture-btn").hidden = false;
  document.getElementById("capture-cancel-btn").hidden = false;
  setStatus(document.getElementById("capture-status"), "");
  state.captureId = null;
  // Back to whatever result you already had, or Setup if nothing's been
  // inferred yet this session.
  showScreen(state.lastResult ? "result" : "settings");
});

function showCaptureControls() {
  document.getElementById("capture-controls").hidden = false;
  document.getElementById("inference-progress").hidden = true;
  document.getElementById("capture-btn").hidden = false;
  document.getElementById("capture-cancel-btn").hidden = false;
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
    setStatus(statusEl, "No pending capture — click Capture first.", true);
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
    setStatus(statusEl, "Inference cancelled. Click Capture to try again.");
    return;
  }

  const totalSeconds = (performance.now() - startTime) / 1000;

  if (!res.ok) {
    showCaptureControls();
    document.getElementById("roi-preview-wrap").hidden = true;
    setStatus(statusEl, "Error: " + data.error, true);
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
  setStatus(document.getElementById("capture-status"), "");
  showScreen("capture");
});

// ---------------------------------------------------------------------------
// Result screen
// ---------------------------------------------------------------------------

function renderResult(data) {
  document.getElementById("result-details").hidden = true;
  const toggleBtn = document.getElementById("toggle-details-btn");
  toggleBtn.classList.remove("expanded");
  toggleBtn.querySelector("span").textContent = "More details";

  document.getElementById("pcb-img").src = data.pcb_image;
  document.getElementById("pcb-img-preview").src = data.pcb_image;
  document.getElementById("yolo-img").src = data.yolo_image;
  document.getElementById("heatmap-img").src = data.heatmap_image;
  document.getElementById("score-value").textContent = data.score.toFixed(4);
  document.getElementById("infer-time-value").textContent =
    (data.clientInferSeconds ?? 0).toFixed(1);

  const t = data.timings || {};
  document.getElementById("timing-breakdown").textContent =
    `Server breakdown — preprocess: ${t.preprocess}s | YOLO: ${t.yolo}s | ` +
    `PatchCore: ${t.patchcore}s | save: ${t.save}s | server total: ${t.total}s`;

  const ngThreshold = state.activeSettings ? state.activeSettings.score_threshold : null;
  document.getElementById("ng-threshold-value").textContent =
    ngThreshold === null ? "none set" : ngThreshold;

  const softIssues = data.soft_issues || [];

  const countsEl = document.getElementById("component-counts");
  countsEl.innerHTML = "<p class=\"eyebrow\">Detected components</p><ul class=\"chip-list\">" +
    Object.entries(data.detected_counts)
      .map(([name, n]) => {
        const hardMismatch = data.issues.some((i) => i.startsWith(name + " ("));
        const softMismatch = !hardMismatch && softIssues.some((i) => i.startsWith(name + " ("));
        const cls = hardMismatch ? " mismatch" : softMismatch ? " mismatch-warn" : "";
        return `<li class="chip${cls}">${name} ${n}</li>`;
      })
      .join("") +
    "</ul>";

  const issuesEl = document.getElementById("issues-list");
  issuesEl.innerHTML =
    (data.issues.length
      ? "<p class=\"eyebrow\">Component issues</p><ul>" + data.issues.map((i) => `<li>${i}</li>`).join("") + "</ul>"
      : "") +
    (softIssues.length
      ? "<p class=\"eyebrow\">Needs review</p><ul class=\"warn\">" + softIssues.map((i) => `<li>${i}</li>`).join("") + "</ul>"
      : "");

  const slider = document.getElementById("threshold-slider");
  slider.min = data.range.min;
  slider.max = data.range.max;
  slider.value = data.suggested_threshold;
  updateThresholdValueLabel(slider.value);
  updateSliderFill(slider);

  const draw = () => renderOverlay(data.grid, parseFloat(slider.value));
  for (const imgEl of [document.getElementById("pcb-img"), document.getElementById("pcb-img-preview")]) {
    if (imgEl.complete) {
      draw();
    } else {
      imgEl.onload = draw;
    }
  }
  // The verdict badge always reflects the real configured NG threshold, not
  // wherever the slider happens to sit -- the slider's own range is clamped
  // to this image's min/max score, so if the NG threshold falls outside
  // that range, slider.value would silently clamp and no longer match the
  // actual threshold being evaluated.
  updateVerdict(data, data.suggested_threshold);

  renderDevPanel(data);
}

document.getElementById("toggle-details-btn").addEventListener("click", () => {
  const details = document.getElementById("result-details");
  const btn = document.getElementById("toggle-details-btn");
  const willShow = details.hidden;
  details.hidden = !willShow;
  btn.classList.toggle("expanded", willShow);
  btn.querySelector("span").textContent = willShow ? "Hide details" : "More details";

  // The image was zero-size while its container was display:none, so the
  // overlay canvas needs to be resized/redrawn now that it can actually
  // measure the image's real rendered dimensions.
  if (willShow && state.lastResult) {
    const slider = document.getElementById("threshold-slider");
    renderOverlay(state.lastResult.grid, parseFloat(slider.value));
  }
});

document.getElementById("threshold-slider").addEventListener("input", (e) => {
  const threshold = parseFloat(e.target.value);
  updateThresholdValueLabel(threshold);
  updateSliderFill(e.target);
  if (state.lastResult) {
    renderOverlay(state.lastResult.grid, threshold);
  }
});

function updateThresholdValueLabel(value) {
  document.getElementById("threshold-value").textContent = parseFloat(value).toFixed(2);
}

function updateSliderFill(slider) {
  const min = parseFloat(slider.min), max = parseFloat(slider.max), val = parseFloat(slider.value);
  const pct = max > min ? ((val - min) / (max - min)) * 100 : 0;
  slider.style.background =
    `linear-gradient(90deg, var(--accent) ${pct}%, var(--border-strong) ${pct}%)`;
}

function checkIcon(cls) {
  if (cls === "ok") {
    return '<svg viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" stroke-width="3" stroke-linecap="round" stroke-linejoin="round"><path d="M20 6 9 17l-5-5"></path></svg>';
  }
  if (cls === "warn") {
    return '<svg viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><path d="M12 3 2 20h20L12 3z"></path><line x1="12" y1="9" x2="12" y2="14"></line><line x1="12" y1="17" x2="12" y2="17.01"></line></svg>';
  }
  return '<svg viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" stroke-width="3" stroke-linecap="round" stroke-linejoin="round"><path d="M18 6 6 18M6 6l12 12"></path></svg>';
}

function updateVerdict(data, threshold) {
  const badge = document.getElementById("verdict-badge");
  const softIssues = data.soft_issues || [];
  const surfaceFail = data.score >= threshold;
  // Only hard issues (data.issues) force NG -- a resistor-only (soft) count
  // mismatch is flagged for human review instead, with PatchCore's score as
  // the primary surface-defect signal. See soft_count_classes in live_config.yaml.
  const componentsFail = data.issues.length > 0;
  const componentsWarn = !componentsFail && softIssues.length > 0;
  const cls = (componentsFail || surfaceFail) ? "ng" : "ok";
  const label = cls === "ng" ? "NG" : "OK";

  badge.innerHTML = checkIcon(cls) + "<span>" + label + "</span>";
  badge.className = "badge badge-lg " + cls;

  const surfaceCls = surfaceFail ? "ng" : "ok";
  const surfaceEl = document.getElementById("surface-status");
  const surfaceText = surfaceFail
    ? `Flagged regions: NG — score ${data.score.toFixed(2)} ≥ ${threshold}`
    : "Flagged regions: OK";
  surfaceEl.innerHTML = checkIcon(surfaceCls) + `<span>${surfaceText}</span>`;
  surfaceEl.className = "status-chip " + surfaceCls;

  const componentsCls = componentsFail ? "ng" : componentsWarn ? "warn" : "ok";
  const componentsEl = document.getElementById("components-status");
  const componentsText = (componentsFail || componentsWarn)
    ? `Components: NG — ${[...data.issues, ...softIssues].join(", ")}`
    : "Components: OK";
  componentsEl.innerHTML = checkIcon(componentsCls) + `<span>${componentsText}</span>`;
  componentsEl.className = "status-chip " + componentsCls;

  document.getElementById("score-value").classList.toggle("score-ng", surfaceFail);
}

function renderOverlay(grid, threshold) {
  // Built once and stamped onto every img/canvas pair below -- the summary
  // view and the "More details" panel both show the same flagged regions.
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

  for (const [imgId, canvasId] of [["pcb-img", "overlay-canvas"], ["pcb-img-preview", "overlay-canvas-preview"]]) {
    const imgEl = document.getElementById(imgId);
    const canvas = document.getElementById(canvasId);
    const displayW = imgEl.clientWidth || imgEl.naturalWidth;
    const displayH = imgEl.clientHeight || imgEl.naturalHeight;
    canvas.width = displayW;
    canvas.height = displayH;

    const ctx = canvas.getContext("2d");
    ctx.imageSmoothingEnabled = false;
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(off, 0, 0, canvas.width, canvas.height);
  }
}

window.addEventListener("resize", () => {
  if (state.lastResult) {
    const slider = document.getElementById("threshold-slider");
    renderOverlay(state.lastResult.grid, parseFloat(slider.value));
  }
});

// ---------------------------------------------------------------------------
// Developer mode: ground-truth verification + debug confidence panel
// ---------------------------------------------------------------------------

function renderDevPanel(data) {
  const panel = document.getElementById("dev-panel");
  panel.hidden = !state.devMode;
  if (!state.devMode) return;

  const caption = document.getElementById("dev-verdict-caption");
  caption.textContent = data.system_verdict === "LIVE"
    ? "System verdict: LIVE — no NG threshold set. This will be recorded but won't count toward TP/FP/TN/FN."
    : `System verdict: ${data.system_verdict}`;

  const okBtn = document.getElementById("verify-ok-btn");
  const ngBtn = document.getElementById("verify-ng-btn");
  const statusEl = document.getElementById("verify-status");
  if (data.verified) {
    okBtn.disabled = true;
    ngBtn.disabled = true;
    setStatus(statusEl, data.verifiedClassification === null
      ? "Logged (unclassified — LIVE mode)."
      : `Logged as ${data.verifiedClassification}.`);
  } else {
    okBtn.disabled = false;
    ngBtn.disabled = false;
    setStatus(statusEl, "");
  }

  // Stale detections from a previous capture shouldn't linger expanded.
  const debugDetails = document.getElementById("debug-details");
  debugDetails.hidden = true;
  debugDetails.innerHTML = "";
  const debugBtn = document.getElementById("toggle-debug-btn");
  debugBtn.classList.remove("expanded");
  debugBtn.querySelector("span").textContent = "Debug: raw detections";

  refreshDevStats(data.profile);
}

function fmtRate(v) {
  return v === null ? "n/a" : (v * 100).toFixed(1) + "%";
}

async function refreshDevStats(profile) {
  const res = await fetch(`/api/stats?profile=${encodeURIComponent(profile)}`);
  const stats = await res.json();
  document.getElementById("dev-stats").textContent =
    `Session stats (${profile}) — TP: ${stats.tp}  FP: ${stats.fp}  TN: ${stats.tn}  FN: ${stats.fn}  |  ` +
    `precision: ${fmtRate(stats.precision)}  recall: ${fmtRate(stats.recall)}  accuracy: ${fmtRate(stats.accuracy)}`;
}

async function submitVerification(humanVerdict) {
  const data = state.lastResult;
  if (!data) return;
  const okBtn = document.getElementById("verify-ok-btn");
  const ngBtn = document.getElementById("verify-ng-btn");
  const statusEl = document.getElementById("verify-status");
  okBtn.disabled = true;
  ngBtn.disabled = true;
  setStatus(statusEl, "Saving...");

  const res = await fetch("/api/verify", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      ts: data.ts,
      profile: data.profile,
      config_path: data.config_path,
      result_path: data.result_path,
      system_verdict: data.system_verdict,
      score: data.score,
      issues: data.issues,
      soft_issues: data.soft_issues,
      human_verdict: humanVerdict,
    }),
  });
  const result = await res.json();
  if (!res.ok) {
    setStatus(statusEl, "Error: " + result.error, true);
    okBtn.disabled = false;
    ngBtn.disabled = false;
    return;
  }

  data.verified = true;
  data.verifiedClassification = result.classification;
  setStatus(statusEl, result.classification === null
    ? "Logged (unclassified — LIVE mode)."
    : `Logged as ${result.classification}.`);
  refreshDevStats(data.profile);
}

document.getElementById("verify-ok-btn").addEventListener("click", () => submitVerification("OK"));
document.getElementById("verify-ng-btn").addEventListener("click", () => submitVerification("NG"));

function renderDebugDetections(detections) {
  return Object.entries(detections).map(([name, dets]) => {
    // A detection only actually counts toward detected_counts (and the verdict)
    // when it BOTH clears the confidence threshold AND survives every NMS/
    // off-board stage -- match that exactly here, rather than pass_threshold
    // alone, so this header can't overstate what was really counted.
    const counted = (d) => d.pass_threshold && d.nms_kept;
    const kept = dets.filter(counted).length;
    const rows = dets.length
      ? dets.map((d) => {
          const isCounted = counted(d);
          const rowCls = isCounted ? "" : "reject";
          const badge = isCounted ? "PASS  " : (d.pass_threshold ? "DROPPED" : "reject ");
          const reasonLine = d.nms_kept
            ? ""
            : `<div class="debug-det-reason">↳ dropped — ${d.drop_reason || "NMS"}</div>`;
          return `<div class="debug-det-row ${rowCls}">` +
            `<div class="debug-det-main">` +
              `<span>${badge}</span>` +
              `<span>conf=${d.conf.toFixed(4)}</span>` +
              `<span>box=[${d.box.join(",")}]</span>` +
            `</div>` +
            reasonLine +
          `</div>`;
        }).join("")
      : `<div class="debug-det-row">0 raw detections</div>`;
    return `<div class="debug-class-block"><h4>${name} — kept ${kept}/${dets.length}</h4>${rows}</div>`;
  }).join("");
}

document.getElementById("toggle-debug-btn").addEventListener("click", async () => {
  const details = document.getElementById("debug-details");
  const btn = document.getElementById("toggle-debug-btn");
  const willShow = details.hidden;
  details.hidden = !willShow;
  btn.classList.toggle("expanded", willShow);
  btn.querySelector("span").textContent = willShow ? "Hide raw detections" : "Debug: raw detections";
  if (!willShow) return;

  details.innerHTML = "Loading...";
  const res = await fetch("/api/debug_confidences");
  const data = await res.json();
  details.innerHTML = res.ok
    ? renderDebugDetections(data.detections)
    : `<p class="status error">${data.error}</p>`;
});

// ---------------------------------------------------------------------------
// History screen
// ---------------------------------------------------------------------------

function formatTs(ts) {
  // "20260731_113659_519011" -> "2026-07-31 11:36:59"
  const m = ts.match(/^(\d{4})(\d{2})(\d{2})_(\d{2})(\d{2})(\d{2})/);
  return m ? `${m[1]}-${m[2]}-${m[3]} ${m[4]}:${m[5]}:${m[6]}` : ts;
}

function renderHistoryCard(entry) {
  const cls = entry.system_verdict === "NG" ? "ng" : entry.system_verdict === "OK" ? "ok" : "";
  const thumb = entry.image_url
    ? `<a href="${entry.image_url}" target="_blank"><img src="${entry.image_url}" alt="Result"></a>`
    : `<div class="history-card-placeholder"></div>`;
  const issuesText = entry.issues && entry.issues.length ? entry.issues.join(", ") : "";
  const softIssuesText = entry.soft_issues && entry.soft_issues.length ? entry.soft_issues.join(", ") : "";
  return `<div class="panel history-card">
    ${thumb}
    <div class="history-card-meta">
      <span class="badge ${cls}">${entry.system_verdict}</span>
      <span class="status mono">${formatTs(entry.ts)}  |  ${entry.profile}  |  score ${entry.score.toFixed(2)}</span>
      ${issuesText ? `<span class="status error">${issuesText}</span>` : ""}
      ${softIssuesText ? `<span class="status warn">review: ${softIssuesText}</span>` : ""}
    </div>
  </div>`;
}

async function loadHistory() {
  const statusEl = document.getElementById("history-status");
  const listEl = document.getElementById("history-list");
  const profile = document.getElementById("history-profile-select").value;
  setStatus(statusEl, "Loading...");

  const res = await fetch(`/api/history?limit=10${profile ? "&profile=" + encodeURIComponent(profile) : ""}`);
  const data = await res.json();
  if (!res.ok) {
    setStatus(statusEl, "Error: " + data.error, true);
    return;
  }

  setStatus(statusEl, `Last ${data.entries.length} capture${data.entries.length === 1 ? "" : "s"}` +
    (profile ? ` — ${profile}` : " — all modules"));
  listEl.innerHTML = data.entries.length
    ? data.entries.map(renderHistoryCard).join("")
    : "";
}

document.getElementById("nav-history").addEventListener("click", () => {
  const current = state.activeSettings ? state.activeSettings.profile : "";
  document.getElementById("history-profile-select").value = current;
  showScreen("history");
  loadHistory();
});

document.getElementById("history-profile-select").addEventListener("change", loadHistory);

document.getElementById("history-close-btn").addEventListener("click", () => {
  showScreen(postSettingsScreen());
});

// ---------------------------------------------------------------------------
// Keyboard-driven capture loop
// ---------------------------------------------------------------------------

// Space/Enter always does "the next obvious thing" on the Capture/Result
// screens (capture -> confirm/run inference -> capture another), so a
// high-volume operator never has to reach for the mouse. Esc cancels/retakes.
// Reads existing DOM .hidden flags as the sole source of truth -- same as
// every button handler above -- rather than tracking parallel state.
function handleCaptureLoopKey(e) {
  if (e.repeat) return;                                    // ignore key-repeat while held
  if (e.ctrlKey || e.altKey || e.metaKey) return;           // don't hijack OS/browser chords
  if (["INPUT", "SELECT", "TEXTAREA"].includes(document.activeElement.tagName)) return;
  if (!screens.settings.hidden) return;                     // no shortcut behavior on Setup

  if (e.code === "Escape") {
    if (screens.capture.hidden) return;                     // only meaningful on Capture
    e.preventDefault();
    if (!document.getElementById("inference-progress").hidden) {
      document.getElementById("cancel-inference-btn").click();      // cancel in-flight inference
    } else if (!document.getElementById("roi-preview-wrap").hidden) {
      document.getElementById("retake-btn").click();                // discard this capture, retake
    } else if (!document.getElementById("capture-cancel-btn").hidden) {
      document.getElementById("capture-cancel-btn").click();        // back out of Capture entirely
    }
    return;
  }

  if (e.code !== "Space" && e.code !== "Enter") return;
  e.preventDefault();                                       // stop Space from scrolling

  if (!screens.result.hidden) {
    document.getElementById("capture-again-btn").click();
  } else if (!screens.capture.hidden) {
    if (!document.getElementById("roi-preview-wrap").hidden) {
      document.getElementById("run-inference-btn").click();
    } else if (!document.getElementById("capture-btn").hidden) {
      document.getElementById("capture-btn").click();
    }
    // else: inference in progress -- no-op, don't interfere with the in-flight request
  }
}

document.addEventListener("keydown", handleCaptureLoopKey);

// ---------------------------------------------------------------------------
// Init
// ---------------------------------------------------------------------------

document.getElementById("dev-mode-toggle").checked = state.devMode;
loadModels().then(restoreLastSettings);
