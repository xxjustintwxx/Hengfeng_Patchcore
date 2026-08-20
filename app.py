"""
Local web UI for the capture -> ROI check -> YOLO -> PatchCore pipeline.

Usage:
  python app.py --device cpu

Reuses live_infer.py's pipeline helpers directly instead of reimplementing them.
"""
import argparse
import base64
import glob
import json
import os
import shutil
import threading
import time
import uuid
from datetime import datetime

# Must happen before `ultralytics` is imported: it defaults OMP_NUM_THREADS to
# 1 on import (to limit CPU contention during its own training runs), which
# also throttles PatchCore's FAISS search and torch's CPU threading once they
# initialize their thread pools, if not set beforehand.
os.environ.setdefault("OMP_NUM_THREADS", str(os.cpu_count()))

import cv2
import numpy as np
import torch
from flask import Flask, jsonify, render_template, request, send_from_directory
from ultralytics import YOLO

import patchcore.common
import patchcore.patchcore

from capture import capture_snapshot
from live_infer import (
    CLASS_NAMES,
    SUPPRESS_VALUE,
    ROOT,
    bgr_to_tensor,
    crop_and_resize,
    load_config,
    save_result,
    _run_yolo,
    _to_colormap,
)

app = Flask(__name__)


@app.context_processor
def inject_static_versioned():
    """Cache-bust static assets by mtime, so browsers never serve a stale
    style.css/app.js after an edit without needing a manual hard refresh."""
    def static_versioned(filename):
        path = os.path.join(app.static_folder, filename)
        try:
            v = int(os.path.getmtime(path))
        except OSError:
            v = 0
        return f"/static/{filename}?v={v}"
    return {"static_versioned": static_versioned}

# Global, single-operator server state (no session/multi-client handling needed
# for this local desktop tool). No profile is loaded at startup -- the user
# picks one (640C, CT11_Power/Front, CT11_Power/Back, ...) on the Settings screen, which
# loads its YOLO+PatchCore models and swaps STATE["cfg"] wholesale so camera/
# ROI/output-dirs/class-taxonomy all switch together as one unit.
STATE = {
    "cfg": None,
    "device": None,
    "device_override": None,  # from --device; None means "use each profile's own inference.device"
    "pc_cache": {},          # patchcore_model path -> PatchCore instance
    "yolo_cache": {},        # yolo weights path -> YOLO instance
    "active_model_path": None,
    "active_config_path": None,
    "active_profile_label": None,
    "suppress_dilation": 0,
    "board_dilation": 20,
    "score_threshold": None,
    "pending_captures": {},  # capture_id -> raw_bgr
    "last_debug_detections": None,  # raw_detections from the most recent /api/infer call
    "last_debug_profile": None,
}

RESULTS_DIR = os.path.join(str(ROOT), "results")
# Narrower than RESULTS_DIR on purpose: RESULTS_DIR also holds the JSONL logs,
# dev_review/ copies, and patchcore/ (the actual trained PatchCore weights) --
# only results/live/**/*_result.jpg needs to be web-servable.
LIVE_RESULTS_DIR = os.path.join(RESULTS_DIR, "live")

GRID_MAX = 320  # cap the interactive threshold-grid's long side, in cells


def encode_jpeg_b64(bgr_img: np.ndarray) -> str:
    ok, buf = cv2.imencode(".jpg", bgr_img, [cv2.IMWRITE_JPEG_QUALITY, 90])
    if not ok:
        raise RuntimeError("cv2.imencode failed")
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode("ascii")


def _profile_label(config_path: str) -> str:
    """Derive the human-readable profile label ("640C", "CT11_Power/Front", ...) from
    a live_config.yaml path -- the path relative to configs/ with the trailing
    /live_config.yaml stripped. Shared by discover_profiles() and
    api_settings() so every caller derives it identically."""
    configs_root = os.path.join(str(ROOT), "configs")
    abs_path = config_path if os.path.isabs(config_path) else os.path.join(str(ROOT), config_path)
    return os.path.dirname(os.path.relpath(abs_path, configs_root)).replace("\\", "/")


def append_jsonl(path: str, record: dict) -> None:
    """Append one JSON-encoded record as a line to path, creating parent dirs
    as needed. Opened/closed per call -- no long-lived file handle across
    requests -- which is fine since this server runs threaded=False (single-
    threaded), so there's no concurrent-writer race to guard against."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def discover_profiles():
    """Scan configs/**/live_config.yaml for inspection profiles.

    Each config fully determines one profile: camera, ROI, YOLO weights +
    class taxonomy, PatchCore model, output dirs. Label is the path relative
    to configs/ with the trailing /live_config.yaml stripped, e.g. "640C",
    "CT11_Power/Front", "CT11_Power/Back".
    """
    pattern = os.path.join(str(ROOT), "configs", "**", "live_config.yaml")
    found = []
    for path in sorted(glob.glob(pattern, recursive=True)):
        rel_path = os.path.relpath(path, str(ROOT)).replace("\\", "/")
        label = _profile_label(path)
        try:
            cfg = load_config(path)
        except Exception:
            continue
        found.append({
            "config_path": rel_path,
            "label": label,
            "suppress_dilation": cfg.get("yolo", {}).get("suppress_dilation", 0),
            "board_dilation": cfg.get("yolo", {}).get("board_dilation", 20)
                              if cfg.get("yolo", {}).get("board_class") else 0,
            "score_threshold": cfg.get("inference", {}).get("score_threshold"),
        })
    return found


def get_patchcore(model_path: str):
    """Load (or reuse a cached) PatchCore instance for model_path."""
    cache = STATE["pc_cache"]
    if model_path in cache:
        return cache[model_path]
    nn_method = patchcore.common.FaissNN(False, os.cpu_count())
    pc = patchcore.patchcore.PatchCore(STATE["device"])
    pc.load_from_path(model_path, STATE["device"], nn_method)
    pc.eval()
    cache[model_path] = pc
    return pc


def get_yolo(weights_path: str):
    """Load (or reuse a cached) YOLO instance for weights_path."""
    cache = STATE["yolo_cache"]
    if weights_path in cache:
        return cache[weights_path]
    model = YOLO(weights_path)
    cache[weights_path] = model
    return model


def downsample_heatmap(raw_heatmap: np.ndarray, max_dim: int = GRID_MAX) -> np.ndarray:
    h, w = raw_heatmap.shape
    scale = min(1.0, max_dim / max(h, w))
    if scale >= 1.0:
        return raw_heatmap
    new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))
    # Nearest-neighbour keeps the SUPPRESS_VALUE sentinel exact (no blending
    # with real values at suppressed/non-suppressed boundaries).
    return cv2.resize(raw_heatmap, (new_w, new_h), interpolation=cv2.INTER_NEAREST)


def crop_square_for_preview(img: np.ndarray, roi, pad_factor: float = 2.0):
    """Crop a square region centered on the ROI (with padding) for the
    browser-facing preview, so it displays larger without the extreme
    portrait aspect ratio of the raw phone photo. Returns (crop, (x0, y0))
    so callers can translate ROI coordinates into the crop's frame.
    """
    h, w = img.shape[:2]
    if roi is None:
        side = min(h, w)
        cx, cy = w // 2, h // 2
    else:
        x, y, rw, rh = roi
        side = min(int(max(rw, rh) * pad_factor), min(h, w))
        cx, cy = x + rw // 2, y + rh // 2

    x0 = max(0, min(cx - side // 2, w - side))
    y0 = max(0, min(cy - side // 2, h - side))
    # .copy() -- a bare slice is a view into img, and STATE["pending_captures"]
    # holds the same raw_bgr array for the later /api/infer crop; drawing on
    # a view here would corrupt that array in place.
    return img[y0:y0 + side, x0:x0 + side].copy(), (x0, y0)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/profiles")
def api_profiles():
    return jsonify({"profiles": discover_profiles()})


@app.route("/api/settings", methods=["POST"])
def api_settings():
    data = request.get_json(force=True)
    config_path = data.get("config_path")
    if not config_path:
        return jsonify({"error": "config_path is required"}), 400

    suppress_dilation = int(data.get("suppress_dilation", 0) or 0)
    board_dilation = int(data.get("board_dilation", 0) or 0)
    score_threshold_raw = data.get("score_threshold")
    score_threshold = (
        float(score_threshold_raw) if score_threshold_raw not in (None, "") else None
    )

    try:
        cfg = load_config(config_path)
        model_path = cfg["output"]["patchcore_model"]
        yolo_weights = cfg["yolo"]["weights"]
    except Exception as e:
        return jsonify({"error": f"Failed to read config: {e}"}), 400

    device_str = STATE["device_override"] or cfg["inference"].get("device", "cuda")
    STATE["device"] = torch.device(device_str)

    t0 = time.time()
    try:
        get_patchcore(model_path)
        get_yolo(yolo_weights)
    except Exception as e:
        return jsonify({"error": f"Failed to load model: {e}"}), 400
    load_time = time.time() - t0

    STATE["cfg"] = cfg
    STATE["active_config_path"] = config_path
    STATE["active_profile_label"] = _profile_label(config_path)
    STATE["active_model_path"] = model_path
    STATE["suppress_dilation"] = suppress_dilation
    STATE["board_dilation"] = board_dilation
    STATE["score_threshold"] = score_threshold

    return jsonify({
        "config_path": config_path,
        "model_path": model_path,
        "suppress_dilation": suppress_dilation,
        "board_dilation": board_dilation,
        "score_threshold": score_threshold,
        "load_time": round(load_time, 2),
    })


@app.route("/api/capture", methods=["POST"])
def api_capture():
    if STATE["cfg"] is None:
        return jsonify({"error": "No profile selected — visit settings first"}), 400
    cam = STATE["cfg"]["camera"]
    try:
        raw_bgr = capture_snapshot(
            url=cam["url"],
            timeout=cam.get("timeout", 3.0),
            retries=cam.get("retries", 3),
            retry_delay=cam.get("retry_delay", 0.5),
            pre_delay=cam.get("pre_delay", 0.3),
        )
    except RuntimeError as e:
        print(f"[CAPTURE ERROR] {e}", flush=True)
        return jsonify({
            "error": f"Can't reach the camera at {cam['url']}. "
                     "Check that IP Webcam is running on the phone and the "
                     "USB/port-forward connection is active."
        }), 502

    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    captures_dir = STATE["cfg"]["output"]["captures_dir"]
    os.makedirs(captures_dir, exist_ok=True)
    cv2.imwrite(os.path.join(captures_dir, f"{ts}_raw.jpg"), raw_bgr)

    roi = STATE["cfg"]["preprocessing"].get("roi")
    preview = raw_bgr.copy()
    if roi is not None:
        x, y, w, h = roi
        cv2.rectangle(preview, (x, y), (x + w, y + h), (0, 255, 0), 6)
        cv2.putText(preview, "ROI", (x, y - 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 255, 0), 4, cv2.LINE_AA)
    cv2.imwrite(os.path.join(captures_dir, f"{ts}_roi_preview.jpg"), preview)

    # Browser-facing preview: crop to a square around the ROI so it displays
    # bigger without the raw phone photo's extreme portrait aspect ratio.
    browser_preview, (ox, oy) = crop_square_for_preview(raw_bgr, roi)
    if roi is not None:
        x, y, w, h = roi
        cv2.rectangle(browser_preview, (x - ox, y - oy), (x - ox + w, y - oy + h),
                      (0, 255, 0), 6)
        cv2.putText(browser_preview, "ROI", (x - ox, y - oy - 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 255, 0), 4, cv2.LINE_AA)

    capture_id = uuid.uuid4().hex
    STATE["pending_captures"][capture_id] = raw_bgr

    return jsonify({
        "capture_id": capture_id,
        "preview_image": encode_jpeg_b64(browser_preview),
    })


@app.route("/api/infer", methods=["POST"])
def api_infer():
    data = request.get_json(force=True)
    capture_id = data.get("capture_id")
    raw_bgr = STATE["pending_captures"].pop(capture_id, None)
    if raw_bgr is None:
        return jsonify({"error": "Unknown or already-consumed capture_id"}), 400

    if not STATE["active_model_path"]:
        return jsonify({"error": "No PatchCore model selected — visit settings first"}), 400

    cfg = STATE["cfg"]
    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    t_start = time.time()
    timings = {}

    roi = cfg["preprocessing"].get("roi")
    output_size = tuple(cfg["preprocessing"]["output_size"])
    t0 = time.time()
    pcb_bgr = crop_and_resize(raw_bgr, roi, output_size)
    timings["preprocess"] = round(time.time() - t0, 2)

    # board_dilation applies INSIDE _run_yolo (unlike suppress_dilation, which
    # applies externally to the already-combined mask below), so the live
    # Settings override has to be injected into the cfg before that call.
    cfg_yolo = dict(cfg.get("yolo", {}))
    cfg_yolo["board_dilation"] = STATE["board_dilation"]
    yolo_model = get_yolo(cfg_yolo["weights"])
    t0 = time.time()
    suppression, issues, soft_issues, detected_counts, yolo_bgr, raw_detections = _run_yolo(
        yolo_model, pcb_bgr, cfg_yolo
    )
    timings["yolo"] = round(time.time() - t0, 2)
    STATE["last_debug_detections"] = raw_detections
    STATE["last_debug_profile"] = STATE["active_profile_label"]

    t0 = time.time()
    pc = get_patchcore(STATE["active_model_path"])
    resize_sz = cropsize_sz = output_size[0]
    tensor = bgr_to_tensor(pcb_bgr, resize_sz, cropsize_sz).to(STATE["device"])
    scores, masks = pc._predict(tensor)
    raw_heatmap = np.array(masks[0], dtype=np.float32)
    timings["patchcore"] = round(time.time() - t0, 2)

    if suppression.shape != raw_heatmap.shape:
        suppression = cv2.resize(
            suppression.astype(np.uint8),
            (raw_heatmap.shape[1], raw_heatmap.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        ).astype(bool)
    dilation = STATE["suppress_dilation"]
    if dilation > 0:
        r = dilation
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
        suppression = cv2.dilate(suppression.astype(np.uint8), kernel).astype(bool)
    raw_heatmap[suppression] = SUPPRESS_VALUE

    valid_px = raw_heatmap[raw_heatmap >= 0]
    score = float(valid_px.max()) if valid_px.size else 0.0

    threshold = STATE["score_threshold"]
    surface_fail = threshold is not None and score >= threshold
    if threshold is not None:
        label = "NG" if (issues or surface_fail) else "OK"
    else:
        label = "NG" if issues else "LIVE"

    t0 = time.time()
    heatmaps_dir = cfg["output"]["heatmaps_dir"]
    result_path = os.path.join(heatmaps_dir, f"{ts}_result.jpg")
    try:
        save_result(pcb_bgr, yolo_bgr, raw_heatmap, score, label, issues,
                    result_path, surface_fail=surface_fail, soft_issues=soft_issues)
    except Exception:
        result_path = None

    cm = _to_colormap(raw_heatmap)
    heatmap_blend = cv2.addWeighted(pcb_bgr, 0.5, cm, 0.5, 0)
    timings["save_and_encode"] = round(time.time() - t0, 2)
    timings["total"] = round(time.time() - t_start, 2)
    print(f"[{ts}] timings: {timings}", flush=True)

    grid = downsample_heatmap(raw_heatmap)
    lo = float(valid_px.min()) if valid_px.size else 0.0
    hi = float(valid_px.max()) if valid_px.size else 1.0
    suggested_threshold = threshold if threshold is not None else float(
        np.percentile(valid_px, 95)
    ) if valid_px.size else hi

    # Always-on structured log of every inference -- independent of developer
    # mode, so score/issues/timings history exists even if dev mode is never
    # turned on.
    append_jsonl(os.path.join(RESULTS_DIR, "inference_log.jsonl"), {
        "ts": ts,
        "profile": STATE["active_profile_label"],
        "config_path": STATE["active_config_path"],
        "score": score,
        "score_threshold": threshold,
        "surface_fail": surface_fail,
        "detected_counts": detected_counts,
        "issues": issues,
        "soft_issues": soft_issues,
        "system_verdict": label,
        "result_path": result_path,
        "timings": timings,
        "suppress_dilation": STATE["suppress_dilation"],
        "board_dilation": STATE["board_dilation"],
    })

    return jsonify({
        "ts": ts,
        "profile": STATE["active_profile_label"],
        "config_path": STATE["active_config_path"],
        "system_verdict": label,
        "pcb_image": encode_jpeg_b64(pcb_bgr),
        "yolo_image": encode_jpeg_b64(yolo_bgr),
        "heatmap_image": encode_jpeg_b64(heatmap_blend),
        "grid": {
            "width": int(grid.shape[1]),
            "height": int(grid.shape[0]),
            "values": [round(float(v), 3) for v in grid.flatten()],
            "suppress_value": SUPPRESS_VALUE,
        },
        "score": score,
        "issues": issues,
        "soft_issues": soft_issues,
        "detected_counts": detected_counts,
        "range": {"min": lo, "max": hi},
        "suggested_threshold": suggested_threshold,
        "result_path": result_path,
        "timings": timings,
    })


@app.route("/api/debug_confidences")
def api_debug_confidences():
    """Raw per-class detection confidences from the most recent /api/infer
    call, for developer mode's debug panel. Served from STATE (cached right
    after _run_yolo already computed it) -- no re-inference, so this is
    near-instant and doesn't add any weight to the normal capture path."""
    if STATE["last_debug_detections"] is None:
        return jsonify({"error": "No inference has run yet this session"}), 400
    return jsonify({
        "profile": STATE["last_debug_profile"],
        "detections": STATE["last_debug_detections"],
    })


def compute_stats(profile=None):
    """Tally TP/FP/TN/FN from verification_log.jsonl, optionally filtered to
    one profile. Tolerates a malformed/truncated last line (e.g. the process
    was killed mid-write)."""
    path = os.path.join(RESULTS_DIR, "verification_log.jsonl")
    tp = fp = tn = fn = 0
    if os.path.isfile(path):
        with open(path, encoding="utf-8") as f:
            for line in f:
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if profile and r.get("profile") != profile:
                    continue
                c = r.get("classification")
                if c == "TP":
                    tp += 1
                elif c == "FP":
                    fp += 1
                elif c == "TN":
                    tn += 1
                elif c == "FN":
                    fn += 1
    total = tp + fp + tn + fn

    def safe_div(n, d):
        return round(n / d, 3) if d else None

    return {
        "tp": tp, "fp": fp, "tn": tn, "fn": fn, "total": total,
        "precision": safe_div(tp, tp + fp),
        "recall": safe_div(tp, tp + fn),
        "specificity": safe_div(tn, tn + fp),
        "accuracy": safe_div(tp + tn, total),
    }


@app.route("/api/verify", methods=["POST"])
def api_verify():
    """Record a human's ground-truth verdict for the most recent inference
    (developer mode). Fields are echoed straight back from the client's
    already-in-hand /api/infer response -- trusted with no server-side
    re-lookup, consistent with every other endpoint here (no auth, single
    local operator), and avoids racing a second capture that starts before
    this one gets verified (verification is intentionally non-blocking)."""
    data = request.get_json(force=True)
    human_verdict = data.get("human_verdict")
    system_verdict = data.get("system_verdict")
    if human_verdict not in ("OK", "NG"):
        return jsonify({"error": "human_verdict must be 'OK' or 'NG'"}), 400
    if system_verdict not in ("OK", "NG", "LIVE"):
        return jsonify({"error": "system_verdict must be 'OK', 'NG', or 'LIVE'"}), 400

    # "LIVE" means the system explicitly declined to make a surface-anomaly
    # call (no NG threshold configured) -- not "the system said OK". Log the
    # verification either way (useful for eyeballing scores while picking a
    # threshold), but only classify into TP/FP/TN/FN when the system actually
    # made a binary call, so LIVE-mode rows don't corrupt the confusion matrix.
    classification = None
    if system_verdict in ("OK", "NG"):
        system_is_ng = system_verdict == "NG"
        human_is_ng = human_verdict == "NG"
        classification = ("TP" if system_is_ng and human_is_ng else
                           "FP" if system_is_ng and not human_is_ng else
                           "FN" if not system_is_ng and human_is_ng else
                           "TN")

    profile = data.get("profile")
    record = {
        "ts": data.get("ts"),
        "profile": profile,
        "config_path": data.get("config_path"),
        "result_path": data.get("result_path"),
        "system_verdict": system_verdict,
        "human_verdict": human_verdict,
        "score": data.get("score"),
        "issues": data.get("issues"),
        "soft_issues": data.get("soft_issues"),
        "classification": classification,
        "verified_at": datetime.now().strftime("%Y%m%d_%H%M%S_%f"),
    }
    append_jsonl(os.path.join(RESULTS_DIR, "verification_log.jsonl"), record)

    # Only copy the image when classified (i.e. not LIVE mode) -- that's the
    # axis the folder structure organizes by. The JSONL record above is
    # written regardless.
    result_path = data.get("result_path")
    if result_path and classification and os.path.isfile(result_path):
        dest_dir = os.path.join(RESULTS_DIR, "dev_review", profile or "unknown", classification)
        os.makedirs(dest_dir, exist_ok=True)
        shutil.copy2(result_path, os.path.join(dest_dir, os.path.basename(result_path)))

    return jsonify({"classification": classification, **compute_stats(profile)})


@app.route("/api/stats")
def api_stats():
    return jsonify(compute_stats(request.args.get("profile")))


def result_image_url(result_path):
    """Translate a logged result_path (e.g. "./results/live/CT11_Power/Front/
    <ts>_result.jpg", relative to ROOT) into a URL servable by
    serve_result_image() below. Returns None if result_path is missing or
    doesn't resolve under LIVE_RESULTS_DIR (e.g. save_result failed at
    capture time and the log recorded a null path)."""
    if not result_path:
        return None
    abs_path = os.path.abspath(os.path.join(str(ROOT), result_path))
    try:
        rel = os.path.relpath(abs_path, LIVE_RESULTS_DIR)
    except ValueError:
        return None  # different drive on Windows -- can't be under LIVE_RESULTS_DIR
    if rel.startswith(".."):
        return None
    return "/results/live/" + rel.replace(os.sep, "/")


def read_inference_log(profile=None, limit=20):
    """Last `limit` entries from inference_log.jsonl, most recent first,
    optionally filtered to one profile, each with an added image_url.
    Full-file scan-then-slice, same pattern as compute_stats() -- confirmed
    fine at real scale (the live log is a few hundred KB after 11 days of
    real use); if this ever needs an efficient tail-read, that should happen
    once centrally for both this and compute_stats(), not bolted on here
    alone."""
    path = os.path.join(RESULTS_DIR, "inference_log.jsonl")
    entries = []
    if os.path.isfile(path):
        with open(path, encoding="utf-8") as f:
            for line in f:
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if profile and r.get("profile") != profile:
                    continue
                entries.append(r)
    entries = entries[-limit:][::-1]
    for e in entries:
        e["image_url"] = result_image_url(e.get("result_path"))
    return entries


@app.route("/api/history")
def api_history():
    profile = request.args.get("profile") or None
    limit = max(1, min(200, int(request.args.get("limit", 20))))
    return jsonify({"entries": read_inference_log(profile, limit)})


@app.route("/results/live/<path:subpath>")
def serve_result_image(subpath):
    return send_from_directory(LIVE_RESULTS_DIR, subpath)


def main():
    parser = argparse.ArgumentParser(
        description="Local web UI for YOLO + PatchCore live inference."
    )
    parser.add_argument("--device", default=None,
                        help="Override inference device for every profile: 'cuda', 'mps', or 'cpu'")
    parser.add_argument("--port", type=int, default=5000)
    args = parser.parse_args()

    # No profile is loaded at startup -- the Settings screen's "Start"/"Save"
    # loads whichever profile (640C, CT11_Power/Front, CT11_Power/Back, ...) is selected.
    STATE["device_override"] = args.device

    app.run(host="127.0.0.1", port=args.port, debug=False, threaded=False)


if __name__ == "__main__":
    main()
