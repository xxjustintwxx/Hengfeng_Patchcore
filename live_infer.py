"""
Triggered live inference: press Enter → capture one photo → YOLO + PatchCore.

Pipeline:
  1. Capture     → phone camera snapshot (BGR numpy array)
  2. ROI crop + resize → PCB image at configured resolution
  3. YOLO        → detect components, check exact counts, build screw suppression mask
  4. PatchCore   → anomaly heatmap with screw regions suppressed
  5. Verdict     → NG if any YOLO count mismatch OR PatchCore score ≥ threshold
  6. Save        → raw photo + 3-panel result (YOLO overlay / anomaly map / peak location)

Usage:
  python live_infer.py --model_path results/IR_Module/<log_group>/models/mvtec_ir_module
  python live_infer.py --model_path ... --config live_config.yaml --device cpu
  python live_infer.py --model_path ... --no_yolo   # PatchCore-only (skip YOLO)
"""
import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

# ultralytics defaults OMP_NUM_THREADS to 1 on import (to limit CPU contention
# during its own training runs), which also throttles PatchCore's FAISS search
# and torch's CPU threading if not set beforehand.
os.environ.setdefault("OMP_NUM_THREADS", str(os.cpu_count()))

import cv2
import numpy as np
import torch
import yaml
import PIL.Image
from torchvision import transforms
from ultralytics import YOLO

import patchcore.common
import patchcore.patchcore

from capture import capture_snapshot

ROOT = Path(__file__).resolve().parent

CLASS_NAMES  = ["Main IC", "connecter", "resistor", "screw"]
CLASS_COLORS = {
    0: (255, 100,   0),   # Main IC   — orange
    1: (  0, 200,   0),   # connecter — green
    2: (  0, 200, 200),   # resistor  — yellow
    3: (  0,  80, 255),   # screw     — red
}
LABEL_CLASSES    = {0, 1, 3}   # resistors excluded (35 labels flood the image)
SUPPRESS_CLASSES = {3}          # screws suppressed from PatchCore heatmap
SUPPRESS_VALUE   = -1.0         # sentinel for suppressed pixels (excluded from colormap scaling)
CLASS_NMS_IOU    = {3: 0.30}    # per-class post-NMS IoU for screw deduplication
CROSS_CLASS_NMS_IOU = 0.5       # class-agnostic: when two DIFFERENT-class boxes
                                # overlap this much (e.g. a connecter and a
                                # spurious Main IC both firing on the same
                                # physical part), keep only the higher-confidence one

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def load_config(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

def crop_and_resize(img: np.ndarray, roi, output_size) -> np.ndarray:
    """Apply ROI crop then resize. Matches preprocess.py behaviour exactly."""
    if roi is not None:
        x, y, w, h = roi
        img = img[y: y + h, x: x + w]
    return cv2.resize(img, tuple(output_size), interpolation=cv2.INTER_AREA)


def bgr_to_tensor(bgr_img: np.ndarray, resize: int, cropsize: int) -> torch.Tensor:
    """BGR numpy array → normalised RGB tensor [1, 3, H, W]."""
    pil = PIL.Image.fromarray(cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB))
    tf = transforms.Compose([
        transforms.Resize(resize),
        transforms.CenterCrop(cropsize),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    return tf(pil).unsqueeze(0)


# ---------------------------------------------------------------------------
# YOLO helpers
# ---------------------------------------------------------------------------

def _per_class_nms(boxes_xyxy: np.ndarray, confs: np.ndarray,
                   cls_ids: np.ndarray, iou_thresholds: dict) -> np.ndarray:
    from torchvision.ops import nms as tv_nms
    keep = np.ones(len(cls_ids), dtype=bool)
    for cls_id, iou_thr in iou_thresholds.items():
        idx = np.where(cls_ids == cls_id)[0]
        if len(idx) <= 1:
            continue
        kept_local = tv_nms(
            torch.tensor(boxes_xyxy[idx], dtype=torch.float32),
            torch.tensor(confs[idx],      dtype=torch.float32),
            iou_thr,
        ).numpy()
        keep[idx] = False
        keep[idx[kept_local]] = True
    return keep


def _cross_class_nms(boxes_xyxy: np.ndarray, confs: np.ndarray,
                     keep_mask: np.ndarray, iou_threshold: float) -> np.ndarray:
    """Suppress lower-confidence boxes that heavily overlap a higher-confidence
    box of a DIFFERENT class. torchvision's nms doesn't look at class at all,
    so running it across every surviving box (regardless of class) is exactly
    class-agnostic dedup: only the highest-confidence box in each overlapping
    cluster survives. Only boxes that already passed _per_class_nms are considered.
    """
    from torchvision.ops import nms as tv_nms
    idx = np.where(keep_mask)[0]
    if len(idx) <= 1:
        return keep_mask
    kept_local = tv_nms(
        torch.tensor(boxes_xyxy[idx], dtype=torch.float32),
        torch.tensor(confs[idx],      dtype=torch.float32),
        iou_threshold,
    ).numpy()
    new_keep = np.zeros_like(keep_mask)
    new_keep[idx[kept_local]] = True
    return new_keep


def _draw_yolo_overlay(bgr: np.ndarray, yolo_res, class_conf: dict,
                       nms_keep: np.ndarray = None) -> np.ndarray:
    """Draw YOLO detections on bgr: filled semi-transparent masks + boxes + labels."""
    out     = bgr.copy()
    overlay = bgr.copy()
    H, W    = bgr.shape[:2]

    if yolo_res.boxes is None:
        return out

    cls_ids  = yolo_res.boxes.cls.cpu().numpy().astype(int)
    confs    = yolo_res.boxes.conf.cpu().numpy()
    xyxy     = yolo_res.boxes.xyxy.cpu().numpy().astype(int)
    masks_np = yolo_res.masks.data.cpu().numpy() if yolo_res.masks is not None else None

    for i, (cls_id, conf) in enumerate(zip(cls_ids, confs)):
        if nms_keep is not None and not nms_keep[i]:
            continue
        if conf < class_conf.get(cls_id, 0.25):
            continue
        color = CLASS_COLORS.get(cls_id, (200, 200, 200))

        if masks_np is not None:
            m = cv2.resize(masks_np[i], (W, H), interpolation=cv2.INTER_LINEAR)
            overlay[m > 0.5] = color

        x1, y1, x2, y2 = xyxy[i]
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)

        if cls_id in LABEL_CLASSES:
            label = f"{CLASS_NAMES[cls_id]} {conf:.2f}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
            cv2.rectangle(out, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
            cv2.putText(out, label, (x1 + 2, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

    cv2.addWeighted(overlay, 0.4, out, 0.6, 0, out)
    return out


def _run_yolo(yolo_model, pcb_bgr: np.ndarray, cfg_yolo: dict):
    """Run YOLO on the preprocessed PCB image (already ROI-cropped + resized).

    Returns (suppression, issues, detected_counts, yolo_bgr).
    """
    H, W    = pcb_bgr.shape[:2]
    pil_img = PIL.Image.fromarray(cv2.cvtColor(pcb_bgr, cv2.COLOR_BGR2RGB))

    yolo_conf  = cfg_yolo.get("conf", 0.25)
    yolo_iou   = cfg_yolo.get("nms_iou", 0.20)
    yolo_imgsz = cfg_yolo.get("imgsz", 1280)
    class_conf = {int(k): v for k, v in cfg_yolo.get("class_conf", {}).items()}
    expected   = {int(k): v for k, v in cfg_yolo.get("expected_counts", {}).items()}

    yolo_res     = yolo_model(pil_img, conf=yolo_conf, iou=yolo_iou,
                              imgsz=yolo_imgsz, verbose=False)[0]
    count_by_cls = {i: 0 for i in range(len(CLASS_NAMES))}
    suppression  = np.zeros((H, W), dtype=bool)

    nms_keep = np.ones(0, dtype=bool)
    if yolo_res.boxes is not None:
        cls_ids    = yolo_res.boxes.cls.cpu().numpy().astype(int)
        confs      = yolo_res.boxes.conf.cpu().numpy()
        boxes_xyxy = yolo_res.boxes.xyxy.cpu().numpy()
        nms_keep   = _per_class_nms(boxes_xyxy, confs, cls_ids, CLASS_NMS_IOU)
        nms_keep   = _cross_class_nms(boxes_xyxy, confs, nms_keep, CROSS_CLASS_NMS_IOU)

        for cls_id, conf, keep in zip(cls_ids, confs, nms_keep):
            if not keep:
                continue
            if conf >= class_conf.get(cls_id, yolo_conf):
                count_by_cls[cls_id] += 1

    if (yolo_res.masks is not None and yolo_res.boxes is not None
            and len(nms_keep)):
        masks_data = yolo_res.masks.data.cpu().numpy()
        cls_ids    = yolo_res.boxes.cls.cpu().numpy().astype(int)
        confs      = yolo_res.boxes.conf.cpu().numpy()
        for mask_raw, cls_id, conf, keep in zip(masks_data, cls_ids, confs, nms_keep):
            if not keep:
                continue
            if conf < class_conf.get(cls_id, yolo_conf):
                continue
            if cls_id in SUPPRESS_CLASSES:
                # Resize YOLO's internal mask to PatchCore resolution
                m = cv2.resize(mask_raw, (W, H), interpolation=cv2.INTER_NEAREST)
                suppression |= m > 0.5

    issues = []
    for cls_id, exp in expected.items():
        got = count_by_cls[cls_id]
        if got != exp:
            tag = "missing" if got < exp else "extra"
            issues.append(f"{CLASS_NAMES[cls_id]} ({got}/{exp} {tag})")

    yolo_bgr        = _draw_yolo_overlay(pcb_bgr, yolo_res, class_conf, nms_keep)
    detected_counts = {CLASS_NAMES[i]: count_by_cls[i] for i in range(len(CLASS_NAMES))}

    return suppression, issues, detected_counts, yolo_bgr


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def _to_colormap(anom_map, lo=None, hi=None):
    m     = anom_map.astype(np.float32)
    valid = m[m >= 0]
    lo    = float(np.percentile(valid, 1))  if lo is None else lo
    hi    = float(np.percentile(valid, 99)) if hi is None else hi
    scaled = np.clip((m - lo) / (hi - lo + 1e-8), 0, 1)
    scaled[m < 0] = 0   # suppressed pixels → blue end of JET colormap
    return cv2.applyColorMap((scaled * 255).astype(np.uint8), cv2.COLORMAP_JET)


def _make_colorbar(height, width=20):
    bar = np.linspace(255, 0, height, dtype=np.uint8).reshape(height, 1)
    return cv2.applyColorMap(np.repeat(bar, width, axis=1), cv2.COLORMAP_JET)


def save_result(orig_bgr, yolo_bgr, anom_map, score, verdict, issues,
                out_path, surface_fail=False):
    """Save 3-panel result: YOLO overlay / anomaly map / peak anomaly location."""
    h, w = orig_bgr.shape[:2]

    # Panel 1 — YOLO component detections
    p1 = cv2.resize(yolo_bgr, (w, h))

    # Panel 2 — Anomaly heatmap blended with original
    cm = _to_colormap(anom_map)
    cm = cv2.resize(cm, (w, h))
    p2 = cv2.addWeighted(orig_bgr, 0.5, cm, 0.5, 0)

    # Panel 3 — Peak anomaly location circled on dimmed original
    valid_px  = anom_map[anom_map >= 0]
    peak_val  = float(valid_px.max()) if valid_px.size else 0.0
    peak_mask = (anom_map >= peak_val)
    ys, xs   = np.where(peak_mask)
    cy = int(ys.mean() * h / anom_map.shape[0])
    cx = int(xs.mean() * w / anom_map.shape[1])
    radius  = max(20, w // 20)
    dimmed  = (orig_bgr * 0.45).astype(np.uint8)
    p3      = dimmed.copy()
    circ_m  = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(circ_m, (cx, cy), radius, 1, -1)
    p3[circ_m == 1] = orig_bgr[circ_m == 1]
    cv2.circle(p3, (cx, cy), radius,     (0, 0, 255), 3, cv2.LINE_AA)
    cv2.circle(p3, (cx, cy), radius + 4, (0, 0, 180), 1, cv2.LINE_AA)
    cv2.line(p3, (cx - 8, cy), (cx + 8, cy), (0, 0, 255), 2, cv2.LINE_AA)
    cv2.line(p3, (cx, cy - 8), (cx, cy + 8), (0, 0, 255), 2, cv2.LINE_AA)

    # Panel wrappers: title bar + image + score bar
    title_h, score_h, gap = 46, 50, 6
    grey = (60, 60, 60)

    def wrap(img, title, sc):
        tbar = np.full((title_h, w, 3), 40, dtype=np.uint8)
        (tw, _), _ = cv2.getTextSize(title, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
        cv2.putText(tbar, title, ((w - tw) // 2, 32),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (210, 210, 210), 2, cv2.LINE_AA)
        sbar = np.full((score_h, w, 3), 255, dtype=np.uint8)
        if sc is not None:
            txt = f"score: {sc:.4f}"
            (sw, _), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
            cv2.putText(sbar, txt, ((w - sw) // 2, 34),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, grey, 2, cv2.LINE_AA)
        return np.vstack([tbar, img, sbar])

    panels = [
        wrap(p1, "YOLO detections",      None),
        wrap(p2, "Anomaly map",           score),
        wrap(p3, "Peak anomaly location", score),
    ]
    div = np.full((panels[0].shape[0], gap, 3), 200, dtype=np.uint8)
    row = np.concatenate([panels[0], div, panels[1], div, panels[2]], axis=1)

    # Colorbar
    panel_h      = panels[0].shape[0]
    cb_w, cb_pad = 20, 50
    cb_col       = np.full((panel_h, cb_w + cb_pad, 3), 255, dtype=np.uint8)
    cb_img_h     = panel_h - title_h - score_h - 20
    y0           = title_h + 10
    cb           = _make_colorbar(cb_img_h, cb_w)
    cb_col[y0: y0 + cb_img_h, 8: 8 + cb_w] = cb
    _valid = anom_map[anom_map >= 0]
    img_hi = float(np.percentile(_valid, 99)) if _valid.size else 0
    img_lo = float(np.percentile(_valid,  1)) if _valid.size else 0
    cv2.putText(cb_col, "High",          (2, y0 - 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, grey, 1, cv2.LINE_AA)
    cv2.putText(cb_col, f"{img_hi:.1f}", (2, y0 -  6), cv2.FONT_HERSHEY_SIMPLEX, 0.50, grey, 1, cv2.LINE_AA)
    cv2.putText(cb_col, f"{img_lo:.1f}", (2, y0 + cb_img_h + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.50, grey, 1, cv2.LINE_AA)
    cv2.putText(cb_col, "Low",           (2, y0 + cb_img_h + 32), cv2.FONT_HERSHEY_SIMPLEX, 0.55, grey, 1, cv2.LINE_AA)
    row = np.concatenate([row, cb_col], axis=1)
    W_tot = row.shape[1]

    # Verdict banner
    if verdict == "NG":
        colour = (0, 0, 200)
    elif verdict == "OK":
        colour = (0, 150, 0)
    else:
        colour = (80, 80, 80)   # "LIVE" — no threshold set
    banner_h = 70
    banner   = np.full((banner_h, W_tot, 3), 255, dtype=np.uint8)
    line1    = f"{verdict}   |   score = {score:.4f}"
    parts    = []
    if issues:
        parts.append("component issues: " + ", ".join(issues))
    if surface_fail:
        parts.append("surface anomaly")
    line2 = "  |  ".join(parts)
    (vw, _), _ = cv2.getTextSize(line1, cv2.FONT_HERSHEY_DUPLEX, 1.0, 2)
    cv2.putText(banner, line1, ((W_tot - vw) // 2, 30),
                cv2.FONT_HERSHEY_DUPLEX, 1.0, colour, 2, cv2.LINE_AA)
    if line2:
        (mw, _), _ = cv2.getTextSize(line2, cv2.FONT_HERSHEY_SIMPLEX, 0.75, 2)
        cv2.putText(banner, line2, ((W_tot - mw) // 2, 58),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, colour, 2, cv2.LINE_AA)
    cv2.rectangle(banner, (0, banner_h - 6), (W_tot, banner_h), colour, -1)

    canvas = np.vstack([row, banner])
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    cv2.imwrite(out_path, canvas)


# ---------------------------------------------------------------------------
# Core inference pipeline (trigger-agnostic)
# ---------------------------------------------------------------------------

def run_inference_pipeline(yolo_model, pc, device, cfg) -> None:
    """One full cycle: capture → ROI crop+resize → YOLO → PatchCore → save.

    "Total time" sums only the processing steps below (capture, preprocess,
    YOLO, PatchCore, save) — it excludes the ROI-confirm prompt, so however
    long you spend deciding whether to proceed doesn't count toward it.
    """
    import time
    processing_time = 0.0
    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")

    # Step 1 — Capture
    cam = cfg["camera"]
    print(f"[{ts}] Step 1/5  Capturing from camera...", flush=True)
    t0 = time.time()
    try:
        raw_bgr = capture_snapshot(
            url=cam["url"],
            timeout=cam.get("timeout", 3.0),
            retries=cam.get("retries", 3),
            retry_delay=cam.get("retry_delay", 0.5),
            pre_delay=cam.get("pre_delay", 0.3),
        )
    except RuntimeError as e:
        print(f"  [CAPTURE ERROR] {e}", flush=True)
        return
    processing_time += time.time() - t0
    print(f"  Done  ({time.time()-t0:.2f}s)  — {raw_bgr.shape[1]}x{raw_bgr.shape[0]}", flush=True)

    # Save raw photo
    captures_dir = cfg["output"]["captures_dir"]
    os.makedirs(captures_dir, exist_ok=True)
    raw_path = os.path.join(captures_dir, f"{ts}_raw.jpg")
    cv2.imwrite(raw_path, raw_bgr)

    # ROI preview
    roi = cfg["preprocessing"].get("roi")
    if roi is not None:
        preview = raw_bgr.copy()
        x, y, w, h = roi
        cv2.rectangle(preview, (x, y), (x + w, y + h), (0, 255, 0), 6)
        cv2.putText(preview, "ROI", (x, y - 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 255, 0), 4, cv2.LINE_AA)
        preview_path = os.path.join(captures_dir, f"{ts}_roi_preview.jpg")
        cv2.imwrite(preview_path, preview)
        if sys.platform == "win32":
            os.startfile(preview_path)
        elif sys.platform == "darwin":
            import subprocess
            subprocess.Popen(["open", preview_path])
        else:
            import subprocess
            subprocess.Popen(["xdg-open", preview_path])
        print(f"\n  ROI preview saved → {preview_path}", flush=True)
        answer = input("  Press Enter to run inference, or type 'q' + Enter to cancel: ").strip().lower()
        if answer == "q":
            print("  Cancelled.", flush=True)
            return

    # Step 2 — ROI crop + resize
    output_size = tuple(cfg["preprocessing"]["output_size"])
    print(f"  Step 2/5  Preprocessing (ROI crop + resize to {output_size})...", flush=True)
    t0 = time.time()
    try:
        pcb_bgr = crop_and_resize(raw_bgr, roi, output_size)
    except Exception as e:
        print(f"  [PREPROCESS ERROR] {e}", flush=True)
        return
    processing_time += time.time() - t0
    print(f"  Done  ({time.time()-t0:.2f}s)", flush=True)

    # Step 3 — YOLO
    print(f"  Step 3/5  YOLO component detection...", flush=True)
    t0 = time.time()
    if yolo_model is not None:
        try:
            cfg_yolo = cfg.get("yolo", {})
            suppression, issues, detected_counts, yolo_bgr = _run_yolo(
                yolo_model, pcb_bgr, cfg_yolo
            )
        except Exception as e:
            print(f"  [YOLO ERROR] {e}", flush=True)
            suppression = np.zeros(output_size, dtype=bool)
            issues, detected_counts, yolo_bgr = [], {}, pcb_bgr
        counts_str = ", ".join(f"{k}:{v}" for k, v in detected_counts.items())
        processing_time += time.time() - t0
        print(f"  Done  ({time.time()-t0:.2f}s)  [{counts_str}]", flush=True)
        if issues:
            print(f"  [COMPONENT ISSUES] {', '.join(issues)}", flush=True)
    else:
        suppression = np.zeros(output_size, dtype=bool)
        issues, detected_counts, yolo_bgr = [], {}, pcb_bgr
        print(f"  Skipped (--no_yolo)", flush=True)

    # Step 4 — PatchCore
    print(f"  Step 4/5  PatchCore inference...", flush=True)
    t0 = time.time()
    try:
        resize_sz = cropsize_sz = output_size[0]
        tensor = bgr_to_tensor(pcb_bgr, resize_sz, cropsize_sz).to(device)
        scores, masks = pc._predict(tensor)
    except Exception as e:
        print(f"  [INFERENCE ERROR] {e}", flush=True)
        return
    processing_time += time.time() - t0
    print(f"  Done  ({time.time()-t0:.2f}s)", flush=True)

    raw_heatmap = np.array(masks[0], dtype=np.float32)

    # Apply screw suppression with optional dilation
    if suppression.shape != raw_heatmap.shape:
        suppression = cv2.resize(
            suppression.astype(np.uint8),
            (raw_heatmap.shape[1], raw_heatmap.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        ).astype(bool)
    suppress_dilation = cfg.get("yolo", {}).get("suppress_dilation", 0)
    if suppress_dilation > 0:
        r      = suppress_dilation
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*r+1, 2*r+1))
        suppression = cv2.dilate(suppression.astype(np.uint8), kernel).astype(bool)
    raw_heatmap[suppression] = SUPPRESS_VALUE

    valid_px = raw_heatmap[raw_heatmap >= 0]
    score    = float(valid_px.max()) if valid_px.size else 0.0

    # Verdict
    threshold    = cfg["inference"].get("score_threshold")
    surface_fail = (threshold is not None and score >= threshold)
    if threshold is not None:
        label = "NG" if (issues or surface_fail) else "OK"
    else:
        label = "NG" if issues else "LIVE"
    yolo_status = "YOLO: " + (", ".join(issues) if issues else "OK")
    print(f"  Score: {score:.4f}  [{label}]  {yolo_status}", flush=True)

    # Step 5 — Save result
    print(f"  Step 5/5  Saving result...", flush=True)
    t0 = time.time()
    heatmaps_dir = cfg["output"]["heatmaps_dir"]
    result_path  = os.path.join(heatmaps_dir, f"{ts}_result.jpg")
    try:
        save_result(pcb_bgr, yolo_bgr, raw_heatmap, score, label, issues,
                    result_path, surface_fail=surface_fail)
        processing_time += time.time() - t0
        print(f"  Done  ({time.time()-t0:.2f}s)", flush=True)
        print(f"  Raw    → {raw_path}", flush=True)
        print(f"  Result → {result_path}", flush=True)
    except Exception as e:
        print(f"  [SAVE ERROR] {e}", flush=True)
    print(f"  Total time: {processing_time:.2f}s\n", flush=True)


# ---------------------------------------------------------------------------
# Trigger implementations
# ---------------------------------------------------------------------------

def cli_trigger_loop(callback) -> None:
    """Press Enter to fire one capture+infer cycle. Ctrl+C to quit."""
    while True:
        try:
            input("Press Enter to capture + infer. Ctrl+C to quit.\n")
        except KeyboardInterrupt:
            print("\nExiting.", flush=True)
            sys.exit(0)
        callback()


def run_trigger_loop(trigger_fn, callback) -> None:
    trigger_fn(callback)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Live YOLO + PatchCore inference triggered by key-press."
    )
    parser.add_argument(
        "--model_path", required=True,
        help="Path to dir containing patchcore_params.pkl "
             "(e.g. results/IR_Module/<log_group>/models/mvtec_ir_module)"
    )
    parser.add_argument(
        "--config", default="live_config.yaml",
        help="Path to YAML config (default: live_config.yaml)"
    )
    parser.add_argument(
        "--device", default=None,
        help="Override inference device: 'cuda', 'mps', or 'cpu'"
    )
    parser.add_argument(
        "--no_yolo", action="store_true",
        help="Skip YOLO step and run PatchCore only"
    )
    parser.add_argument(
        "--suppress_dilation", type=int, default=None,
        help="Override yolo.suppress_dilation: expand screw mask edge outward "
             "by N px (e.g. 20 if background screw halos appear in the heatmap)"
    )
    args = parser.parse_args()

    cfg = load_config(args.config)

    device_str = args.device or cfg["inference"].get("device", "cuda")
    device     = torch.device(device_str)

    if args.suppress_dilation is not None:
        cfg.setdefault("yolo", {})["suppress_dilation"] = args.suppress_dilation

    # Load YOLO
    yolo_model = None
    if not args.no_yolo:
        cfg_yolo   = cfg.get("yolo", {})
        yolo_path  = cfg_yolo.get("weights",
                                  str(ROOT / "models/yolo/pcb_seg/weights/best.pt"))
        print(f"Loading YOLO from: {yolo_path}", flush=True)
        yolo_model = YOLO(yolo_path)
        print("YOLO ready.", flush=True)

    # Load PatchCore
    print(f"Loading PatchCore from: {args.model_path}", flush=True)
    nn_method = patchcore.common.FaissNN(False, os.cpu_count())
    pc        = patchcore.patchcore.PatchCore(device)
    pc.load_from_path(args.model_path, device, nn_method)
    pc.eval()
    print(f"PatchCore ready on {device}.\n", flush=True)

    run_trigger_loop(
        trigger_fn=cli_trigger_loop,
        callback=lambda: run_inference_pipeline(yolo_model, pc, device, cfg),
    )


if __name__ == "__main__":
    main()
