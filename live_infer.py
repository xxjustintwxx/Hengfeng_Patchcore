"""
Triggered live inference: press Enter → capture one photo → ROI crop → PatchCore.

Usage:
  python live_infer.py --model_path results/IR_Module/<log_group>/models/mvtec_ir_module
  python live_infer.py --model_path ... --config live_config.yaml --device cpu

The trigger loop is decoupled from the core pipeline so that the trigger source
(CLI key-press, sensor signal, motion detector, etc.) can be swapped by passing
a different trigger function to run_trigger_loop().
"""
import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml
from PIL import Image
from torchvision import transforms

import patchcore.backbones
import patchcore.common
import patchcore.patchcore

from capture import capture_snapshot

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def load_config(path: str) -> dict:
    with open(path) as f:
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
    """BGR numpy array → normalised RGB tensor [1, 3, H, W].

    Applies the same transform chain as MVTecDataset / test_patchcore_heatmap.py.
    When the image is already (cropsize × cropsize) the Resize and CenterCrop
    are no-ops; they are kept for safety in case the input size ever differs.
    """
    pil = Image.fromarray(cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB))
    tf = transforms.Compose([
        transforms.Resize(resize),
        transforms.CenterCrop(cropsize),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    return tf(pil).unsqueeze(0)


# ---------------------------------------------------------------------------
# Visualisation (mirrors test_patchcore_heatmap.py — kept self-contained)
# ---------------------------------------------------------------------------

def _to_colormap(anom_map, lo=None, hi=None):
    m = anom_map.astype(np.float32)
    if lo is None: lo = np.percentile(m, 1)
    if hi is None: hi = np.percentile(m, 99)
    m = np.clip((m - lo) / (hi - lo + 1e-8), 0, 1)
    return cv2.applyColorMap((m * 255).astype(np.uint8), cv2.COLORMAP_JET)


def _make_colorbar(height, width=20):
    bar = np.linspace(255, 0, height, dtype=np.uint8).reshape(height, 1)
    return cv2.applyColorMap(np.repeat(bar, width, axis=1), cv2.COLORMAP_JET)


def save_heatmap(orig_bgr, anom_map, score, label, out_path,
                 shared_lo=None, shared_hi=None, top_pct=5):
    """Save a 3-panel side-by-side heatmap (original / anomaly map / top-N%)."""
    h, w = orig_bgr.shape[:2]

    cm  = _to_colormap(anom_map, shared_lo, shared_hi)
    cm  = cv2.resize(cm, (w, h))
    ov  = cv2.addWeighted(orig_bgr, 0.5, cm, 0.5, 0)

    threshold = np.percentile(anom_map, 100 - top_pct)
    mask = cv2.resize((anom_map >= threshold).astype(np.uint8), (w, h),
                      interpolation=cv2.INTER_NEAREST)
    dimmed = (orig_bgr * 0.35).astype(np.uint8)
    red_overlay = dimmed.copy()
    red_overlay[mask == 1] = orig_bgr[mask == 1]
    red_highlight = np.zeros_like(orig_bgr)
    red_highlight[:, :, 2] = 180
    red_overlay[mask == 1] = cv2.addWeighted(
        orig_bgr, 0.4, red_highlight, 0.6, 0)[mask == 1]

    if label == "NG":
        label_color = (0, 0, 200)
    elif label == "OK":
        label_color = (0, 150, 0)
    else:
        label_color = (80, 80, 80)

    grey = (60, 60, 60)
    title_h, score_h, gap = 46, 50, 6

    def make_panel(img, title, sc):
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
        make_panel(orig_bgr,    "Original",               None),
        make_panel(ov,          "Anomaly map",             score),
        make_panel(red_overlay, f"Top {top_pct}% anomaly", score),
    ]
    div = np.full((panels[0].shape[0], gap, 3), 200, dtype=np.uint8)
    row = np.concatenate([panels[0], div, panels[1], div, panels[2]], axis=1)

    panel_h = panels[0].shape[0]
    cb_w, cb_pad = 20, 38
    cb_col = np.full((panel_h, cb_w + cb_pad, 3), 255, dtype=np.uint8)
    cb_img_h = panel_h - title_h - score_h - 20
    y0 = title_h + 10
    cb = _make_colorbar(cb_img_h, cb_w)
    cb_col[y0: y0 + cb_img_h, 8: 8 + cb_w] = cb
    img_hi = float(np.percentile(anom_map, 99))
    img_lo = float(np.percentile(anom_map, 1))
    cv2.putText(cb_col, "High",          (2, y0 - 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (60, 60, 60), 1, cv2.LINE_AA)
    cv2.putText(cb_col, f"{img_hi:.1f}", (2, y0 - 6),  cv2.FONT_HERSHEY_SIMPLEX, 0.50, (100, 100, 100), 1, cv2.LINE_AA)
    cv2.putText(cb_col, f"{img_lo:.1f}", (2, y0 + cb_img_h + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (100, 100, 100), 1, cv2.LINE_AA)
    cv2.putText(cb_col, "Low",           (2, y0 + cb_img_h + 32), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (60, 60, 60), 1, cv2.LINE_AA)
    row = np.concatenate([row, cb_col], axis=1)
    W = row.shape[1]

    banner_h = 60
    banner = np.full((banner_h, W, 3), 255, dtype=np.uint8)
    txt = f"{label}   |   score = {score:.4f}"
    (vw, _), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_DUPLEX, 1.0, 2)
    cv2.putText(banner, txt, ((W - vw) // 2, 38),
                cv2.FONT_HERSHEY_DUPLEX, 1.0, label_color, 2, cv2.LINE_AA)
    cv2.rectangle(banner, (0, banner_h - 6), (W, banner_h), label_color, -1)

    canvas = np.vstack([row, banner])
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    cv2.imwrite(out_path, canvas)


# ---------------------------------------------------------------------------
# Core inference pipeline (trigger-agnostic)
# ---------------------------------------------------------------------------

def run_inference_pipeline(pc, device, cfg) -> None:
    """One full cycle: capture → ROI crop → tensor → infer → save outputs.

    This function contains the complete processing logic. The trigger source
    (CLI, sensor, motion detector) calls this as a callback and does not need
    to know anything about the camera or model.
    """
    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")

    # --- Capture ---
    cam = cfg["camera"]
    print(f"[{ts}] Capturing...", flush=True)
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

    print(f"  Captured {raw_bgr.shape[1]}x{raw_bgr.shape[0]}.", flush=True)

    # --- Save raw photo ---
    captures_dir = cfg["output"]["captures_dir"]
    os.makedirs(captures_dir, exist_ok=True)
    raw_path = os.path.join(captures_dir, f"{ts}_raw.jpg")
    cv2.imwrite(raw_path, raw_bgr)

    # --- Preprocess ---
    roi = cfg["preprocessing"].get("roi")
    output_size = tuple(cfg["preprocessing"]["output_size"])
    try:
        preprocessed_bgr = crop_and_resize(raw_bgr, roi, output_size)
        resize = cropsize = output_size[0]
        tensor = bgr_to_tensor(preprocessed_bgr, resize, cropsize).to(device)
    except Exception as e:
        print(f"  [PREPROCESS ERROR] {e}", flush=True)
        return

    # --- Infer ---
    print("  Running inference...", flush=True)
    try:
        scores, masks = pc._predict(tensor)
    except Exception as e:
        print(f"  [INFERENCE ERROR] {e}", flush=True)
        return

    score    = float(scores[0])
    anom_map = np.array(masks[0])

    # --- Verdict ---
    threshold = cfg["inference"].get("score_threshold")
    if threshold is not None:
        label = "NG" if score >= threshold else "OK"
        print(f"  Score: {score:.4f}  [{label}]", flush=True)
    else:
        label = "LIVE"
        print(f"  Score: {score:.4f}", flush=True)

    # --- Save heatmap ---
    heatmaps_dir = cfg["output"]["heatmaps_dir"]
    heatmap_path = os.path.join(heatmaps_dir, f"{ts}_heatmap.jpg")
    try:
        save_heatmap(preprocessed_bgr, anom_map, score, label, heatmap_path)
        print(f"  Raw   → {raw_path}", flush=True)
        print(f"  Map   → {heatmap_path}", flush=True)
    except Exception as e:
        print(f"  [SAVE ERROR] {e}", flush=True)


# ---------------------------------------------------------------------------
# Trigger implementations
# ---------------------------------------------------------------------------

def cli_trigger_loop(callback) -> None:
    """Press Enter to fire one capture+infer cycle. Ctrl+C to quit.

    To swap in a different trigger (e.g. GPIO sensor, motion detector), replace
    this function with one that calls callback() on the desired event.
    """
    print("Ready. Press Enter to capture + infer. Ctrl+C to quit.\n", flush=True)
    while True:
        try:
            input()
        except KeyboardInterrupt:
            print("\nExiting.", flush=True)
            sys.exit(0)
        callback()


def run_trigger_loop(trigger_fn, callback) -> None:
    """Run the given trigger function, passing the inference callback to it."""
    trigger_fn(callback)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Live PatchCore inference triggered by key-press."
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
        help="Override inference device: 'cuda' or 'cpu'"
    )
    args = parser.parse_args()

    cfg = load_config(args.config)

    device_str = args.device or cfg["inference"].get("device", "cuda")
    device = torch.device(device_str)

    print(f"Loading PatchCore from: {args.model_path}", flush=True)
    nn_method = patchcore.common.FaissNN(False, 4)
    pc = patchcore.patchcore.PatchCore(device)
    pc.load_from_path(args.model_path, device, nn_method)
    pc.eval()
    print(f"Model loaded on {device}.\n", flush=True)

    run_trigger_loop(
        trigger_fn=cli_trigger_loop,
        callback=lambda: run_inference_pipeline(pc, device, cfg),
    )


if __name__ == "__main__":
    main()
