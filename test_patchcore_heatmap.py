"""
Load saved PatchCore model and generate side-by-side heatmaps for test images.

Usage (from patchcore-inspection/):
  python test_patchcore_heatmap.py
  python test_patchcore_heatmap.py --model_path results/IR_Module/.../models/mvtec_ir_module
"""
import argparse
import os
import glob
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

import patchcore.backbones
import patchcore.common
import patchcore.patchcore

OUTPUT_DIR  = None  # derived from model path at runtime
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]
VALID_EXTS    = {".jpg", ".jpeg", ".png", ".bmp"}


def find_latest_model(results_root="./results"):
    """Find the most recently modified patchcore model directory."""
    pattern = os.path.join(results_root, "**", "patchcore_params.pkl")
    candidates = glob.glob(pattern, recursive=True)
    if not candidates:
        return None
    latest = max(candidates, key=os.path.getmtime)
    return os.path.dirname(latest)


def collect_test_images(data_root):
    items = []
    for f in sorted(Path(data_root, "test", "good").iterdir()):
        if f.suffix.lower() in VALID_EXTS:
            items.append((str(f), 0))
    defect_dir = Path(data_root, "test", "defect")
    if defect_dir.exists():
        for f in sorted(defect_dir.iterdir()):
            if f.suffix.lower() in VALID_EXTS:
                items.append((str(f), 1))
    return items


def preprocess(img_path, resize=256, cropsize=224):
    tf = transforms.Compose([
        transforms.Resize(resize),
        transforms.CenterCrop(cropsize),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    pil = Image.open(img_path).convert("RGB")
    return tf(pil).unsqueeze(0), pil


def pil_to_display(pil, resize=256, cropsize=224):
    """Return a BGR numpy array matching exactly what the model sees (Resize then CenterCrop)."""
    tf = transforms.Compose([
        transforms.Resize(resize),
        transforms.CenterCrop(cropsize),
    ])
    arr = np.array(tf(pil))
    return arr[:, :, ::-1].copy()


def to_colormap(anom_map, lo=None, hi=None):
    m = anom_map.astype(np.float32)
    if lo is None: lo = np.percentile(m, 1)
    if hi is None: hi = np.percentile(m, 99)
    m = np.clip((m - lo) / (hi - lo + 1e-8), 0, 1)
    return cv2.applyColorMap((m * 255).astype(np.uint8), cv2.COLORMAP_JET)


def make_colorbar(height, width=20):
    bar = np.linspace(255, 0, height, dtype=np.uint8).reshape(height, 1)
    return cv2.applyColorMap(np.repeat(bar, width, axis=1), cv2.COLORMAP_JET)


def save_heatmap(orig_bgr, anom_map, score, true_label, out_path,
                 shared_lo=None, shared_hi=None):
    h, w = orig_bgr.shape[:2]

    cm  = to_colormap(anom_map, shared_lo, shared_hi)
    cm  = cv2.resize(cm, (w, h))
    ov  = cv2.addWeighted(orig_bgr, 0.5, cm, 0.5, 0)

    is_ng       = true_label == "NG"
    label_color = (0, 0, 200) if is_ng else (0, 150, 0)
    grey        = (60, 60, 60)

    title_h, score_h, gap = 30, 34, 6

    def make_panel(img, title, sc):
        tbar = np.full((title_h, w, 3), 40, dtype=np.uint8)
        (tw, _), _ = cv2.getTextSize(title, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.putText(tbar, title, ((w - tw) // 2, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (210, 210, 210), 1, cv2.LINE_AA)
        sbar = np.full((score_h, w, 3), 255, dtype=np.uint8)
        if sc is not None:
            txt = f"score: {sc:.4f}"
            (sw, _), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 1)
            cv2.putText(sbar, txt, ((w - sw) // 2, 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.52, grey, 1, cv2.LINE_AA)
        return np.vstack([tbar, img, sbar])

    panels = [
        make_panel(orig_bgr, "Original",     None),
        make_panel(ov,       "Anomaly map",  score),
    ]
    div = np.full((panels[0].shape[0], gap, 3), 200, dtype=np.uint8)
    row = np.concatenate([panels[0], div, panels[1]], axis=1)

    # Colorbar on the right
    panel_h = panels[0].shape[0]
    cb_w, cb_pad = 20, 38
    cb_col = np.full((panel_h, cb_w + cb_pad, 3), 255, dtype=np.uint8)
    cb_img_h = panel_h - title_h - score_h - 20
    y0 = title_h + 10
    cb = make_colorbar(cb_img_h, cb_w)
    cb_col[y0:y0 + cb_img_h, 8:8 + cb_w] = cb
    # Show this image's actual value range on the colorbar (not the shared range)
    img_hi = float(np.percentile(anom_map, 99))
    img_lo = float(np.percentile(anom_map, 1))
    cv2.putText(cb_col, "High",           (2, y0 - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (60,60,60), 1, cv2.LINE_AA)
    cv2.putText(cb_col, f"{img_hi:.1f}",  (2, y0 - 6),  cv2.FONT_HERSHEY_SIMPLEX, 0.30, (100,100,100), 1, cv2.LINE_AA)
    cv2.putText(cb_col, f"{img_lo:.1f}",  (2, y0 + cb_img_h + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.30, (100,100,100), 1, cv2.LINE_AA)
    cv2.putText(cb_col, "Low",            (2, y0 + cb_img_h + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (60,60,60), 1, cv2.LINE_AA)
    row = np.concatenate([row, cb_col], axis=1)
    W = row.shape[1]

    # Verdict banner
    banner_h = 42
    banner = np.full((banner_h, W, 3), 255, dtype=np.uint8)
    txt = f"{true_label}   |   score = {score:.4f}"
    (vw, _), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_DUPLEX, 0.65, 1)
    cv2.putText(banner, txt, ((W - vw) // 2, 26),
                cv2.FONT_HERSHEY_DUPLEX, 0.65, label_color, 2, cv2.LINE_AA)
    cv2.rectangle(banner, (0, banner_h - 5), (W, banner_h), label_color, -1)

    canvas = np.vstack([row, banner])
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    cv2.imwrite(out_path, canvas)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default=None,
                        help="Path to dir containing patchcore_params.pkl (auto-detected if omitted)")
    parser.add_argument("--data_root",  type=str, default=None,
                        help="Path to ir_module/ir_module/ test folder (auto-detected from log_group if omitted)")
    parser.add_argument("--resize",    type=int, default=None,
                        help="Resize resolution (auto-detected from log_group if omitted)")
    parser.add_argument("--cropsize",  type=int, default=None,
                        help="Center crop size (defaults to resize * 224//256)")
    parser.add_argument("--device",    type=str, default="cuda")
    args = parser.parse_args()

    model_path = args.model_path or find_latest_model()
    if model_path is None:
        print("ERROR: No saved model found. Run a run_ir_module*.sh script first.")
        return
    print(f"Loading model from: {model_path}")

    # Derive output dir and resolution from log_group name
    # model_path is like: results/IR_Module/<log_group>/models/mvtec_ir_module
    log_group = Path(model_path).parts[-3]
    output_dir = os.path.join("results", "heatmaps", log_group)

    # Auto-detect resize from log_group suffix (e.g. _512 → 512, else 256)
    resize = args.resize
    if resize is None:
        parts = log_group.split("_")
        last = parts[-1]
        resize = int(last) if last.isdigit() else 256
    cropsize = args.cropsize or resize

    # Auto-detect data_root from resize
    data_root = args.data_root or \
        f"/work/xxjustin77xx/patchcore-inspection/data/ir_module_{resize}/ir_module"
    print(f"Data root:  {data_root}")
    print(f"Resize: {resize}  Cropsize: {cropsize}")

    device = torch.device(args.device)
    nn_method = patchcore.common.FaissNN(False, 4)
    pc = patchcore.patchcore.PatchCore(device)
    pc.load_from_path(model_path, device, nn_method)
    pc.eval()

    items = collect_test_images(data_root)
    if not items:
        print(f"No test images found under {DATA_ROOT}/test/")
        return

    print(f"\nRunning inference on {len(items)} images...\n")

    # First pass: collect all anomaly maps for shared colorbar scale
    all_maps, all_scores, all_labels, all_paths, all_origs = [], [], [], [], []
    for img_path, label in items:
        tensor, pil = preprocess(img_path, resize, cropsize)
        scores, masks = pc._predict(tensor.to(device))
        anom_map = np.array(masks[0])
        score    = float(scores[0])
        all_maps.append(anom_map)
        all_scores.append(score)
        all_labels.append("NG" if label else "OK")
        all_paths.append(img_path)
        all_origs.append(pil_to_display(pil, resize, cropsize))

    # Shared scale across all images so colours are comparable
    shared_lo = float(np.percentile(np.concatenate([m.ravel() for m in all_maps]), 1))
    shared_hi = float(np.percentile(np.concatenate([m.ravel() for m in all_maps]), 99))

    os.makedirs(output_dir, exist_ok=True)
    for img_path, orig_bgr, anom_map, score, label in zip(
            all_paths, all_origs, all_maps, all_scores, all_labels):
        fname = Path(img_path).stem
        out_path = os.path.join(output_dir, f"{fname}_patchcore.jpg")
        save_heatmap(orig_bgr, anom_map, score, label, out_path, shared_lo, shared_hi)
        marker = "!!" if label == "NG" else "  "
        print(f"  {marker} {Path(img_path).name:35s}  score={score:.4f}  [{label}]  → {fname}_patchcore.jpg")

    print(f"\nHeatmaps → {output_dir}/")


if __name__ == "__main__":
    main()
