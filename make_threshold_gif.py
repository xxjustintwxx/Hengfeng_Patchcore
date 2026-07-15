"""
Generate threshold-sweep GIFs for PatchCore anomaly maps.

For each test image, sweeps the highlighted region from top 20% anomaly
pixels down to top 0.5%, showing exactly where the highest-scoring
patches concentrate. Useful for verifying that the max score comes from
the actual defect region rather than screws.

Output: results/gifs/<model_name>/<image>_sweep.gif
        Left panel = original, Right panel = red overlay at current threshold.

Usage:
  python make_threshold_gif.py
  python make_threshold_gif.py --model_path results/IR_Module/.../models/mvtec_ir_module
  python make_threshold_gif.py --fps 6 --no_boomerang
"""
import argparse
import glob
import os
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

import patchcore.backbones
import patchcore.common
import patchcore.patchcore

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]
VALID_EXTS    = {".jpg", ".jpeg", ".png", ".bmp"}

# Sweep from broad (top 20%) down to the single highest-scored patch (0 = max only)
THRESHOLDS = [20, 15, 12, 10, 8, 6, 5, 4, 3, 2, 1.5, 1, 0.75, 0.5, 0.25, 0.1, 0]


def find_latest_model(results_root="./results"):
    pattern = os.path.join(results_root, "**", "patchcore_params.pkl")
    candidates = glob.glob(pattern, recursive=True)
    if not candidates:
        return None
    return os.path.dirname(max(candidates, key=os.path.getmtime))


def collect_test_images(data_root, originals_only=True):
    items = []
    for f in sorted(Path(data_root, "test", "good").iterdir()):
        if f.suffix.lower() in VALID_EXTS:
            if originals_only and "_aug_" in f.stem:
                continue
            items.append((str(f), "OK"))
    defect_dir = Path(data_root, "test", "defect")
    if defect_dir.exists():
        for f in sorted(defect_dir.iterdir()):
            if f.suffix.lower() in VALID_EXTS:
                if originals_only and "_aug_" in f.stem:
                    continue
                items.append((str(f), "NG"))
    return items


def preprocess(img_path, resize, cropsize):
    tf = transforms.Compose([
        transforms.Resize(resize),
        transforms.CenterCrop(cropsize),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    pil = Image.open(img_path).convert("RGB")
    return tf(pil).unsqueeze(0), pil


def pil_to_bgr(pil, resize, cropsize):
    tf = transforms.Compose([
        transforms.Resize(resize),
        transforms.CenterCrop(cropsize),
    ])
    return np.array(tf(pil))[:, :, ::-1].copy()


def make_frame(orig_bgr, anom_map, pct, score):
    h, w = orig_bgr.shape[:2]

    # pct == 0 means show only the single highest-scored patch
    if pct == 0:
        patch_mask = (anom_map >= anom_map.max()).astype(np.uint8)
        threshold_val = float(anom_map.max())
        label_pct = "MAX (top 1 patch)"
    else:
        threshold_val = np.percentile(anom_map, 100 - pct)
        patch_mask = (anom_map >= threshold_val).astype(np.uint8)
        label_pct = f"Top {pct:.2f}%"

    mask     = cv2.resize(patch_mask, (w, h), interpolation=cv2.INTER_NEAREST)
    n_pixels = int(mask.sum())

    # Right panel: keep original brightness everywhere, overlay red on highlighted pixels
    overlay = orig_bgr.copy()
    red = np.zeros_like(orig_bgr)
    red[:, :, 2] = 255
    overlay[mask == 1] = cv2.addWeighted(orig_bgr, 0.25, red, 0.75, 0)[mask == 1]

    # Title bar
    gap    = np.full((h, 12, 3), 50, dtype=np.uint8)
    canvas = np.concatenate([orig_bgr, gap, overlay], axis=1)
    W      = canvas.shape[1]

    bar_h = 56
    bar   = np.full((bar_h, W, 3), 25, dtype=np.uint8)
    txt   = f"{label_pct}  |  {n_pixels} pixels  |  anomaly cutoff>={threshold_val:.2f}  |  image score={score:.2f}"
    (tw, _), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.62, 1)
    cv2.putText(bar, txt, ((W - tw) // 2, 34),
                cv2.FONT_HERSHEY_SIMPLEX, 0.62, (200, 200, 200), 1, cv2.LINE_AA)

    # Panel labels
    label_h = 32
    labels  = np.full((label_h, W, 3), 40, dtype=np.uint8)
    for txt_l, x_center in [("Original", w // 2), ("Anomaly highlight", w + 12 + w // 2)]:
        (lw, _), _ = cv2.getTextSize(txt_l, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 1)
        cv2.putText(labels, txt_l, (x_center - lw // 2, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (180, 180, 180), 1, cv2.LINE_AA)

    return np.vstack([bar, labels, canvas])


def save_gif(frames_bgr, out_path, fps, boomerang):
    # Convert BGR → RGB for PIL
    pil_frames = [Image.fromarray(f[:, :, ::-1]) for f in frames_bgr]
    if boomerang:
        # play forward then backward so the loop is smooth
        pil_frames = pil_frames + pil_frames[-2:0:-1]
    duration_ms = max(1, int(1000 / fps))
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    pil_frames[0].save(
        out_path, save_all=True, append_images=pil_frames[1:],
        duration=duration_ms, loop=0, optimize=False,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path",      type=str, default=None)
    parser.add_argument("--data_root",       type=str, default=None)
    parser.add_argument("--resize",          type=int, default=None)
    parser.add_argument("--device",          type=str, default="cuda")
    parser.add_argument("--fps",             type=int, default=4,
                        help="Frames per second (default 4 → 250ms per step)")
    parser.add_argument("--no_boomerang",    action="store_true",
                        help="Don't play in reverse on loop")
    parser.add_argument("--all_rotations",   action="store_true",
                        help="Include augmented (rotated) test images too")
    args = parser.parse_args()

    model_path = args.model_path or find_latest_model()
    if model_path is None:
        print("ERROR: No saved model found. Run a training script first.")
        return
    print(f"Loading model from: {model_path}")

    log_group  = Path(model_path).parts[-3]
    output_dir = os.path.join("results", "gifs", log_group)

    resize = args.resize
    if resize is None:
        digit_parts = [p for p in log_group.split("_") if p.isdigit()]
        resize = int(digit_parts[-1]) if digit_parts else 256
    cropsize = resize

    data_root = args.data_root or \
        f"/work/xxjustin77xx/Hengfeng_Patchcore/data/ir_module_{resize}/ir_module"
    print(f"Data root : {data_root}")
    print(f"Resize    : {resize}  Output dir: {output_dir}")

    device    = torch.device(args.device)
    nn_method = patchcore.common.FaissNN(False, 4)
    pc        = patchcore.patchcore.PatchCore(device)
    pc.load_from_path(model_path, device, nn_method)
    pc.eval()

    originals_only = not args.all_rotations
    items = collect_test_images(data_root, originals_only)
    print(f"\nGenerating {len(THRESHOLDS)}-frame GIFs for {len(items)} images "
          f"({'originals only' if originals_only else 'all including rotations'})...\n")

    os.makedirs(output_dir, exist_ok=True)

    for img_path, label in items:
        tensor, pil = preprocess(img_path, resize, cropsize)
        scores, masks = pc._predict(tensor.to(device))
        anom_map = np.array(masks[0])
        score    = float(scores[0])
        orig_bgr = pil_to_bgr(pil, resize, cropsize)

        frames = [make_frame(orig_bgr, anom_map, pct, score) for pct in THRESHOLDS]

        fname    = Path(img_path).stem
        out_path = os.path.join(output_dir, f"{fname}_sweep.gif")
        save_gif(frames, out_path, args.fps, not args.no_boomerang)

        marker = "!!" if label == "NG" else "  "
        print(f"  {marker} {Path(img_path).name:35s}  score={score:.2f}  [{label}]"
              f"  → {fname}_sweep.gif")

    print(f"\nGIFs saved to {output_dir}/")


if __name__ == "__main__":
    main()
