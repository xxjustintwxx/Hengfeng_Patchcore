"""
Generate threshold-sweep GIFs for new defect images using the full
YOLO+PatchCore pipeline (screw suppression + dilation applied).

Sweeps the highlighted region from top 20% anomaly pixels down to the
single highest patch, so you can verify the max score comes from the
actual defect rather than screw corners.

Suppressed (screw) regions are shown as dark grey in the right panel.

Usage:
  python yolo/make_sweep_gif.py --image_dir data/ir_module_1024/ir_module/test/defect_type1
  python yolo/make_sweep_gif.py --image_dir data/ir_module_1024/ir_module/test/defect_type2
  python yolo/make_sweep_gif.py --image_dir ... --suppress_dilation 30 --fps 6
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

sys.path.insert(0, str(ROOT / "yolo"))
from infer_pcb import PCBInspector, ANOMALY_THRESH, SUPPRESS_DILATION_PX, PATCHCORE_PATH

VALID_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
THRESHOLDS = [20, 15, 12, 10, 8, 6, 5, 4, 3, 2, 1.5, 1, 0.75, 0.5, 0.25, 0.1, 0]


def make_frame(orig_bgr, anom_map, pct, score, suppress_dilation):
    h, w = orig_bgr.shape[:2]
    valid_mask = anom_map >= 0          # False where suppressed

    valid_vals = anom_map[valid_mask]
    if valid_vals.size == 0:
        return None

    if pct == 0:
        cutoff     = float(valid_vals.max())
        patch_mask = (anom_map >= cutoff)
        label_pct  = "MAX (top 1 patch)"
    else:
        cutoff     = float(np.percentile(valid_vals, 100 - pct))
        patch_mask = (anom_map >= cutoff) & valid_mask
        label_pct  = f"Top {pct:.2f}%"

    hot  = cv2.resize(patch_mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
    supp = cv2.resize((~valid_mask).astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)

    # Right panel: suppress regions → dark grey; hot pixels → red; rest → original
    right = orig_bgr.copy()
    grey  = np.full_like(orig_bgr, 40)
    red   = np.zeros_like(orig_bgr); red[:, :, 2] = 255
    right[supp == 1] = grey[supp == 1]
    right[hot  == 1] = cv2.addWeighted(orig_bgr, 0.25, red, 0.75, 0)[hot == 1]

    n_pixels = int(hot.sum())
    gap      = np.full((h, 12, 3), 50, dtype=np.uint8)
    canvas   = np.concatenate([orig_bgr, gap, right], axis=1)
    W        = canvas.shape[1]

    bar_h = 56
    bar   = np.full((bar_h, W, 3), 25, dtype=np.uint8)
    txt   = (f"{label_pct}  |  {n_pixels} px  |  cutoff≥{cutoff:.1f}"
             f"  |  score={score:.2f}  |  dilation={suppress_dilation}px")
    (tw, _), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.58, 1)
    cv2.putText(bar, txt, ((W - tw) // 2, 34),
                cv2.FONT_HERSHEY_SIMPLEX, 0.58, (200, 200, 200), 1, cv2.LINE_AA)

    label_h = 32
    labels  = np.full((label_h, W, 3), 40, dtype=np.uint8)
    for lbl, xc in [("Original", w // 2), ("Anomaly highlight (grey=suppressed)", w + 12 + w // 2)]:
        (lw, _), _ = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.62, 1)
        cv2.putText(labels, lbl, (xc - lw // 2, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.62, (180, 180, 180), 1, cv2.LINE_AA)

    return np.vstack([bar, labels, canvas])


def save_gif(frames_bgr, out_path, fps, boomerang):
    pil_frames = [Image.fromarray(f[:, :, ::-1]) for f in frames_bgr]
    if boomerang:
        pil_frames = pil_frames + pil_frames[-2:0:-1]
    duration_ms = max(1, int(1000 / fps))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pil_frames[0].save(
        str(out_path), save_all=True, append_images=pil_frames[1:],
        duration=duration_ms, loop=0, optimize=False,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir",         type=str, required=True)
    parser.add_argument("--device",            type=str, default="cuda:0")
    parser.add_argument("--suppress_dilation", type=int, default=SUPPRESS_DILATION_PX)
    parser.add_argument("--patchcore_path",    type=str, default=str(PATCHCORE_PATH),
                        help="Path to PatchCore model directory")
    parser.add_argument("--fps",               type=int, default=4)
    parser.add_argument("--no_boomerang",      action="store_true")
    args = parser.parse_args()

    image_dir  = Path(args.image_dir)
    images     = sorted(p for p in image_dir.iterdir()
                        if p.suffix.lower() in VALID_EXTS and "_aug_" not in p.name)
    if not images:
        print(f"No images found in {image_dir}")
        return

    pc_path    = Path(args.patchcore_path)
    model_name = pc_path.parent.parent.name
    out_dir    = ROOT / "results" / "gifs" / "pcb_inspection" / model_name / image_dir.name
    print(f"Images    : {len(images)} in {image_dir.name}")
    print(f"Model     : {model_name}")
    print(f"Output    : {out_dir}")
    print(f"Dilation  : {args.suppress_dilation}px\n")

    inspector = PCBInspector(device=args.device, anomaly_thresh=ANOMALY_THRESH,
                             suppress_dilation=args.suppress_dilation,
                             patchcore_path=pc_path)

    for img_path in images:
        r = inspector.run(str(img_path))
        anom_map = r["raw_heatmap"]   # already has suppression applied (sentinel=-1)
        score    = r["anomaly_score"]
        orig_bgr = r["orig_bgr"]

        frames = []
        for pct in THRESHOLDS:
            f = make_frame(orig_bgr, anom_map, pct, score, args.suppress_dilation)
            if f is not None:
                frames.append(f)

        out_path = out_dir / f"{img_path.stem}_sweep.gif"
        save_gif(frames, out_path, args.fps, not args.no_boomerang)
        print(f"  {img_path.name:35s}  score={score:.2f}  → {out_path.name}")

    print(f"\nGIFs saved to {out_dir}/")


if __name__ == "__main__":
    main()
