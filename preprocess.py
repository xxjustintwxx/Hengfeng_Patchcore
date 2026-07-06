"""
Preprocess raw images: crop ROI and resize to target resolution.

Workflow:
  1. Drop raw photos into data/raw/
  2. Preview the ROI setting:
       python preprocess.py --calibrate data/raw/sample.jpg
     → inspect calibrate_preview.jpg and calibrate_crop.jpg
  3. Adjust roi/output_size in preprocess_config.yaml, repeat step 2
  4. Batch process a split — output folder is auto-named by resolution:
       python preprocess.py --src data/raw       --split train/good
       python preprocess.py --src data/raw       --split test/good   --files Unknown-15.jpg Unknown-16.jpg Unknown-17.jpg
       python preprocess.py --src data/raw       --split test/defect --files Unknown-18.jpg
     → writes to data/ir_module_<W>x<H>/ir_module/{split}/
  5. Copy ground_truth mask manually if needed.
"""
import argparse
import sys
from pathlib import Path
import cv2
import yaml

PROJECT_ROOT = Path(__file__).parent
CONFIG_PATH  = PROJECT_ROOT / "preprocess_config.yaml"
VALID_EXTS   = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}


def load_config():
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def crop_and_resize(img, roi, output_size):
    if roi is not None:
        x, y, w, h = roi
        img = img[y : y + h, x : x + w]
    return cv2.resize(img, tuple(output_size), interpolation=cv2.INTER_AREA)


def calibrate(image_path: Path, roi, output_size):
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"ERROR: cannot read {image_path}")
        sys.exit(1)

    h, w = img.shape[:2]
    print(f"Image size: {w}x{h}")

    preview = img.copy()
    if roi is not None:
        x, y, rw, rh = roi
        cv2.rectangle(preview, (x, y), (x + rw, y + rh), (0, 255, 0), 3)
        cv2.putText(preview, f"ROI: x={x} y={y} w={rw} h={rh}", (x, max(y - 8, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cropped = crop_and_resize(img, roi, output_size)
        cv2.imwrite(str(PROJECT_ROOT / "calibrate_crop.jpg"), cropped)
        print(f"ROI applied: x={x} y={y} w={rw} h={rh}  →  {output_size[0]}x{output_size[1]}")
        print("Saved → calibrate_crop.jpg")
    else:
        print("No ROI set — full image will be used.")

    cv2.imwrite(str(PROJECT_ROOT / "calibrate_preview.jpg"), preview)
    print("Saved → calibrate_preview.jpg")


def process_one(src_path: Path, dst_path: Path, roi, output_size) -> bool:
    img = cv2.imread(str(src_path))
    if img is None:
        print(f"  SKIP (unreadable): {src_path.name}")
        return False
    result = crop_and_resize(img, roi, output_size)
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(dst_path), result)
    return True


def batch_process(src: Path, dst: Path, roi, output_size, only_files=None):
    if src.is_file():
        files = [src]
    else:
        files = sorted(f for f in src.iterdir() if f.suffix.lower() in VALID_EXTS)
        if only_files:
            keep = set(only_files)
            files = [f for f in files if f.name in keep]

    if not files:
        print(f"No valid images found in {src}")
        sys.exit(1)

    dst.mkdir(parents=True, exist_ok=True)
    ok = 0
    for f in files:
        out = dst / f.name
        if process_one(f, out, roi, output_size):
            ok += 1
            print(f"  {f.name}  →  {out.relative_to(PROJECT_ROOT)}")

    print(f"\nDone: {ok}/{len(files)} images → {dst.relative_to(PROJECT_ROOT)}")


def main():
    parser = argparse.ArgumentParser(description="Preprocess raw IR module images for PatchCore")
    parser.add_argument("--calibrate", metavar="IMAGE", help="Preview ROI on a sample image")
    parser.add_argument("--src",   metavar="PATH",  help="Source image or folder (e.g. data/raw)")
    parser.add_argument("--split", metavar="SPLIT", help="Dataset split subfolder (e.g. train/good, test/defect)")
    parser.add_argument("--files", metavar="FILE", nargs="+", help="Only process these filenames (from --src folder)")
    args = parser.parse_args()

    cfg         = load_config()
    roi         = cfg["preprocessing"]["roi"]
    output_size = tuple(cfg["preprocessing"]["output_size"])
    w, h        = output_size

    if args.calibrate:
        calibrate(Path(args.calibrate), roi, output_size)
    elif args.src and args.split:
        # Auto-name destination by resolution: data/ir_module_<W>x<H>/ir_module/<split>/
        dst = PROJECT_ROOT / "data" / f"ir_module_{w}" / "ir_module" / args.split
        batch_process(PROJECT_ROOT / args.src, dst, roi, output_size, args.files)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
