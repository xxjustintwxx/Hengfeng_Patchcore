"""
YOLOv11-seg training script for PCB component segmentation.

Trains a YOLOv11-small-seg model to detect and segment PCB components:
  0: Main IC
  1: connector
  2: resistor
  3: screw

Augmentation is tuned aggressively for small datasets (13 train images).
Trained weights land at:  models/yolo/640C/pcb_seg/weights/best.pt
Training logs land at:    results/yolo/640C/
"""
import os
import shutil
from pathlib import Path

from ultralytics import YOLO

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT        = Path(__file__).resolve().parent.parent.parent  # Hengfeng_Patchcore/
DATA_YAML   = ROOT / "data" / "PCB labeling" / "data.yaml"
PRETRAINED  = ROOT / "models" / "yolo" / "yolo11s-seg.pt"
WEIGHTS_DIR = ROOT / "models" / "yolo" / "640C" / "pcb_seg" / "weights"
RESULTS_DIR = ROOT / "results" / "yolo" / "640C"

# ── Download base weights if not present ───────────────────────────────────────
if not PRETRAINED.exists():
    print(f"Downloading yolo11s-seg.pt to {PRETRAINED} ...")
    _tmp = YOLO("yolo11s-seg.pt")          # ultralytics downloads to CWD
    downloaded = Path("yolo11s-seg.pt")
    if downloaded.exists():
        shutil.move(str(downloaded), str(PRETRAINED))
    print("Download complete.")

# ── Training ───────────────────────────────────────────────────────────────────
model = YOLO(str(PRETRAINED))

model.train(
    data    = str(DATA_YAML),
    epochs  = 300,
    imgsz   = 1280,
    batch   = 2,               # halved from 640-run: 1280px uses ~4x more GPU memory
    workers = 4,
    device  = 0,

    # Output routing
    project = str(RESULTS_DIR),
    name    = "pcb_seg_1280_s",
    exist_ok= True,

    # Optimizer
    optimizer = "AdamW",
    lr0       = 0.001,
    lrf       = 0.01,          # final lr = lr0 * lrf
    weight_decay = 0.0005,
    warmup_epochs = 10,        # longer warmup helps on tiny datasets

    # ── Augmentation (aggressive for 13-image dataset) ────────────────────────
    # Geometric
    degrees   = 45.0,          # rotation ± 45° (screws appear at all angles)
    translate = 0.1,
    scale     = 0.5,           # zoom ± 50%
    shear     = 5.0,
    perspective = 0.0005,
    flipud    = 0.5,           # vertical flip
    fliplr    = 0.5,           # horizontal flip

    # Colour / photometric
    hsv_h     = 0.015,
    hsv_s     = 0.7,
    hsv_v     = 0.4,

    # Mosaic & mix — maximise synthetic variety from 13 images
    mosaic    = 1.0,           # always mosaic 4 images together
    mixup     = 0.15,
    copy_paste= 0.3,           # paste component instances across images
    erasing   = 0.4,           # random patch erasing — helps detect partially occluded resistors

    # Turn mosaic off for the last N epochs so the model sees clean images
    close_mosaic = 30,

    # Early stopping — saves best.pt before overfitting sets in
    patience  = 50,

    # Save
    save      = True,
    save_period = 50,          # checkpoint every 50 epochs in addition to best/last
)

# ── Copy best weights to models/yolo/640C/pcb_seg/weights/ ────────────────────
trained_best = RESULTS_DIR / "pcb_seg_1280_s" / "weights" / "best.pt"
trained_last = RESULTS_DIR / "pcb_seg_1280_s" / "weights" / "last.pt"

for src in [trained_best, trained_last]:
    if src.exists():
        dst = WEIGHTS_DIR / src.name
        shutil.copy2(str(src), str(dst))
        print(f"Copied {src.name} → {dst}")

print(f"\nDone. Best weights: {WEIGHTS_DIR / 'best.pt'}")
