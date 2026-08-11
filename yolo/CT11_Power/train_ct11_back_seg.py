"""
YOLOv11-seg training script for CT11_Power Back PCB component segmentation.

Classes (4):
  0: board      — full PCB outline (used for PatchCore heatmap masking)
  1: component  — expected count: 1
  2: connector  — expected count: 1
  3: resistor   — expected count: 61

Trained weights land at:  models/yolo/CT11_Power/Back/pcb_seg/weights/best.pt
Training logs land at:    results/yolo/CT11_Power/Back/
"""
import shutil
from pathlib import Path

from ultralytics import YOLO

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT        = Path(__file__).resolve().parent.parent.parent
DATA_YAML   = ROOT / "data" / "CT11_Power" / "Back" / "CT11 back_padded" / "data.yaml"
PRETRAINED  = ROOT / "models" / "yolo" / "yolo11s-seg.pt"
WEIGHTS_DIR = ROOT / "models" / "yolo" / "CT11_Power" / "Back" / "pcb_seg" / "weights"
RESULTS_DIR = ROOT / "results" / "yolo" / "CT11_Power" / "Back"
RUN_NAME    = "ct11_back_seg_1280_s"

# ── Download base weights if not present ───────────────────────────────────────
if not PRETRAINED.exists():
    print(f"Downloading yolo11m-seg.pt to {PRETRAINED} ...")
    _tmp = YOLO("yolo11s-seg.pt")
    downloaded = Path("yolo11m-seg.pt")
    if downloaded.exists():
        shutil.move(str(downloaded), str(PRETRAINED))
    print("Download complete.")

# ── Training ───────────────────────────────────────────────────────────────────
model = YOLO(str(PRETRAINED))

model.train(
    data    = str(DATA_YAML),
    epochs  = 500,
    imgsz   = 1280,
    batch   = 2,
    workers = 4,
    device  = 0,

    # Output routing
    project  = str(RESULTS_DIR),
    name     = RUN_NAME,
    exist_ok = True,

    # Optimizer
    optimizer    = "AdamW",
    lr0          = 0.001,
    lrf          = 0.01,
    weight_decay = 0.0005,
    warmup_epochs = 10,

    # ── Augmentation — toned down for dense small objects (61 resistors) ─────────
    degrees   = 10.0,
    translate = 0.03,   # reduced from 0.1 — board bottom near image edge; large translate pushes it off-screen and degrades boundary prediction
    scale     = 0.2,
    shear     = 2.0,
    perspective = 0.0005,
    flipud    = 0.5,
    fliplr    = 0.5,

    hsv_h = 0.015,
    hsv_s = 0.7,
    hsv_v = 0.4,

    mosaic     = 0.3,
    mixup      = 0.15,
    copy_paste = 0.3,
    erasing    = 0.2,

    close_mosaic = 80,
    patience     = 150,

    save        = True,
    save_period = 50,
)

# ── Copy best/last weights ─────────────────────────────────────────────────────
WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
for fname in ["best.pt", "last.pt"]:
    src = RESULTS_DIR / RUN_NAME / "weights" / fname
    if src.exists():
        shutil.copy2(str(src), str(WEIGHTS_DIR / fname))
        print(f"Copied {fname} → {WEIGHTS_DIR / fname}")

print(f"\nDone. Best weights: {WEIGHTS_DIR / 'best.pt'}")
