# Hengfeng PCB Anomaly Detection

Two-stage automated visual inspection for Hengfeng IR module PCBs:

1. **YOLOv11-seg** — detect components and verify exact counts (missing / extra → NG)
2. **PatchCore (WideResNet50)** — surface anomaly heatmap with screw regions suppressed

**Verdict:** NG if YOLO count mismatch **or** PatchCore score ≥ threshold.

---

## Pipeline Overview

```
Phone camera (IP Webcam)
        ↓
  ROI crop + resize to 1024×1024   ← preprocess_config.yaml
        ↓
  YOLOv11-seg component detection
    • count check (Main IC=1, connector=4, resistor=35, screw=4)
    • build screw mask for suppression
        ↓
  PatchCore anomaly scoring
    • screw regions zeroed out before scoring
    • score = max patch distance over non-screw pixels
        ↓
  Verdict + 3-panel result image saved
```

---

## Directory Structure

```
data/
  640C/                         # IR module 640C (current production module)
    raw/                        # original photos (renamed with _ws/_ns suffix)
    ir_module_1024/ir_module/
      train/good/               # training images + augmented variants
      test/good/                # held-out good boards
      test/defect/              # defect boards (with screw, _ws)
      test/defect_type1/        # defect boards (no screw, _ns)
      test/defect_type2/        # defect boards (no screw, _ns, different defect type)
      ground_truth/             # pixel-level masks (blank = no annotation)
  CT11/                         # CT11 module (new — to be populated)
    raw/
    ir_module_1024/

models/
  yolo/
    640C/pcb_seg/weights/best.pt  # trained YOLOv11s-seg (small-1280) for 640C
    CT11/                         # (to be trained)
    yolo11s-seg.pt                # base weights used for YOLO training

results/
  IR_Module/
    640C/
      ir_module_WR50_L2-3_PS3_1024_m4_p0.1/   # ← current best PatchCore model
    CT11/
  live/
    640C/                       # live inference result images
    CT11/

configs/
  640C/
    preprocess_config.yaml      # ROI crop + output resolution
    live_config.yaml            # camera URL, YOLO weights, score threshold, output dirs
  CT11/

notes/
  640C/
    model_experiment_log.md     # full experiment history and score analysis
    patchcore_resolution_analysis.md
  CT11/
```

### Image naming convention

Suffix encodes whether the PCB has screws installed:

| Suffix | Meaning |
|--------|---------|
| `_ws`  | with screw |
| `_ns`  | no screw (without screw) |

Example: `IMG-07_ws.jpg`, `IMG-defect-01_ns_aug_rot90.jpg`

---

## Setup

```bash
conda activate aoi
pip install -r requirements.txt
pip install -e .          # installs the patchcore package from src/
```

Camera: Android phone running **IP Webcam** app. Set the URL in `configs/640C/live_config.yaml`.

---

## Workflow

### 1. Add new raw images

Drop photos into `data/raw/`. Rename with `_ws` / `_ns` suffix to mark screw presence.

### 2. Preprocess (crop + resize to 1024×1024)

```bash
# Preview ROI on a sample image first
python preprocess.py --calibrate data/640C/raw/IMG-01_ws.jpg

# Batch process into the appropriate split (defaults to configs/640C/preprocess_config.yaml)
python preprocess.py --src data/640C/raw --split train/good \
  --files IMG-17_ws.jpg IMG-18_ws.jpg ...

python preprocess.py --src data/640C/raw --split test/defect \
  --files IMG-defect-27_ws.jpg

# For CT11, pass its own config
python preprocess.py --config configs/CT11/preprocess_config.yaml \
  --src data/CT11/raw --split train/good
```

ROI and output resolution are set in `configs/640C/preprocess_config.yaml`:
```yaml
roi: [640, 1620, 950, 820]   # [x, y, w, h] for new 2268×4032 camera
output_size: [1024, 1024]
```

### 3. Augment training images

```bash
# Current augmentation used for Model 4 (full 360° rotation coverage)
python augment_train.py \
  --no_fixed_rot --n_random_rot 8 --max_deg 180 \
  --n_jitter 1 --strong_jitter --n_noise 1 \
  --src data/ir_module_1024/ir_module/train/good
```

This generates `_aug_*` variants alongside each original. `augment_train.py` skips files
that already have `_aug_` in the name so it is safe to re-run.

### 4. Train PatchCore

```bash
# Submit to TWCC (SLURM)
sbatch run_ir_module_1024_aug_rot360_p01.sh
```

Model is saved under `results/IR_Module/640C/<log_group>/models/mvtec_ir_module/`.

### 5. Batch inference (offline evaluation)

```bash
# Single image
python yolo/infer_pcb.py \
  --image data/640C/ir_module_1024/ir_module/test/defect/IMG-defect-01_ns.jpg

# Full test folder (saves 3-panel result images)
python yolo/infer_pcb.py \
  --image_dir data/640C/ir_module_1024/ir_module/test/defect --save
```

### 6. Live inference (Enter-to-capture)

```bash
python live_infer.py \
  --model_path results/IR_Module/640C/ir_module_WR50_L2-3_PS3_1024_m4_p0.1/models/mvtec_ir_module
```

Each Enter keypress captures one frame from the phone camera and runs the full pipeline.
Results saved to `results/live/`.

### 7. Web UI

```bash
python app.py --device cpu
```

Opens a Flask dashboard at `http://localhost:5000`. No profile is loaded at startup —
pick a module (640C / CT11 Front / CT11 Back) on the Setup screen, which loads its
camera, YOLO weights, and PatchCore model together as one unit. `--device` overrides
every profile's own `inference.device` (use `cpu` if the machine has no GPU, otherwise
omit it to use each profile's own configured device); `--port` picks a different port
(default 5000).

---

## Key Config Files

| File | Purpose |
|------|---------|
| `configs/640C/preprocess_config.yaml` | ROI crop + output resolution for 640C |
| `configs/640C/live_config.yaml` | Camera URL, YOLO weights, output dirs for 640C |
| `run_ir_module_1024_aug_rot360_p01.sh` | SLURM job for PatchCore training (640C Model 4) |
