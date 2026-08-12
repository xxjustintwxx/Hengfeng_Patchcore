# Hengfeng PCB Anomaly Detection

Two-stage automated visual inspection for Hengfeng IR module PCBs, covering four
inspection profiles (one board family, three module/side variants — see
[Profiles](#profiles)):

1. **YOLOv11-seg** — detect components and verify exact counts (missing / extra → NG)
2. **PatchCore (WideResNet50)** — surface anomaly heatmap, with screw regions and
   (where configured) everything outside the board's own outline suppressed

**Verdict:** NG if the YOLO component count mismatches **or** the PatchCore score
≥ the profile's `score_threshold`; otherwise OK.

---

## Pipeline Overview

```
Phone camera (IP Webcam)
        ↓
  ROI crop + resize   ← preprocessing.roi / output_size in live_config.yaml
        ↓
  YOLOv11-seg component detection
    • per-class confidence filtering + 3-stage NMS (same-class, cross-class,
      containment) + off-board rejection (board-masking profiles only)
    • exact count check per class (see expected_counts)
    • build suppression mask (screws and/or off-board area)
        ↓
  PatchCore anomaly scoring
    • suppressed regions zeroed out before scoring
    • score = max patch distance over non-suppressed pixels
        ↓
  Verdict (OK / NG) + 3-panel result image saved
```

---

## Profiles

Everything about a profile — camera, ROI, YOLO weights, class taxonomy, PatchCore
model, output paths — lives entirely in that profile's `live_config.yaml`. The app
discovers every `configs/**/live_config.yaml` at startup with no per-module code;
adding a new module is a matter of adding a new config directory (see
[Config reference](#config-reference)).

| Profile label       | Config path                              | Classes                                                          | Screw suppress | Board masking |
|----------------------|-------------------------------------------|-------------------------------------------------------------------|:---:|:---:|
| `640C`               | `configs/640C/`                           | Main IC(1), connector(4), resistor(35), screw(4)                  | ✓ | — |
| `CT11_Power/Front`   | `configs/CT11_Power/Front/`               | Main IC(1), board, component(6), connector(2), resistor(21)       | — | ✓ |
| `CT11_Power/Back`    | `configs/CT11_Power/Back/`                | board, component(1), connector(1), resistor(61)                   | — | ✓ |
| `CT11_Image`         | `configs/CT11_Image/`                     | Main IC(2), board, component(10), connector(1), resistor(40), screw(1) | ✓ | ✓ |

Numbers in parentheses are `expected_counts` — the exact count that class must hit
or the board is flagged NG for that class. `board` itself is structural (used only
to mask the heatmap to the PCB's own outline), not counted.

"Screw suppress" (`suppress_classes`) zeroes screw regions out of the PatchCore
heatmap, so a screw's own strong edges/reflections never register as a surface
anomaly — only applies to profiles with a `screw` class (`640C`, `CT11_Image`;
neither `CT11_Power` side has one).

"Board masking" profiles segment the PCB's own outline via a dedicated `board`
class and suppress everything outside it from the PatchCore heatmap, so the
background/fixture never contributes to the anomaly score. `640C` predates this
and has no `board` class.

---

## Directory Structure

```
data/                            # gitignored (data/.gitkeep only) — populate locally
  640C/raw/                      # legacy flat layout — see Workflow §2
  ir_module_1024/ir_module/{train,test}/...
  <Module>/<Side?>/raw/          # e.g. CT11_Power/Front/raw, CT11_Image/raw
    ir_module_1024/ir_module/{train,test}/...

models/                          # gitignored (models/.gitkeep only) — populate locally
  yolo/
    640C/pcb_seg/weights/best.pt
    CT11_Power/{Front,Back}/pcb_seg/weights/best.pt
    CT11_Image/pcb_seg/weights/best.pt
    yolo11s-seg.pt                            # base weights used for YOLO training
  patchcore/
    640C/ir_module_WR50_L2-3_PS3_1024_m4_p0.1/models/mvtec_ir_module
    CT11_Power/CT11_{Front,Back}_WR50_L2-3_PS3_1024_m4_p0.1/models/mvtec_ir_module
    CT11_Image/CT11_Image_WR50_L2-3_PS3_1024_m4_p0.1/models/mvtec_ir_module

configs/
  640C/{preprocess_config.yaml, live_config.yaml}
  CT11_Power/{Front,Back}/{preprocess_config.yaml, live_config.yaml}
  CT11_Image/{preprocess_config.yaml, live_config.yaml}

yolo/                            # offline YOLO training + batch-eval scripts
  640C/{train_yolo_seg.py, infer_pcb.py, ...}
  CT11_Power/{train_ct11_front_seg.py, train_ct11_back_seg.py,
              infer_ct11_front.py, infer_ct11_back.py, ...}
  CT11_Image/{train_ct11_image_seg.py, infer_ct11_image.py, ...}

run_*.sh                         # SLURM jobs for PatchCore training, one per
                                  # module (run_ir_module_1024_aug_rot360_p01.sh,
                                  # run_ct11_power_front_1024_m4_p01.sh,
                                  # run_ct11_power_back_1024_m4_p01.sh,
                                  # run_ct11_image_1024_m4_p01.sh)

captures/                        # gitignored — raw phone photos from live/web capture
  <Profile>/<ts>_raw.jpg
  <Profile>/<ts>_roi_preview.jpg

results/
  live/<Profile>/<ts>_result.jpg # gitignored — 3-panel result images
  dev_review/<Profile>/{TP,FP,TN,FN}/   # gitignored — developer-mode verified copies
  inference_log.jsonl            # gitignored — every /api/infer call (score, counts, timings)
  verification_log.jsonl         # gitignored — every developer-mode human verdict

notes/
  640C/{model_experiment_log.md, patchcore_resolution_analysis.md}

src/                             # the `patchcore` package (pip install -e .)
static/, templates/              # Flask web UI
app.py                           # web UI server
live_infer.py                    # shared pipeline (also used by app.py) + CLI
capture.py                       # phone-camera HTTP snapshot helper
preprocess.py                    # ROI crop + resize, per-module
augment_train.py                 # PatchCore training-image augmentation
check_yolo_confidence.py         # inspect raw YOLO confidences for a capture
test_patchcore_heatmap.py        # side-by-side heatmaps for a saved PatchCore model
time_inference.py                # FlatL2 vs IVF-PQ speed/accuracy comparison
```

`data/` and `models/` are entirely gitignored (only `.gitkeep` is tracked) — clone
the repo, then populate them locally or point configs at wherever your trained
weights/models actually live.

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
# Create and activate the environment (any recent Python 3.10+ works)
conda create -n aoi python=3.10 -y
conda activate aoi

pip install -r requirements.txt
pip install -e .          # installs the patchcore package from src/
```

Camera: Android phone running the **IP Webcam** app. Set `camera.url` in each
profile's `live_config.yaml` (they can point at different URLs/ports if you run
multiple phones).

---

## Workflow

### 1. Collect raw images

Drop photos into `data/<Module>[/<Side>]/raw/` (or `data/640C/raw/` for the legacy
flat layout). Rename with `_ws` / `_ns` suffix to mark screw presence.

### 2. Preprocess (ROI crop + resize)

```bash
# Preview the ROI on a sample image first
python preprocess.py --config configs/CT11_Power/Front/preprocess_config.yaml \
  --calibrate data/CT11_Power/Front/raw/IMG-01.jpg
# → inspect calibrate_preview.jpg and calibrate_crop.jpg, adjust roi/output_size
#   in the config, repeat until it looks right

# Batch process a split — --module nests the output under data/<module>/
python preprocess.py --config configs/CT11_Power/Front/preprocess_config.yaml \
  --module CT11_Power/Front \
  --src data/CT11_Power/Front/raw --split train/good
# → writes to data/CT11_Power/Front/ir_module_1024/ir_module/train/good/

# Legacy flat layout (640C only) — omit --module and --config
python preprocess.py --src data/640C/raw --split train/good \
  --files IMG-17_ws.jpg IMG-18_ws.jpg
# → writes to data/ir_module_1024/ir_module/train/good/
```

Copy `ground_truth/` masks in manually if a split needs pixel-level annotations.

### 3. Augment training images (PatchCore only, optional)

```bash
python augment_train.py \
  --no_fixed_rot --n_random_rot 8 --max_deg 180 \
  --n_jitter 1 --strong_jitter --n_noise 1 \
  --src data/CT11_Power/Front/ir_module_1024/ir_module/train/good
```

Generates `_aug_*` variants alongside each original (rotation + color jitter +
noise, no flip/crop/zoom). Skips files that already have `_aug_` in the name, so
it's safe to re-run. Skip this step entirely if the raw training set is large
enough on its own — the current CT11_Power/CT11_Image models were trained
directly on 36 raw (unaugmented) images.

### 4. Train YOLO (per module)

```bash
python yolo/CT11_Power/train_ct11_front_seg.py    # or train_ct11_back_seg.py,
                                                   # yolo/CT11_Image/train_ct11_image_seg.py,
                                                   # yolo/640C/train_yolo_seg.py
```

Weights land at `models/yolo/<module>/[<side>/]pcb_seg/weights/best.pt`.

### 5. Train PatchCore (per module, SLURM)

```bash
sbatch run_ct11_power_front_1024_m4_p01.sh   # or run_ct11_power_back_1024_m4_p01.sh,
                                              # run_ct11_image_1024_m4_p01.sh,
                                              # run_ir_module_1024_aug_rot360_p01.sh (640C)
```

Model lands at `models/patchcore/<module>/<log_group>/models/mvtec_ir_module/`.

### 6. Offline batch evaluation

```bash
# Single image
python yolo/CT11_Power/infer_ct11_front.py \
  --image data/CT11_Power/Front/ir_module_1024/ir_module/test/defect/IMG-defect-01.jpg

# Full test folder (saves 3-panel result images)
python yolo/CT11_Power/infer_ct11_front.py \
  --image_dir data/CT11_Power/Front/ir_module_1024/ir_module/test/defect --save
```

Same pattern for `yolo/CT11_Image/infer_ct11_image.py` and `yolo/640C/infer_pcb.py`.

### 7. Live inference — CLI (Enter-to-capture)

```bash
python live_infer.py --config configs/CT11_Power/Front/live_config.yaml \
  --model_path models/patchcore/CT11_Power/CT11_Front_WR50_L2-3_PS3_1024_m4_p0.1/models/mvtec_ir_module

python live_infer.py --config configs/CT11_Power/Front/live_config.yaml \
  --model_path models/patchcore/CT11_Power/CT11_Front_WR50_L2-3_PS3_1024_m4_p0.1/models/mvtec_ir_module \
  --device cpu

python live_infer.py --config configs/CT11_Power/Front/live_config.yaml \
  --model_path models/patchcore/CT11_Power/CT11_Front_WR50_L2-3_PS3_1024_m4_p0.1/models/mvtec_ir_module \
  --no_yolo   # PatchCore-only
```

`--model_path` is required — unlike the web UI, this CLI script doesn't read
`output.patchcore_model` from the config; it's an independent flag. YOLO weights
*do* come from the config's `yolo.weights`. Each Enter keypress captures one frame
from the phone camera and runs the full pipeline. Results saved to
`results/live/<profile>/` and `captures/<profile>/`.

### 8. Web UI

A local Flask dashboard that walks through capture → inspect for whichever
profile you pick, without touching the command line per-run. Before starting it,
make sure at least one profile has its YOLO weights and PatchCore model actually
present on disk (see [Directory Structure](#directory-structure)) and the phone's
IP Webcam app is running and reachable at that profile's `camera.url`.

```bash
python app.py --device cpu
```

Opens the dashboard at `http://localhost:5000`. `--device` overrides every
profile's own `inference.device` (use `cpu` if the machine has no GPU, otherwise
omit it to use each profile's own configured device); `--port` picks a different
port (default `5000`).

No profile is loaded at startup. The UI walks through four screens:

- **Setup** — pick a profile (`640C` / `CT11_Power/Front` / `CT11_Power/Back` /
  `CT11_Image`), which loads its camera, YOLO weights, and PatchCore model
  together as one unit. `suppress_dilation`, `board_dilation`, and
  `score_threshold` are editable here too, per session (they don't persist back
  to the config file — edit the YAML directly for a permanent change). Last-used
  settings auto-restore into the form on reload.
- **Capture** — triggers the phone camera, shows the ROI preview before
  committing to inference (retake if it's off), then runs the pipeline. Space /
  Enter / Escape work as keyboard shortcuts throughout.
- **Result** — verdict badge (OK / NG) with separate surface and
  component-count status chips, YOLO overlay, anomaly heatmap, an interactive
  threshold slider, and per-class detected counts / issues.
- **History** — browse recent captures (`/api/history`), optionally filtered by
  profile.

**Developer mode** (toggle on the Setup screen) adds, on the Result screen:
- **Verify buttons** (`Actually OK` / `Actually NG`) — record your ground-truth
  call against the system's verdict into `results/verification_log.jsonl`,
  classified as TP/FP/TN/FN. Running TP/FP/TN/FN stats for the active profile are
  shown live.
- **Debug panel** — every raw YOLO detection per class, not just the ones that
  survived filtering: confidence, box, and (for anything dropped) exactly *why* —
  which of the four filtering stages caught it (same-class NMS, cross-class NMS,
  containment NMS, or off-board rejection) and which other detection caused it.
  Use this to calibrate `class_conf` / `cross_class_nms_iou` / `containment_thresh`
  / `off_board_overlap_thresh` against real captures.
- Verified results also get copied to `results/dev_review/<profile>/<TP|FP|TN|FN>/`
  for later review.

Every `/api/infer` call is logged to `results/inference_log.jsonl` regardless of
developer mode (score, counts, issues, timings) — a history of runs even if
developer mode is never turned on.

---

## Config reference

Every profile's `configs/<...>/live_config.yaml` has four top-level sections:

`camera`: `url`, `timeout`, `retries`, `retry_delay`, `pre_delay` — phone snapshot HTTP request.

`preprocessing`: `roi` (`[x, y, w, h]`, must match `preprocess_config.yaml`), `output_size`.

`yolo`:

| Key | Purpose | Default if omitted |
|---|---|---|
| `weights` | path to trained `.pt` | — (required) |
| `conf` | global confidence floor | `0.25` |
| `nms_iou` | Ultralytics' own internal NMS IoU | `0.20` |
| `imgsz` | YOLO inference resolution | `1280` |
| `class_names` | ordered class name list | built-in 4-class fallback (see below) |
| `class_colors` | BGR color per class index, for overlay drawing | built-in fallback colors |
| `label_classes` | which class indices get text labels drawn (avoid clutter from dense classes like resistor) | built-in fallback set |
| `class_conf` | per-class confidence override (must be ≥ `conf`) | — |
| `expected_counts` | exact count check per class index — omit a class to skip its count check | — |
| `suppress_classes` | classes whose masks are zeroed out of the PatchCore heatmap (e.g. screws) | built-in fallback (screw class) |
| `class_nms_iou` | per-class same-class-duplicate NMS IoU override | built-in fallback (screw dedup) |
| `suppress_dilation` | px to expand the suppression mask before use | `0` |
| `cross_class_nms_iou` | class-agnostic IoU above which a lower-confidence, different-class box is dropped for overlapping a higher-confidence one | `0.50` |
| `containment_thresh` | mask-overlap ratio (relative to the smaller detection's own area) above which a smaller different-class detection sitting inside a bigger one gets dropped | `0.10` |
| `off_board_overlap_thresh` | fraction of a non-board detection's own mask that must fall inside the board outline to be trusted (board-masking profiles only) | `0.50` |
| `board_class` | class name whose segmented outline masks the heatmap | — (no board masking) |
| `board_dilation` | px to expand the board mask outward before use | `20` |
| `board_pad` | gray-border padding added before YOLO (only if `board_class` is set — these models were trained on padded images) | `128` |

`inference`: `device` (`cuda` / `mps` / `cpu`), `score_threshold` (`null` skips
surface NG labelling; component counts are still checked).

`output`: `patchcore_model`, `captures_dir`, `heatmaps_dir`.

Every key past `weights`/`conf`/`nms_iou`/`imgsz` is optional — a config that
omits `class_names`/`class_colors`/`label_classes`/`suppress_classes`/
`class_nms_iou` falls back to a hardcoded 4-class default in `live_infer.py`. The NMS/masking tunables (`cross_class_nms_iou`,
`containment_thresh`, `off_board_overlap_thresh`) are meant to be calibrated
per-profile against real captures using developer mode's debug panel, which
names exactly which threshold caused a given detection to be dropped.

---

## Utility scripts

| Script | Purpose |
|---|---|
| `check_yolo_confidence.py --config <live_config.yaml> --image <raw.jpg>` | Inspect raw YOLO confidences for a capture, using the exact same crop/resize + taxonomy as the live pipeline — for calibrating `class_conf` without running the full app. |
| `test_patchcore_heatmap.py --model_path <mvtec_ir_module>` | Side-by-side heatmaps for a saved PatchCore model against its test set. |
| `time_inference.py` | Compares FlatL2 vs IVF-PQ FAISS index speed/accuracy for a saved model. |
