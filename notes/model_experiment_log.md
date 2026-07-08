# PatchCore Model Experiment Log

Tracks configuration changes, score evolution, and key findings across model versions.

---

## Summary Table

| Model | Resolution | Camera | Train images | Test (OK/NG) | AUROC | Threshold gap |
|-------|-----------|--------|-------------|--------------|-------|---------------|
| `WR50_L2-3_PS3_512` | 512×512 | Old 1152×2048 | Unknown-2~14 | 3 OK / 1 NG | **1.000** | 349 vs 191 (large) |
| `WR50_L2-3_PS3_1024` | 1024×1024 | New 2268×4032 | IMG-01~13 | 3 OK / 13 NG | **0.949** | 236 vs 216 (tight, 2 misses) |
| `WR50_L2-3_PS3_1024_aug_p0.1` | 1024×1024 | New 2268×4032 | IMG-01~13 + 91 aug | 3 OK / 13 NG | **1.000** | 214 vs 217 (tight, 0 misses) |
| `WR50_L2-3_PS3_1024_aug_p0.1` (rotation test) | 1024×1024 | New 2268×4032 | same | 12 OK / 52 NG (4× orientations) | **0.984** | TTA max: 223 vs 240 (17pt gap) |
| `WR50_L2-3_PS3_1024_aug_rot360_p0.1` | 1024×1024 | New 2268×4032 | IMG-01~13 + 169 aug | 12 OK / 52 NG (4× orientations) | **0.992** | TTA max: 216 vs 234 (18pt gap) |

---

## Model 1 — `ir_module_WR50_L2-3_PS3_512`

### Configuration
- **Backbone:** WideResNet50, Layer2 + Layer3
- **Input:** resize=512, imagesize=512 (no center crop)
- **Patchsize:** 3
- **Sampler:** identity, p=1.0 (no coreset reduction)
- **Camera:** Old 1152×2048
- **ROI:** `[320, 860, 475, 410]` (x, y, w, h)
- **Dataset:** Unknown-2 to Unknown-18 renamed from raw camera dumps
- **Train/Test split:** Unknown-2~14 train, Unknown-15~17 test OK, Unknown-18 test NG

### Scores
| Image | Score | Label |
|-------|-------|-------|
| Unknown-15 | 191.46 | OK |
| Unknown-16 | 157.69 | OK |
| Unknown-17 | 150.99 | OK |
| Unknown-18 | **349.26** | NG |

### AUROC: 1.000
- 3 OK × 1 NG = 3 pairs, all correctly ordered (349 >> 191)
- **Caveat:** only 1 defect image makes this statistically unreliable — AUROC is either 0, 0.33, 0.67, or 1.0 with no granularity

### Key observations
- Screws consistently flagged as high anomaly (hot spots in corners)
- Large score gap between OK and NG because Unknown-18 was a strongly abnormal board
- Top 5% heatmap localized anomaly roughly to the defect region but screw noise was prominent

---

## Model 2 — `ir_module_WR50_L2-3_PS3_1024`

### Configuration changes vs Model 1
- **Resolution:** 512 → **1024×1024** (4× more patches: 64×64 → 128×128)
- **Camera:** upgraded to new **2268×4032** camera
- **ROI:** recalibrated to `[640, 1620, 950, 820]` — y shifted up 100px from initial `[640, 1720, 950, 820]` to include all 4 corner screws
- **Dataset:** 16 new IMG-* images (IMG-01~16 + IMG-defect-01~13); 13 defect test images instead of 1
- **Sampler:** identity, p=1.0

### Scores
| Image | Score | Label |
|-------|-------|-------|
| IMG-14 | 236.25 | OK |
| IMG-15 | 187.26 | OK |
| IMG-16 | 178.20 | OK |
| IMG-defect-01 | 381.96 | NG |
| IMG-defect-02 | 238.38 | NG |
| IMG-defect-03 | 236.95 | NG |
| IMG-defect-04 | 306.54 | NG |
| IMG-defect-05 | 272.34 | NG |
| IMG-defect-06 | 284.55 | NG |
| **IMG-defect-07** | **216.27** | NG ← scored BELOW OK max |
| IMG-defect-08 | 265.47 | NG |
| **IMG-defect-09** | **225.94** | NG ← scored BELOW OK max |
| IMG-defect-10 | 259.93 | NG |
| IMG-defect-11 | 321.02 | NG |
| IMG-defect-12 | 241.32 | NG |
| IMG-defect-13 | 310.19 | NG |

### AUROC: 0.949
- 3 OK × 13 NG = 39 pairs; 2 pairs inverted (defect-07 and defect-09 scored below IMG-14)
- AUROC = 37/39 = 0.949

### Root cause of misses
- IMG-14 scored unusually high (236.25) because it had screw orientations the memory bank hadn't seen well
- defect-07 and defect-09 had subtle defects that blended close to normal appearance
- The 13-image memory bank did not cover the full range of normal screw appearances

### Key observations
- Much sharper anomaly map than 512 (128×128 vs 64×64 patches)
- Screws still prominently flagged — memory bank learned from only 13 board images with limited screw orientation variety
- No threshold exists that separates all OK from all NG

---

## Model 3 — `ir_module_WR50_L2-3_PS3_1024_aug_p0.1`

### Configuration changes vs Model 2
- **Augmentation:** 7 variants generated per training image × 13 originals = **91 new images**
- **Total training set:** 13 original + 91 augmented = **104 images**
- **Augmentations applied** (no flip, no crop/zoom):
  - Rotate 90°, 180°, 270°
  - 2× random small rotation ±20° (bicubic, reflection fill)
  - Brightness + contrast jitter (0.7–1.4× brightness, 0.8–1.3× contrast)
  - Gaussian noise (σ=8)
- **Sampler:** `approx_greedy_coreset`, **p=0.1**
  - Raw patch features: 104 images × 128×128 patches = ~1.7M vectors
  - After coreset: ~170k vectors (10%) — similar bank size to Model 2 but far more diverse
- **Script:** `run_ir_module_1024_aug_p01.sh`

### Scores
| Image | Score | Label |
|-------|-------|-------|
| IMG-14 | **213.79** | OK ↓ from 236.25 |
| IMG-15 | **183.23** | OK ↓ from 187.26 |
| IMG-16 | **181.65** | OK ↓ from 178.20 |
| IMG-defect-01 | 280.32 | NG |
| IMG-defect-02 | 242.54 | NG |
| IMG-defect-03 | 252.15 | NG |
| IMG-defect-04 | 292.53 | NG |
| IMG-defect-05 | 258.03 | NG |
| IMG-defect-06 | 292.39 | NG |
| **IMG-defect-07** | **216.91** | NG ← was 216.27, now above OK max |
| IMG-defect-08 | 255.29 | NG |
| **IMG-defect-09** | **234.18** | NG ← was 225.94, now above OK max |
| IMG-defect-10 | 254.48 | NG |
| IMG-defect-11 | 287.06 | NG |
| IMG-defect-12 | 255.05 | NG |
| IMG-defect-13 | 294.69 | NG |

### AUROC: 1.000
- All 39 pairs correctly ordered
- Suggested threshold: **~215** (midpoint between max OK=213.79 and min NG=216.91)

### Why scores changed
- **OK scores dropped** (especially IMG-14: 236 → 214): augmentation added many screw orientations to the memory bank, so screws are now "known normal" and no longer inflate good-board scores
- **Some NG scores dropped too** (defect-01: 382 → 280): the memory bank is more general, so even defective boards match some augmented patches. However all NG scores remain above all OK scores
- **Tight margin (3 points)**: the threshold is not robust yet — more good training images or more defect variety would widen the gap

### Remaining concern
- defect-07 (216.91) and defect-09 (234.18) are still the weakest detections — these likely have subtle defects close in appearance to normal variation
- The 3-point threshold gap means a single borderline board could flip. Collecting more good images and retraining would widen this gap

---

## Rotation Consistency Test — `ir_module_WR50_L2-3_PS3_1024_aug_p0.1`

### Motivation
In production the PCB can enter the camera in any of 4 orientations (0°/90°/180°/270°). The model was trained with all 4 rotations of each good image (via augmentation), but the test set only had 0°. This test adds rot90/180/270 to every test image to evaluate true production-condition robustness.

### Test set expansion
- `test/good/`: 3 → **12 images** (3 originals + 9 rotated)
- `test/defect/`: 13 → **52 images** (13 originals + 39 rotated)
- Blank ground truth masks created for all 39 new defect images

### AUROC on expanded test set: **0.984** (was 1.000)
- 12 OK × 52 NG = 624 pairs; ~10 pairs inverted
- Root cause: two images fail at specific orientations (see below)

### Rotation consistency scores

| Image | 0° | rot A | rot B | rot C | Spread | Label |
|-------|-----|-------|-------|-------|--------|-------|
| IMG-14 | 213.79 | **222.64** | 217.39 | 215.59 | 8.85 | OK |
| IMG-15 | 183.23 | 180.61 | 185.64 | **219.10** | **38.49** | OK ← spike |
| IMG-16 | 181.65 | 207.25 | 213.10 | 191.32 | 31.45 | OK |
| defect-01 | 280.32 | 300.40 | 282.25 | 294.68 | 20.08 | NG |
| defect-02 | 242.54 | 239.66 | 245.32 | 242.65 | 5.66 | NG |
| defect-03 | 252.15 | 251.01 | 241.43 | 243.66 | 10.72 | NG |
| defect-04 | 292.53 | 299.41 | 279.27 | 311.87 | 32.60 | NG |
| defect-05 | 258.03 | 260.77 | 268.16 | 251.34 | 16.82 | NG |
| defect-06 | 292.39 | 267.49 | 265.70 | 290.94 | 26.69 | NG |
| defect-07 | **216.91** | 239.62 | 221.46 | 233.57 | 22.71 | NG ← miss at 0° |
| defect-08 | 255.29 | 259.15 | 257.96 | 254.15 | 5.00 | NG |
| defect-09 | 234.18 | 225.79 | **213.01** | 244.19 | **31.18** | NG ← miss at rot B |
| defect-10 | 254.48 | 259.04 | 243.32 | 278.39 | 35.07 | NG |
| defect-11 | 287.06 | 314.09 | 276.21 | 292.81 | 37.87 | NG |
| defect-12 | 255.05 | 235.14 | 230.55 | 248.73 | 24.51 | NG |
| defect-13 | 294.69 | 278.96 | 278.15 | 272.84 | 21.85 | NG |

*(rot A/B/C = the three non-original orientations; exact angle mapping is alphabetically sorted, not guaranteed 90→180→270)*

### Key findings

**IMG-15 spike (spread 38.49):** At three orientations scores 180–186, but at one specific orientation jumps to **219.10** — almost triggering a false alarm. This is screw-orientation sensitivity: that rotation places a screw slot in a position the memory bank hasn't learned well, despite including all 4 fixed rotations in training. The ±20° random augmentation didn't cover this particular alignment.

**IMG-16 variation (spread 31.45, max 213.10):** Large spread but all 4 scores remain below the NG floor (239.62 with TTA). Variation is screw-driven, not a defect signal. IMG-16 is a true good board.

**defect-07 at 0° = 216.91:** Still the hardest NG case. At 0° it gets beaten by IMG-14 and IMG-15's spike rotation. At other orientations it scores 221–240, well clear of OK.

**defect-09 min = 213.01:** The most dangerous miss — at one orientation it scores 213.01, below most OK scores. The defect is very subtle and effectively invisible from this angle. At other orientations it scores 225–244.

### Test-Time Augmentation (TTA) — max over 4 rotations

If in production the board is captured at all 4 orientations and we take the **maximum score**, the threshold analysis becomes:

| | Max score across 4 rotations |
|--|--|
| OK ceiling (worst good board) | **222.64** (IMG-14) |
| NG floor (easiest-to-miss defect) | **239.62** (defect-07) |
| Gap | **~17 points** |

TTA AUROC = **1.000** — all pairs correctly ordered with a 17-point margin (vs the previous 3-point margin). This is the recommended production approach if multi-orientation capture is feasible.

### Remaining concerns
- The 17-point TTA gap is still not wide. More good training images would widen it.
- defect-09 is barely detectable even with TTA (max=244 vs OK ceiling=222, gap=21). Its defect may be too subtle for patch-level features alone.
- IMG-15's 38-point spread shows the memory bank still has orientation gaps. Denser random rotation augmentation (8 random samples across full 360° instead of 2 at ±20°) is the next logical training improvement.

---

## Model 4 — `ir_module_WR50_L2-3_PS3_1024_aug_rot360_p0.1`

### Configuration changes vs Model 3
- **Random rotations:** 2 × ±20° → **8 × ±180°** (uniform sampling across full 360°)
- **Total aug per image:** 7 → **13** (rot90/180/270 + 8 random + jitter + noise)
- **Total training set:** 104 → **182 images** (13 originals + 169 augmented)
- **Patch vectors (raw):** ~1.7M → ~3.0M; after coreset p=0.1: ~170k → ~300k
- **Script:** `run_ir_module_1024_aug_rot360_p01.sh`
- **`augment_train.py` change:** added `--max_deg` argument (default 20°, set to 180° here)

### Rotation consistency scores (4-orientation test set)

| Image | 0° | rot A | rot B | rot C | Spread | Label |
|-------|-----|-------|-------|-------|--------|-------|
| IMG-14 | 199.88 | 203.06 | 213.61 | **215.51** | 15.63 | OK |
| IMG-15 | 187.26 | 188.28 | 180.08 | **205.83** | 25.75 | OK ↓ spread from 38.49 |
| IMG-16 | 184.93 | 205.17 | **211.32** | 187.77 | 26.39 | OK |
| defect-01 | 286.51 | 295.62 | 284.48 | 298.78 | 14.30 | NG |
| defect-02 | 250.71 | 246.55 | 245.32 | 251.40 | 6.08 | NG |
| defect-03 | 243.84 | 253.96 | 241.43 | 242.25 | 12.53 | NG |
| defect-04 | 288.92 | 298.58 | 286.79 | 305.33 | 18.54 | NG |
| defect-05 | 266.25 | 277.71 | 274.94 | 261.72 | 15.99 | NG |
| defect-06 | 285.44 | 272.06 | 262.87 | 269.09 | 22.57 | NG |
| defect-07 | 230.13 | 225.95 | **221.88** | 233.57 | 11.69 | NG ✓ now above OK ceiling |
| defect-08 | 244.18 | 258.91 | 257.41 | 262.55 | 18.37 | NG |
| defect-09 | 232.26 | **210.26** | 213.01 | 241.75 | 31.49 | NG ← sole remaining problem |
| defect-10 | 251.50 | 258.56 | 255.51 | 272.16 | 20.66 | NG |
| defect-11 | 263.36 | 311.82 | 267.92 | 290.73 | 48.46 | NG |
| defect-12 | 248.36 | 224.92 | 232.47 | 243.76 | 23.44 | NG |
| defect-13 | 293.34 | 260.87 | 286.47 | 267.93 | 32.47 | NG |

### AUROC: 0.992 (4-orientation test set)
- OK max = **215.51** (IMG-14), NG min = **210.26** (defect-09 at worst rotation)
- ~5 inverted pairs, all from defect-09 at its two worst orientations (210.26 and 213.01)

### What improved vs Model 3

| Metric | Model 3 aug_p0.1 | Model 4 rot360_p0.1 | Change |
|--------|------------------|---------------------|--------|
| AUROC (4-rotation test) | 0.984 | **0.992** | +0.008 ✓ |
| OK ceiling | 222.64 | **215.51** | −7.13 ✓ |
| IMG-15 spread | 38.49 | **25.75** | −12.74 ✓ |
| IMG-15 spike | 219.10 | **205.83** | −13.27 ✓ |
| defect-07 min | 216.91 | **221.88** | +4.97 ✓ now safely above ceiling |
| defect-09 min | 213.01 | **210.26** | −2.75 ✗ slightly worse |

### TTA analysis (max score across 4 rotations per image)

| | Model 3 | Model 4 |
|--|---------|---------|
| OK TTA ceiling | 222.64 | **215.51** |
| NG TTA floor (defect-07 max) | 239.62 | **233.57** |
| TTA gap | 17 pts | **18 pts** |

TTA AUROC = **1.000**. Suggested production threshold with Model 4: **224** (midpoint of 215.51–233.57).

### Conclusion — rotation augmentation ceiling reached

Denser rotation augmentation fixed defect-07 and significantly reduced OK spread, but **defect-09 remains the sole failure** and its minimum score (210.26) worsened slightly. This confirms the problem is not orientation coverage — the defect in defect-09 is visually indistinguishable from a normal board at certain orientations. Further rotation augmentation will not help.

Paths forward for defect-09:
1. Relabel as OK if the defect is below acceptable severity
2. Collect more diverse good training images to push the OK ceiling below 210
3. Apply a targeted strategy (higher-resolution ROI crop of defect region, or a separate classifier)

---

## ROI Calibration History

| Version | ROI [x, y, w, h] | Note |
|---------|------------------|------|
| Old camera (1152×2048) | `[320, 860, 475, 410]` | Original calibration |
| New camera initial | `[640, 1720, 950, 820]` | ~2× scale, but top screws cut off |
| **New camera final** | **`[640, 1620, 950, 820]`** | y moved up 100px, all 4 screws visible |

---

## Suggested Next Steps

1. **TTA in production** — capture board at all 4 orientations, flag as NG if max score ≥ **224** (midpoint of Model 4 TTA gap: 215.51–233.57); use Model 4 (`aug_rot360_p0.1`)
2. **Investigate defect-09 visually** — check its heatmap at the two worst orientations; if the anomaly hotspot is on a screw rather than the PCB body, the defect may be untreatable by PatchCore alone
3. **More good training images** — increasing from 13 to 30+ originals is the most reliable way to push the OK ceiling below 210 and create safe separation from defect-09
4. **Ground truth masks** — drawing pixel-level defect masks enables `full_pixel_auroc` which validates whether heatmaps localize to the correct area
