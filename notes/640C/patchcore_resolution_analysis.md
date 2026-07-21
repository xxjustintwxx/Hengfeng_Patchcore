# PatchCore Resolution & Receptive Field Analysis

## Background

This project uses PatchCore with a WideResNet50 backbone for PCB anomaly detection.
Two key questions arose during development:

1. How does input resolution affect the anomaly map precision?
2. How many pixels does each patch actually "see" (receptive field)?

---

## 1. Feature Map Size vs Input Resolution

PatchCore extracts features from **Layer2** and **Layer3** of WideResNet50.
These layers have cumulative strides of 8 and 16 respectively, so the feature map
is always a fixed fraction of the input size.

| Stage | Stride | 256px input | 512px input | 1024px input |
|-------|--------|-------------|-------------|--------------|
| Input | — | 256×256 | 512×512 | 1024×1024 |
| Conv1 | ×2 | 128×128 | 256×256 | 512×512 |
| MaxPool | ×2 | 64×64 | 128×128 | 256×256 |
| Layer1 | ×1 | 64×64 | 128×128 | 256×256 |
| **Layer2** | ×2 | **32×32** | **64×64** | **128×128** |
| **Layer3** | ×2 | **16×16** | **32×32** | **64×64** |
| Layer4 | ×2 | 8×8 | 16×16 | 32×32 |

PatchCore combines Layer2 + Layer3 features. The effective anomaly map resolution
is driven by Layer2 (the finer of the two).

**Number of scoring patches (Layer2):**
- 256px → 32×32 = **1,024 patches**
- 512px → 64×64 = **4,096 patches**
- 1024px → 128×128 = **16,384 patches** (16× more than 256px)

This is why higher resolution gives a sharper, more localized anomaly map.

---

## 2. Receptive Field Calculation

The receptive field (RF) is how many pixels of the **original input image** each
feature vector "sees." Larger RF = less precise localization.

WideResNet50 bottleneck blocks each contribute one 3×3 conv to the RF.

**Formula:** `RF_new = RF_prev + (kernel_size - 1) × cumulative_stride_before_this_layer`

| Stage | Kernel | Stride | Cumul. Stride | Receptive Field |
|-------|--------|--------|---------------|-----------------|
| Input | — | — | 1 | 1 px |
| Conv1 | 7×7 | 2 | 1 | **7 px** |
| MaxPool | 3×3 | 2 | 2 | **11 px** |
| Layer1 block 1 | 3×3 | 1 | 4 | 19 px |
| Layer1 block 2 | 3×3 | 1 | 4 | 27 px |
| Layer1 block 3 | 3×3 | 1 | 4 | **35 px** |
| Layer2 block 1 | 3×3 | 2 | 4 | 43 px |
| Layer2 block 2 | 3×3 | 1 | 8 | 59 px |
| Layer2 block 3 | 3×3 | 1 | 8 | 75 px |
| Layer2 block 4 | 3×3 | 1 | 8 | **91 px** |
| Layer3 block 1 | 3×3 | 2 | 8 | 107 px |
| Layer3 block 2 | 3×3 | 1 | 16 | 139 px |
| Layer3 block 3 | 3×3 | 1 | 16 | 171 px |
| Layer3 block 4 | 3×3 | 1 | 16 | 203 px |
| Layer3 block 5 | 3×3 | 1 | 16 | 235 px |
| Layer3 block 6 | 3×3 | 1 | 16 | **267 px** |

> Note: Only the 3×3 conv in each bottleneck contributes to RF.
> The 1×1 convs (channel squeeze/expand) have kernel=1 and do not increase RF.

### PatchCore patchsize=3 adjustment

PatchCore aggregates a 3×3 neighborhood of feature map cells into each patch descriptor.
This extends the effective RF by `(patchsize - 1) × stride`:

| Layer | Backbone RF | + patchsize=3 | Effective RF |
|-------|-------------|---------------|--------------|
| Layer2 | 91 px | + (3-1)×8 = 16 px | **107 px** |
| Layer3 | 267 px | + (3-1)×16 = 32 px | **299 px** |

---

## 3. What Fraction of the Image Each Patch Covers

This is the key metric for localization precision.

| Input | Layer2 RF (107px) | Layer3 RF (299px) |
|-------|-------------------|-------------------|
| **256px** | 107/256 = **42%** of image | 299/256 = **>100%** ⚠️ sees whole image |
| **512px** | 107/512 = **21%** of image | 299/512 = **58%** of image |
| **1024px** | 107/1024 = **10%** of image | 299/1024 = **29%** of image |

### Key insight

At **256px input**, Layer3's RF (299px) is larger than the entire image — every
Layer3 feature sees the whole PCB. It can detect *that* something is wrong but
has no ability to localize *where*. Layer2 at 256px covers 42% of the image,
which is also too coarse for precise component-level localization.

At **1024px input**, Layer2 covers only ~10% of the image. For our PCB crop
(~950×820 px native), this corresponds to roughly **100×100 px** of actual PCB
surface — tight enough to pinpoint individual components.

---

## 4. Why We Moved to 1024px

| Config | Input | Camera | ROI (native) | Anomaly map |
|--------|-------|--------|--------------|-------------|
| Old | 256×256 | 1152×2048 | 475×410 px | 32×32 patches |
| Current | 1024×1024 | 2268×4032 | 950×820 px | 128×128 patches |

The new higher-resolution camera images (2268×4032) provide a 950×820 px ROI
crop that maps almost 1:1 to 1024×1024 output (only ~1.08× upscale), preserving
nearly all original detail. Combined with removing the center crop (imagesize=1024,
no 448-crop), the full PCB is visible without cutting off corner screws.

---

## 5. Run Configurations Summary

| Script | Resolution | Layers | Patches (Layer2) | Notes |
|--------|------------|--------|-------------------|-------|
| `run_ir_module.sh` | 256 | L2 only | 32×32 | Old images, old ROI |
| `run_ir_module_512.sh` | 512 | L2+L3 | 64×64 | Old images, no center crop |
| `run_ir_module_1024.sh` | 1024 | L2+L3 | 128×128 | **New images, best localization** |
