"""
PCB inspection: YOLOv11-seg component detection + PatchCore surface anomaly detection.

Pipeline:
  1. YOLO  → detect components, check exact per-class counts (no more, no less)
  2. PatchCore → raw anomaly heatmap on unmodified image
  3. Set screw regions to SUPPRESS_VALUE (-1) so they are excluded from
     colormap scaling — fixes the "everything looks yellow" problem
  4. Visualisation matches test_patchcore_heatmap.py style:
       Panel 1: Original + YOLO overlay
       Panel 2: Anomaly heatmap blended with original
       Panel 3: Top-5% anomaly region highlighted red
       + Colorbar  +  Verdict banner (raw score)

Usage:
  # Single image
  python yolo/infer_pcb.py --image data/640C/ir_module_1024/ir_module/test/defect/IMG-defect-01_ns.jpg

  # Whole folder (shared colorbar scale, originals only)
  python yolo/infer_pcb.py --image_dir data/640C/ir_module_1024/ir_module/test/defect --save

  # Include augmented images too
  python yolo/infer_pcb.py --image_dir data/640C/ir_module_1024/ir_module/test/defect --save --include_aug
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import PIL.Image
import torch
from torchvision import transforms
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))

import patchcore.common
import patchcore.patchcore

# ── Config ─────────────────────────────────────────────────────────────────────
YOLO_WEIGHTS   = ROOT / "models/yolo/640C/pcb_seg/weights/best.pt"
PATCHCORE_PATH = ROOT / "results/IR_Module/640C/ir_module_WR50_L2-3_PS3_1024_m4_p0.1/models/mvtec_ir_module"
OUTPUT_BASE    = ROOT / "results/yolo/640C/pcb_inspection"

CLASS_NAMES    = ["Main IC", "connecter", "resistor", "screw"]

# Per-class colour (BGR) for the custom YOLO overlay
CLASS_COLORS = {
    0: (255, 100,   0),   # Main IC   — orange
    1: (  0, 200,   0),   # connecter — green
    2: (  0, 200, 200),   # resistor  — yellow (mask only, no text label)
    3: (  0,  80, 255),   # screw     — red
}
# Resistors are excluded from text labels — 35 labels would flood the image
LABEL_CLASSES = {0, 1, 3}

# Exact expected count for every component class (no more, no less).
# Calibrate resistor count by running YOLO on several known-good boards
# and picking the stable count at the chosen CLASS_CONF threshold.
EXPECTED_EXACT_COUNT = {
    0: 1,    # Main IC
    1: 4,    # connecter
    2: 35,   # resistor  ← calibrate this value
    3: 4,    # screw
}

# Per-class confidence threshold — raised for resistors to cut false alarms
# while keeping sensitive detection for larger, critical components.
CLASS_CONF = {
    0: 0.25,   # Main IC
    1: 0.25,   # connecter
    2: 0.55,   # resistor  ← higher = fewer false detections
    3: 0.25,   # screw
}

SUPPRESS_CLASSES      = {3}    # screw: exclude from PatchCore heatmap
SUPPRESS_VALUE        = -1.0   # sentinel written into suppressed pixels
SUPPRESS_DILATION_PX  = 0      # expand screw mask edge outward by N px; use 20 for type1/type2 (background screws)

# Per-class post-NMS IoU threshold applied AFTER YOLO's global NMS.
# Screws sometimes produce two overlapping detections on the same fastener
# that slip through the global IoU=0.45 threshold. A stricter per-class NMS
# deduplicates them while keeping genuinely separate screws (IoU ~0.0–0.1).
CLASS_NMS_IOU = {
    3: 0.30,   # screw
}

# PatchCore preprocessing must match training resolution
PC_RESIZE = 1024
PC_CROP   = 1024
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

YOLO_CONF      = 0.25
TOP_PCT        = 1          # top-N% anomaly highlight panel
ANOMALY_THRESH = 190.0      # PatchCore score >= this → surface NG (override with --threshold)



# ── Helpers ────────────────────────────────────────────────────────────────────
def _per_class_nms(boxes_xyxy: np.ndarray, confs: np.ndarray,
                   cls_ids: np.ndarray, iou_thresholds: dict) -> np.ndarray:
    """Return a boolean keep-mask after applying stricter per-class NMS.

    Only classes listed in iou_thresholds are re-filtered; all other
    detections pass through unchanged.
    """
    from torchvision.ops import nms as tv_nms
    keep = np.ones(len(cls_ids), dtype=bool)
    for cls_id, iou_thr in iou_thresholds.items():
        idx = np.where(cls_ids == cls_id)[0]
        if len(idx) <= 1:
            continue
        kept_local = tv_nms(
            torch.tensor(boxes_xyxy[idx], dtype=torch.float32),
            torch.tensor(confs[idx],      dtype=torch.float32),
            iou_thr,
        ).numpy()
        keep[idx] = False
        keep[idx[kept_local]] = True
    return keep


# ── Model loader ───────────────────────────────────────────────────────────────
class PCBInspector:
    def __init__(self, device="cuda:0", anomaly_thresh=ANOMALY_THRESH,
                 suppress_dilation=SUPPRESS_DILATION_PX,
                 patchcore_path=PATCHCORE_PATH,
                 pc_resize=PC_RESIZE, pc_crop=PC_CROP):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.anomaly_thresh = anomaly_thresh
        self.suppress_dilation = suppress_dilation
        self.patchcore_path = Path(patchcore_path)
        self.pc_resize = pc_resize
        self.pc_crop   = pc_crop

        print(f"Loading YOLO   : {YOLO_WEIGHTS}")
        self.yolo = YOLO(str(YOLO_WEIGHTS))

        print(f"Loading PatchCore: {self.patchcore_path}")
        nn_method = patchcore.common.FaissNN(False, 4)
        self.pc = patchcore.patchcore.PatchCore(self.device)
        self.pc.load_from_path(str(self.patchcore_path), self.device, nn_method)

        self._pc_img_transform = transforms.Compose([
            transforms.Resize(pc_resize),
            transforms.CenterCrop(pc_crop),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])
        self._pc_mask_transform = transforms.Compose([
            transforms.Resize(pc_resize, interpolation=transforms.InterpolationMode.NEAREST),
            transforms.CenterCrop(pc_crop),
            transforms.ToTensor(),
        ])
        print("Ready.\n")

    # ── Core inference (returns raw values; caller handles visualisation) ──────
    def run(self, image_path: str) -> dict:
        """
        Returns:
            yolo_result    : ultralytics Results object (for .plot())
            issues        : list[str]  component classes below expected count
            detected_counts: dict       class_name -> detected count
            raw_heatmap    : np.ndarray (H, W) raw PatchCore scores, screws zeroed
            anomaly_score  : float      max raw score in non-suppressed region
            orig_bgr       : np.ndarray original image cropped to PatchCore space, BGR
        """
        pil_img = PIL.Image.open(image_path).convert("RGB")

        # ── YOLO ──────────────────────────────────────────────────────────────
        # iou=0.45: NMS removes a box if it overlaps an existing one by >45%.
        # Default 0.7 is too lenient for dense resistors — same-resistor duplicates
        # overlap ~50-60% and slip through. Adjacent resistors share only an edge
        # (IoU ~0.1-0.2) so they are not merged at this threshold.
        yolo_res      = self.yolo(pil_img, conf=YOLO_CONF, iou=0.20, imgsz=1280, verbose=False)[0]
        count_by_cls  = {i: 0 for i in range(len(CLASS_NAMES))}
        suppression   = np.zeros((self.pc_crop, self.pc_crop), dtype=bool)

        if yolo_res.boxes is not None:
            cls_ids   = yolo_res.boxes.cls.cpu().numpy().astype(int)
            confs     = yolo_res.boxes.conf.cpu().numpy()
            boxes_xyxy = yolo_res.boxes.xyxy.cpu().numpy()
            # Per-class stricter NMS to remove duplicate detections (e.g. double-counted screws)
            nms_keep  = _per_class_nms(boxes_xyxy, confs, cls_ids, CLASS_NMS_IOU)
            for cls_id, conf, keep in zip(cls_ids, confs, nms_keep):
                if not keep:
                    continue
                # Apply per-class confidence threshold
                if conf >= CLASS_CONF.get(cls_id, 0.25):
                    count_by_cls[cls_id] += 1

        if yolo_res.masks is not None and yolo_res.boxes is not None:
            masks_data = yolo_res.masks.data.cpu().numpy()
            cls_ids    = yolo_res.boxes.cls.cpu().numpy().astype(int)
            confs      = yolo_res.boxes.conf.cpu().numpy()
            for mask_raw, cls_id, conf, keep in zip(masks_data, cls_ids, confs, nms_keep):
                if not keep:
                    continue
                if conf < CLASS_CONF.get(cls_id, 0.25):
                    continue
                if cls_id in SUPPRESS_CLASSES:
                    m_pil = PIL.Image.fromarray(
                        (mask_raw * 255).astype(np.uint8), mode="L"
                    )
                    m = self._pc_mask_transform(m_pil)[0].numpy()
                    suppression |= m > 0.5

        # Exact component count check: flag any class that is over or under
        issues = []
        for cls_id, expected in EXPECTED_EXACT_COUNT.items():
            got = count_by_cls[cls_id]
            if got != expected:
                tag = "missing" if got < expected else "extra"
                issues.append(f"{CLASS_NAMES[cls_id]} ({got}/{expected} {tag})")

        # ── PatchCore ─────────────────────────────────────────────────────────
        tensor = self._pc_img_transform(pil_img).unsqueeze(0)
        scores, masks = self.pc._predict(tensor)
        raw_heatmap   = np.array(masks[0], dtype=np.float32)

        # Apply screw suppression — use sentinel (-1) not zero so colormap
        # scaling is computed from valid pixels only (fixes "all yellow" issue)
        if suppression.shape != raw_heatmap.shape:
            suppression = cv2.resize(
                suppression.astype(np.uint8),
                (raw_heatmap.shape[1], raw_heatmap.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            ).astype(bool)
        # Dilate outward from mask edge to suppress corner halos from background screws
        if self.suppress_dilation > 0:
            r = self.suppress_dilation
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*r+1, 2*r+1))
            suppression = cv2.dilate(suppression.astype(np.uint8), kernel).astype(bool)
        raw_heatmap[suppression] = SUPPRESS_VALUE
        valid_pixels  = raw_heatmap[raw_heatmap >= 0]
        anomaly_score = float(valid_pixels.max()) if valid_pixels.size else 0.0

        # Original image in BGR, cropped to same space PatchCore sees
        orig_bgr = _pil_to_bgr(pil_img, self.pc_resize, self.pc_crop)

        # Custom YOLO overlay: masks + boxes for all classes,
        # text labels only for LABEL_CLASSES (not resistors)
        orig_full = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        yolo_bgr  = _draw_yolo_overlay(orig_full, yolo_res)

        detected_counts = {CLASS_NAMES[i]: count_by_cls[i] for i in range(len(CLASS_NAMES))}

        return dict(
            yolo_result     = yolo_res,
            yolo_bgr        = yolo_bgr,
            issues         = issues,
            detected_counts = detected_counts,
            raw_heatmap     = raw_heatmap,
            anomaly_score   = anomaly_score,
            orig_bgr        = orig_bgr,
        )

    def inspect(self, image_path: str,
                shared_lo=None, shared_hi=None) -> dict:
        """Run + visualise a single image. Returns result dict with 'vis' key."""
        r = self.run(image_path)

        surface_fail = (self.anomaly_thresh is not None
                        and r["anomaly_score"] > self.anomaly_thresh)
        verdict = "NG" if (r["issues"] or surface_fail) else "OK"

        vis = _render(
            r["orig_bgr"], r["yolo_bgr"], r["raw_heatmap"],
            r["anomaly_score"], verdict, r["issues"],
            surface_fail=surface_fail, shared_lo=shared_lo, shared_hi=shared_hi,
        )
        return {**r, "verdict": verdict, "vis": vis}


# ── Visualisation helpers ──────────────────────────────────────────────────────
def _draw_yolo_overlay(bgr: np.ndarray, yolo_res) -> np.ndarray:
    """
    Draw YOLO detections on bgr without using results.plot().
    - Filled semi-transparent mask for every class
    - Bounding box outline for every class
    - Text label + confidence only for LABEL_CLASSES (not resistors)
    """
    out      = bgr.copy()
    overlay  = bgr.copy()
    H, W     = bgr.shape[:2]

    if yolo_res.boxes is None:
        return out

    boxes   = yolo_res.boxes
    cls_ids = boxes.cls.cpu().numpy().astype(int)
    confs   = boxes.conf.cpu().numpy()
    xyxy    = boxes.xyxy.cpu().numpy().astype(int)
    masks   = yolo_res.masks.data.cpu().numpy() if yolo_res.masks is not None else None

    for i, (cls_id, conf) in enumerate(zip(cls_ids, confs)):
        if conf < CLASS_CONF.get(cls_id, 0.25):
            continue
        color = CLASS_COLORS.get(cls_id, (200, 200, 200))

        # Filled mask (semi-transparent)
        if masks is not None:
            m = cv2.resize(masks[i], (W, H), interpolation=cv2.INTER_LINEAR)
            overlay[m > 0.5] = color

        # Bounding box
        x1, y1, x2, y2 = xyxy[i]
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)

        # Text label only for non-resistor classes
        if cls_id in LABEL_CLASSES:
            label = f"{CLASS_NAMES[cls_id]} {conf:.2f}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
            cv2.rectangle(out, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
            cv2.putText(out, label, (x1 + 2, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

    # Blend mask overlay at 40% opacity
    cv2.addWeighted(overlay, 0.4, out, 0.6, 0, out)
    return out


def _pil_to_bgr(pil, resize, crop):
    tf = transforms.Compose([transforms.Resize(resize), transforms.CenterCrop(crop)])
    arr = np.array(tf(pil))
    return arr[:, :, ::-1].copy()


def _to_colormap(anom_map, lo=None, hi=None):
    m     = anom_map.astype(np.float32)
    valid = m[m >= 0]
    lo    = float(np.percentile(valid, 1))  if lo is None else lo
    hi    = float(np.percentile(valid, 99)) if hi is None else hi
    scaled = np.clip((m - lo) / (hi - lo + 1e-8), 0, 1)
    scaled[m < 0] = 0   # suppressed pixels → blue (low end of JET)
    return cv2.applyColorMap((scaled * 255).astype(np.uint8), cv2.COLORMAP_JET)


def _make_colorbar(height, width=20):
    bar = np.linspace(255, 0, height, dtype=np.uint8).reshape(height, 1)
    return cv2.applyColorMap(np.repeat(bar, width, axis=1), cv2.COLORMAP_JET)


def _render(orig_bgr, yolo_bgr, anom_map, score, verdict, issues,
            surface_fail=False, shared_lo=None, shared_hi=None, top_pct=TOP_PCT):
    h, w = orig_bgr.shape[:2]

    # Panel 1 — YOLO overlay resized to PatchCore crop size
    p1 = cv2.resize(yolo_bgr, (w, h))

    # Panel 2 — Anomaly heatmap blended with original
    cm = _to_colormap(anom_map, shared_lo, shared_hi)
    cm = cv2.resize(cm, (w, h))
    p2 = cv2.addWeighted(orig_bgr, 0.5, cm, 0.5, 0)

    # Panel 3 — Highest-score patch circled on dimmed original
    # Find the single peak location in the heatmap (max of valid pixels)
    valid_px  = anom_map[anom_map >= 0]
    peak_val  = float(valid_px.max()) if valid_px.size else 0
    peak_mask = (anom_map >= peak_val)
    ys, xs    = np.where(peak_mask)
    cy = int(ys.mean() * h / anom_map.shape[0])
    cx = int(xs.mean() * w / anom_map.shape[1])
    # Radius = ~5% of image width so the circle is visible
    radius = max(20, w // 20)
    dimmed = (orig_bgr * 0.45).astype(np.uint8)
    p3 = dimmed.copy()
    # Bright patch inside circle stays at full colour
    circle_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(circle_mask, (cx, cy), radius, 1, -1)
    p3[circle_mask == 1] = orig_bgr[circle_mask == 1]
    # Red ring outline
    cv2.circle(p3, (cx, cy), radius,     (0, 0, 255), 3, cv2.LINE_AA)
    cv2.circle(p3, (cx, cy), radius + 4, (0, 0, 180), 1, cv2.LINE_AA)
    # Small crosshair at exact peak
    cv2.line(p3, (cx - 8, cy), (cx + 8, cy), (0, 0, 255), 2, cv2.LINE_AA)
    cv2.line(p3, (cx, cy - 8), (cx, cy + 8), (0, 0, 255), 2, cv2.LINE_AA)

    # Panel wrappers (title bar + image + score bar)
    title_h, score_h, gap = 46, 50, 6
    grey = (60, 60, 60)

    def wrap(img, title, sc):
        tbar = np.full((title_h, w, 3), 40, dtype=np.uint8)
        (tw, _), _ = cv2.getTextSize(title, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
        cv2.putText(tbar, title, ((w - tw) // 2, 32),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (210, 210, 210), 2, cv2.LINE_AA)
        sbar = np.full((score_h, w, 3), 255, dtype=np.uint8)
        if sc is not None:
            txt = f"score: {sc:.4f}"
            (sw, _), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
            cv2.putText(sbar, txt, ((w - sw) // 2, 34),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, grey, 2, cv2.LINE_AA)
        return np.vstack([tbar, img, sbar])

    panels = [
        wrap(p1, "YOLO detections",         None),
        wrap(p2, "Anomaly map",              score),
        wrap(p3, "Peak anomaly location",    score),
    ]
    div = np.full((panels[0].shape[0], gap, 3), 200, dtype=np.uint8)
    row = np.concatenate([panels[0], div, panels[1], div, panels[2]], axis=1)

    # Colorbar
    panel_h  = panels[0].shape[0]
    cb_w, cb_pad = 20, 50
    cb_col   = np.full((panel_h, cb_w + cb_pad, 3), 255, dtype=np.uint8)
    cb_img_h = panel_h - title_h - score_h - 20
    y0       = title_h + 10
    cb       = _make_colorbar(cb_img_h, cb_w)
    cb_col[y0: y0 + cb_img_h, 8: 8 + cb_w] = cb
    _valid = anom_map[anom_map >= 0]
    img_hi = float(np.percentile(_valid, 99)) if _valid.size else 0
    img_lo = float(np.percentile(_valid,  1)) if _valid.size else 0
    cv2.putText(cb_col, "High",          (2, y0 - 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, grey, 1, cv2.LINE_AA)
    cv2.putText(cb_col, f"{img_hi:.1f}", (2, y0 -  6), cv2.FONT_HERSHEY_SIMPLEX, 0.50, grey, 1, cv2.LINE_AA)
    cv2.putText(cb_col, f"{img_lo:.1f}", (2, y0 + cb_img_h + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.50, grey, 1, cv2.LINE_AA)
    cv2.putText(cb_col, "Low",           (2, y0 + cb_img_h + 32), cv2.FONT_HERSHEY_SIMPLEX, 0.55, grey, 1, cv2.LINE_AA)
    row = np.concatenate([row, cb_col], axis=1)
    W = row.shape[1]

    # Verdict banner
    is_ng  = verdict == "NG"
    colour = (0, 0, 200) if is_ng else (0, 150, 0)
    banner_h = 70
    banner = np.full((banner_h, W, 3), 255, dtype=np.uint8)
    line1  = f"{verdict}   |   score = {score:.4f}"
    parts  = []
    if issues:
        parts.append("component issues: " + ", ".join(issues))
    if surface_fail:
        parts.append("surface anomaly")
    line2  = "  |  ".join(parts)
    (vw, _), _ = cv2.getTextSize(line1, cv2.FONT_HERSHEY_DUPLEX, 1.0, 2)
    cv2.putText(banner, line1, ((W - vw) // 2, 30),
                cv2.FONT_HERSHEY_DUPLEX, 1.0, colour, 2, cv2.LINE_AA)
    if line2:
        (mw, _), _ = cv2.getTextSize(line2, cv2.FONT_HERSHEY_SIMPLEX, 0.75, 2)
        cv2.putText(banner, line2, ((W - mw) // 2, 58),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, colour, 2, cv2.LINE_AA)
    cv2.rectangle(banner, (0, banner_h - 6), (W, banner_h), colour, -1)

    return np.vstack([row, banner])


# ── CLI ────────────────────────────────────────────────────────────────────────
def _collect_images(folder, include_aug=False):
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    paths = sorted(p for p in Path(folder).iterdir() if p.suffix.lower() in exts)
    if not include_aug:
        paths = [p for p in paths if "_aug_" not in p.name]
    return paths


def main():
    parser = argparse.ArgumentParser(description="PCB defect inspection")
    grp = parser.add_mutually_exclusive_group(required=True)
    grp.add_argument("--image",     type=str)
    grp.add_argument("--image_dir", type=str)
    parser.add_argument("--device",      type=str,  default="cuda:0")
    parser.add_argument("--threshold",        type=float, default=ANOMALY_THRESH,
                        help=f"Raw anomaly score threshold for NG verdict (default: {ANOMALY_THRESH})")
    parser.add_argument("--suppress_dilation", type=int, default=SUPPRESS_DILATION_PX,
                        help=f"Pixels to expand screw mask outward (default: {SUPPRESS_DILATION_PX})")
    parser.add_argument("--patchcore_path", type=str, default=str(PATCHCORE_PATH),
                        help="Path to PatchCore model directory")
    parser.add_argument("--resize",   type=int, default=PC_RESIZE,
                        help=f"PatchCore resize resolution (default: {PC_RESIZE})")
    parser.add_argument("--cropsize", type=int, default=PC_CROP,
                        help=f"PatchCore center-crop size (default: {PC_CROP})")
    parser.add_argument("--save",        action="store_true")
    parser.add_argument("--include_aug", action="store_true",
                        help="Include _aug_ rotated images (excluded by default)")
    args = parser.parse_args()

    pc_path    = Path(args.patchcore_path)
    model_name = pc_path.parent.parent.name   # e.g. ir_module_WR50_L2-3_PS3_1024_aug_rot360_p0.1
    out_dir    = OUTPUT_BASE / model_name

    inspector = PCBInspector(device=args.device, anomaly_thresh=args.threshold,
                             suppress_dilation=args.suppress_dilation,
                             patchcore_path=pc_path,
                             pc_resize=args.resize, pc_crop=args.cropsize)

    if args.save:
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"Output dir: {out_dir}\n")

    if args.image:
        r = inspector.inspect(args.image)
        print(f"[{r['verdict']}] score={r['anomaly_score']:.4f}"
              + (f"  issues={r['issues']}" if r["issues"] else ""))
        if args.save:
            out = out_dir / f"{Path(args.image).stem}_result.jpg"
            cv2.imwrite(str(out), r["vis"])
            print(f"Saved → {out}")
        return

    # ── Folder mode ───────────────────────────────────────────────────────────
    image_paths = _collect_images(args.image_dir, args.include_aug)
    if not image_paths:
        print("No images found.")
        return

    print(f"Running inference on {len(image_paths)} images...")
    all_raw = []
    all_results = []
    for p in image_paths:
        r = inspector.run(str(p))
        all_raw.append(r["raw_heatmap"])
        all_results.append((p, r))

    print("Rendering...")
    for p, r in all_results:
        surface_fail = (inspector.anomaly_thresh is not None
                        and r["anomaly_score"] > inspector.anomaly_thresh)
        verdict = "NG" if (r["issues"] or surface_fail) else "OK"

        vis = _render(
            r["orig_bgr"], r["yolo_bgr"], r["raw_heatmap"],
            r["anomaly_score"], verdict, r["issues"],
            surface_fail=surface_fail, shared_lo=None, shared_hi=None,
        )
        marker = "!!" if verdict == "NG" else "  "
        miss   = f"  issues={r['issues']}" if r["issues"] else ""
        counts = r["detected_counts"]
        print(f"  {marker} [{verdict}] {p.name:35s}  score={r['anomaly_score']:.4f}"
              f"  det={counts}{miss}")

        if args.save:
            out = out_dir / f"{p.stem}_result.jpg"
            cv2.imwrite(str(out), vis)


if __name__ == "__main__":
    main()
