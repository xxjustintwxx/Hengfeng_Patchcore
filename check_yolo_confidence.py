"""
Inspect YOLO detection confidences for a raw capture, using the exact same
crop/resize + class taxonomy as the live pipeline (live_infer.py). Useful for
calibrating per-class confidence thresholds (configs/<module>/live_config.yaml's
yolo.class_conf) directly, without re-running the whole app or asking for help.

Usage:
  python check_yolo_confidence.py --config configs/CT11/Front/live_config.yaml --image captures/CT11/Front/<ts>_raw.jpg
  python check_yolo_confidence.py --config configs/CT11/Front/live_config.yaml --image captures/CT11/Front/<ts>_raw.jpg --class resistor
"""
import argparse
import os

os.environ.setdefault("OMP_NUM_THREADS", str(os.cpu_count()))

import cv2
import numpy as np
import PIL.Image
from ultralytics import YOLO

from live_infer import CLASS_NAMES as DEFAULT_CLASS_NAMES
from live_infer import crop_and_resize, load_config


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True, help="Path to a live_config.yaml profile")
    parser.add_argument("--image", required=True,
                        help="Path to a raw capture (e.g. captures/<module>/<ts>_raw.jpg)")
    parser.add_argument("--class", dest="class_filter", default=None,
                        help="Only show this class name (default: show every class)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    raw = cv2.imread(args.image)
    if raw is None:
        raise SystemExit(f"Could not read image: {args.image}")

    roi = cfg["preprocessing"].get("roi")
    output_size = tuple(cfg["preprocessing"]["output_size"])
    pcb_bgr = crop_and_resize(raw, roi, output_size)

    cfg_yolo = cfg["yolo"]
    class_names = cfg_yolo.get("class_names", DEFAULT_CLASS_NAMES)
    class_conf = {int(k): v for k, v in cfg_yolo.get("class_conf", {}).items()}
    global_conf = cfg_yolo.get("conf", 0.25)

    # Board masking (CT11) pads the image with a gray border before running
    # YOLO -- the models were retrained expecting this exact padding, so
    # matching it here keeps these confidence numbers representative of what
    # the live pipeline (_run_yolo) actually sees.
    board_pad = cfg_yolo.get("board_pad", 128) if cfg_yolo.get("board_class") else 0
    if board_pad > 0:
        H, W = pcb_bgr.shape[:2]
        padded = cv2.copyMakeBorder(pcb_bgr, board_pad, board_pad, board_pad, board_pad,
                                    cv2.BORDER_CONSTANT, value=(114, 114, 114))
        pil_img = PIL.Image.fromarray(cv2.cvtColor(padded, cv2.COLOR_BGR2RGB))
        run_imgsz = W + 2 * board_pad
    else:
        pil_img = PIL.Image.fromarray(cv2.cvtColor(pcb_bgr, cv2.COLOR_BGR2RGB))
        run_imgsz = cfg_yolo.get("imgsz", 1280)

    model = YOLO(cfg_yolo["weights"])
    res = model(pil_img, conf=global_conf, iou=cfg_yolo.get("nms_iou", 0.20),
                imgsz=run_imgsz, verbose=False)[0]

    print(f"Config:  {args.config}")
    print(f"Image:   {args.image}")
    print(f"Weights: {cfg_yolo['weights']}")
    expected = {int(k): v for k, v in cfg_yolo.get("expected_counts", {}).items()}
    print()

    if res.boxes is None or len(res.boxes) == 0:
        print("No detections at all (below the global conf floor).")
        return

    cls_ids = res.boxes.cls.cpu().numpy().astype(int)
    confs = res.boxes.conf.cpu().numpy()
    boxes = res.boxes.xyxy.cpu().numpy()
    if board_pad > 0:
        boxes = boxes - board_pad   # back to pcb_bgr's coordinate space

    for cid in sorted(set(cls_ids)):
        name = class_names[cid] if cid < len(class_names) else f"class_{cid}"
        if args.class_filter and name != args.class_filter:
            continue
        thresh = class_conf.get(cid, global_conf)
        idx = np.where(cls_ids == cid)[0]
        order = idx[np.argsort(-confs[idx])]
        kept = sum(1 for i in order if confs[i] >= thresh)
        exp_str = f" (expected {expected[cid]})" if cid in expected else ""
        print(f"=== {name} (class {cid})  threshold={thresh}  kept={kept}/{len(order)}{exp_str} ===")
        for i in order:
            x1, y1, x2, y2 = boxes[i].astype(int)
            flag = "PASS  " if confs[i] >= thresh else "reject"
            near = "  <-- within 0.05 of threshold" if abs(confs[i] - thresh) < 0.05 else ""
            print(f"  {flag}  conf={confs[i]:.4f}  box=[{x1},{y1},{x2},{y2}]{near}")
        print()


if __name__ == "__main__":
    main()
