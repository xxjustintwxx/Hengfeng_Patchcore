"""
Inspect YOLO detection confidences for a raw capture, using the exact same
crop/resize + class taxonomy as the live pipeline (live_infer.py). Useful for
calibrating per-class confidence thresholds (configs/<module>/live_config.yaml's
yolo.class_conf) directly, without re-running the whole app or asking for help.

Usage:
  python check_yolo_confidence.py --config configs/CT11_Power/Front/live_config.yaml --image captures/CT11_Power/Front/<ts>_raw.jpg
  python check_yolo_confidence.py --config configs/CT11_Power/Front/live_config.yaml --image captures/CT11_Power/Front/<ts>_raw.jpg --class resistor
"""
import argparse
import os

os.environ.setdefault("OMP_NUM_THREADS", str(os.cpu_count()))

import cv2
from ultralytics import YOLO

from live_infer import CLASS_NAMES as DEFAULT_CLASS_NAMES
from live_infer import crop_and_resize, load_config, _run_yolo


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
    expected = {int(k): v for k, v in cfg_yolo.get("expected_counts", {}).items()}

    # A fresh, uncached load -- this is a one-shot CLI script, no reason to
    # share app.py's model cache. _run_yolo (not a manual model() call) so
    # this always sees exactly what the live pipeline sees, including the
    # board-masking gray-border padding, per-class/cross-class/containment
    # NMS, etc. -- no separate copy of that logic to keep in sync.
    model = YOLO(cfg_yolo["weights"])
    _, _, _, _, _, raw_detections = _run_yolo(model, pcb_bgr, cfg_yolo)

    print(f"Config:  {args.config}")
    print(f"Image:   {args.image}")
    print(f"Weights: {cfg_yolo['weights']}")
    print()

    for cid, name in enumerate(class_names):
        if args.class_filter and name != args.class_filter:
            continue
        thresh = class_conf.get(cid, global_conf)
        dets = raw_detections.get(name, [])
        exp_str = f" (expected {expected[cid]})" if cid in expected else ""
        if not dets:
            print(f"=== {name} (class {cid})  threshold={thresh}  0 raw detections{exp_str} ===")
            print()
            continue
        kept = sum(1 for d in dets if d["pass_threshold"])
        print(f"=== {name} (class {cid})  threshold={thresh}  kept={kept}/{len(dets)}{exp_str} ===")
        for d in dets:
            flag = "PASS  " if d["pass_threshold"] else "reject"
            near = "  <-- within 0.05 of threshold" if abs(d["conf"] - thresh) < 0.05 else ""
            nms = "" if d["nms_kept"] else "  <-- dropped by NMS"
            print(f"  {flag}  conf={d['conf']:.4f}  box={d['box']}{near}{nms}")
        print()


if __name__ == "__main__":
    main()
