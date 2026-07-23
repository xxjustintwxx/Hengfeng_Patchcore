"""
CT11 Back inspection: YOLOv11-seg + PatchCore heatmap.

Classes (3):  0=component  1=connecter  2=resistor
Expected:     1            1            61

Usage:
  # Single image
  python yolo/infer_ct11_back.py --image data/CT11/Back/ir_module_1024/ir_module/test/good/IMG-37.jpg

  # Full test folder, save heatmaps
  python yolo/infer_ct11_back.py --image_dir data/CT11/Back/ir_module_1024/ir_module/test/good --save
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "yolo" / "640C"))
sys.path.insert(0, str(ROOT / "src"))

import infer_pcb

# ── CT11 Back overrides ────────────────────────────────────────────────────────
infer_pcb.YOLO_WEIGHTS   = ROOT / "models/yolo/CT11/Back/pcb_seg/weights/best.pt"
infer_pcb.PATCHCORE_PATH = ROOT / "results/IR_Module/CT11/CT11_Back_WR50_L2-3_PS3_1024_m4_p0.1/models/mvtec_ir_module"
infer_pcb.OUTPUT_BASE    = ROOT / "results/heatmaps/CT11/Back"

infer_pcb.CLASS_NAMES = ["component", "connecter", "resistor"]

infer_pcb.CLASS_COLORS = {
    0: ( 50, 180, 255),   # component — light blue
    1: (  0, 200,   0),   # connecter — green
    2: (  0, 200, 200),   # resistor  — yellow
}

infer_pcb.LABEL_CLASSES = {0, 1}   # skip resistor labels (61 is too many)

infer_pcb.EXPECTED_EXACT_COUNT = {
    0: 1,    # component
    1: 1,    # connecter
    2: 61,   # resistor
}

infer_pcb.CLASS_CONF = {
    0: 0.25,   # component
    1: 0.25,   # connecter
    2: 0.40,   # resistor — raised to reduce false detections
}

infer_pcb.SUPPRESS_CLASSES     = set()   # no suppression for CT11 Back
infer_pcb.SUPPRESS_DILATION_PX = 0
infer_pcb.CLASS_NMS_IOU        = {}

infer_pcb.ANOMALY_THRESH = 190.0   # placeholder — calibrate after seeing scores

if __name__ == "__main__":
    infer_pcb.main()
