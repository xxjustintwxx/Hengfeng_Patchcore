"""
CT11_Power Back inspection: YOLOv11-seg + PatchCore heatmap.

Classes (4):  0=board  1=component  2=connector  3=resistor
Expected:     —        1            1             61

Usage:
  # Single image
  python yolo/infer_ct11_back.py --image data/CT11_Power/Back/ir_module_1024/ir_module/test/good/IMG-37.jpg

  # Full test folder, save heatmaps
  python yolo/infer_ct11_back.py --image_dir data/CT11_Power/Back/ir_module_1024/ir_module/test/good --save
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "yolo" / "640C"))
sys.path.insert(0, str(ROOT / "src"))

import infer_pcb

# ── CT11_Power Back overrides ────────────────────────────────────────────────────────
infer_pcb.YOLO_WEIGHTS   = ROOT / "models/yolo/CT11_Power/Back/pcb_seg/weights/best.pt"
infer_pcb.PATCHCORE_PATH = ROOT / "models/patchcore/CT11_Power/CT11_Back_WR50_L2-3_PS3_1024_m4_p0.1/models/mvtec_ir_module"
infer_pcb.OUTPUT_BASE    = ROOT / "results/heatmaps/CT11_Power/Back"

infer_pcb.CLASS_NAMES = ["board", "component", "connector", "resistor"]

infer_pcb.CLASS_COLORS = {
    0: (150, 150, 150),   # board     — gray (structural outline)
    1: ( 50, 180, 255),   # component — light blue
    2: (  0, 200,   0),   # connector — green
    3: (  0, 200, 200),   # resistor  — yellow
}

infer_pcb.LABEL_CLASSES = {1, 2}   # skip board (structural) and resistor (61 is too many)

infer_pcb.EXPECTED_EXACT_COUNT = {
    1: 1,    # component
    2: 1,    # connector
    3: 61,   # resistor
}

infer_pcb.CLASS_CONF = {
    0: 0.50,   # board
    1: 0.25,   # component
    2: 0.25,   # connector
    3: 0.40,   # resistor — raised to reduce false detections
}

infer_pcb.SUPPRESS_CLASSES     = set()   # no suppression for CT11_Power Back
infer_pcb.SUPPRESS_DILATION_PX = 0
infer_pcb.CLASS_NMS_IOU        = {}

infer_pcb.BOARD_CLASS_ID    = 0    # board polygon → suppress background in heatmap
infer_pcb.BOARD_DILATION_PX = 20   # grow board mask outward by 20 px before masking
infer_pcb.BOARD_PAD_PX      = 128  # pad 1024→1280 so PCB boundary is never at image edge

infer_pcb.ANOMALY_THRESH = 190.0   # placeholder — calibrate after seeing scores

if __name__ == "__main__":
    infer_pcb.main()
