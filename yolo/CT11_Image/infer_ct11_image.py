"""
CT11_Image inspection: YOLOv11-seg + PatchCore heatmap.

Classes (6):  0=Main IC  1=board  2=component  3=connector  4=resistor  5=screw
Expected:     2          —        10            1            40          1

Usage:
  # Single image
  python yolo/CT11_Image/infer_ct11_image.py --image data/CT11_Image/ir_module_1024/ir_module/test/defect/IMG-defect-01.jpg

  # Full test folder (good + defect), save heatmaps
  python yolo/CT11_Image/infer_ct11_image.py --image_dir data/CT11_Image/ir_module_1024/ir_module/test/defect --save
  python yolo/CT11_Image/infer_ct11_image.py --image_dir data/CT11_Image/ir_module_1024/ir_module/test/good  --save
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "yolo" / "640C"))
sys.path.insert(0, str(ROOT / "src"))

import infer_pcb

# ── CT11_Image overrides ───────────────────────────────────────────────────────
infer_pcb.YOLO_WEIGHTS   = ROOT / "models/yolo/CT11_Image/pcb_seg/weights/best.pt"
infer_pcb.PATCHCORE_PATH = ROOT / "models/patchcore/CT11_Image/CT11_Image_WR50_L2-3_PS3_1024_m4_p0.1/models/mvtec_ir_module"
infer_pcb.OUTPUT_BASE    = ROOT / "results/heatmaps/CT11_Image"

infer_pcb.CLASS_NAMES = ["Main IC", "board", "component", "connector", "resistor", "screw"]

infer_pcb.CLASS_COLORS = {
    0: (255, 100,   0),   # Main IC   — orange
    1: (150, 150, 150),   # board     — gray (structural outline)
    2: ( 50, 180, 255),   # component — light blue
    3: (  0, 200,   0),   # connector — green
    4: (  0, 200, 200),   # resistor  — yellow
    5: (180,   0, 255),   # screw     — purple
}

infer_pcb.LABEL_CLASSES = {0, 2, 3, 5}   # skip board (structural) and resistor (40 is too many)

infer_pcb.EXPECTED_EXACT_COUNT = {
    0: 2,    # Main IC
    2: 10,   # component
    3: 1,    # connector
    4: 40,   # resistor
    5: 1,    # screw
}

infer_pcb.CLASS_CONF = {
    0: 0.25,   # Main IC
    1: 0.50,   # board
    2: 0.25,   # component
    3: 0.25,   # connector
    4: 0.40,   # resistor — raised to reduce false detections
    5: 0.25,   # screw
}

infer_pcb.SUPPRESS_CLASSES     = set()   # no suppression for CT11_Image
infer_pcb.SUPPRESS_DILATION_PX = 0
infer_pcb.CLASS_NMS_IOU        = {}

infer_pcb.BOARD_CLASS_ID    = 1    # board polygon → suppress background in heatmap
infer_pcb.BOARD_DILATION_PX = 20   # grow board mask outward by 20 px before masking
infer_pcb.BOARD_PAD_PX      = 128  # pad 1024→1280 so PCB boundary is never at image edge

infer_pcb.ANOMALY_THRESH = 190.0   # placeholder — calibrate after seeing scores

if __name__ == "__main__":
    infer_pcb.main()
