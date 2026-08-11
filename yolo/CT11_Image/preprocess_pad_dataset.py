"""
Pad CT11_Image YOLO dataset with a 128px gray border on all sides.

  1024×1024 → 1280×1280  (matches imgsz=1280, no internal rescaling during training)

Output folder:
  data/CT11_Image/CT11_Image_padded/
"""
from pathlib import Path
import numpy as np
import cv2

ROOT = Path(__file__).resolve().parent.parent.parent
PAD  = 128

SRC = ROOT / "data/CT11_Image/CT11_Image"
DST = ROOT / "data/CT11_Image/CT11_Image_padded"

NC    = 6
NAMES = "['Main IC', 'board', 'component', 'connector', 'resistor', 'screw']"

print(f"Processing {SRC.name} → {DST.name}")

for split in ["train", "test"]:
    img_src = SRC / split / "images"
    lbl_src = SRC / split / "labels"
    img_dst = DST / split / "images"
    lbl_dst = DST / split / "labels"
    img_dst.mkdir(parents=True, exist_ok=True)
    lbl_dst.mkdir(parents=True, exist_ok=True)

    img_paths = sorted(img_src.glob("*.jpg")) + sorted(img_src.glob("*.png"))
    print(f"  {split}: {len(img_paths)} images")

    for img_path in img_paths:
        img = cv2.imread(str(img_path))
        h, w = img.shape[:2]
        new_w, new_h = w + 2*PAD, h + 2*PAD

        padded = np.full((new_h, new_w, 3), 114, dtype=np.uint8)
        padded[PAD:PAD+h, PAD:PAD+w] = img
        cv2.imwrite(str(img_dst / img_path.name), padded)

        lbl_path = lbl_src / (img_path.stem + ".txt")
        if not lbl_path.exists():
            continue
        new_lines = []
        for line in lbl_path.read_text().strip().splitlines():
            parts = line.split()
            if not parts:
                continue
            cls_id = parts[0]
            coords = list(map(float, parts[1:]))
            new_coords = []
            for i in range(0, len(coords), 2):
                x_new = (coords[i]   * w + PAD) / new_w
                y_new = (coords[i+1] * h + PAD) / new_h
                new_coords.extend([x_new, y_new])
            new_lines.append(cls_id + " " + " ".join(f"{v:.6f}" for v in new_coords))
        (lbl_dst / lbl_path.name).write_text("\n".join(new_lines) + "\n")

yaml_text = (
    f"train: train/images\n"
    f"val: test/images\n"
    f"test: test/images\n"
    f"\n"
    f"nc: {NC}\n"
    f"names: {NAMES}\n"
)
(DST / "data.yaml").write_text(yaml_text)
print(f"  data.yaml written → {DST / 'data.yaml'}")
print("\nDone. Padded dataset ready.")
