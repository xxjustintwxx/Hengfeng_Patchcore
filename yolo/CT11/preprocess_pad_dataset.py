"""
Pad CT11 Front and Back YOLO datasets with a 128px gray border on all sides.

  1024×1024 → 1280×1280  (matches imgsz=1280, no internal rescaling during training)

The board boundary is no longer at the image edge, giving YOLO background context
to predict it accurately.  Label coordinates are shifted proportionally.

Output folders:
  data/CT11/Front/CT11 front_padded/
  data/CT11/Back/CT11 back_padded/
"""
from pathlib import Path
import numpy as np
import cv2

ROOT = Path(__file__).resolve().parent.parent.parent
PAD  = 128   # pixels added on each side

SIDES = [
    {
        "src":   ROOT / "data/CT11/Front/CT11 front",
        "dst":   ROOT / "data/CT11/Front/CT11 front_padded",
        "nc":    5,
        "names": "['Main IC', 'board', 'component', 'connecter', 'resistor']",
    },
    {
        "src":   ROOT / "data/CT11/Back/CT11 back",
        "dst":   ROOT / "data/CT11/Back/CT11 back_padded",
        "nc":    4,
        "names": "['board', 'component', 'connecter', 'resistor']",
    },
]

for cfg in SIDES:
    src_base = cfg["src"]
    dst_base = cfg["dst"]
    print(f"\nProcessing {src_base.name} → {dst_base.name}")

    for split in ["train", "test"]:
        img_src = src_base / split / "images"
        lbl_src = src_base / split / "labels"
        img_dst = dst_base / split / "images"
        lbl_dst = dst_base / split / "labels"
        img_dst.mkdir(parents=True, exist_ok=True)
        lbl_dst.mkdir(parents=True, exist_ok=True)

        img_paths = sorted(img_src.glob("*.jpg")) + sorted(img_src.glob("*.png"))
        print(f"  {split}: {len(img_paths)} images")

        for img_path in img_paths:
            img = cv2.imread(str(img_path))
            h, w = img.shape[:2]
            new_w, new_h = w + 2*PAD, h + 2*PAD

            # Gray canvas (114 = ultralytics letterbox fill colour)
            padded = np.full((new_h, new_w, 3), 114, dtype=np.uint8)
            padded[PAD:PAD+h, PAD:PAD+w] = img
            cv2.imwrite(str(img_dst / img_path.name), padded)

            # Shift label coordinates
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
                # coords alternate x y (both normalised to original image size)
                new_coords = []
                for i in range(0, len(coords), 2):
                    x_new = (coords[i]   * w + PAD) / new_w
                    y_new = (coords[i+1] * h + PAD) / new_h
                    new_coords.extend([x_new, y_new])
                new_lines.append(cls_id + " " + " ".join(f"{v:.6f}" for v in new_coords))
            (lbl_dst / lbl_path.name).write_text("\n".join(new_lines) + "\n")

    # Write data.yaml for padded dataset
    yaml = (
        f"train: train/images\n"
        f"val: test/images\n"
        f"test: test/images\n"
        f"\n"
        f"nc: {cfg['nc']}\n"
        f"names: {cfg['names']}\n"
    )
    (dst_base / "data.yaml").write_text(yaml)
    print(f"  data.yaml written → {dst_base / 'data.yaml'}")

print("\nDone. Padded datasets ready.")
