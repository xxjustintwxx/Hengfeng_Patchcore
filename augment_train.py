"""
Generate augmented training images for PatchCore.

Augmentations applied (no flip, no crop/zoom per project rules):
  - Rotate 90 / 180 / 270 degrees  (skip with --no_fixed_rot)
  - N random small rotations (-20 to +20 deg, reflection padding)
  - Color jitter: brightness + contrast + saturation  (stronger than before)
  - Gaussian noise

Usage:
  # Original rotation-heavy augmentation
  python augment_train.py

  # No rotation, heavy color jitter (for lighting/color-shift robustness)
  python augment_train.py --no_fixed_rot --n_random_rot 0 --n_jitter 6 --n_noise 1
"""
import argparse
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance

VALID_EXTS = {".jpg", ".jpeg", ".png"}


def rotate_fixed(img: Image.Image, angle: int) -> Image.Image:
    return img.rotate(angle, expand=False)


def rotate_random(img: Image.Image, max_deg: float = 20.0) -> Image.Image:
    angle = random.uniform(-max_deg, max_deg)
    return img.rotate(angle, resample=Image.BICUBIC, expand=False)


def jitter(img: Image.Image, strong: bool = False) -> Image.Image:
    if strong:
        img = ImageEnhance.Brightness(img).enhance(random.uniform(0.5, 1.8))
        img = ImageEnhance.Contrast(img).enhance(random.uniform(0.6, 1.6))
        img = ImageEnhance.Color(img).enhance(random.uniform(0.5, 1.6))   # saturation
    else:
        img = ImageEnhance.Brightness(img).enhance(random.uniform(0.7, 1.4))
        img = ImageEnhance.Contrast(img).enhance(random.uniform(0.8, 1.3))
    return img


def add_noise(img: Image.Image, std: float = 8.0) -> Image.Image:
    arr = np.array(img).astype(np.float32)
    noise = np.random.normal(0, std, arr.shape)
    arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def augment_image(img: Image.Image, n_random_rot: int, n_jitter: int, n_noise: int,
                  max_deg: float = 20.0, no_fixed_rot: bool = False,
                  strong_jitter: bool = False):
    variants = []
    if not no_fixed_rot:
        for angle in [90, 180, 270]:
            variants.append((f"rot{angle}", rotate_fixed(img, angle)))
    for i in range(n_random_rot):
        variants.append((f"randrot{i+1}", rotate_random(img, max_deg)))
    for i in range(n_jitter):
        variants.append((f"jitter{i+1}", jitter(img.copy(), strong=strong_jitter)))
    for i in range(n_noise):
        variants.append((f"noise{i+1}", add_noise(img.copy())))
    return variants


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", type=str,
                        default="data/ir_module_1024/ir_module/train/good")
    parser.add_argument("--n_random_rot",  type=int,   default=2)
    parser.add_argument("--max_deg",       type=float, default=20.0,
                        help="Max rotation angle for random rotations (use 180 for full 360° coverage)")
    parser.add_argument("--no_fixed_rot",  action="store_true",
                        help="Skip fixed 90/180/270 rotations")
    parser.add_argument("--n_jitter",      type=int,   default=1)
    parser.add_argument("--strong_jitter", action="store_true",
                        help="Wider brightness/contrast + saturation jitter")
    parser.add_argument("--n_noise",       type=int,   default=1)
    parser.add_argument("--seed",          type=int,   default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    src = Path(args.src)
    originals = sorted([f for f in src.iterdir() if f.suffix.lower() in VALID_EXTS
                        and "_aug_" not in f.stem])

    n_fixed   = 0 if args.no_fixed_rot else 3
    n_aug_per = n_fixed + args.n_random_rot + args.n_jitter + args.n_noise
    max_deg   = args.max_deg
    print(f"Source: {src}")
    print(f"Original images : {len(originals)}")
    print(f"Fixed rotations : {'none (--no_fixed_rot)' if args.no_fixed_rot else '90/180/270'}")
    print(f"Jitter strength : {'strong (brightness+contrast+saturation)' if args.strong_jitter else 'normal (brightness+contrast)'}")
    print(f"Augmentations   : {n_aug_per} per image")
    print(f"New images added: {len(originals) * n_aug_per}")
    print(f"Total training  : {len(originals) * (1 + n_aug_per)}")
    print()

    generated = 0
    for f in originals:
        img = Image.open(f).convert("RGB")
        variants = augment_image(img, args.n_random_rot, args.n_jitter, args.n_noise,
                                 max_deg, args.no_fixed_rot, args.strong_jitter)
        for tag, aug_img in variants:
            out = src / f"{f.stem}_aug_{tag}{f.suffix}"
            aug_img.save(out, quality=95)
            print(f"  {out.name}")
            generated += 1

    print(f"\nDone. {generated} augmented images written to {src}")


if __name__ == "__main__":
    main()
