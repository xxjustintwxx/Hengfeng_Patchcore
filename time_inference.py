"""
Compare FlatL2 vs IVF-PQ inference speed and score accuracy.

Loads the saved model, converts the memory bank to IVF-PQ in-memory,
then times inference on all original test images with both index types.

Usage:
  python time_inference.py
  python time_inference.py --model_path models/patchcore/<module>/.../models/mvtec_ir_module
  python time_inference.py --nprobe 32
"""
import argparse
import glob
import os
import time
from pathlib import Path

import faiss
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

import patchcore.backbones
import patchcore.common
import patchcore.patchcore

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]
VALID_EXTS    = {".jpg", ".jpeg", ".png", ".bmp"}


def find_latest_model(results_root="./results"):
    pattern = os.path.join(results_root, "**", "patchcore_params.pkl")
    candidates = glob.glob(pattern, recursive=True)
    if not candidates:
        return None
    return os.path.dirname(max(candidates, key=os.path.getmtime))


def collect_test_images(data_root):
    items = []
    for f in sorted(Path(data_root, "test", "good").iterdir()):
        if f.suffix.lower() in VALID_EXTS and "_aug_" not in f.stem:
            items.append((str(f), "OK"))
    defect_dir = Path(data_root, "test", "defect")
    if defect_dir.exists():
        for f in sorted(defect_dir.iterdir()):
            if f.suffix.lower() in VALID_EXTS and "_aug_" not in f.stem:
                items.append((str(f), "NG"))
    return items


def preprocess(img_path, resize, cropsize):
    tf = transforms.Compose([
        transforms.Resize(resize),
        transforms.CenterCrop(cropsize),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    pil = Image.open(img_path).convert("RGB")
    return tf(pil).unsqueeze(0)


def extract_vectors(flat_index):
    n = flat_index.ntotal
    dim = flat_index.d
    print(f"  Memory bank: {n:,} vectors x {dim}-dim")
    return flat_index.reconstruct_n(0, n), dim


def build_ivfpq_index(vectors, dim, nprobe):
    """IVF-PQ: approximate cell search + quantized distances."""
    n_centroids = 512
    index = faiss.IndexIVFPQ(faiss.IndexFlatL2(dim), dim, n_centroids, 64, 8)
    print(f"  Training IVF-PQ (n_centroids={n_centroids})...")
    t0 = time.time()
    index.train(vectors)
    index.add(vectors)
    index.nprobe = nprobe
    print(f"  IVF-PQ built in {time.time() - t0:.1f}s  (nprobe={nprobe})")
    return index


def build_ivfflat_index(vectors, dim, nprobe):
    """IVF-Flat: approximate cell search + exact distances (no quantization error)."""
    n_centroids = 512
    index = faiss.IndexIVFFlat(faiss.IndexFlatL2(dim), dim, n_centroids, faiss.METRIC_L2)
    print(f"  Training IVF-Flat (n_centroids={n_centroids})...")
    t0 = time.time()
    index.train(vectors)
    index.add(vectors)
    index.nprobe = nprobe
    print(f"  IVF-Flat built in {time.time() - t0:.1f}s  (nprobe={nprobe})")
    return index


def run_inference(pc, items, resize, cropsize, device, label):
    scores = []
    times = []
    for img_path, _ in items:
        tensor = preprocess(img_path, resize, cropsize).to(device)
        t0 = time.time()
        s, _ = pc._predict(tensor)
        elapsed = time.time() - t0
        scores.append(float(s[0]))
        times.append(elapsed)
    print(f"\n[{label}]")
    print(f"  Per-image time: mean={np.mean(times):.1f}s  min={np.min(times):.1f}s  max={np.max(times):.1f}s")
    return scores, times


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--data_root",  type=str, default=None)
    parser.add_argument("--resize",     type=int, default=None)
    parser.add_argument("--device",     type=str, default="cuda")
    parser.add_argument("--nprobe",     type=int, default=16,
                        help="IVF-PQ nprobe (cells to search, higher=more accurate)")
    args = parser.parse_args()

    model_path = args.model_path or find_latest_model()
    if model_path is None:
        print("ERROR: No saved model found.")
        return
    print(f"Model: {model_path}")

    log_group = Path(model_path).parts[-3]
    resize = args.resize
    if resize is None:
        digit_parts = [p for p in log_group.split("_") if p.isdigit()]
        resize = int(digit_parts[-1]) if digit_parts else 256
    cropsize = resize

    data_root = args.data_root or \
        f"/work/xxjustin77xx/Hengfeng_Patchcore/data/ir_module_{resize}/ir_module"
    print(f"Data root: {data_root}  resize={resize}")

    device = torch.device(args.device)
    nn_method = patchcore.common.FaissNN(False, 4)
    pc = patchcore.patchcore.PatchCore(device)
    pc.load_from_path(model_path, device, nn_method)
    pc.eval()

    items = collect_test_images(data_root)
    print(f"Test images: {len(items)} (originals only)\n")

    flat_index = pc.anomaly_scorer.nn_method.search_index

    # FlatL2 baseline (exact, p=0.1)
    scores_flat, times_flat = run_inference(pc, items, resize, cropsize, device, "FlatL2 p=0.1 (exact)")

    # Extract vectors once for all index variants
    print("\nExtracting memory bank vectors...")
    vectors, dim = extract_vectors(flat_index)

    # IVF-Flat
    print("\nBuilding IVF-Flat index...")
    ivfflat_index = build_ivfflat_index(vectors, dim, args.nprobe)
    pc.anomaly_scorer.nn_method.search_index = ivfflat_index
    scores_ivfflat, times_ivfflat = run_inference(
        pc, items, resize, cropsize, device, f"IVF-Flat nprobe={args.nprobe}"
    )

    # Subsampled FlatL2 at p=0.05 and p=0.01
    results_sub = {}
    for p in [0.05, 0.01]:
        n_keep = int(len(vectors) * p / 0.1)
        idx = np.random.choice(len(vectors), n_keep, replace=False)
        sub_index = faiss.IndexFlatL2(dim)
        sub_index.add(vectors[idx])
        pc.anomaly_scorer.nn_method.search_index = sub_index
        scores_sub, times_sub = run_inference(
            pc, items, resize, cropsize, device,
            f"FlatL2 p={p} ({n_keep:,} vectors)"
        )
        results_sub[p] = (scores_sub, times_sub)

    # Score comparison table
    all_scores = [scores_flat, scores_ivfflat, results_sub[0.05][0], results_sub[0.01][0]]
    all_times  = [times_flat,  times_ivfflat,  results_sub[0.05][1], results_sub[0.01][1]]
    cols = ["p=0.1", f"IVF-Flat(np={args.nprobe})", "p=0.05", "p=0.01"]

    print(f"\n--- Score comparison ---")
    print(f"  {'Image':35s}  {'p=0.1':>8s}  {'IVF-Flat':>8s}  {'p=0.05':>8s}  {'p=0.01':>8s}  label")
    for i, (img_path, lbl) in enumerate(items):
        marker = "!!" if lbl == "NG" else "  "
        print(f"  {marker} {Path(img_path).name:33s}"
              f"  {scores_flat[i]:8.2f}"
              f"  {scores_ivfflat[i]:8.2f}"
              f"  {results_sub[0.05][0][i]:8.2f}"
              f"  {results_sub[0.01][0][i]:8.2f}  [{lbl}]")

    print(f"\n{'':38s}  {'  '.join(f'{c:>14s}' for c in cols)}")
    ok_lists = [[s for s, (_, l) in zip(sc, items) if l == "OK"] for sc in all_scores]
    ng_lists = [[s for s, (_, l) in zip(sc, items) if l == "NG"] for sc in all_scores]
    for label, fn, lst in [("OK ceiling", max, ok_lists), ("NG floor", min, ng_lists)]:
        print(f"  {label:36s}  {'  '.join(f'{fn(sl):14.2f}' for sl in lst)}")
    gaps = [min(ng) - max(ok) for ok, ng in zip(ok_lists, ng_lists)]
    print(f"  {'gap':36s}  {'  '.join(f'{g:14.2f}' for g in gaps)}")
    print(f"  {'time/image':36s}  {'  '.join(f'{np.mean(t):13.1f}s' for t in all_times)}")


if __name__ == "__main__":
    main()
