"""Download EuroSAT (RGB) and prepare train/test split for the constraint-loss pipeline.

EuroSAT: 27,000 Sentinel-2 satellite tiles, 10 land cover classes, 64x64 RGB.
Source: Helber et al. 2018, https://github.com/phelber/EuroSAT
Published benchmark accuracy:
  ResNet-50 (ImageNet pretrain, 64x64): 98.57%   [Helber et al. 2018]
  ConvNeXt / EfficientNet-B0:           97-98%
  Lower-bound (Linear-SVM on raw pixels): ~70%

Strategy:
  1. Download 27000-sample EuroSAT-RGB via torchvision
  2. Stratified 80/20 train/test split (21600 train / 5400 test)
  3. Resize 64x64 -> 224x224 for pretrained-model compatibility
  4. Fabricate balanced binary group column for local-constraint protocol
     parity with TissueMNIST/DermMNIST. Real geographic groups deferred.

Output (matches existing TissueMNIST/DermMNIST layout):
  data/eurosat/train_images.npy   (N, 3, 224, 224) float32
  data/eurosat/train_labels.npy   (N,) int64
  data/eurosat/train_meta.csv     label, class_name, synth_group
  data/eurosat/test_*.npy / *.csv

Usage:
  python data/eurosat/download_data.py
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedShuffleSplit
from torchvision import transforms
from torchvision.datasets import EuroSAT
from torch.utils.data import DataLoader

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
DOWNLOAD_DIR = os.path.join(DATA_DIR, "_torchvision_cache")
DEFAULT_TARGET_SIZE = 224
DEFAULT_N_SAMPLES = 12000  # match TissueMNIST scale; disk-budget aware
TEST_FRACTION = 0.2
SEED = 42

CLASS_NAMES = {
    0: "AnnCrop", 1: "Forest", 2: "HerbVeg", 3: "Highway", 4: "Industrial",
    5: "Pasture", 6: "PermCrop", 7: "Resid", 8: "River", 9: "SeaLake",
}
CLASS_FULL_NAMES = {
    0: "AnnualCrop", 1: "Forest", 2: "HerbaceousVegetation",
    3: "Highway", 4: "Industrial", 5: "Pasture",
    6: "PermanentCrop", 7: "Residential", 8: "River", 9: "SeaLake",
}


def download_and_assemble(target_size: int):
    os.makedirs(DOWNLOAD_DIR, exist_ok=True)
    print(f"Downloading EuroSAT to {DOWNLOAD_DIR} (one-time, ~90 MB)...")
    tx = transforms.Compose([
        transforms.Resize((target_size, target_size)),
        transforms.ToTensor(),  # -> [0,1] float, channels-first
    ])
    ds = EuroSAT(root=DOWNLOAD_DIR, download=True, transform=tx)
    print(f"  loaded {len(ds)} samples")
    loader = DataLoader(ds, batch_size=256, shuffle=False, num_workers=4)
    images = np.empty((len(ds), 3, target_size, target_size), dtype=np.float32)
    labels = np.empty(len(ds), dtype=np.int64)
    idx = 0
    for batch_imgs, batch_lbls in loader:
        n = batch_imgs.shape[0]
        images[idx:idx + n] = batch_imgs.numpy()
        labels[idx:idx + n] = batch_lbls.numpy()
        idx += n
        if (idx // 256) % 20 == 0:
            print(f"  {idx}/{len(ds)}")
    print(f"  assembled images: {images.shape} {images.dtype} "
          f"range=[{images.min():.3f}, {images.max():.3f}]")
    return images, labels


def subsample(images, labels, n_samples):
    if n_samples >= len(labels):
        return images, labels
    sss = StratifiedShuffleSplit(n_splits=1, train_size=n_samples, random_state=SEED)
    keep_idx, _ = next(sss.split(images, labels))
    print(f"  subsampled {len(labels)} -> {len(keep_idx)} (stratified)")
    return images[keep_idx], labels[keep_idx]


def make_split(images, labels):
    sss = StratifiedShuffleSplit(n_splits=1, test_size=TEST_FRACTION, random_state=SEED)
    train_idx, test_idx = next(sss.split(images, labels))
    rng = np.random.default_rng(SEED)

    def _meta(lbls):
        df = pd.DataFrame({
            "label": lbls,
            "class_name": [CLASS_NAMES[int(l)] for l in lbls],
            "synth_group": rng.integers(0, 2, size=len(lbls), dtype=np.int64),
        })
        return df

    return (images[train_idx], labels[train_idx], _meta(labels[train_idx]),
            images[test_idx], labels[test_idx], _meta(labels[test_idx]))


def save(prefix, images, labels, meta):
    np.save(os.path.join(DATA_DIR, f"{prefix}_images.npy"), images)
    np.save(os.path.join(DATA_DIR, f"{prefix}_labels.npy"), labels)
    meta.to_csv(os.path.join(DATA_DIR, f"{prefix}_meta.csv"), index=False)
    print(f"  wrote {prefix}: {images.shape}")


def shannon(labels):
    _, c = np.unique(labels, return_counts=True)
    p = c / c.sum()
    H = -np.sum(p * np.log(p))
    return H / np.log(len(c))


def print_summary():
    print("\n" + "=" * 70)
    print("EuroSAT-RGB — Dataset Summary")
    print("=" * 70)
    for split in ("train", "test"):
        labels = np.load(os.path.join(DATA_DIR, f"{split}_labels.npy"))
        images = np.load(os.path.join(DATA_DIR, f"{split}_images.npy"))
        meta = pd.read_csv(os.path.join(DATA_DIR, f"{split}_meta.csv"))
        print(f"\n{split.upper()} ({len(labels):,} samples)  images={images.shape}")
        _, counts = np.unique(labels, return_counts=True)
        for c in range(10):
            n = counts[c] if c < len(counts) else 0
            print(f"  Class {c} ({CLASS_NAMES[c]:>8}): {n:>5,} "
                  f"({100 * n / len(labels):5.1f}%)  {CLASS_FULL_NAMES[c]}")
        print(f"  Shannon equitability: {shannon(labels):.4f}")
        print(f"  Synth groups: 0={sum(meta['synth_group'] == 0)}, "
              f"1={sum(meta['synth_group'] == 1)}")
    all_labels = np.concatenate([
        np.load(os.path.join(DATA_DIR, f"{s}_labels.npy")) for s in ("train", "test")
    ])
    _, counts = np.unique(all_labels, return_counts=True)
    sorted_classes = np.argsort(counts)
    print("\nConstrained class candidates (minority first):")
    for c in sorted_classes[:4]:
        print(f"  Class {c} ({CLASS_NAMES[c]}): {counts[c]:,}  "
              f"{CLASS_FULL_NAMES[c]}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=DEFAULT_TARGET_SIZE)
    ap.add_argument("--n_samples", type=int, default=DEFAULT_N_SAMPLES)
    args = ap.parse_args()
    images, labels = download_and_assemble(args.size)
    images, labels = subsample(images, labels, args.n_samples)
    train_x, train_y, train_meta, test_x, test_y, test_meta = make_split(images, labels)
    save("train", train_x, train_y, train_meta)
    save("test", test_x, test_y, test_meta)
    # Also point slice_1 at the same data (cheap symlink-style copy via os.symlink)
    slice1 = os.path.join(DATA_DIR, "slice_1")
    os.makedirs(slice1, exist_ok=True)
    for f in ("train_images.npy", "train_labels.npy", "train_meta.csv",
              "test_images.npy", "test_labels.npy", "test_meta.csv"):
        src = os.path.join(DATA_DIR, f)
        dst = os.path.join(slice1, f)
        if os.path.exists(dst) or os.path.islink(dst):
            os.remove(dst)
        os.symlink(src, dst)
    print(f"slice_1/ -> top-level data/ (symlinked)")
    print_summary()


if __name__ == "__main__":
    main()
