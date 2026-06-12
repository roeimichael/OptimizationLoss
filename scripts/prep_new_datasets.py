"""Prep new dataset candidates: MedMNIST (retinamnist/bloodmnist) and CIFAR-100.

Saves as uint8 NHWC -> data/<ds>/slice_1/{train,test}_{images,labels}.npy + meta.csv.
Group column = synth_group (round-robin 3 partitions) for local-cap support.

Usage:
  python scripts/prep_new_datasets.py retinamnist
  python scripts/prep_new_datasets.py bloodmnist
  python scripts/prep_new_datasets.py cifar100
  python scripts/prep_new_datasets.py cifar100 --sigma 0.20  # contamination
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
from PIL import Image

IMG_SIZE = 224


def save_slice(ds_name, train_x, train_y, test_x, test_y, suffix=""):
    out = f"data/{ds_name}{suffix}/slice_1"
    os.makedirs(out, exist_ok=True)
    for name, x, y in (("train", train_x, train_y), ("test", test_x, test_y)):
        np.save(f"{out}/{name}_images.npy", x)
        np.save(f"{out}/{name}_labels.npy", y)
        groups = np.arange(len(y)) % 3
        meta = pd.DataFrame({
            "label": y, "class_name": [f"c{int(c)}" for c in y],
            "filename": [f"{name}_{i:06d}.jpg" for i in range(len(y))],
            "synth_group": groups,
        })
        meta.to_csv(f"{out}/{name}_meta.csv", index=False)
    print(f"  saved {ds_name}{suffix}: train={len(train_y)} test={len(test_y)} "
          f"classes={len(np.unique(np.concatenate([train_y, test_y])))} "
          f"size={os.popen(f'du -sh {out}').read().strip()}")


def resize_uint8(arr_nhwc):
    """Resize NHWC uint8 to 224x224x3 uint8."""
    if arr_nhwc.ndim == 3:                     # (N,H,W) grayscale
        arr_nhwc = np.repeat(arr_nhwc[..., None], 3, axis=-1)
    if arr_nhwc.shape[-1] == 1:
        arr_nhwc = np.repeat(arr_nhwc, 3, axis=-1)
    n, H, W, C = arr_nhwc.shape
    out = np.empty((n, IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
    for i in range(n):
        out[i] = np.asarray(Image.fromarray(arr_nhwc[i]).resize(
            (IMG_SIZE, IMG_SIZE), Image.BILINEAR))
        if (i+1) % 5000 == 0: print(f"    resize {i+1}/{n}", flush=True)
    return out


def prep_medmnist(name):
    import medmnist
    from medmnist import INFO
    info = INFO[name]
    DataClass = getattr(medmnist, info["python_class"])
    print(f"  downloading {name} ({info['n_samples']})...")
    tr = DataClass(split="train", download=True, size=28)
    te = DataClass(split="test",  download=True, size=28)
    val= DataClass(split="val",   download=True, size=28)
    train_x_raw = tr.imgs
    test_x_raw  = te.imgs
    val_x_raw   = val.imgs
    train_y = tr.labels.squeeze().astype(np.int64)
    test_y  = te.labels.squeeze().astype(np.int64)
    val_y   = val.labels.squeeze().astype(np.int64)
    print(f"  raw shapes: train={train_x_raw.shape} test={test_x_raw.shape}")
    print(f"  resizing to 224x224 uint8...")
    train_x = resize_uint8(np.concatenate([train_x_raw, val_x_raw], axis=0))
    test_x  = resize_uint8(test_x_raw)
    train_y = np.concatenate([train_y, val_y])
    print(f"  class distribution train: {np.bincount(train_y).tolist()}")
    print(f"  class distribution test:  {np.bincount(test_y).tolist()}")
    save_slice(name, train_x, train_y, test_x, test_y)


def prep_cifar100():
    import torchvision.datasets as tvd
    print(f"  downloading CIFAR-100...")
    tr = tvd.CIFAR100(root="/tmp/cifar100", train=True, download=True)
    te = tvd.CIFAR100(root="/tmp/cifar100", train=False, download=True)
    train_y = np.array(tr.targets, dtype=np.int64)
    test_y  = np.array(te.targets, dtype=np.int64)
    train_x = tr.data  # (50000, 32, 32, 3) uint8
    test_x  = te.data  # (10000, 32, 32, 3) uint8
    print(f"  raw shapes: train={train_x.shape} test={test_x.shape}")
    print(f"  resizing to 224x224 uint8...")
    train_x = resize_uint8(train_x)
    test_x  = resize_uint8(test_x)
    print(f"  class distribution train: {np.bincount(train_y).min()} min / "
          f"{np.bincount(train_y).max()} max (balanced)")
    save_slice("cifar100", train_x, train_y, test_x, test_y)


def add_contamination(ds_name, sigma_pct):
    """Apply Gaussian noise to TEST images only (transductive contamination).
    Saved as data/<ds>_sigma<NN>/slice_1/"""
    src = f"data/{ds_name}/slice_1"
    suffix = f"_sigma{sigma_pct:02d}"
    if not os.path.isdir(src):
        print(f"  ERROR: {src} not found"); return
    print(f"  loading clean test from {src}")
    test_x = np.load(f"{src}/test_images.npy")
    test_y = np.load(f"{src}/test_labels.npy")
    train_x = np.load(f"{src}/train_images.npy")
    train_y = np.load(f"{src}/train_labels.npy")
    sigma = sigma_pct / 100.0 * 255.0
    rng = np.random.RandomState(42)
    noisy = test_x.astype(np.float32) + rng.normal(0, sigma, test_x.shape)
    noisy = np.clip(noisy, 0, 255).astype(np.uint8)
    save_slice(ds_name, train_x, train_y, noisy, test_y, suffix=suffix)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", choices=["retinamnist","bloodmnist","organamnist",
                                          "breastmnist","cifar100"])
    ap.add_argument("--sigma", type=int, default=0,
                    help="If non-zero, add this contamination sigma (% × 100) on the test side.")
    args = ap.parse_args()
    if args.sigma > 0:
        add_contamination(args.dataset, args.sigma)
    elif args.dataset == "cifar100":
        prep_cifar100()
    else:
        prep_medmnist(args.dataset)


if __name__ == "__main__":
    main()
