"""Universal torchvision dataset prep for the overnight dataset hunt.

Usage:
  python -m data.dataset_hunt_prep <ds_name> <tv_class>

Where:
  ds_name   = our internal name (lowercase, underscore-separated)
  tv_class  = torchvision class name (e.g. 'Caltech256', 'FGVCAircraft'),
              or 'TinyImageNet' for the special URL-download case.

Writes to: data/<ds_name>/slice_1/{train,test}_{images,labels}.npy + meta.csv
Format matches the project's other imagery datasets: uint8 NHWC, label int,
synth_group integer (round-robin partition into 3 groups for local-constraint
probing).

If download fails, raises and bails. If dataset has no canonical train/test
split, takes the first 80%/20% by index.
"""
import os
import sys
import urllib.request
import zipfile
from io import BytesIO

import numpy as np
import pandas as pd
from PIL import Image

import torchvision.datasets as tvd

IMG_SIZE = 224


def _resize_and_collect(dataset, ds_name: str, split_name: str):
    n = len(dataset)
    images = np.empty((n, IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
    labels = np.empty(n, dtype=np.int64)
    for i in range(n):
        item = dataset[i]
        # Torchvision returns (img, label) where img is PIL or tensor
        if len(item) >= 2:
            img, lbl = item[0], item[1]
        else:
            img, lbl = item[0], 0
        if hasattr(img, 'convert'):  # PIL
            if img.mode != 'RGB':
                img = img.convert('RGB')
            img_r = img.resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
            images[i] = np.asarray(img_r)
        else:  # tensor (C,H,W) float
            arr = np.array(img)
            if arr.ndim == 3 and arr.shape[0] in (1, 3):
                arr = np.transpose(arr, (1, 2, 0))
            if arr.dtype != np.uint8:
                arr = np.clip(arr * 255, 0, 255).astype(np.uint8)
            pil = Image.fromarray(arr if arr.shape[-1] == 3
                                  else np.broadcast_to(arr, (*arr.shape[:2], 3)))
            images[i] = np.asarray(pil.resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR))
        labels[i] = int(lbl) if isinstance(lbl, (int, np.integer)) else 0
        if (i + 1) % 2000 == 0:
            print(f"  [{split_name}] {i+1}/{n}", flush=True)
    return images, labels


def _save_slice(ds_name: str, train_x, train_y, test_x, test_y):
    slice_dir = os.path.join("data", ds_name, "slice_1")
    os.makedirs(slice_dir, exist_ok=True)
    for name, x, y in (("train", train_x, train_y), ("test", test_x, test_y)):
        np.save(os.path.join(slice_dir, f"{name}_images.npy"), x)
        np.save(os.path.join(slice_dir, f"{name}_labels.npy"), y)
        groups = np.arange(len(y)) % 3
        meta = pd.DataFrame({
            "label": y,
            "class_name": [f"class_{int(c)}" for c in y],
            "filename": [f"{name}_{i:06d}.jpg" for i in range(len(y))],
            "synth_group": groups,
        })
        meta.to_csv(os.path.join(slice_dir, f"{name}_meta.csv"), index=False)
    print(f"{ds_name}: train={len(train_y)} test={len(test_y)} classes={len(np.unique(np.concatenate([train_y, test_y])))}", flush=True)


def _split_8020(images, labels, seed=42):
    rng = np.random.RandomState(seed)
    n = len(labels)
    idx = rng.permutation(n)
    cut = int(0.8 * n)
    tr, te = idx[:cut], idx[cut:]
    return images[tr], labels[tr], images[te], labels[te]


def prep_torchvision(ds_name: str, tv_class: str):
    """Try various torchvision init signatures."""
    cls = getattr(tvd, tv_class)
    root = os.path.join("data", ds_name, "raw")
    os.makedirs(root, exist_ok=True)
    # Strategy A: split='train'/'test'
    try:
        train_ds = cls(root=root, split="train", download=True)
        test_ds  = cls(root=root, split="test",  download=True)
        tr_x, tr_y = _resize_and_collect(train_ds, ds_name, "train")
        te_x, te_y = _resize_and_collect(test_ds,  ds_name, "test")
        _save_slice(ds_name, tr_x, tr_y, te_x, te_y)
        return
    except Exception as e:
        print(f"  split='train'/'test' failed: {e}", flush=True)
    # Strategy B: train=True/False
    try:
        train_ds = cls(root=root, train=True,  download=True)
        test_ds  = cls(root=root, train=False, download=True)
        tr_x, tr_y = _resize_and_collect(train_ds, ds_name, "train")
        te_x, te_y = _resize_and_collect(test_ds,  ds_name, "test")
        _save_slice(ds_name, tr_x, tr_y, te_x, te_y)
        return
    except Exception as e:
        print(f"  train=True/False failed: {e}", flush=True)
    # Strategy C: no split arg, 80/20 stratified
    try:
        full_ds = cls(root=root, download=True)
        x, y = _resize_and_collect(full_ds, ds_name, "full")
        tr_x, tr_y, te_x, te_y = _split_8020(x, y)
        _save_slice(ds_name, tr_x, tr_y, te_x, te_y)
        return
    except Exception as e:
        print(f"  no-split failed: {e}", flush=True)
    raise RuntimeError(f"Could not load {tv_class} via any torchvision strategy")


def prep_tiny_imagenet(ds_name: str):
    """Tiny ImageNet -- direct URL download, then ImageFolder-style."""
    url = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"
    raw_dir = os.path.join("data", ds_name, "raw")
    os.makedirs(raw_dir, exist_ok=True)
    zip_path = os.path.join(raw_dir, "tiny-imagenet-200.zip")
    if not os.path.exists(zip_path):
        print(f"Downloading Tiny ImageNet from {url} ...", flush=True)
        urllib.request.urlretrieve(url, zip_path)
    extracted = os.path.join(raw_dir, "tiny-imagenet-200")
    if not os.path.exists(extracted):
        print("Unzipping...", flush=True)
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(raw_dir)

    # Train: <root>/tiny-imagenet-200/train/<wnid>/images/*.JPEG
    # Val:   <root>/tiny-imagenet-200/val/images/*.JPEG + val_annotations.txt
    train_root = os.path.join(extracted, "train")
    val_root   = os.path.join(extracted, "val")
    wnids = sorted(os.listdir(train_root))
    wnid_to_idx = {w: i for i, w in enumerate(wnids)}

    train_imgs, train_lbls = [], []
    for w in wnids:
        img_dir = os.path.join(train_root, w, "images")
        for fn in sorted(os.listdir(img_dir)):
            pil = Image.open(os.path.join(img_dir, fn))
            if pil.mode != "RGB":
                pil = pil.convert("RGB")
            train_imgs.append(np.asarray(pil.resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)))
            train_lbls.append(wnid_to_idx[w])
    train_x = np.stack(train_imgs).astype(np.uint8)
    train_y = np.array(train_lbls, dtype=np.int64)

    test_imgs, test_lbls = [], []
    with open(os.path.join(val_root, "val_annotations.txt")) as f:
        for line in f:
            parts = line.strip().split("\t")
            fn, w = parts[0], parts[1]
            pil = Image.open(os.path.join(val_root, "images", fn))
            if pil.mode != "RGB":
                pil = pil.convert("RGB")
            test_imgs.append(np.asarray(pil.resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)))
            test_lbls.append(wnid_to_idx[w])
    test_x = np.stack(test_imgs).astype(np.uint8)
    test_y = np.array(test_lbls, dtype=np.int64)
    _save_slice(ds_name, train_x, train_y, test_x, test_y)


def main():
    ds_name = sys.argv[1]
    tv_class = sys.argv[2]
    if tv_class == "TinyImageNet":
        prep_tiny_imagenet(ds_name)
    else:
        prep_torchvision(ds_name, tv_class)


if __name__ == "__main__":
    main()
