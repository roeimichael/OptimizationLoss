"""Prep OctMNIST with stratified train subsample to fit disk + match
other MedMNIST scales.

  train: 12,000 (3,000 per class, stratified — matches dermmnist size)
  test:  1,000  (250 per class, balanced, full)
  format: NHWC uint8 224x224, broadcast grayscale to 3-channel
"""
import os
import numpy as np
import pandas as pd
from PIL import Image
import medmnist

IMG_SIZE = 224
N_PER_CLASS_TRAIN = 3000
SEED = 42


def resize_to_uint8(arr_nhwc):
    n = len(arr_nhwc)
    out = np.empty((n, IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
    for i in range(n):
        img = arr_nhwc[i]
        if img.ndim == 2:
            img = np.repeat(img[..., None], 3, axis=-1)
        if img.shape[-1] == 1:
            img = np.repeat(img, 3, axis=-1)
        out[i] = np.asarray(Image.fromarray(img).resize(
            (IMG_SIZE, IMG_SIZE), Image.BILINEAR))
        if (i+1) % 2000 == 0: print(f"    resize {i+1}/{n}", flush=True)
    return out


def main():
    tr = medmnist.OCTMNIST(split="train", download=True, size=28)
    va = medmnist.OCTMNIST(split="val",   download=True, size=28)
    te = medmnist.OCTMNIST(split="test",  download=True, size=28)
    train_x_all = np.concatenate([tr.imgs, va.imgs], axis=0)
    train_y_all = np.concatenate([tr.labels.squeeze(), va.labels.squeeze()]).astype(np.int64)
    test_x  = te.imgs
    test_y  = te.labels.squeeze().astype(np.int64)

    print(f"raw: train+val={len(train_y_all)} test={len(test_y)}")
    print(f"train class counts pre-subsample: {np.bincount(train_y_all).tolist()}")

    # Stratified subsample: N_PER_CLASS_TRAIN per class
    rng = np.random.RandomState(SEED)
    sel_idx = []
    for c in sorted(np.unique(train_y_all)):
        cls_idx = np.where(train_y_all == c)[0]
        n_take = min(N_PER_CLASS_TRAIN, len(cls_idx))
        chosen = rng.choice(cls_idx, size=n_take, replace=False)
        sel_idx.append(chosen)
    sel_idx = np.concatenate(sel_idx)
    rng.shuffle(sel_idx)
    train_x = train_x_all[sel_idx]
    train_y = train_y_all[sel_idx]

    print(f"after stratified subsample: train={len(train_y)} (per-class={np.bincount(train_y).tolist()})")
    print(f"test (full): {np.bincount(test_y).tolist()}")

    print(f"resizing train ({len(train_x)} samples) to 224x224 uint8...")
    train_x = resize_to_uint8(train_x)
    print(f"resizing test ({len(test_x)} samples)...")
    test_x = resize_to_uint8(test_x)

    out_dir = "data/octmnist/slice_1"
    os.makedirs(out_dir, exist_ok=True)
    for name, x, y in (("train", train_x, train_y), ("test", test_x, test_y)):
        np.save(f"{out_dir}/{name}_images.npy", x)
        np.save(f"{out_dir}/{name}_labels.npy", y)
        groups = np.arange(len(y)) % 3
        meta = pd.DataFrame({
            "label": y, "class_name": [f"c{int(c)}" for c in y],
            "filename": [f"{name}_{i:06d}.jpg" for i in range(len(y))],
            "synth_group": groups,
        })
        meta.to_csv(f"{out_dir}/{name}_meta.csv", index=False)
    print(f"saved {out_dir}: " + os.popen(f"du -sh {out_dir}").read().strip())


if __name__ == "__main__":
    main()
