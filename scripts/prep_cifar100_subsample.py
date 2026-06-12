"""Create stratified train-subsample variants of CIFAR-100.

Tests the 'limit training data quantity' headroom hypothesis: with fewer
train samples per class, MobileNetV3 can't memorize the whole train set
even with 50 warmup epochs -> train_acc caps below 1.0 -> CE gradient
stays alive for the constraint phase -> TraLO has room.

Output: data/cifar100_subset<N>/slice_1/{train,test}_{images,labels}.npy
Train = N samples per class (stratified). Test = full 10000.

Usage:
  python scripts/prep_cifar100_subsample.py 50
  python scripts/prep_cifar100_subsample.py 20
"""
import os
import sys

import numpy as np
import pandas as pd

SEED = 42


def main():
    n_per_class = int(sys.argv[1])
    src = "data/cifar100/slice_1"
    out = f"data/cifar100_subset{n_per_class}/slice_1"
    os.makedirs(out, exist_ok=True)

    print(f"loading {src}/train_*.npy ...")
    train_x = np.load(f"{src}/train_images.npy", mmap_mode="r")
    train_y = np.load(f"{src}/train_labels.npy")
    test_x  = np.load(f"{src}/test_images.npy",  mmap_mode="r")
    test_y  = np.load(f"{src}/test_labels.npy")
    print(f"  full train: {train_x.shape}  test: {test_x.shape}")

    rng = np.random.RandomState(SEED)
    sel_idx = []
    for c in sorted(np.unique(train_y)):
        cls_idx = np.where(train_y == c)[0]
        chosen = rng.choice(cls_idx, size=min(n_per_class, len(cls_idx)),
                            replace=False)
        sel_idx.append(chosen)
    sel_idx = np.concatenate(sel_idx)
    rng.shuffle(sel_idx)
    print(f"  selected {len(sel_idx)} train samples ({n_per_class}/class)")

    train_x_sub = np.ascontiguousarray(train_x[sel_idx])
    train_y_sub = train_y[sel_idx]

    np.save(f"{out}/train_images.npy", train_x_sub)
    np.save(f"{out}/train_labels.npy", train_y_sub)
    np.save(f"{out}/test_images.npy",  np.ascontiguousarray(test_x))
    np.save(f"{out}/test_labels.npy",  test_y)

    for name, y in (("train", train_y_sub), ("test", test_y)):
        groups = np.arange(len(y)) % 3
        meta = pd.DataFrame({
            "label": y, "class_name": [f"c{int(c)}" for c in y],
            "filename": [f"{name}_{i:06d}.jpg" for i in range(len(y))],
            "synth_group": groups,
        })
        meta.to_csv(f"{out}/{name}_meta.csv", index=False)

    size = os.popen(f"du -sh {out}").read().strip()
    print(f"saved {out}: {size}")


if __name__ == "__main__":
    main()
