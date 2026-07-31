"""TMLR Track B / B2: build a NATIVE-resolution OCT slice matched to OctMNIST.

The paper's OctMNIST is 28x28 upsampled to 224 (blurry). This builds an
apples-to-apples native-resolution counterpart: the ORIGINAL Kermany OCT2017
images (native ~1024x496, from HF `zacharielegault/Kermany2017-OCT`), decoded
and resized to 224, subsampled to the SAME shape as `data/octmnist/slice_1`
(12000 train / 1000 test, balanced 250/class test) so the ONLY difference from
the frozen OctMNIST experiment is real image detail vs. upsampled thumbnails.

Label order is identical (0=CNV,1=DME,2=DRUSEN,3=NORMAL) so constrained_class=2
(DRUSEN) matches the paper. `synth_group` = 3 balanced synthetic groups on the
test set (mirrors OctMNIST's 334/333/333), seeded + deterministic.

Output: data/octnative/slice_1/{train,test}_images.npy + labels + test_meta.csv
Idempotent: skips if the output already exists.

Run from repo root (after the parquet shards are downloaded):
    python3 scripts/prep_oct_native.py
"""

import glob
import io
import os

import numpy as np
import pandas as pd
from PIL import Image

SRC = "data/oct_native_src/data"
OUT = "data/octnative/slice_1"
SIZE = 224
N_TRAIN_PER_CLASS = 3000   # 4 x 3000 = 12000  (matches octmnist slice_1 train)
N_TEST_PER_CLASS = 250     # 4 x 250  = 1000   (= OCT2017 official test)
N_GROUPS = 3
SEED = 12345
CLASS_NAMES = {0: "CNV", 1: "DME", 2: "DRUSEN", 3: "NORMAL"}


def _collect(parquets):
    """Return (labels list, image-bytes list) from a set of parquet shards."""
    labs, imgs = [], []
    for p in sorted(parquets):
        df = pd.read_parquet(p)
        col_img = "image" if "image" in df.columns else df.columns[0]
        col_lab = "label" if "label" in df.columns else df.columns[-1]
        for v, l in zip(df[col_img].tolist(), df[col_lab].tolist()):
            b = v["bytes"] if isinstance(v, dict) else v
            imgs.append(b)
            labs.append(int(l))
    return np.array(labs, dtype=np.int64), imgs


def _decode(b):
    """Native OCT jpeg bytes -> (224,224,3) uint8 grayscale-replicated (OctMNIST layout)."""
    im = Image.open(io.BytesIO(b)).convert("L").resize((SIZE, SIZE), Image.LANCZOS)
    a = np.asarray(im, dtype=np.uint8)
    return np.stack([a, a, a], axis=-1)


def _balanced_sample(labels, per_class, rng):
    idx = []
    for c in range(4):
        ci = np.where(labels == c)[0]
        take = min(per_class, len(ci))
        idx.extend(rng.permutation(ci)[:take].tolist())
    rng.shuffle(idx)
    return np.array(idx, dtype=np.int64)


def main():
    if os.path.exists(os.path.join(OUT, "test_meta.csv")):
        print(f"[prep_oct_native] {OUT} already built -- skipping")
        return
    train_pq = glob.glob(f"{SRC}/train-*.parquet")
    test_pq = glob.glob(f"{SRC}/test-*.parquet")
    assert train_pq, f"no train parquet in {SRC} (download not finished?)"
    assert test_pq, f"no test parquet in {SRC}"

    rng = np.random.RandomState(SEED)
    print(f"[prep_oct_native] reading {len(train_pq)} train + {len(test_pq)} test shards")
    ytr_all, xtr_all = _collect(train_pq)
    yte_all, xte_all = _collect(test_pq)
    print(f"  pooled train={len(ytr_all)} test={len(yte_all)} "
          f"train-class-counts={np.bincount(ytr_all).tolist()}")

    tr_idx = _balanced_sample(ytr_all, N_TRAIN_PER_CLASS, rng)
    te_idx = _balanced_sample(yte_all, N_TEST_PER_CLASS, np.random.RandomState(SEED + 2))
    print(f"  sampled train={len(tr_idx)} test={len(te_idx)} (decoding+resize to {SIZE})")

    Xtr = np.stack([_decode(xtr_all[i]) for i in tr_idx]).astype(np.uint8)
    ytr = ytr_all[tr_idx]
    Xte = np.stack([_decode(xte_all[i]) for i in te_idx]).astype(np.uint8)
    yte = yte_all[te_idx]

    # 3 balanced synthetic groups on the test set (mirrors octmnist synth_group)
    g = np.empty(len(yte), dtype=np.int64)
    perm = np.random.RandomState(SEED + 1).permutation(len(yte))
    for k, i in enumerate(perm):
        g[i] = k % N_GROUPS

    os.makedirs(OUT, exist_ok=True)
    np.save(f"{OUT}/train_images.npy", Xtr)
    np.save(f"{OUT}/train_labels.npy", ytr)
    np.save(f"{OUT}/test_images.npy", Xte)
    np.save(f"{OUT}/test_labels.npy", yte)
    pd.DataFrame({"label": yte,
                  "class_name": [CLASS_NAMES[l] for l in yte],
                  "synth_group": g}).to_csv(f"{OUT}/test_meta.csv", index=False)
    pd.DataFrame({"label": ytr,
                  "class_name": [CLASS_NAMES[l] for l in ytr]}).to_csv(
                      f"{OUT}/train_meta.csv", index=False)
    print(f"[prep_oct_native] DONE  train{Xtr.shape} test{Xte.shape} "
          f"test-class={np.bincount(yte).tolist()} group={np.bincount(g).tolist()}")


if __name__ == "__main__":
    main()
