"""Native-Resolution Campaign (docs/NATIVE_RES_CAMPAIGN.md) — stage MedMNIST v2
datasets at GENUINE native size=224 into the octnative-style .npy layout.

Why fresh: the pre-staged data/{retinamnist,bloodmnist}/ have low native-MSE
(likely 28px-upsampled) and NO test_meta.csv. medmnist size=224 is genuinely
native, so we re-stage into data/native224/<key>/slice_1/ with a synthetic
group column, matched to the pipeline the loader expects.

Layout written (matches prep_oct_native.py):
  data/native224/<key>/slice_1/{train,test}_images.npy  (uint8 NHWC, 3-channel)
                                {train,test}_labels.npy   (int64)
                                test_meta.csv             (label,class_name,synth_group)

Subsampling: train stratified-proportional capped at N_TRAIN; test stratified
capped at N_TEST (full if smaller) so cc-F1 K stays stable. Reports per-class
counts and the suggested constrained class (rarest with test-count >= 75, i.e.
K>=30 at the loosest cap L40). Idempotent (skips if slice already built).

Run ON THE SERVER from repo root (needs internet for the size=224 npz):
    python scripts/prep_medmnist224.py <key> [<key> ...]
    keys: retinamnist bloodmnist tissuemnist organamnist   (dermamnist optional)
"""
import os
import sys

import numpy as np
import pandas as pd

SIZE = 224
N_TRAIN = 15000     # cap; smaller datasets use all
N_TEST = 4000       # cap; smaller datasets use all
N_GROUPS = 3
SEED = 12345
OUT_ROOT = "data/native224"

# medmnist python-class name per key
CLS = {
    "retinamnist": "RetinaMNIST", "bloodmnist": "BloodMNIST",
    "tissuemnist": "TissueMNIST", "organamnist": "OrganAMNIST",
    "dermamnist": "DermaMNIST", "pathmnist": "PathMNIST",
}


def _to_hwc3_u8(imgs):
    """medmnist .imgs -> (N,224,224,3) uint8. Handles grayscale (N,H,W) or
    (N,H,W,1) by channel-replication, and (N,H,W,3) passthrough."""
    a = np.asarray(imgs)
    if a.ndim == 3:                      # (N,H,W) grayscale
        a = a[..., None]
    if a.shape[-1] == 1:                 # (N,H,W,1) -> replicate
        a = np.repeat(a, 3, axis=-1)
    if a.dtype != np.uint8:
        a = a.astype(np.uint8)
    return a


def _strat_sample(labels, cap, rng):
    """Stratified proportional subsample to <= cap, preserving class ratios."""
    n = len(labels)
    if n <= cap:
        return np.arange(n)
    keep = []
    for c in np.unique(labels):
        ci = np.where(labels == c)[0]
        take = max(1, int(round(len(ci) * cap / n)))
        keep.extend(rng.permutation(ci)[:take].tolist())
    keep = np.array(keep, dtype=np.int64)
    rng.shuffle(keep)
    return keep


def build(key):
    import medmnist
    from medmnist import INFO
    out = f"{OUT_ROOT}/{key}/slice_1"
    if os.path.exists(os.path.join(out, "test_meta.csv")):
        print(f"[{key}] already built at {out} -- skipping")
        return
    info = INFO[key]
    names = {int(k): v for k, v in info["label"].items()}
    DS = getattr(medmnist, CLS[key])
    rng = np.random.RandomState(SEED)

    # merge official train+val -> train (val unused), keep official test
    parts = {}
    for split in ("train", "val", "test"):
        d = DS(split=split, size=SIZE, download=True)
        parts[split] = (_to_hwc3_u8(d.imgs), np.asarray(d.labels).ravel().astype(np.int64))
    Xtr = np.concatenate([parts["train"][0], parts["val"][0]], 0)
    ytr = np.concatenate([parts["train"][1], parts["val"][1]], 0)
    Xte, yte = parts["test"]
    print(f"[{key}] pooled train={len(ytr)} test={len(yte)} classes={len(names)}")

    tr_idx = _strat_sample(ytr, N_TRAIN, rng)
    te_idx = _strat_sample(yte, N_TEST, np.random.RandomState(SEED + 2))
    Xtr, ytr = Xtr[tr_idx], ytr[tr_idx]
    Xte, yte = Xte[te_idx], yte[te_idx]

    # synthetic balanced groups on test (mirrors octnative)
    g = np.empty(len(yte), dtype=np.int64)
    for k, i in enumerate(np.random.RandomState(SEED + 1).permutation(len(yte))):
        g[i] = k % N_GROUPS

    os.makedirs(out, exist_ok=True)
    np.save(f"{out}/train_images.npy", Xtr)
    np.save(f"{out}/train_labels.npy", ytr)
    np.save(f"{out}/test_images.npy", Xte)
    np.save(f"{out}/test_labels.npy", yte)
    pd.DataFrame({"label": yte, "class_name": [names[int(l)] for l in yte],
                  "synth_group": g}).to_csv(f"{out}/test_meta.csv", index=False)

    trd = np.bincount(ytr, minlength=len(names))
    ted = np.bincount(yte, minlength=len(names))
    # suggested constrained class: rarest in test with count>=75 (K>=30 at L40)
    elig = [(int(c), int(ted[c])) for c in range(len(names)) if ted[c] >= 75]
    sugg = min(elig, key=lambda t: t[1]) if elig else (int(np.argmax(ted)), int(ted.max()))
    print(f"[{key}] DONE train{Xtr.shape} test{Xte.shape}")
    print(f"[{key}] train_dist={trd.tolist()}")
    print(f"[{key}] test_dist ={ted.tolist()}")
    print(f"[{key}] class_names={names}")
    print(f"[{key}] SUGGESTED constrained_class={sugg[0]} "
          f"('{names[sugg[0]]}', test_count={sugg[1]}, "
          f"K@L40={int(round(sugg[1]*0.4))}, K@L30={int(round(sugg[1]*0.3))}, "
          f"K@L20={int(round(sugg[1]*0.2))})  group_column=synth_group")


if __name__ == "__main__":
    keys = sys.argv[1:] or ["retinamnist", "bloodmnist"]
    for k in keys:
        assert k in CLS, f"unknown key {k}; known={list(CLS)}"
        build(k)
