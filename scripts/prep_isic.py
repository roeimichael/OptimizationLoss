"""Build a HELD-OUT-SUBPOPULATION ISIC 2019 slice -- the first NON-camera-trap
dataset to clear the FRAMEWORK 2(n) criterion.

WHY THIS DATASET. Every dataset that has cleared 2(n) so far is a camera-trap
corpus read through the same COCO-CameraTraps schema, several by the same
authors, so a reviewer may fairly call the whole family ONE generalization
unit. 2(n) names ISIC 2019 as the standing non-camera-trap candidate and
records that it had never been screened. It has now.

WHAT THE GROUP IS, and it is not the obvious one. ISIC 2019 pools three
archives (BCN_20000 / Barcelona, HAM10000 / Vienna, MSK), and the obvious group
is the archive. That FAILS: with one held-out archive the per-group and the
global shift are the same object by construction, and NET comes back NEGATIVE
(-141 items, z=-2.7) against a GLOBAL of +5581. A cross-institution split of
this dataset is a pure prior shift, which 2(j) shut permanently.

What works is a WITHIN-INSTITUTION subpopulation split: BCN_20000 only, groups
= (anatomical site x age band), held out entire. Both factors are acquisition
metadata, so the group is knowable at inference without the label.

    NET +1705 items, z=47.7, NET/LOCAL 84.9%, 10 unseen groups (n_test 3793)

⚠️ AND READ `scripts.factorial_control` BESIDE IT. A product group is not an
atomic one: the screen credits an unseen group with the GLOBAL prior, but a
model that has seen (head/neck, 60s) and (upper extremity, 70s) can interpolate
(head/neck, 70s). Pooling BCN with HAM makes almost all of the signal
interpolable -- 98.4% NET/LOCAL collapses to 17.6% surviving. BCN-only survives
83.6% (+1424 items, 50-128% over four seeds), which is why the slice is
single-institution and not the larger pooled one.

LEAKAGE. ISIC ships up to 31 images of the SAME lesion; a random split would
reproduce the dermmnist 38.7% leak exactly. Lesions are held disjoint here and
the count of images dropped from TRAIN to enforce it is printed. BCN-only also
contains ZERO HAM10000 images, so the slice does not overlap dermmnist, which
this project removed after it nulled.

    python -m scripts.prep_isic --out data/isic/oodslice --meta-only   # 2.5 MB
    python -m scripts.dataset_screen data/isic/oodslice
    python -m scripts.factorial_control data/isic/oodslice
    python -m scripts.prep_isic --out data/isic/oodslice               # +9.8 GB
"""
import argparse
import io
import os
import subprocess
import zipfile

import numpy as np
import pandas as pd

from scripts.prep_iwildcam import write_meta

BASE = "https://isic-challenge-data.s3.amazonaws.com/2019/"
GT = "ISIC_2019_Training_GroundTruth.csv"
MD = "ISIC_2019_Training_Metadata.csv"
IMAGES = "ISIC_2019_Training_Input.zip"          # 9,771,618,190 bytes
AGE_BINS = [-1, 29, 39, 49, 59, 69, 79, 200]
AGE_NAMES = ["<30", "30s", "40s", "50s", "60s", "70s", "80+"]


def fetch(cache, name):
    path = os.path.join(cache, name)
    if not os.path.exists(path):
        os.makedirs(cache, exist_ok=True)
        rc = subprocess.call(["curl", "-sL", "--max-time", "3600", "-o", path,
                              BASE + name])
        if rc != 0 or not os.path.exists(path):
            raise SystemExit("download failed: %s (rc=%d)" % (name, rc))
    return path


def load(cache):
    gt = pd.read_csv(fetch(cache, GT))
    md = pd.read_csv(fetch(cache, MD))
    cls = [c for c in gt.columns if c != "image"]
    gt["raw"] = [cls[i] for i in gt[cls].values.argmax(1)]
    assert (gt["raw"] == "UNK").sum() == 0, "UNK in the training ground truth"
    df = md.merge(gt[["image", "raw"]], on="image")
    # the archive is not a column: it is the lesion_id prefix, and the rows
    # with no lesion_id are the legacy ISIC-archive submissions.
    df["src"] = df["lesion_id"].fillna("LEGACY").astype(str).str.split("_").str[0]
    df["site"] = df["anatom_site_general"].fillna("unknown")
    df["age"] = pd.cut(df["age_approx"], bins=AGE_BINS,
                       labels=AGE_NAMES).astype(str)
    df["group"] = df["site"] + "|" + df["age"]
    # a lesion with no id is its own singleton, never a shared identity
    df["lesion"] = df["lesion_id"].fillna("").astype(str)
    solo = df["lesion"] == ""
    df.loc[solo, "lesion"] = "SOLO_" + df.loc[solo, "image"]
    return df


def build_split(df, sources, n_classes, min_per_group, max_per_group,
                test_target, seed=0, tries=400):
    """Pick the classes, then hold out whole (site x age) groups for test.

    Groups are held out ENTIRE and LESIONS are held disjoint: ISIC ships up to
    31 images of one lesion, so a lesion on both sides is the dermmnist leak.
    """
    df = df[df["src"].isin(sources)]
    df = df[(df["site"] != "unknown") & (df["age"] != "nan")].copy()
    top = df["raw"].value_counts().head(n_classes).index.tolist()
    df = df[df["raw"].isin(top)].copy()
    remap = {c: i for i, c in enumerate(top)}
    df["label"] = df["raw"].map(remap)

    sz = df["group"].value_counts()
    cand = sz[(sz >= min_per_group) & (sz <= max_per_group)].index.tolist()
    rng = np.random.default_rng(seed)
    best = None
    for _ in range(tries):
        sel, n = [], 0
        for g in rng.permutation(cand):
            sel.append(g)
            n += int(sz[g])
            if n >= test_target:
                break
        sub = df[df["group"].isin(sel)]
        if sub["label"].nunique() < n_classes:
            continue
        # maximise the RAREST class in test, exactly as prep_iwildcam does
        rarest = int(sub["label"].value_counts().min())
        if best is None or rarest > best[0]:
            best = (rarest, list(sel))
    if best is None:
        raise SystemExit("no (site x age) set holds all %d classes" % n_classes)
    test_g = set(best[1])
    te = df[df["group"].isin(test_g)].copy()
    tr = df[~df["group"].isin(test_g)].copy()
    assert not (set(te["group"]) & set(tr["group"])), "group leaked"
    shared = set(te["lesion"]) & set(tr["lesion"])
    n_drop = int(tr["lesion"].isin(shared).sum())
    tr = tr[~tr["lesion"].isin(shared)]
    assert not (set(te["lesion"]) & set(tr["lesion"])), "lesion leaked"
    return tr, te, {remap[c]: c for c in top}, sorted(test_g), n_drop


def collect(tr, te, out_dir, cache):
    """Extract ONLY the slice's members from the training zip, then resize."""
    from PIL import Image
    zpath = fetch(cache, IMAGES)
    want = {}
    for split, d in (("train", tr), ("test", te)):
        for img, lab, grp in zip(d["image"], d["label"], d["group"]):
            want[img + ".jpg"] = (split, int(lab), grp)
    got = {"train": [], "test": []}
    with zipfile.ZipFile(zpath) as z:
        for member in z.namelist():
            name = os.path.basename(member)
            if name not in want:
                continue
            split, lab, grp = want.pop(name)
            with z.open(member) as fh:
                im = Image.open(io.BytesIO(fh.read())).convert("RGB")
            got[split].append((np.asarray(im.resize((224, 224)), np.uint8),
                               lab, grp, name))
    if want:
        print("  WARNING: %d wanted images not found in the zip" % len(want))
    os.makedirs(out_dir, exist_ok=True)
    for split in ("train", "test"):
        rows = got[split]
        if not rows:
            raise SystemExit("collected 0 %s images" % split)
        np.save(os.path.join(out_dir, "%s_images.npy" % split),
                np.stack([r[0] for r in rows]))
        y = np.asarray([r[1] for r in rows], np.int64)
        np.save(os.path.join(out_dir, "%s_labels.npy" % split), y)
        write_meta(out_dir, split, y, [r[3] for r in rows], [r[2] for r in rows])
        print("  wrote %s: %d images" % (split, len(rows)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/isic/oodslice")
    ap.add_argument("--cache", default="data/isic/_cache")
    ap.add_argument("--sources", default="BCN",
                    help="archives to KEEP. BCN alone is the screened slice; "
                         "pooling BCN,HAM makes the novelty interpolable")
    ap.add_argument("--classes", type=int, default=8)
    ap.add_argument("--min-per-group", type=int, default=150)
    ap.add_argument("--max-per-group", type=int, default=1200)
    ap.add_argument("--test-target", type=int, default=2900)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--meta-only", action="store_true",
                    help="stop after the split: 2.5 MB of CSV, no images, no GPU")
    args = ap.parse_args()

    df = load(args.cache)
    tr, te, names, test_g, n_drop = build_split(
        df, set(args.sources.split(",")), args.classes, args.min_per_group,
        args.max_per_group, args.test_target, args.seed)
    print("  classes: %s" % names)
    print("  train %d / test %d" % (len(tr), len(te)))
    print("  test groups (%d, all unseen in training): %s" % (len(test_g), test_g))
    print("  lesion-leak images dropped from TRAIN: %d" % n_drop)
    print("  test class counts: %s"
          % te["label"].value_counts().sort_index().to_dict())

    if args.meta_only:
        os.makedirs(args.out, exist_ok=True)
        for split, d in (("train", tr), ("test", te)):
            write_meta(args.out, split, d["label"].values,
                       (d["image"] + ".jpg").values, d["group"].values)
        print("  META ONLY: wrote the two CSVs dataset_screen reads. No images.")
        print("  A meta-only NET is the INTENDED slice, so it is an UPPER "
              "bound: good enough to REJECT, never to accept a borderline one.")
        return
    collect(tr, te, args.out, args.cache)


if __name__ == "__main__":
    main()
