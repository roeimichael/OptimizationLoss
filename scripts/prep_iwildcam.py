"""Build a HELD-OUT-CAMERA iWildCam slice -- the first dataset that clears the
FRAMEWORK 2(n) criterion.

WHY THIS DATASET. Post-hoc top-K is optimal given the probabilities, so a count
cap can only help by carrying information the training set does not. 2(j) shut
the global route permanently (one multiplier per class is monotone and cannot
reorder). The only live route is a PER-GROUP cap, and it needs per-group counts
the model could not have learned. On dermmnist it could: same cameras, same
sites, residual correction 1.68x, 6 items moved, null.

Here the test cameras are DISJOINT from the training cameras. The median camera
sees 10 of 216 species, and camera 501 is impala/elephant/cattle while camera 408
is ocellated turkey/tapir/puma -- different continents. A species that dominates
one camera is ABSENT from another, so the correction is unbounded rather than
1.68x. `scripts.dataset_screen` measures the difference:

    dermmnist/slice_1     NET   +65 items  (z=2.9)   unseen groups 0
    iwildcam/oodslice     NET +3131 items  (z=97.4)  unseen groups 7

WHERE THE DATA COMES FROM. The official WILDS archive is served from a CodaLab
bundle that returns HTTP 500 (verified 2026-08-22, three attempts; the CodaLab
root itself answers 200, so the bundle is gone rather than the site being down).
This reads the iWildCam 2020 COCO annotations plus the HuggingFace parquet
mirror of the images instead. That is not a workaround but an improvement: the
raw annotations let us define the camera split ourselves, which 2(n) requires
anyway -- `data/dermmnist/create_slices.py` stratifies on the LABEL, which forces
test prevalence to match train and is exactly why the global cap carries nothing.

The parquet shards hold one column, `image`, a struct of raw JPEG bytes and a
path; labels and cameras come from the annotations, joined on the file name.
Shards are streamed and DELETED after each one, so peak disk stays near a single
shard rather than the 90GB the full mirror would need.

    python -m scripts.prep_iwildcam --out data/iwildcam/oodslice
"""
import argparse
import io
import json
import os
import subprocess

import numpy as np
import pandas as pd

SHARD_URL = ("https://huggingface.co/datasets/anngrosha/iWildCam2020/resolve/"
             "main/data/train-%05d-of-00190.parquet")
N_SHARDS = 190
DROP = ("empty", "unknown", "misfire", "start", "end")


def build_split(ann_path, n_classes, min_per_camera, test_target, seed=0):
    """Pick the classes, then hold out whole cameras for test.

    Cameras are held out ENTIRE -- never split across train and test -- because
    a camera appearing on both sides would let the model learn that camera's
    species prior, which is the single property this dataset exists to remove.
    """
    d = json.load(open(ann_path, encoding="utf-8"))
    cat = {c["id"]: c["name"] for c in d["categories"]}
    lab = {a["image_id"]: a["category_id"] for a in d["annotations"]}
    df = pd.DataFrame([{"file_name": os.path.basename(im["file_name"]),
                        "location": im["location"], "raw": lab.get(im["id"])}
                       for im in d["images"]]).dropna(subset=["raw"])
    df["raw"] = df["raw"].astype(int)
    drop_ids = [k for k, v in cat.items() if v in DROP]
    df = df[~df["raw"].isin(drop_ids)]

    top = df["raw"].value_counts().head(n_classes).index.tolist()
    df = df[df["raw"].isin(top)].copy()
    remap = {c: i for i, c in enumerate(top)}
    df["label"] = df["raw"].map(remap)
    names = {remap[c]: cat[c] for c in top}

    sz = df["location"].value_counts()
    cand = sz[sz >= min_per_camera].index.tolist()
    rng = np.random.default_rng(seed)
    best = None
    for _ in range(400):
        sel, n = [], 0
        for loc in rng.permutation(cand):
            sel.append(loc)
            n += int(sz[loc])
            if n >= test_target:
                break
        sub = df[df["location"].isin(sel)]
        if sub["label"].nunique() < n_classes:
            continue
        # maximise the RAREST class in test: a per-group budget that rounds to
        # zero is silently SKIPPED in the loss, so a cell with a near-empty
        # class would quietly disable its own constraint.
        rarest = int(sub["label"].value_counts().min())
        if best is None or rarest > best[0]:
            best = (rarest, list(sel))
    if best is None:
        raise SystemExit("no camera set holds all %d classes" % n_classes)
    test_cams = set(best[1])
    te = df[df["location"].isin(test_cams)]
    tr = df[~df["location"].isin(test_cams)]
    assert not (set(te["location"]) & set(tr["location"])), "camera leaked"
    return tr, te, names, sorted(test_cams)


def collect(targets, out_dir, cache):
    """Stream shards, keep only wanted images, delete each shard after."""
    from PIL import Image
    import pyarrow.parquet as pq

    want = dict(targets)                    # file_name -> (split, label, loc)
    got = {"train": [], "test": []}
    shard = os.path.join(cache, "_shard.parquet")
    for i in range(N_SHARDS):
        if not want:
            break
        rc = subprocess.call(["curl", "-sL", "--max-time", "900", "-o", shard,
                              SHARD_URL % i])
        if rc != 0 or not os.path.exists(shard):
            print("  shard %d: download failed (rc=%d), skipping" % (i, rc))
            continue
        try:
            tbl = pq.read_table(shard)
        except Exception as exc:                       # a truncated shard
            print("  shard %d: unreadable (%s), skipping" % (i, exc))
            os.remove(shard)
            continue
        col = tbl.column("image").to_pylist()
        hits = 0
        for cell in col:
            name = os.path.basename(cell["path"])
            if name not in want:
                continue
            split, label, loc = want.pop(name)
            img = Image.open(io.BytesIO(cell["bytes"])).convert("RGB")
            got[split].append((np.asarray(img.resize((224, 224)), np.uint8),
                               label, loc, name))
            hits += 1
        os.remove(shard)
        print("  shard %3d/%d: +%4d kept (train %d, test %d, %d still wanted)"
              % (i + 1, N_SHARDS, hits, len(got["train"]), len(got["test"]),
                 len(want)), flush=True)

    os.makedirs(out_dir, exist_ok=True)
    for split in ("train", "test"):
        rows = got[split]
        if not rows:
            raise SystemExit("collected 0 %s images" % split)
        x = np.stack([r[0] for r in rows])
        y = np.asarray([r[1] for r in rows], np.int64)
        np.save(os.path.join(out_dir, "%s_images.npy" % split), x)
        np.save(os.path.join(out_dir, "%s_labels.npy" % split), y)
        pd.DataFrame({"label": y,
                      "class_name": ["c%d" % v for v in y],
                      "filename": [r[3] for r in rows],
                      "location": [r[2] for r in rows]}).to_csv(
            os.path.join(out_dir, "%s_meta.csv" % split), index=False)
        print("  wrote %s: %s images" % (split, x.shape))
    return got


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--annotations",
                    default="data/iwildcam/train_annotations.json")
    ap.add_argument("--out", default="data/iwildcam/oodslice")
    ap.add_argument("--classes", type=int, default=8)
    ap.add_argument("--min-per-camera", type=int, default=120)
    ap.add_argument("--test-target", type=int, default=1800)
    ap.add_argument("--train-per-class", type=int, default=2500)
    args = ap.parse_args()

    tr, te, names, cams = build_split(args.annotations, args.classes,
                                      args.min_per_camera, args.test_target)
    rng = np.random.default_rng(0)
    keep = []
    for c in sorted(tr["label"].unique()):
        sub = tr[tr["label"] == c]
        take = min(len(sub), args.train_per_class)
        keep.append(sub.iloc[rng.permutation(len(sub))[:take]])
    tr = pd.concat(keep)

    print("iWildCam held-out-camera slice")
    print("  classes      : %s" % {k: v[:22] for k, v in sorted(names.items())})
    print("  test cameras : %s (held out ENTIRE)" % cams)
    print("  train        : %d images, %d cameras" % (len(tr),
                                                      tr["location"].nunique()))
    print("  test         : %d images, %d cameras" % (len(te),
                                                      te["location"].nunique()))
    print("  overlap      : %d cameras (must be 0)"
          % len(set(te["location"]) & set(tr["location"])))
    print("  test per class: %s" % te["label"].value_counts().sort_index().to_dict())
    print("")

    targets = {}
    for split, frame in (("train", tr), ("test", te)):
        for _, r in frame.iterrows():
            targets[r["file_name"]] = (split, int(r["label"]), int(r["location"]))
    print("  streaming %d shards for %d wanted images" % (N_SHARDS, len(targets)))
    collect(targets, args.out, os.path.dirname(args.annotations))
    print("")
    print("  now run:  python -m scripts.dataset_screen %s" % args.out)


if __name__ == "__main__":
    main()
