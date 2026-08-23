"""Build a HELD-OUT-COUNTRY fMoW slice -- the strongest candidate FRAMEWORK 2(n)
has measured, and the only one independent of both camera traps and dermatology.

WHY THIS DATASET. Every dataset that has cleared 2(n) is a camera-trap corpus read
through the same COCO-CameraTraps schema, so the whole family is arguably ONE
generalization unit. fMoW is satellite imagery, the group is a COUNTRY, and the
class mix genuinely differs between countries (Egypt in the screened slice is 192
places of worship and 0 recreational facilities; Japan is 253 recreational
facilities and 13 places of worship).

    NET +2969 items, z=79.7, NET/LOCAL 95.8%, 10 unseen countries (n_test 3442)

🟢 IT BEATS IWILDCAM ON THE PROPERTY 2(n) CHOSE IWILDCAM FOR. A K=0 per-group
ceiling binds regardless of sum slack. Capping {single-unit_residential,
military_facility} gives 11 of 20 per-group ceilings at zero on classes of 408
and 511 test items; iwildcam gives 7 of 14 on 370 and 456, and the screened ISIC
slice only 4 of 20 on 86 and 135. The group is also ATOMIC, so it scores 100.0%
under `scripts.factorial_control` -- none of its NET is the screen's baseline.

WHERE THE DATA COMES FROM, and the two obvious routes are both wrong.

  * ⛔ The WILDS CodaLab bundle is gone (2(n)).
  * ⛔ `EVER-Z/fMoW_rgb` has a CORRUPT `category` column, truncated at the first
    underscore, which MERGES `airport`/`airport_hangar`/`airport_terminal` (2(n)).
  * ⛔ `danielz01/fMoW`, the byte-exact WILDS parquet, is GATED: HTTP 401,
    `x-error-code: GatedRepo`. A columnar range read is not possible anonymously.
  * ✅ `jbourcier/fmow-rgb-baseline` publishes per-image JSON METADATA SEPARATELY
    from the images, so the whole of stage 1 costs 52 MB and 8 seconds, and the
    images are ALREADY cropped to the AOI and resized to 224x224 JPEG.

🛑 THE LABEL COMES FROM THE PATH, NEVER FROM A `category` FIELD. Paths are
`<split>/<class>/<class>_<seq>/<class>_<aoi>/<class>_<seq>_<i>_rgb.{jpg,json}`,
so the 2(n) truncation trap cannot bite here.

ONLY THE `val` SPLIT IS USED, and that is deliberate: it yields a 17,670-image
training side, close to iwildcam's 20,000, so there is no compute confound -- and
we re-split by country ourselves regardless of the original train/val boundary.
`train-images.tar.gz` is 9.57 GB and is not needed.

    python -m scripts.prep_fmow --out data/fmow/oodslice --meta-only   # 52 MB
    python -m scripts.dataset_screen data/fmow/oodslice
    python -m scripts.factorial_control data/fmow/oodslice             # expect ~100%
    python -m scripts.prep_fmow --out data/fmow/oodslice               # +1.65 GB
"""
import argparse
import io
import json
import os
import subprocess
import tarfile

import numpy as np
import pandas as pd

from scripts.prep_iwildcam import write_meta

BASE = ("https://huggingface.co/datasets/jbourcier/fmow-rgb-baseline/resolve/main/")
META = "val-metadata.tar.gz"          # 51,905,314 B
IMAGES = "val-images.tar.gz"          # 1,652,231,185 B
DROP = ("false_detection",)


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
    """Stream the sidecar tarball into a (file, class, site, country) frame."""
    rows = []
    with tarfile.open(fetch(cache, META), "r:gz") as t:
        for m in t:
            if not m.isfile() or not m.name.endswith(".json"):
                continue
            parts = m.name.split("/")        # split / class / class_seq / aoi / file
            try:
                d = json.load(t.extractfile(m))
            except Exception:
                continue
            rows.append({"file": os.path.basename(m.name)[:-5] + ".jpg",
                         "raw": parts[1],             # the label, from the PATH
                         "site": parts[2],
                         "group": d.get("country_code"),
                         "timestamp": d.get("timestamp")})
    df = pd.DataFrame(rows).dropna(subset=["group"])
    return df[~df["raw"].isin(DROP)].copy()


def build_split(df, n_classes, min_per_group, max_per_group, test_target,
                seed=0, tries=400):
    """Pick the classes, then hold out whole COUNTRIES for test.

    Countries are held out ENTIRE. An fMoW site sits inside one country, so
    holding countries out also holds sites out -- verified 0 site overlap.
    """
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
        rarest = int(sub["label"].value_counts().min())
        if best is None or rarest > best[0]:
            best = (rarest, list(sel))
    if best is None:
        raise SystemExit("no country set holds all %d classes" % n_classes)
    test_g = set(best[1])
    te = df[df["group"].isin(test_g)].copy()
    tr = df[~df["group"].isin(test_g)].copy()
    assert not (set(te["group"]) & set(tr["group"])), "country leaked"
    assert not (set(te["site"]) & set(tr["site"])), "site leaked"
    return tr, te, {remap[c]: c for c in top}, sorted(test_g)


def collect(tr, te, out_dir, cache):
    """Stream the image tarball, keeping only the slice's members."""
    from PIL import Image
    want = {}
    for split, d in (("train", tr), ("test", te)):
        for f, lab, grp in zip(d["file"], d["label"], d["group"]):
            want[f] = (split, int(lab), grp)
    got = {"train": [], "test": []}
    with tarfile.open(fetch(cache, IMAGES), "r:gz") as t:
        for m in t:
            if not m.isfile():
                continue
            name = os.path.basename(m.name)
            if name not in want:
                continue
            split, lab, grp = want.pop(name)
            im = Image.open(io.BytesIO(t.extractfile(m).read())).convert("RGB")
            if im.size != (224, 224):        # the mirror ships 224x224 already
                im = im.resize((224, 224))
            got[split].append((np.asarray(im, np.uint8), lab, grp, name))
    if want:
        print("  WARNING: %d wanted images not found in the tarball" % len(want))
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
    ap.add_argument("--out", default="data/fmow/oodslice")
    ap.add_argument("--cache", default="data/fmow/_cache")
    ap.add_argument("--classes", type=int, default=8)
    ap.add_argument("--min-per-group", type=int, default=150)
    ap.add_argument("--max-per-group", type=int, default=1200)
    ap.add_argument("--test-target", type=int, default=2900)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--meta-only", action="store_true",
                    help="stop after the split: 52 MB, no images, no GPU")
    args = ap.parse_args()

    df = load(args.cache)
    print("  %d sidecars, %d classes, %d countries"
          % (len(df), df["raw"].nunique(), df["group"].nunique()))
    tr, te, names, test_g = build_split(
        df, args.classes, args.min_per_group, args.max_per_group,
        args.test_target, args.seed)
    print("  classes: %s" % names)
    print("  train %d / test %d" % (len(tr), len(te)))
    print("  test countries (%d, all unseen): %s" % (len(test_g), test_g))
    print("  test class counts: %s"
          % te["label"].value_counts().sort_index().to_dict())

    if args.meta_only:
        os.makedirs(args.out, exist_ok=True)
        for split, d in (("train", tr), ("test", te)):
            write_meta(args.out, split, d["label"].values, d["file"].values,
                       d["group"].values)
        print("  META ONLY: wrote the two CSVs dataset_screen reads. No images.")
        return
    collect(tr, te, args.out, args.cache)


if __name__ == "__main__":
    main()
