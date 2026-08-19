"""The receipt for the dermmnist leakage numbers FRAMEWORK section 1 asserts.

`docs/FRAMEWORK.md` states that 38.7% of the dermmnist test set, and 67.3% of
its melanoma, share a `lesion_id` with a training image. Those were prose-only:
no committed script emitted them, and the standing instruction was "replay it
yourself". A number that only exists in prose has already been wrong once in
this project (the ALM lambda, 701.2 -> 22.62), so this is the receipt.

WHY THE LEAKAGE EXISTS. HAM10000 photographs many lesions more than once --
10,015 images over 7,470 lesions. DermaMNIST-C ships leakage-free official
splits. `data/dermmnist/create_slices.py` POOLS all three official splits and
re-splits them stratified on the LABEL alone, so the two images of one lesion
land on opposite sides of the new boundary whenever the shuffle says so. It is
a design choice in our prep, not a defect in the source.

WHAT IT DOES AND DOES NOT INVALIDATE. Every arm in a campaign trains on the same
slice and is scored on the same test set, so a PAIRED arm-vs-arm delta is
unaffected -- the memorized items are memorized identically on both sides.
ABSOLUTE quality numbers on dermmnist are inflated and should not be read as
generalization.

This needs only the metadata CSV, never the images: `StratifiedShuffleSplit`
consumes `len(X)` and `y`, so the exact indices reproduce from labels alone.

    python -m scripts.check_lesion_leakage
    python -m scripts.check_lesion_leakage --data-dir /path/to/dermmnist --slice 1
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

# Kept in step with data/dermmnist/create_slices.py. If that file's constants
# change and these do not, the reproduced split is not the one on disk -- so
# they are asserted against it below rather than trusted.
DX_TO_CLASS = {"akiec": 0, "bcc": 1, "bkl": 2, "df": 3, "mel": 4, "nv": 5, "vasc": 6}
MEL = 4
BASE_SEED = 42
TEST_FRACTION = 0.2
SPLIT_ORDER = ("train", "val", "test")


def _check_constants_against_prep():
    """Refuse to report a number derived from a stale copy of the split rule."""
    prep = os.path.join("data", "dermmnist", "create_slices.py")
    if not os.path.exists(prep):
        return "create_slices.py not found -- constants unverified"
    src = open(prep, encoding="utf-8").read()
    for name, val in (("BASE_SEED", BASE_SEED), ("TEST_FRACTION", TEST_FRACTION)):
        needle = "%s = %r" % (name, val)
        if needle not in src:
            raise SystemExit(
                "%s disagrees with create_slices.py on %s. This script would "
                "reproduce a DIFFERENT split from the one on disk and report a "
                "leakage figure for a slice nobody trained on." % (__file__, name))
    return "constants match create_slices.py"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", default=os.path.join("data", "dermmnist"))
    ap.add_argument("--slice", type=int, default=1)
    args = ap.parse_args(argv)

    print(_check_constants_against_prep())

    meta_path = os.path.join(args.data_dir, "dermmnist_c_metadata.csv")
    if not os.path.exists(meta_path):
        print("\n%s not found." % meta_path)
        print("This is the only input needed -- no images, no GPU. It ships with")
        print("the DermaMNIST-C download (data/dermmnist/download_data.py) and")
        print("lives on the server. Re-run there, or point --data-dir at it.")
        return 2

    meta = pd.read_csv(meta_path)
    for col in ("dx", "split", "lesion_id"):
        if col not in meta.columns:
            raise SystemExit("%s has no `%s` column: %s"
                             % (meta_path, col, list(meta.columns)))
    meta["label"] = meta["dx"].map(DX_TO_CLASS)
    if meta["label"].isna().any():
        raise SystemExit("unmapped dx value(s): %s"
                         % sorted(meta.loc[meta["label"].isna(), "dx"].unique()))

    # Pool in the SAME order create_slices.py does. Order decides the indices.
    pooled = pd.concat([meta[meta["split"] == s].reset_index(drop=True)
                        for s in SPLIT_ORDER], ignore_index=True)
    y = pooled["label"].to_numpy()
    lesion = pooled["lesion_id"].to_numpy()
    print("pooled %d images over %d distinct lesions"
          % (len(pooled), len(set(lesion))))

    from sklearn.model_selection import StratifiedShuffleSplit
    seed = BASE_SEED + args.slice
    sss = StratifiedShuffleSplit(n_splits=1, test_size=TEST_FRACTION,
                                 random_state=seed)
    train_idx, test_idx = next(sss.split(np.zeros((len(y), 1)), y))

    train_lesions = set(lesion[train_idx])
    test_lesion = lesion[test_idx]
    test_y = y[test_idx]
    leaked = np.array([lid in train_lesions for lid in test_lesion])

    n, k = len(test_idx), int(leaked.sum())
    print("\nslice %d (seed %d), test set of %d:" % (args.slice, seed, n))
    print("  %d of %d test images share a lesion_id with TRAIN  =  %.1f%%"
          % (k, n, 100.0 * k / n))
    mel = test_y == MEL
    km, nm = int((leaked & mel).sum()), int(mel.sum())
    print("  %d of %d MELANOMA test images do  =  %.1f%%   <-- the capped class"
          % (km, nm, 100.0 * km / max(1, nm)))

    print("\n  per class:")
    inv = {v: k2 for k2, v in DX_TO_CLASS.items()}
    for c in sorted(set(test_y.tolist())):
        m = test_y == c
        print("    %-6s %4d test, %4d leaked  (%.1f%%)%s"
              % (inv[c], int(m.sum()), int((leaked & m).sum()),
                 100.0 * (leaked & m).sum() / max(1, m.sum()),
                 "   <-- capped" if c == MEL else ""))

    print("\nPaired arm-vs-arm deltas survive this. Absolute dermmnist quality")
    print("numbers do not. See docs/FRAMEWORK.md section 1.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
