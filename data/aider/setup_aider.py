"""Setup AIDER: read raw class dirs, resize to 224x224, save train/test npy + meta.

Source: ~/OptimizationLoss/data/aider_raw/AIDER/<class>/*.jpg
Output: ~/OptimizationLoss/data/aider/{train,test}_{images,labels}.npy + meta.csv

After this runs, use create_slices.py (mirrors eurosat) for 5 stratified slices.
"""
import os

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import StratifiedShuffleSplit

CLASSES = {  # consistent with AIDER paper class order
    "collapsed_building": 0,
    "fire": 1,
    "flooded_areas": 2,
    "normal": 3,
}
CLASS_NAMES = {v: k for k, v in CLASSES.items()}
IMG_SIZE = 224
DATA_DIR = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(os.path.dirname(DATA_DIR), "aider_raw", "AIDER")
TEST_FRACTION = 0.2
SEED = 42


def main():
    rows, images = [], []
    for cls_name, lbl in CLASSES.items():
        cls_dir = os.path.join(SRC, cls_name)
        files = sorted(f for f in os.listdir(cls_dir) if f.endswith(".jpg"))
        print(f"  {cls_name}: {len(files)} files")
        for fn in files:
            img = Image.open(os.path.join(cls_dir, fn)).convert("RGB").resize(
                (IMG_SIZE, IMG_SIZE), Image.BILINEAR)
            arr = np.asarray(img, dtype=np.float32) / 255.0  # (H, W, 3) in [0,1]
            arr = np.transpose(arr, (2, 0, 1))  # → (3, H, W)
            images.append(arr)
            rows.append({"label": lbl, "class_name": cls_name, "filename": fn})

    X = np.stack(images).astype(np.float32)  # (N, 3, 224, 224)
    df = pd.DataFrame(rows)
    print(f"\nPooled: {X.shape} | {X.dtype} | range [{X.min():.3f}, {X.max():.3f}]")

    # Synthetic binary group (AIDER subset has no geographic metadata)
    rng = np.random.default_rng(SEED)
    df["synth_group"] = rng.integers(0, 2, size=len(df))

    sss = StratifiedShuffleSplit(n_splits=1, test_size=TEST_FRACTION,
                                  random_state=SEED)
    tr_idx, te_idx = next(sss.split(X, df["label"]))

    for prefix, idx in (("train", tr_idx), ("test", te_idx)):
        np.save(os.path.join(DATA_DIR, f"{prefix}_images.npy"), X[idx])
        np.save(os.path.join(DATA_DIR, f"{prefix}_labels.npy"),
                df["label"].values[idx].astype(np.int64))
        df.iloc[idx].reset_index(drop=True).to_csv(
            os.path.join(DATA_DIR, f"{prefix}_meta.csv"), index=False)
        print(f"  {prefix}: N={len(idx)}")

    print("\nClass balance per split:")
    for prefix, idx in (("train", tr_idx), ("test", te_idx)):
        print(f"  {prefix}: {df.iloc[idx]['class_name'].value_counts().to_dict()}")


if __name__ == "__main__":
    main()
