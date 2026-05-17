"""5 stratified slices of EuroSAT — same protocol as TissueMNIST/DermMNIST.
Reads pooled train+test, then makes 5 independent stratified 80/20 splits with
different seeds. Output: data/eurosat/slice_{1..5}/<train|test>_*.{npy,csv}.
"""
import os

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
NUM_SLICES = 5
TEST_FRACTION = 0.2
BASE_SEED = 42
NUM_CLASSES = 10
CLASS_NAMES = {
    0: "AnnCrop", 1: "Forest", 2: "HerbVeg", 3: "Highway", 4: "Industrial",
    5: "Pasture", 6: "PermCrop", 7: "Resid", 8: "River", 9: "SeaLake",
}


def main():
    train_x = np.load(os.path.join(DATA_DIR, "train_images.npy"))
    train_y = np.load(os.path.join(DATA_DIR, "train_labels.npy"))
    test_x = np.load(os.path.join(DATA_DIR, "test_images.npy"))
    test_y = np.load(os.path.join(DATA_DIR, "test_labels.npy"))
    train_meta = pd.read_csv(os.path.join(DATA_DIR, "train_meta.csv"))
    test_meta = pd.read_csv(os.path.join(DATA_DIR, "test_meta.csv"))

    pooled_x = np.concatenate([train_x, test_x], axis=0)
    pooled_y = np.concatenate([train_y, test_y], axis=0)
    pooled_meta = pd.concat([train_meta, test_meta], ignore_index=True)
    print(f"Pooled: {pooled_x.shape} labels={len(pooled_y)}")

    for slice_idx in range(1, NUM_SLICES + 1):
        seed = BASE_SEED + slice_idx
        sss = StratifiedShuffleSplit(n_splits=1, test_size=TEST_FRACTION, random_state=seed)
        tr_idx, te_idx = next(sss.split(pooled_x, pooled_y))
        slice_dir = os.path.join(DATA_DIR, f"slice_{slice_idx}")
        os.makedirs(slice_dir, exist_ok=True)
        for prefix, idx in (("train", tr_idx), ("test", te_idx)):
            np.save(os.path.join(slice_dir, f"{prefix}_images.npy"), pooled_x[idx])
            np.save(os.path.join(slice_dir, f"{prefix}_labels.npy"), pooled_y[idx])
            pooled_meta.iloc[idx].reset_index(drop=True).to_csv(
                os.path.join(slice_dir, f"{prefix}_meta.csv"), index=False)
        print(f"slice_{slice_idx}: train={len(tr_idx)} test={len(te_idx)}")


if __name__ == "__main__":
    main()
