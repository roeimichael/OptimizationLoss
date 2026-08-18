"""Create 5 stratified train/test splits (slices) of TissueMNIST for statistical significance.

Reads from the existing train/test npy files, pools them, then creates 5 independent
stratified 80/20 train/test splits with different random seeds.

Output structure:
    data/tissuemnist/slice_1/  train_images.npy, test_images.npy, train_labels.npy,
                               test_labels.npy, train_meta.csv, test_meta.csv
    ...
    data/tissuemnist/slice_5/  (same files)

Usage:
    python data/tissuemnist/create_slices.py
"""

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
NUM_SLICES = 5
TEST_FRACTION = 0.2  # 80/20 split
BASE_SEED = 42
NUM_CLASSES = 8

CLASS_NAMES = {
    0: 'CDI', 1: 'CDS', 2: 'CST', 3: 'EPI',
    4: 'GE', 5: 'PTC', 6: 'STR', 7: 'TUB',
}


def load_all_data():
    """Load all TissueMNIST data, pool train + test into single dataset."""
    train_images = np.load(os.path.join(DATA_DIR, 'train_images.npy'))
    train_labels = np.load(os.path.join(DATA_DIR, 'train_labels.npy')).ravel()
    test_images = np.load(os.path.join(DATA_DIR, 'test_images.npy'))
    test_labels = np.load(os.path.join(DATA_DIR, 'test_labels.npy')).ravel()

    train_meta = pd.read_csv(os.path.join(DATA_DIR, 'train_meta.csv'))
    test_meta = pd.read_csv(os.path.join(DATA_DIR, 'test_meta.csv'))

    all_images = np.concatenate([train_images, test_images], axis=0)
    all_labels = np.concatenate([train_labels, test_labels], axis=0)
    all_meta = pd.concat([train_meta, test_meta], ignore_index=True)

    print("Pooled data: %d samples, images shape %s" % (len(all_labels), all_images.shape))
    print("Class distribution:")
    for c in range(NUM_CLASSES):
        n = (all_labels == c).sum()
        print("  %d (%4s): %5d (%.1f%%)" % (c, CLASS_NAMES[c], n, n / len(all_labels) * 100))

    return all_images, all_labels, all_meta


def create_slices(all_images, all_labels, all_meta):
    """Create NUM_SLICES stratified splits and save each to its own directory."""
    for slice_idx in range(1, NUM_SLICES + 1):
        seed = BASE_SEED + slice_idx
        slice_dir = os.path.join(DATA_DIR, 'slice_%d' % slice_idx)
        os.makedirs(slice_dir, exist_ok=True)

        sss = StratifiedShuffleSplit(n_splits=1, test_size=TEST_FRACTION, random_state=seed)
        train_idx, test_idx = next(sss.split(all_images, all_labels))

        train_images = all_images[train_idx]
        train_labels = all_labels[train_idx]
        test_images = all_images[test_idx]
        test_labels = all_labels[test_idx]

        train_meta = all_meta.iloc[train_idx].reset_index(drop=True)
        test_meta = all_meta.iloc[test_idx].reset_index(drop=True)

        np.save(os.path.join(slice_dir, 'train_images.npy'), train_images)
        np.save(os.path.join(slice_dir, 'train_labels.npy'), train_labels)
        np.save(os.path.join(slice_dir, 'test_images.npy'), test_images)
        np.save(os.path.join(slice_dir, 'test_labels.npy'), test_labels)

        train_meta.to_csv(os.path.join(slice_dir, 'train_meta.csv'), index=False)
        test_meta.to_csv(os.path.join(slice_dir, 'test_meta.csv'), index=False)

        print("\n--- Slice %d (seed=%d) ---" % (slice_idx, seed))
        print("  Train: %d  Test: %d" % (len(train_labels), len(test_labels)))

        assert len(set(train_idx) & set(test_idx)) == 0, "Train/test overlap!"

        for c in range(NUM_CLASSES):
            train_pct = (train_labels == c).sum() / len(train_labels) * 100
            test_pct = (test_labels == c).sum() / len(test_labels) * 100
            diff = abs(train_pct - test_pct)
            marker = " *" if diff > 1.0 else ""
            print("  %4s: train %5.1f%%  test %5.1f%%%s" % (CLASS_NAMES[c], train_pct, test_pct, marker))

    print("\nCreated %d slices in %s/slice_1..%d/" % (NUM_SLICES, DATA_DIR, NUM_SLICES))


if __name__ == '__main__':
    all_images, all_labels, all_meta = load_all_data()
    create_slices(all_images, all_labels, all_meta)
