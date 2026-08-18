"""Create 5 stratified train/test splits (slices) of DermMNIST for statistical significance.

Reads directly from dermamnist_corrected_224.npz + metadata CSV (downloaded by
download_data.py). Pools all splits (train+val+test) into a single dataset, then
creates 5 independent stratified 80/20 train/test splits with different random seeds.

Output structure:
    data/dermmnist/slice_1/  train_images.npy, test_images.npy, train_labels.npy,
                             test_labels.npy, train_meta.csv, test_meta.csv
    ...
    data/dermmnist/slice_5/  (same files)

Each slice is independent and can be used by setting dataset_config.data_dir to
the slice directory, e.g., 'data/dermmnist/slice_1'.

Usage:
    python data/dermmnist/download_data.py   # First: download .npz + metadata
    python data/dermmnist/create_slices.py   # Then: create 5 slices
"""

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
NUM_SLICES = 5
TEST_FRACTION = 0.2  # 80/20 split
BASE_SEED = 42

CLASS_NAMES = {
    0: 'AKIEC', 1: 'BCC', 2: 'BKL', 3: 'DF',
    4: 'MEL', 5: 'NV', 6: 'VASC',
}

DX_TO_CLASS = {
    'akiec': 0, 'bcc': 1, 'bkl': 2, 'df': 3, 'mel': 4, 'nv': 5, 'vasc': 6,
}

SEX_MAP = {'male': 0, 'female': 1, 'unknown': 0}

LOC_MAP = {
    'back': 'torso', 'trunk': 'torso', 'abdomen': 'torso', 'chest': 'torso',
    'genital': 'torso', 'unknown': 'torso',  # class-distribution closest to torso
    'lower extremity': 'extremity', 'upper extremity': 'extremity',
    'foot': 'extremity', 'hand': 'extremity', 'acral': 'extremity',
    'face': 'head_neck', 'neck': 'head_neck', 'scalp': 'head_neck', 'ear': 'head_neck',
}
LOC_ENCODE = {'torso': 0, 'extremity': 1, 'head_neck': 2}


def load_all_data():
    """Load all DermMNIST data from .npz + metadata CSV, pool all splits."""
    npz_path = os.path.join(DATA_DIR, 'dermamnist_corrected_224.npz')
    meta_path = os.path.join(DATA_DIR, 'dermmnist_c_metadata.csv')

    if not os.path.exists(npz_path):
        raise FileNotFoundError(
            f"{npz_path} not found. Run 'python data/dermmnist/download_data.py' first.")

    data = np.load(npz_path)
    meta_df = pd.read_csv(meta_path)
    meta_df['label'] = meta_df['dx'].map(DX_TO_CLASS)

    # Pool all splits (train + val + test)
    all_images_list = []
    all_labels_list = []
    all_meta_list = []

    for split in ['train', 'val', 'test']:
        images = data[f'{split}_images']       # (N, 224, 224, 3) uint8
        labels = data[f'{split}_labels'].flatten()

        # Transpose to channels-first and normalize: (N, 3, H, W) float32
        images_chw = images.transpose(0, 3, 1, 2).astype(np.float32) / 255.0

        # Get metadata for this split (row-aligned with npz)
        split_meta = meta_df[meta_df['split'] == split].reset_index(drop=True)
        assert np.array_equal(labels, split_meta['label'].values), \
            f"{split}: npz labels don't match CSV labels!"

        sex_encoded = split_meta['sex'].map(SEX_MAP).values.astype(np.int64)
        loc_grouped = split_meta['localization'].str.lower().map(LOC_MAP)
        # 'other' is NOT in LOC_ENCODE, so an unmapped localization used to
        # become NaN and then, through .astype(np.int64), the sentinel
        # -9223372036854775808 -- a group id that silently defines a local cap.
        # The loader's null guard cannot catch it: the bad cast happens here,
        # upstream, and produces a perfectly valid int64. This is the only real
        # group column in the project, so it fails loudly instead.
        unmapped = split_meta.loc[loc_grouped.isna(), 'localization'].unique()
        if len(unmapped):
            raise ValueError(
                "%s: localization value(s) %s are not in LOC_MAP. Every local "
                "cap is defined over this column; an unmapped value would cast "
                "to an int64 sentinel and become a phantom group."
                % (split, sorted(map(str, unmapped))))
        loc_encoded = loc_grouped.map(LOC_ENCODE).values.astype(np.int64)
        meta_out = pd.DataFrame({
            'label': labels,
            'class_name': [CLASS_NAMES[l] for l in labels],
            'sex': sex_encoded,
            'loc_group': loc_encoded,
        })

        all_images_list.append(images_chw)
        all_labels_list.append(labels)
        all_meta_list.append(meta_out)

    all_images = np.concatenate(all_images_list, axis=0)
    all_labels = np.concatenate(all_labels_list, axis=0)
    all_meta = pd.concat(all_meta_list, ignore_index=True)

    print(f"Pooled data: {len(all_labels)} samples, images shape {all_images.shape}")
    print(f"Class distribution:")
    for c in range(7):
        n = (all_labels == c).sum()
        print(f"  {c} ({CLASS_NAMES[c]:>5}): {n:>5} ({n/len(all_labels)*100:.1f}%)")

    return all_images, all_labels, all_meta


def create_slices(all_images, all_labels, all_meta):
    """Create NUM_SLICES stratified splits and save each to its own directory."""
    for slice_idx in range(1, NUM_SLICES + 1):
        seed = BASE_SEED + slice_idx
        slice_dir = os.path.join(DATA_DIR, f'slice_{slice_idx}')
        os.makedirs(slice_dir, exist_ok=True)

        # Stratified split
        sss = StratifiedShuffleSplit(n_splits=1, test_size=TEST_FRACTION, random_state=seed)
        train_idx, test_idx = next(sss.split(all_images, all_labels))

        train_images = all_images[train_idx]
        train_labels = all_labels[train_idx]
        test_images = all_images[test_idx]
        test_labels = all_labels[test_idx]

        train_meta = all_meta.iloc[train_idx].reset_index(drop=True)
        test_meta = all_meta.iloc[test_idx].reset_index(drop=True)

        # Save arrays
        np.save(os.path.join(slice_dir, 'train_images.npy'), train_images)
        np.save(os.path.join(slice_dir, 'train_labels.npy'), train_labels)
        np.save(os.path.join(slice_dir, 'test_images.npy'), test_images)
        np.save(os.path.join(slice_dir, 'test_labels.npy'), test_labels)

        # Save metadata CSVs
        train_meta.to_csv(os.path.join(slice_dir, 'train_meta.csv'), index=False)
        test_meta.to_csv(os.path.join(slice_dir, 'test_meta.csv'), index=False)

        # Verify stratification
        print(f"\n--- Slice {slice_idx} (seed={seed}) ---")
        print(f"  Train: {len(train_labels)}  Test: {len(test_labels)}")

        # Check no overlap
        assert len(set(train_idx) & set(test_idx)) == 0, "Train/test overlap!"

        # Check class distribution preserved
        for c in range(7):
            train_pct = (train_labels == c).sum() / len(train_labels) * 100
            test_pct = (test_labels == c).sum() / len(test_labels) * 100
            diff = abs(train_pct - test_pct)
            marker = " *" if diff > 1.0 else ""
            print(f"  {CLASS_NAMES[c]:>5}: train {train_pct:5.1f}%  test {test_pct:5.1f}%{marker}")

    print(f"\nCreated {NUM_SLICES} slices in {DATA_DIR}/slice_1..{NUM_SLICES}/")


def verify_independence():
    """Verify that test sets across slices have different samples."""
    test_sets = []
    for i in range(1, NUM_SLICES + 1):
        labels = np.load(os.path.join(DATA_DIR, f'slice_{i}', 'test_labels.npy'))
        images = np.load(os.path.join(DATA_DIR, f'slice_{i}', 'test_images.npy'))
        # Use a hash of each image as identity
        hashes = set()
        for img in images:
            hashes.add(img.tobytes()[:64])  # First 64 bytes as fingerprint
        test_sets.append(hashes)

    print("\nSlice independence check (test set overlap %):")
    for i in range(NUM_SLICES):
        for j in range(i + 1, NUM_SLICES):
            overlap = len(test_sets[i] & test_sets[j])
            total = len(test_sets[i])
            print(f"  Slice {i+1} vs {j+1}: {overlap}/{total} overlap ({overlap/total*100:.1f}%)")


if __name__ == '__main__':
    all_images, all_labels, all_meta = load_all_data()
    create_slices(all_images, all_labels, all_meta)
    verify_independence()
