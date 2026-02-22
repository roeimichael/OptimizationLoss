"""Download DermMNIST 64x64 and prepare train/val/test splits.

DermMNIST is from the MedMNIST collection, sourced from HAM10000 (10,015 skin lesion images).
7 classes: AKIEC, BCC, BKL, DF, MEL, NV, VASC.
Predefined splits: train 7007 / val 1003 / test 2005.

Note: The MedMNIST npz does not include HAM10000 image IDs, so demographic
metadata (age, sex) cannot be joined. Group column for local constraints
must be derived from another source or omitted.
"""

import os
import numpy as np
import pandas as pd

CLASS_NAMES = {
    0: 'AKIEC',  # Actinic keratoses
    1: 'BCC',    # Basal cell carcinoma
    2: 'BKL',    # Benign keratosis
    3: 'DF',     # Dermatofibroma
    4: 'MEL',    # Melanoma
    5: 'NV',     # Melanocytic nevi
    6: 'VASC',   # Vascular lesions
}

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
NPZ_FILE = os.path.join(DATA_DIR, 'dermamnist_64.npz')


def download_if_needed():
    """Download DermMNIST 64x64 via medmnist if not present."""
    if os.path.exists(NPZ_FILE):
        print(f"NPZ already exists: {NPZ_FILE}")
        return

    print("Downloading DermMNIST 64x64...")
    from medmnist import DermaMNIST
    DermaMNIST(split='train', download=True, size=64, root=DATA_DIR)
    print(f"Downloaded to: {NPZ_FILE}")


def process_and_save():
    """Extract splits from npz, normalize images, save as npy + csv."""
    data = np.load(NPZ_FILE)

    for split in ['train', 'val', 'test']:
        images = data[f'{split}_images']   # (N, 64, 64, 3) uint8
        labels = data[f'{split}_labels'].flatten()  # (N,) uint8

        # Transpose to channels-first: (N, 3, 64, 64) and normalize to [0, 1]
        images_chw = images.transpose(0, 3, 1, 2).astype(np.float32) / 255.0

        # Save arrays
        np.save(os.path.join(DATA_DIR, f'{split}_images.npy'), images_chw)
        np.save(os.path.join(DATA_DIR, f'{split}_labels.npy'), labels)

        # Save metadata CSV (label + class name, extensible for future group columns)
        meta = pd.DataFrame({
            'label': labels,
            'class_name': [CLASS_NAMES[l] for l in labels],
        })
        meta.to_csv(os.path.join(DATA_DIR, f'{split}_meta.csv'), index=False)

        print(f"\n{split}: {len(labels)} samples, images shape {images_chw.shape}")


def print_summary():
    """Print class distribution and dataset statistics."""
    print("\n" + "=" * 70)
    print("DermMNIST 64x64 — Dataset Summary")
    print("=" * 70)

    for split in ['train', 'val', 'test']:
        labels = np.load(os.path.join(DATA_DIR, f'{split}_labels.npy'))
        images = np.load(os.path.join(DATA_DIR, f'{split}_images.npy'))

        print(f"\n--- {split.upper()} ({len(labels)} samples) ---")
        print(f"Images: {images.shape} dtype={images.dtype} range=[{images.min():.2f}, {images.max():.2f}]")

        for c in range(7):
            count = (labels == c).sum()
            pct = count / len(labels) * 100
            print(f"  Class {c} ({CLASS_NAMES[c]:>5}): {count:>5} ({pct:5.1f}%)")

    # Overall
    all_labels = np.concatenate([
        np.load(os.path.join(DATA_DIR, f'{s}_labels.npy'))
        for s in ['train', 'val', 'test']
    ])
    print(f"\n--- TOTAL ({len(all_labels)} samples) ---")
    for c in range(7):
        count = (all_labels == c).sum()
        pct = count / len(all_labels) * 100
        print(f"  Class {c} ({CLASS_NAMES[c]:>5}): {count:>5} ({pct:5.1f}%)")

    print("\nNote: No demographic metadata (sex/age) available from MedMNIST npz.")
    print("Local constraints require an alternative group source.")


if __name__ == '__main__':
    download_if_needed()
    process_and_save()
    print_summary()
