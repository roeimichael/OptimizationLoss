"""Download DermaMNIST-C 224x224 (corrected) and prepare splits.

DermaMNIST-C is from SFU's corrected version of MedMNIST's DermaMNIST.
Source: HAM10000 (10,015 skin lesion images), with fixed train/val/test splits
that prevent same-lesion leakage across splits.

7 classes: AKIEC, BCC, BKL, DF, MEL, NV, VASC.
Corrected splits: train 8215 / val 573 / test 1227.

Supports native 224x224 or resized 64x64. Metadata (sex, age, localization)
comes from the corrected CSV which is row-aligned with the npz.

Usage:
    python download_data.py          # Default: 224x224 (native)
    python download_data.py --size 64  # Resized to 64x64
"""

import os
import numpy as np
import pandas as pd
from PIL import Image

CLASS_NAMES = {
    0: 'AKIEC',  # Actinic keratoses
    1: 'BCC',    # Basal cell carcinoma
    2: 'BKL',    # Benign keratosis
    3: 'DF',     # Dermatofibroma
    4: 'MEL',    # Melanoma (constrained class)
    5: 'NV',     # Melanocytic nevi
    6: 'VASC',   # Vascular lesions
}

DX_TO_CLASS = {
    'akiec': 0, 'bcc': 1, 'bkl': 2, 'df': 3, 'mel': 4, 'nv': 5, 'vasc': 6,
}

SEX_MAP = {'male': 0, 'female': 1, 'unknown': 0}  # 57 unknowns → majority group

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
NPZ_FILE = os.path.join(DATA_DIR, 'dermamnist_corrected_224.npz')
META_CSV = os.path.join(DATA_DIR, 'dermmnist_c_metadata.csv')

NPZ_URL = 'https://zenodo.org/records/11101338/files/dermamnist_corrected_224.npz?download=1'
META_URL = ('https://raw.githubusercontent.com/kakumarabhishek/Corrected-Skin-Image-Datasets'
            '/main/DermaMNIST/DermaMNIST_Analysis/CSV_files'
            '/combined_metadata_corrected-HAM10000_corrected.csv')


def download_if_needed():
    """Download DermaMNIST-C 224x224 npz and metadata CSV if not present."""
    import urllib.request

    if not os.path.exists(NPZ_FILE):
        print(f"Downloading DermaMNIST-C 224x224 (~1.1 GB)...")
        urllib.request.urlretrieve(NPZ_URL, NPZ_FILE)
        size_mb = os.path.getsize(NPZ_FILE) / 1024 / 1024
        print(f"Downloaded: {NPZ_FILE} ({size_mb:.0f} MB)")
    else:
        print(f"NPZ already exists: {NPZ_FILE}")

    if not os.path.exists(META_CSV):
        print("Downloading corrected metadata CSV...")
        urllib.request.urlretrieve(META_URL, META_CSV)
        print(f"Downloaded: {META_CSV}")
    else:
        print(f"Metadata CSV already exists: {META_CSV}")


TARGET_SIZE = 224  # Native resolution (use --size 64 for reduced)


def resize_images(images_hwc, target_size):
    """Resize batch of (N, H, W, 3) uint8 images to (N, target_size, target_size, 3)."""
    resized = np.empty((len(images_hwc), target_size, target_size, 3), dtype=np.uint8)
    for i in range(len(images_hwc)):
        img = Image.fromarray(images_hwc[i])
        resized[i] = np.array(img.resize((target_size, target_size), Image.LANCZOS))
    return resized


def process_and_save(target_size=TARGET_SIZE):
    """Extract splits from npz, optionally resize, build metadata, save as npy + csv."""
    data = np.load(NPZ_FILE)
    meta_df = pd.read_csv(META_CSV)
    meta_df['label'] = meta_df['dx'].map(DX_TO_CLASS)

    for split in ['train', 'val', 'test']:
        images = data[f'{split}_images']   # (N, 224, 224, 3) uint8
        labels = data[f'{split}_labels'].flatten()  # (N,) uint8

        # Resize only if target is not native 224
        if target_size != 224:
            images = resize_images(images, target_size)

        # Transpose to channels-first: (N, 3, H, W) and normalize to [0, 1]
        images_chw = images.transpose(0, 3, 1, 2).astype(np.float32) / 255.0

        # Get metadata for this split (row-aligned with npz)
        split_meta = meta_df[meta_df['split'] == split].reset_index(drop=True)

        # Verify alignment: labels must match
        csv_labels = split_meta['label'].values
        assert np.array_equal(labels, csv_labels), (
            f"{split}: npz labels don't match CSV labels! Data alignment broken."
        )

        # Build sex column (encoded: male=0, female=1, unknown→0)
        sex_encoded = split_meta['sex'].map(SEX_MAP).values.astype(np.int64)

        # Save arrays
        np.save(os.path.join(DATA_DIR, f'{split}_images.npy'), images_chw)
        np.save(os.path.join(DATA_DIR, f'{split}_labels.npy'), labels)

        # Save metadata CSV with sex
        out_meta = pd.DataFrame({
            'label': labels,
            'class_name': [CLASS_NAMES[l] for l in labels],
            'sex': sex_encoded,
        })
        out_meta.to_csv(os.path.join(DATA_DIR, f'{split}_meta.csv'), index=False)

        print(f"\n{split}: {len(labels)} samples, images shape {images_chw.shape}")
        print(f"  sex: male(0)={sum(sex_encoded == 0)}, female(1)={sum(sex_encoded == 1)}")


def print_summary():
    """Print class distribution and sex distribution per split."""
    images = np.load(os.path.join(DATA_DIR, 'train_images.npy'))
    img_size = images.shape[-1]
    print("\n" + "=" * 70)
    print(f"DermaMNIST-C {img_size}x{img_size} (Corrected) - Dataset Summary")
    print("=" * 70)

    for split in ['train', 'val', 'test']:
        labels = np.load(os.path.join(DATA_DIR, f'{split}_labels.npy'))
        images = np.load(os.path.join(DATA_DIR, f'{split}_images.npy'))
        meta = pd.read_csv(os.path.join(DATA_DIR, f'{split}_meta.csv'))

        print(f"\n--- {split.upper()} ({len(labels)} samples) ---")
        print(f"Images: {images.shape} dtype={images.dtype} "
              f"range=[{images.min():.2f}, {images.max():.2f}]")

        for c in range(7):
            count = (labels == c).sum()
            pct = count / len(labels) * 100
            print(f"  Class {c} ({CLASS_NAMES[c]:>5}): {count:>5} ({pct:5.1f}%)")

        sex = meta['sex'].values
        print(f"  Sex: male(0)={sum(sex == 0)}, female(1)={sum(sex == 1)}")

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


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--size', type=int, default=TARGET_SIZE,
                        help=f'Image size (default: {TARGET_SIZE})')
    args = parser.parse_args()

    download_if_needed()
    process_and_save(target_size=args.size)
    print_summary()
