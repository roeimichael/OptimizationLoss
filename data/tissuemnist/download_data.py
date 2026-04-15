"""Download TissueMNIST and prepare subsampled train/test splits.

TissueMNIST: 236,386 grayscale microscopy images of kidney cortex cells.
8 cell-type classes. Original 32x32 patches, MedMNIST standard 28x28.
Source: Broad Bioimage Benchmark Collection (BBBC051).

Published benchmark accuracy (ResNet-18, 28x28): 67.6%.
SOTA (as of 2024): ~75.6%. Suitable for constraint optimization research
because non-trivial classification difficulty leads to meaningful trade-offs.

Strategy:
  1. Download compact 28x28 npz (~25 MB) from Zenodo
  2. Combine train+val+test into one pool (236K samples)
  3. Stratified random subsample to ~12K (comparable to DermMNIST's ~10K)
  4. Resize 28x28 -> 224x224 for pretrained model compatibility
  5. Stratified 80/20 train/test split
  6. Fabricate balanced binary group column (no real demographics available)

The 224x224 MedMNIST+ version is just the 28x28 upscaled — no extra detail.
Resizing ourselves after subsampling is equivalent and avoids a multi-GB download.

Usage:
    python download_data.py                    # 12K samples, 224x224 (default)
    python download_data.py --n_samples 15000  # 15K samples
    python download_data.py --size 28          # Keep native 28x28
"""

import os

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split

# ── Class definitions ─────────────────────────────────────────────────────────

CLASS_NAMES = {
    0: 'CDI',    # Collecting Duct, Intercalated
    1: 'CDP',    # Collecting Duct, Principal
    2: 'CT',     # Connecting Tubule
    3: 'DCT',    # Distal Convoluted Tubule
    4: 'GE',     # Glomerular Endothelial
    5: 'INT',    # Interstitial
    6: 'PTC',    # Proximal Tubule, Convoluted
    7: 'PTS',    # Proximal Tubule, Straight
}

CLASS_FULL_NAMES = {
    0: 'Collecting Duct, Intercalated',
    1: 'Collecting Duct, Principal',
    2: 'Connecting Tubule',
    3: 'Distal Convoluted Tubule',
    4: 'Glomerular Endothelial',
    5: 'Interstitial',
    6: 'Proximal Tubule, Convoluted',
    7: 'Proximal Tubule, Straight',
}

# ── Paths & URLs ──────────────────────────────────────────────────────────────

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
NPZ_FILE = os.path.join(DATA_DIR, 'tissuemnist.npz')
NPZ_URL = 'https://zenodo.org/records/10519652/files/tissuemnist.npz?download=1'

DEFAULT_N_SAMPLES = 12000   # Similar to DermMNIST (~10K)
DEFAULT_TARGET_SIZE = 224   # Match pretrained model expectations


# ── Download ──────────────────────────────────────────────────────────────────

def download_if_needed():
    """Download TissueMNIST npz from Zenodo if not present."""
    import urllib.request

    if not os.path.exists(NPZ_FILE):
        print("Downloading TissueMNIST 28x28 from Zenodo (~25 MB)...")
        urllib.request.urlretrieve(NPZ_URL, NPZ_FILE)
        size_mb = os.path.getsize(NPZ_FILE) / 1024 / 1024
        print(f"Downloaded: {NPZ_FILE} ({size_mb:.0f} MB)")
    else:
        print(f"NPZ already exists: {NPZ_FILE}")


# ── Validation helpers ────────────────────────────────────────────────────────

def check_duplicates(images):
    """Count duplicate images via pixel-hash. Returns duplicate count."""
    hashes = set()
    duplicates = 0
    for i in range(len(images)):
        h = hash(images[i].tobytes())
        if h in hashes:
            duplicates += 1
        hashes.add(h)
    return duplicates


def validate_data(images, labels, n_classes=8):
    """Run integrity checks on raw images and labels."""
    assert images.min() >= 0 and images.max() <= 255, \
        f"Image values out of uint8 range: [{images.min()}, {images.max()}]"
    unique_labels = set(np.unique(labels))
    expected = set(range(n_classes))
    assert unique_labels <= expected, \
        f"Unexpected label values: {unique_labels - expected}"
    assert len(images) == len(labels), \
        f"Image/label count mismatch: {len(images)} vs {len(labels)}"
    print(f"  Validation passed: {len(labels):,} samples, "
          f"labels {min(unique_labels)}..{max(unique_labels)}, "
          f"pixel range [{images.min()}, {images.max()}]")


# ── Resize ────────────────────────────────────────────────────────────────────

def resize_images_chunked(images, target_size, chunk_size=2000):
    """Resize (N, H, W) grayscale uint8 images in chunks to limit memory."""
    n = len(images)
    resized = np.empty((n, target_size, target_size), dtype=np.uint8)
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        for i in range(start, end):
            img = Image.fromarray(images[i], mode='L')
            resized[i] = np.array(
                img.resize((target_size, target_size), Image.LANCZOS))
        print(f"  Resized {end:,}/{n:,} images...", end='\r')
    print()
    return resized


# ── Main processing ──────────────────────────────────────────────────────────

def process_and_save(n_samples=DEFAULT_N_SAMPLES, target_size=DEFAULT_TARGET_SIZE):
    """Subsample, resize, stratified 80/20 split, save as npy+csv.

    Pipeline:
      1. Load all 236K labels + images at 28x28 (~185 MB uint8 — fits in memory)
      2. Stratified random subsample to n_samples
      3. Validate + duplicate check on the subsample
      4. Resize from 28x28 to target_size (only 12K images — fast)
      5. Stratified 80/20 train/test split
      6. Convert to (N, 1, H, W) float32 [0, 1]
      7. Add synthetic group column
      8. Save npy + csv
    """
    data = np.load(NPZ_FILE)

    # ── 1. Gather all images and labels (28x28, ~185 MB total — no memory issue) ──
    all_images_list = []
    all_labels_list = []
    for split in ['train', 'val', 'test']:
        imgs = data[f'{split}_images']        # (N, 28, 28) uint8
        lbls = data[f'{split}_labels'].flatten()  # (N,1) -> (N,)
        all_images_list.append(imgs)
        all_labels_list.append(lbls)
        print(f"Loaded {split:>5}: {len(lbls):>7,} samples, shape {imgs.shape}")

    all_images = np.concatenate(all_images_list, axis=0)  # (236K, 28, 28)
    all_labels = np.concatenate(all_labels_list, axis=0)   # (236K,)
    del all_images_list, all_labels_list
    print(f"\nFull dataset: {len(all_labels):,} samples, shape {all_images.shape}")

    # ── 2. Stratified subsample ──
    if n_samples >= len(all_labels):
        print(f"n_samples={n_samples:,} >= total={len(all_labels):,}, using all.")
        keep_idx = np.arange(len(all_labels))
    else:
        print(f"\nStratified subsample: {len(all_labels):,} -> {n_samples:,}")
        keep_idx, _ = train_test_split(
            np.arange(len(all_labels)),
            train_size=n_samples,
            random_state=42,
            stratify=all_labels,
        )
        keep_idx.sort()  # Maintain original ordering

    all_images = all_images[keep_idx]
    all_labels = all_labels[keep_idx]
    print(f"Subsampled: {len(all_labels):,} samples")

    # Print class distribution of subsample
    _, counts = np.unique(all_labels, return_counts=True)
    for c in range(8):
        pct = counts[c] / len(all_labels) * 100
        print(f"  Class {c} ({CLASS_NAMES[c]:>3}): {counts[c]:>5,} ({pct:5.1f}%)")

    # ── 3. Validate & check duplicates ──
    validate_data(all_images, all_labels)
    n_dupes = check_duplicates(all_images)
    print(f"Duplicate images in subsample: {n_dupes:,} "
          f"({n_dupes / len(all_labels) * 100:.2f}%)")
    if n_dupes > 0:
        print("  Note: kept — cell patches may legitimately look identical "
              "at this resolution.")

    # ── 4. Resize if target != 28 ──
    if target_size != 28:
        print(f"\nResizing {len(all_images):,} images from 28x28 to "
              f"{target_size}x{target_size}...")
        all_images = resize_images_chunked(all_images, target_size)
        print(f"Resized shape: {all_images.shape}")

    # ── 5. Stratified 80/20 train/test split ──
    train_idx, test_idx = train_test_split(
        np.arange(len(all_labels)),
        test_size=0.2,
        random_state=42,
        stratify=all_labels,
    )

    train_images_raw = all_images[train_idx]
    train_labels = all_labels[train_idx]
    test_images_raw = all_images[test_idx]
    test_labels = all_labels[test_idx]
    del all_images, all_labels

    print(f"\nStratified 80/20 split:")
    print(f"  Train: {len(train_labels):,} samples")
    print(f"  Test:  {len(test_labels):,} samples")

    # ── 6. Convert to channels-first float32 ──
    # (N, H, W) -> (N, 1, H, W) float32 [0, 1]
    train_chw = train_images_raw[:, np.newaxis, :, :].astype(np.float32) / 255.0
    test_chw = test_images_raw[:, np.newaxis, :, :].astype(np.float32) / 255.0
    del train_images_raw, test_images_raw

    # ── 7. Synthetic binary group column (deterministic) ──
    rng = np.random.RandomState(42)
    train_groups = rng.randint(0, 2, size=len(train_labels)).astype(np.int64)
    test_groups = rng.randint(0, 2, size=len(test_labels)).astype(np.int64)

    # ── 8. Save npy arrays ──
    np.save(os.path.join(DATA_DIR, 'train_images.npy'), train_chw)
    np.save(os.path.join(DATA_DIR, 'train_labels.npy'), train_labels)
    np.save(os.path.join(DATA_DIR, 'test_images.npy'), test_chw)
    np.save(os.path.join(DATA_DIR, 'test_labels.npy'), test_labels)

    # ── 9. Save metadata CSVs ──
    for split, labels, groups in [('train', train_labels, train_groups),
                                   ('test', test_labels, test_groups)]:
        meta = pd.DataFrame({
            'label': labels,
            'class_name': [CLASS_NAMES[l] for l in labels],
            'synth_group': groups,
        })
        meta.to_csv(os.path.join(DATA_DIR, f'{split}_meta.csv'), index=False)

    # ── 10. Print storage summary ──
    print(f"\nSaved to {DATA_DIR}/")
    print(f"  train_images.npy: {train_chw.shape}  "
          f"({train_chw.nbytes / 1024**2:.0f} MB)")
    print(f"  test_images.npy:  {test_chw.shape}  "
          f"({test_chw.nbytes / 1024**2:.0f} MB)")
    print(f"  dtype={train_chw.dtype}, range=[{train_chw.min():.2f}, "
          f"{train_chw.max():.2f}]")
    print(f"  Train groups: 0={sum(train_groups == 0):,}, "
          f"1={sum(train_groups == 1):,}")
    print(f"  Test groups:  0={sum(test_groups == 0):,}, "
          f"1={sum(test_groups == 1):,}")


# ── Analysis ──────────────────────────────────────────────────────────────────

def shannon_equitability(labels):
    """Compute Shannon equitability index (0=one class dominates, 1=uniform)."""
    _, counts = np.unique(labels, return_counts=True)
    proportions = counts / counts.sum()
    H = -np.sum(proportions * np.log(proportions))
    H_max = np.log(len(counts))
    return H / H_max if H_max > 0 else 0.0


def print_summary():
    """Print class distribution, Shannon equitability, and dataset info."""
    print("\n" + "=" * 70)
    print("TissueMNIST — Dataset Summary")
    print("=" * 70)

    for split in ['train', 'test']:
        labels = np.load(os.path.join(DATA_DIR, f'{split}_labels.npy'))
        images = np.load(os.path.join(DATA_DIR, f'{split}_images.npy'))
        meta = pd.read_csv(os.path.join(DATA_DIR, f'{split}_meta.csv'))

        print(f"\n{'-' * 50}")
        print(f"{split.upper()} ({len(labels):,} samples)")
        print(f"{'-' * 50}")
        print(f"Images: {images.shape} dtype={images.dtype} "
              f"range=[{images.min():.2f}, {images.max():.2f}]")

        _, counts = np.unique(labels, return_counts=True)
        for c in range(8):
            count = counts[c] if c < len(counts) else 0
            pct = count / len(labels) * 100
            print(f"  Class {c} ({CLASS_NAMES[c]:>3}): "
                  f"{count:>5,} ({pct:5.1f}%)  {CLASS_FULL_NAMES[c]}")

        eq = shannon_equitability(labels)
        print(f"  Shannon equitability: {eq:.4f}")

        groups = meta['synth_group'].values
        print(f"  Synth groups: 0={sum(groups == 0):,}, "
              f"1={sum(groups == 1):,}")

    # Overall
    all_labels = np.concatenate([
        np.load(os.path.join(DATA_DIR, f'{s}_labels.npy'))
        for s in ['train', 'test']
    ])
    print(f"\n{'-' * 50}")
    print(f"TOTAL ({len(all_labels):,} samples)")
    print(f"{'-' * 50}")
    _, counts = np.unique(all_labels, return_counts=True)
    for c in range(8):
        count = counts[c] if c < len(counts) else 0
        pct = count / len(all_labels) * 100
        print(f"  Class {c} ({CLASS_NAMES[c]:>3}): "
              f"{count:>5,} ({pct:5.1f}%)  {CLASS_FULL_NAMES[c]}")
    eq = shannon_equitability(all_labels)
    print(f"  Shannon equitability: {eq:.4f}")

    # Suggest constrained class
    print(f"\n{'-' * 50}")
    print("Constrained class candidates (minority classes):")
    print(f"{'-' * 50}")
    sorted_classes = np.argsort(counts)
    for c in sorted_classes[:3]:
        pct = counts[c] / len(all_labels) * 100
        print(f"  Class {c} ({CLASS_NAMES[c]}): {counts[c]:,} ({pct:.1f}%) "
              f"— {CLASS_FULL_NAMES[c]}")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(
        description='Download and prepare TissueMNIST dataset')
    parser.add_argument('--n_samples', type=int, default=DEFAULT_N_SAMPLES,
                        help=f'Number of samples to subsample '
                             f'(default: {DEFAULT_N_SAMPLES:,})')
    parser.add_argument('--size', type=int, default=DEFAULT_TARGET_SIZE,
                        help=f'Image size (default: {DEFAULT_TARGET_SIZE})')
    args = parser.parse_args()

    download_if_needed()
    process_and_save(n_samples=args.n_samples, target_size=args.size)
    print_summary()
