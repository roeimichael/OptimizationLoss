"""Download CIFAR-100 and prepare subsampled train/test splits.

CIFAR-100: 60,000 32x32 RGB images across 100 fine classes and 20 superclasses.
Each superclass groups 5 fine classes. Standard train/test: 50K/10K.

Strategy:
  1. Download CIFAR-100 via torchvision (auto-cached ~170 MB)
  2. Pool train+test into one set (60K samples)
  3. Stratified random subsample to ~12K (comparable to TissueMNIST)
  4. Resize 32x32 -> 224x224 for pretrained model compatibility
  5. Stratified 80/20 train/test split
  6. Use coarse_label (superclass 0-19) as the group column for local constraints

DO NOT apply ImageNet normalization here -- data_loader.py handles that.

Usage:
    python data/cifar100/download_data.py                    # 12K samples, 224x224
    python data/cifar100/download_data.py --n_samples 15000  # 15K samples
    python data/cifar100/download_data.py --size 32          # Keep native 32x32
"""

import os

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split

# -- Paths -----------------------------------------------------------------

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
SLICE_DIR = os.path.join(DATA_DIR, 'slice_1')

DEFAULT_N_SAMPLES = 12000
DEFAULT_TARGET_SIZE = 224

# -- CIFAR-100 superclass mapping ------------------------------------------
# 20 superclasses, each containing 5 fine classes.

COARSE_NAMES = {
    0: 'aquatic_mammals', 1: 'fish', 2: 'flowers', 3: 'food_containers',
    4: 'fruit_and_vegetables', 5: 'household_electrical_devices',
    6: 'household_furniture', 7: 'insects', 8: 'large_carnivores',
    9: 'large_man-made_outdoor_things', 10: 'large_natural_outdoor_scenes',
    11: 'large_omnivores_and_herbivores', 12: 'medium_mammals',
    13: 'non-insect_invertebrates', 14: 'people', 15: 'reptiles',
    16: 'small_mammals', 17: 'trees', 18: 'vehicles_1', 19: 'vehicles_2',
}

# Fine class names in label order (0-99), from CIFAR-100 metadata.
FINE_NAMES = [
    'apple', 'aquarium_fish', 'baby', 'bear', 'beaver',
    'bed', 'bee', 'beetle', 'bicycle', 'bottle',
    'bowl', 'boy', 'bridge', 'bus', 'butterfly',
    'camel', 'can', 'castle', 'caterpillar', 'cattle',
    'chair', 'chimpanzee', 'clock', 'cloud', 'cockroach',
    'couch', 'crab', 'crocodile', 'cup', 'dinosaur',
    'dolphin', 'elephant', 'flatfish', 'forest', 'fox',
    'girl', 'hamster', 'house', 'kangaroo', 'keyboard',
    'lamp', 'lawn_mower', 'leopard', 'lion', 'lizard',
    'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain',
    'mouse', 'mushroom', 'oak_tree', 'orange', 'orchid',
    'otter', 'palm_tree', 'pear', 'pickup_truck', 'pine_tree',
    'plain', 'plate', 'poppy', 'porcupine', 'possum',
    'rabbit', 'raccoon', 'ray', 'road', 'rocket',
    'rose', 'sea', 'seal', 'shark', 'shrew',
    'skunk', 'skyscraper', 'snail', 'snake', 'spider',
    'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table',
    'tank', 'telephone', 'television', 'tiger', 'tractor',
    'train', 'trout', 'tulip', 'turtle', 'wardrobe',
    'whale', 'willow_tree', 'wolf', 'woman', 'worm',
]


# -- Validation helpers ----------------------------------------------------

def validate_data(images, labels, n_classes=100):
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


# -- Resize ----------------------------------------------------------------

def resize_images_chunked(images, target_size, chunk_size=2000):
    """Resize (N, H, W, 3) RGB uint8 images in chunks to limit memory."""
    n = len(images)
    resized = np.empty((n, target_size, target_size, 3), dtype=np.uint8)
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        for i in range(start, end):
            img = Image.fromarray(images[i], mode='RGB')
            resized[i] = np.array(
                img.resize((target_size, target_size), Image.LANCZOS))
        print(f"  Resized {end:,}/{n:,} images...", end='\r')
    print()
    return resized


# -- Main processing -------------------------------------------------------

def process_and_save(n_samples=DEFAULT_N_SAMPLES, target_size=DEFAULT_TARGET_SIZE):
    """Subsample, resize, stratified 80/20 split, save as npy+csv.

    Pipeline:
      1. Download CIFAR-100 via torchvision (auto-cached)
      2. Pool train+test into single array (60K samples)
      3. Stratified random subsample to n_samples
      4. Validate the subsample
      5. Resize from 32x32 to target_size
      6. Stratified 80/20 train/test split
      7. Convert to (N, 3, H, W) float32 [0, 1]
      8. Save npy + csv with coarse_label as group column
    """
    import torchvision

    print("Downloading/loading CIFAR-100 via torchvision...")
    train_ds = torchvision.datasets.CIFAR100(
        root=DATA_DIR, train=True, download=True)
    test_ds = torchvision.datasets.CIFAR100(
        root=DATA_DIR, train=False, download=True)

    # Extract arrays: torchvision stores data as (N, 32, 32, 3) uint8
    train_images = np.array(train_ds.data)       # (50000, 32, 32, 3) uint8
    train_fine = np.array(train_ds.targets)       # (50000,) int
    test_images = np.array(test_ds.data)          # (10000, 32, 32, 3) uint8
    test_fine = np.array(test_ds.targets)          # (10000,) int

    # Coarse labels: torchvision exposes coarse_targets when loaded
    # but only via the internal attribute. Build from fine->coarse mapping.
    # torchvision stores the mapping in the meta dict after loading.
    # Access it through the dataset's internal structure.
    # The fine-to-coarse mapping is stored in the pickle metadata.
    train_coarse = np.array(
        [train_ds.class_to_idx.get(c, -1) for c in train_fine]
        if not hasattr(train_ds, 'coarse_targets') else train_ds.coarse_targets
    )
    test_coarse = np.array(
        [test_ds.class_to_idx.get(c, -1) for c in test_fine]
        if not hasattr(test_ds, 'coarse_targets') else test_ds.coarse_targets
    )

    # If coarse_targets is not directly available, build from the meta file.
    # torchvision CIFAR100 always has the mapping in self.targets (fine) but
    # coarse labels require reading the meta pickle. Let's just read it directly.
    import pickle
    meta_path = os.path.join(DATA_DIR, 'cifar-100-python', 'meta')
    with open(meta_path, 'rb') as f:
        meta_dict = pickle.load(f, encoding='latin1')
    fine_to_coarse = meta_dict['fine_label_names']  # not what we need

    # Read the actual train/test batches which have coarse_labels
    train_batch_path = os.path.join(DATA_DIR, 'cifar-100-python', 'train')
    with open(train_batch_path, 'rb') as f:
        train_batch = pickle.load(f, encoding='latin1')
    train_coarse = np.array(train_batch['coarse_labels'])

    test_batch_path = os.path.join(DATA_DIR, 'cifar-100-python', 'test')
    with open(test_batch_path, 'rb') as f:
        test_batch = pickle.load(f, encoding='latin1')
    test_coarse = np.array(test_batch['coarse_labels'])

    print(f"Loaded train: {len(train_fine):,} samples, shape {train_images.shape}")
    print(f"Loaded test:  {len(test_fine):,} samples, shape {test_images.shape}")

    # -- 1. Pool train + test --
    all_images = np.concatenate([train_images, test_images], axis=0)
    all_fine = np.concatenate([train_fine, test_fine], axis=0)
    all_coarse = np.concatenate([train_coarse, test_coarse], axis=0)
    del train_images, test_images, train_fine, test_fine, train_coarse, test_coarse
    print(f"\nPooled dataset: {len(all_fine):,} samples, shape {all_images.shape}")

    # -- 2. Stratified subsample --
    if n_samples >= len(all_fine):
        print(f"n_samples={n_samples:,} >= total={len(all_fine):,}, using all.")
        keep_idx = np.arange(len(all_fine))
    else:
        print(f"Stratified subsample: {len(all_fine):,} -> {n_samples:,}")
        keep_idx, _ = train_test_split(
            np.arange(len(all_fine)),
            train_size=n_samples,
            random_state=42,
            stratify=all_fine,
        )
        keep_idx.sort()

    all_images = all_images[keep_idx]
    all_fine = all_fine[keep_idx]
    all_coarse = all_coarse[keep_idx]
    print(f"Subsampled: {len(all_fine):,} samples")

    # Print superclass distribution summary
    print(f"\nSuperclass distribution ({len(np.unique(all_coarse))} superclasses):")
    coarse_vals, coarse_counts = np.unique(all_coarse, return_counts=True)
    for cv, cc in zip(coarse_vals, coarse_counts):
        pct = cc / len(all_fine) * 100
        print(f"  Superclass {cv:>2} ({COARSE_NAMES[cv]:<35}): {cc:>4,} ({pct:5.1f}%)")

    # Print fine class distribution summary (compact)
    fine_vals, fine_counts = np.unique(all_fine, return_counts=True)
    print(f"\nFine class distribution ({len(fine_vals)} classes):")
    print(f"  Min count: {fine_counts.min():,} (class {fine_vals[fine_counts.argmin()]} "
          f"= {FINE_NAMES[fine_vals[fine_counts.argmin()]]})")
    print(f"  Max count: {fine_counts.max():,} (class {fine_vals[fine_counts.argmax()]} "
          f"= {FINE_NAMES[fine_vals[fine_counts.argmax()]]})")
    print(f"  Mean count: {fine_counts.mean():.1f}, Std: {fine_counts.std():.1f}")

    # -- 3. Validate --
    validate_data(all_images, all_fine, n_classes=100)

    # -- 4. Resize if target != 32 --
    if target_size != 32:
        print(f"\nResizing {len(all_images):,} images from 32x32 to "
              f"{target_size}x{target_size}...")
        all_images = resize_images_chunked(all_images, target_size)
        print(f"Resized shape: {all_images.shape}")

    # -- 5. Stratified 80/20 train/test split --
    train_idx, test_idx = train_test_split(
        np.arange(len(all_fine)),
        test_size=0.2,
        random_state=42,
        stratify=all_fine,
    )

    tr_images_raw = all_images[train_idx]
    tr_fine = all_fine[train_idx]
    tr_coarse = all_coarse[train_idx]
    te_images_raw = all_images[test_idx]
    te_fine = all_fine[test_idx]
    te_coarse = all_coarse[test_idx]
    del all_images, all_fine, all_coarse

    print(f"\nStratified 80/20 split:")
    print(f"  Train: {len(tr_fine):,} samples")
    print(f"  Test:  {len(te_fine):,} samples")

    # -- 6. Convert to channels-first float32 --
    # (N, H, W, 3) -> (N, 3, H, W) float32 [0, 1]
    train_chw = np.transpose(tr_images_raw, (0, 3, 1, 2)).astype(np.float32) / 255.0
    test_chw = np.transpose(te_images_raw, (0, 3, 1, 2)).astype(np.float32) / 255.0
    del tr_images_raw, te_images_raw

    # -- 7. Save npy arrays to slice_1 --
    os.makedirs(SLICE_DIR, exist_ok=True)

    np.save(os.path.join(SLICE_DIR, 'train_images.npy'), train_chw)
    np.save(os.path.join(SLICE_DIR, 'train_labels.npy'), tr_fine)
    np.save(os.path.join(SLICE_DIR, 'test_images.npy'), test_chw)
    np.save(os.path.join(SLICE_DIR, 'test_labels.npy'), te_fine)

    # -- 8. Save metadata CSVs --
    for split, fine, coarse in [('train', tr_fine, tr_coarse),
                                ('test', te_fine, te_coarse)]:
        meta = pd.DataFrame({
            'label': fine,
            'class_name': [FINE_NAMES[f] for f in fine],
            'coarse_label': coarse,
        })
        meta.to_csv(os.path.join(SLICE_DIR, f'{split}_meta.csv'), index=False)

    # -- 9. Print storage summary --
    print(f"\nSaved to {SLICE_DIR}/")
    print(f"  train_images.npy: {train_chw.shape}  "
          f"({train_chw.nbytes / 1024**2:.0f} MB)")
    print(f"  test_images.npy:  {test_chw.shape}  "
          f"({test_chw.nbytes / 1024**2:.0f} MB)")
    print(f"  dtype={train_chw.dtype}, range=[{train_chw.min():.2f}, "
          f"{train_chw.max():.2f}]")

    # -- 10. Print superclass mapping --
    print(f"\n{'=' * 70}")
    print("Superclass -> Fine class mapping:")
    print(f"{'=' * 70}")
    # Build coarse->fine mapping from the full dataset
    coarse_to_fine = {}
    all_fine_reload = np.concatenate([tr_fine, te_fine])
    all_coarse_reload = np.concatenate([tr_coarse, te_coarse])
    for f, c in zip(all_fine_reload, all_coarse_reload):
        coarse_to_fine.setdefault(int(c), set()).add(int(f))
    for c_idx in sorted(coarse_to_fine.keys()):
        fine_classes = sorted(coarse_to_fine[c_idx])
        fine_str = ', '.join(f"{fc}={FINE_NAMES[fc]}" for fc in fine_classes)
        print(f"  {c_idx:>2} ({COARSE_NAMES[c_idx]:<35}): {fine_str}")

    # Group stats
    print(f"\nGroup (coarse_label) stats:")
    for split, coarse in [('Train', tr_coarse), ('Test', te_coarse)]:
        vals, counts = np.unique(coarse, return_counts=True)
        print(f"  {split}: {len(vals)} groups, "
              f"min={counts.min()}, max={counts.max()}, "
              f"mean={counts.mean():.1f}")


# -- Analysis --------------------------------------------------------------

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
    print("CIFAR-100 -- Dataset Summary")
    print("=" * 70)

    for split in ['train', 'test']:
        labels = np.load(os.path.join(SLICE_DIR, f'{split}_labels.npy'))
        images = np.load(os.path.join(SLICE_DIR, f'{split}_images.npy'))
        meta = pd.read_csv(os.path.join(SLICE_DIR, f'{split}_meta.csv'))

        print(f"\n{'-' * 50}")
        print(f"{split.upper()} ({len(labels):,} samples)")
        print(f"{'-' * 50}")
        print(f"Images: {images.shape} dtype={images.dtype} "
              f"range=[{images.min():.2f}, {images.max():.2f}]")

        _, counts = np.unique(labels, return_counts=True)
        eq = shannon_equitability(labels)
        print(f"  Fine classes: {len(counts)}, Shannon equitability: {eq:.4f}")
        print(f"  Counts: min={counts.min()}, max={counts.max()}, "
              f"mean={counts.mean():.1f}")

        coarse = meta['coarse_label'].values
        coarse_vals, coarse_counts = np.unique(coarse, return_counts=True)
        print(f"  Superclass groups: {len(coarse_vals)}, "
              f"min={coarse_counts.min()}, max={coarse_counts.max()}")

    # Overall
    all_labels = np.concatenate([
        np.load(os.path.join(SLICE_DIR, f'{s}_labels.npy'))
        for s in ['train', 'test']
    ])
    print(f"\n{'-' * 50}")
    print(f"TOTAL ({len(all_labels):,} samples)")
    print(f"{'-' * 50}")
    _, counts = np.unique(all_labels, return_counts=True)
    eq = shannon_equitability(all_labels)
    print(f"  Fine classes: {len(counts)}, Shannon equitability: {eq:.4f}")
    print(f"  Counts: min={counts.min()}, max={counts.max()}")

    # Suggest constrained classes (smallest fine classes)
    print(f"\n{'-' * 50}")
    print("Constrained class candidates (minority fine classes):")
    print(f"{'-' * 50}")
    sorted_classes = np.argsort(counts)
    for c in sorted_classes[:5]:
        pct = counts[c] / len(all_labels) * 100
        print(f"  Class {c} ({FINE_NAMES[c]}): {counts[c]:,} ({pct:.1f}%)")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(
        description='Download and prepare CIFAR-100 dataset')
    parser.add_argument('--n_samples', type=int, default=DEFAULT_N_SAMPLES,
                        help=f'Number of samples to subsample '
                             f'(default: {DEFAULT_N_SAMPLES:,})')
    parser.add_argument('--size', type=int, default=DEFAULT_TARGET_SIZE,
                        help=f'Image size (default: {DEFAULT_TARGET_SIZE})')
    args = parser.parse_args()

    process_and_save(n_samples=args.n_samples, target_size=args.size)
    print_summary()
