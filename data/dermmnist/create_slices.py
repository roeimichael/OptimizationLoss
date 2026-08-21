"""Create group-aware train/test slices of DermMNIST. LESION-DISJOINT, by construction.

WHAT WAS WRONG, and why every absolute dermmnist number before 2026-08-21 is void.

HAM10000 -- which DermaMNIST derives from -- photographs many lesions more than
once. Measured here: 10,015 images over 7,470 unique lesions, 26.2% of lesions
carrying more than one image and some carrying six. The previous version of this
script pooled DermaMNIST-C's train/val/test back together and re-split on the
LABEL alone (`StratifiedShuffleSplit`), which puts two photographs of the SAME
lesion on opposite sides of the split. Measured consequence: **38.7% of the test
set, and 67.3% of the MELANOMA test set, shared a lesion with a training image.**
The model had already seen the lesion it was being tested on.

DermaMNIST-C ships leakage-free splits for exactly this reason. Pooling them away
threw out the one thing the corrected release existed to provide.

THE FIX. `StratifiedGroupKFold` over `lesion_id`: every image of a lesion lands
on the same side, while the label distribution is kept as close as the grouping
allows. The five slices are the five FOLDS of one split, so their test sets are
disjoint by construction rather than by luck. Three asserts fail the build rather
than write a bad slice: no shared lesion, no shared image, and meta rows aligned
to labels.

!! WHAT THIS COSTS. Regenerating the slices invalidates every stored dermmnist
result -- the models were trained on the leaked split. That is the point: the
alternative is continuing to measure on data we know is wrong.

!! WHAT IT DOES NOT FIX. A stratified split still forces test prevalence to match
train prevalence, so the transductive budget K stays inferable from the TRAINING
distribution and carries almost no information (see docs/FRAMEWORK.md section 4).
That is a separate axis, and `--shift` is the separate knob: it subsamples the
TEST split only, so the capped class appears at a different rate than training
predicts. Kept as its own output directory (`shift_N/`) precisely so a result on
shifted data cannot be confused with a result on fixed data.

    python data/dermmnist/download_data.py    # first: .npz + metadata
    python data/dermmnist/create_slices.py    # then: slice_1..5  (leakage fix)
    python data/dermmnist/create_slices.py --shift --shift-class 4 --shift-factor 0.5
                                              # and: shift_1..5  (prevalence shift)
"""

import argparse
import os

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
NUM_SLICES = 5
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
    """Load all DermMNIST data from .npz + metadata CSV, pool all splits.

    Pooling is still correct here: the split below is group-aware, so it can
    reconstruct a lesion-disjoint partition from the pool. What was wrong was
    pooling and then splitting on the LABEL.
    """
    npz_path = os.path.join(DATA_DIR, 'dermamnist_corrected_224.npz')
    meta_path = os.path.join(DATA_DIR, 'dermmnist_c_metadata.csv')

    if not os.path.exists(npz_path):
        raise FileNotFoundError(
            f"{npz_path} not found. Run 'python data/dermmnist/download_data.py' first.")

    data = np.load(npz_path)
    meta_df = pd.read_csv(meta_path)
    meta_df['label'] = meta_df['dx'].map(DX_TO_CLASS)
    for col in ('lesion_id', 'image_id'):
        if col not in meta_df.columns:
            raise ValueError(
                "%s has no `%s` column. The split is grouped by lesion; without "
                "it this script cannot guarantee a lesion-disjoint partition, "
                "and silently falling back to a label-only split is the exact "
                "bug this file exists to fix." % (meta_path, col))

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
            # Carried into every slice so the leakage check is reproducible from
            # the slice alone, without re-joining the source metadata.
            'lesion_id': split_meta['lesion_id'].values,
            'image_id': split_meta['image_id'].values,
        })

        all_images_list.append(images_chw)
        all_labels_list.append(labels)
        all_meta_list.append(meta_out)

    all_images = np.concatenate(all_images_list, axis=0)
    all_labels = np.concatenate(all_labels_list, axis=0)
    all_meta = pd.concat(all_meta_list, ignore_index=True)

    n_les = all_meta['lesion_id'].nunique()
    print(f"Pooled data: {len(all_labels)} images over {n_les} unique lesions")
    print(f"Class distribution:")
    for c in range(7):
        n = (all_labels == c).sum()
        print(f"  {c} ({CLASS_NAMES[c]:>5}): {n:>5} ({n/len(all_labels)*100:.1f}%)")

    return all_images, all_labels, all_meta


def shift_test(test_idx, all_labels, cls, factor, rng):
    """Drop a fraction of ONE class from the TEST split, to move its prevalence.

    The transductive budget K is a fraction of the capped class's true count in
    the test set. When the split is stratified, that count is recoverable from
    the TRAINING prevalence to within about one item, so K tells the model
    something it could already compute and the constraint has no information to
    add. Removing part of the class from the test split -- and only from the
    test split -- breaks that correspondence, which is the one setup in which
    the budget is genuinely news.

    Drops whole IMAGES, from the test side only. It cannot introduce leakage:
    the surviving items were already test items, and no training item moves.
    """
    keep = np.ones(len(test_idx), dtype=bool)
    is_c = all_labels[test_idx] == cls
    n_drop = int(round(is_c.sum() * (1.0 - factor)))
    if n_drop > 0:
        where = np.flatnonzero(is_c)
        keep[rng.choice(where, size=n_drop, replace=False)] = False
    return test_idx[keep]


def create_slices(all_images, all_labels, all_meta, shift=False,
                  shift_class=4, shift_factor=0.5, folds=None):
    """Lesion-disjoint slices, as the folds of one grouped split.

    `folds` selects which folds to materialise. The partition is fixed by
    BASE_SEED, so writing fold 1 today and fold 3 next month yields exactly the
    slices that a single full run would have produced -- each slice is ~6GB, and
    the project uses one at a time.
    """
    groups = all_meta['lesion_id'].to_numpy()
    prefix = 'shift' if shift else 'slice'
    sgkf = StratifiedGroupKFold(n_splits=NUM_SLICES, shuffle=True,
                                random_state=BASE_SEED)
    rng = np.random.default_rng(BASE_SEED)

    for slice_idx, (train_idx, test_idx) in enumerate(
            sgkf.split(all_images, all_labels, groups=groups), start=1):
        if folds and slice_idx not in folds:
            continue
        if shift:
            test_idx = shift_test(test_idx, all_labels, shift_class,
                                  shift_factor, rng)
        slice_dir = os.path.join(DATA_DIR, f'{prefix}_{slice_idx}')
        os.makedirs(slice_dir, exist_ok=True)

        train_labels = all_labels[train_idx]
        test_labels = all_labels[test_idx]
        train_meta = all_meta.iloc[train_idx].reset_index(drop=True)
        test_meta = all_meta.iloc[test_idx].reset_index(drop=True)

        # THE CHECK THIS FILE EXISTS FOR. Not a warning -- a slice that fails it
        # must not reach disk, because nothing downstream can detect it.
        shared = set(train_meta['lesion_id']) & set(test_meta['lesion_id'])
        if shared:
            raise AssertionError(
                "slice %d: %d lesion(s) appear in BOTH train and test (e.g. %s). "
                "The split is supposed to be grouped by lesion."
                % (slice_idx, len(shared), sorted(shared)[:3]))
        assert not (set(train_meta['image_id']) & set(test_meta['image_id'])), \
            f"slice {slice_idx}: an image_id is in both splits"
        assert np.array_equal(train_meta['label'].to_numpy(), train_labels)
        assert np.array_equal(test_meta['label'].to_numpy(), test_labels)

        np.save(os.path.join(slice_dir, 'train_images.npy'), all_images[train_idx])
        np.save(os.path.join(slice_dir, 'train_labels.npy'), train_labels)
        np.save(os.path.join(slice_dir, 'test_images.npy'), all_images[test_idx])
        np.save(os.path.join(slice_dir, 'test_labels.npy'), test_labels)
        train_meta.to_csv(os.path.join(slice_dir, 'train_meta.csv'), index=False)
        test_meta.to_csv(os.path.join(slice_dir, 'test_meta.csv'), index=False)

        print(f"\n--- {prefix} {slice_idx} ---")
        print(f"  Train: {len(train_labels)} images / "
              f"{train_meta['lesion_id'].nunique()} lesions   "
              f"Test: {len(test_labels)} images / "
              f"{test_meta['lesion_id'].nunique()} lesions   LESION-DISJOINT")
        for c in range(7):
            train_pct = (train_labels == c).sum() / len(train_labels) * 100
            test_pct = (test_labels == c).sum() / len(test_labels) * 100
            diff = test_pct - train_pct
            # >1pt drift is worth seeing: grouping constrains stratification, so
            # unlike the old label-only split these will NOT agree to 0.02pt.
            marker = " *" if abs(diff) > 1.0 else ""
            print(f"  {CLASS_NAMES[c]:>5}: train {train_pct:5.2f}%  "
                  f"test {test_pct:5.2f}%  ({diff:+.2f}){marker}")

    print(f"\nCreated {NUM_SLICES} slices in {DATA_DIR}/{prefix}_1..{NUM_SLICES}/")


def verify_independence(prefix='slice'):
    """Test sets must not overlap across slices, and none may share a lesion."""
    les = []
    for i in range(1, NUM_SLICES + 1):
        d = os.path.join(DATA_DIR, f'{prefix}_{i}')
        if not os.path.exists(d):
            continue
        les.append(set(pd.read_csv(os.path.join(d, 'test_meta.csv'))['lesion_id']))
    if len(les) < 2:
        return
    print(f"\nSlice independence (shared TEST lesions between slices):")
    for i in range(NUM_SLICES):
        for j in range(i + 1, NUM_SLICES):
            n = len(les[i] & les[j])
            print(f"  {prefix} {i+1} vs {j+1}: {n} shared lesion(s)"
                  + ("" if n == 0 else "   <-- NOT DISJOINT"))


if __name__ == '__main__':
    a = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    a.add_argument('--shift', action='store_true',
                   help='also write shift_N/: test split subsampled so the '
                        'capped class appears at a different rate than training '
                        'predicts, which is the only setup where the budget K '
                        'carries information')
    a.add_argument('--shift-class', type=int, default=4, help='default 4 = MEL')
    a.add_argument('--shift-factor', type=float, default=0.5,
                   help='keep this fraction of the class in TEST (default 0.5)')
    a.add_argument('--folds', type=int, nargs='+',
                   help='which folds to write (default all %d). Each is ~6GB '
                        'and the project uses one at a time; the partition is '
                        'fixed by BASE_SEED, so folds written separately match '
                        'a single full run exactly.' % NUM_SLICES)
    args = a.parse_args()

    imgs, labels, meta = load_all_data()
    create_slices(imgs, labels, meta, folds=args.folds)
    verify_independence('slice')
    if args.shift:
        create_slices(imgs, labels, meta, shift=True, folds=args.folds,
                      shift_class=args.shift_class, shift_factor=args.shift_factor)
        verify_independence('shift')
