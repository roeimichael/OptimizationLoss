"""Prepare RetinaMNIST into the slice format used by the pipeline loader.

RetinaMNIST: 1,600 fundus images, 28x28 RGB, 5-class DR severity grading
(ordinal task, but we treat it as multi-class).
Train 1080 / val 120 / test 400 in MedMNIST. We use full train (< 15k cap)
and full test (< 3k cap).
Constrained class: chosen at runtime as the second-most-frequent test class
(typical DR distribution: 0 dominates, 2 is the "moderate NPDR" middle bucket;
constraining 2 forces the network not to over-call moderate disease).
Group = synthetic balanced 2-split (no real metadata in MedMNIST).
"""
import os
import numpy as np
from PIL import Image

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
NPZ = os.path.join(DATA_DIR, 'retinamnist.npz')

TARGET = 224
SEED = 43

CLASS_NAMES = {0: 'DR0', 1: 'DR1', 2: 'DR2', 3: 'DR3', 4: 'DR4'}
N_CLASSES = 5


def upscale(images_28):
    """images_28: (N, 28, 28, 3) uint8 RGB -> (N, 3, 224, 224) float32 [0,1]."""
    n = len(images_28)
    arr = np.empty((n, 3, TARGET, TARGET), dtype=np.float32)
    for i in range(n):
        img = Image.fromarray(images_28[i]).resize(
            (TARGET, TARGET), Image.LANCZOS)
        arr[i] = np.asarray(img, dtype=np.float32).transpose(2, 0, 1) / 255.0
    return arr


def main():
    d = np.load(NPZ)
    rng = np.random.RandomState(SEED)
    out = os.path.join(DATA_DIR, 'slice_1')
    os.makedirs(out, exist_ok=True)

    import pandas as pd
    for name in ('train', 'test'):
        X = d[f'{name}_images']
        y = d[f'{name}_labels'].reshape(-1).astype(np.int64)
        print(f"\n{name}: {len(y)} images -> upscaling 28->{TARGET}")
        for c in range(N_CLASSES):
            cnt = int((y == c).sum())
            print(f"  {c} ({CLASS_NAMES[c]:>4}): {cnt}")
        images = upscale(X)
        groups = rng.randint(0, 2, size=len(y)).astype(np.int64)
        np.save(os.path.join(out, f'{name}_images.npy'), images)
        np.save(os.path.join(out, f'{name}_labels.npy'), y)
        pd.DataFrame({'label': y,
                      'class_name': [CLASS_NAMES[int(v)] for v in y],
                      'synth_group': groups}).to_csv(
            os.path.join(out, f'{name}_meta.csv'), index=False)
        print(f"  saved {name}_images {images.shape} {images.dtype} "
              f"range[{images.min():.2f},{images.max():.2f}]")

    print(f"\nDone -> {out}/  (group col 'synth_group')")


if __name__ == '__main__':
    main()
