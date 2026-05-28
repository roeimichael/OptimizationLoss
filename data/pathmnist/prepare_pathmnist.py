"""Prepare PathMNIST into the slice format used by the pipeline loader.

Mirrors data/isic2019/prepare_isic.py:
  - images saved as (N, 3, 224, 224) float32 in [0,1], channels-first
  - labels saved as (N,) int64
  - meta.csv columns: label, class_name, synth_group

PathMNIST (MedMNIST v2): 107k colon-histology 28x28 RGB tiles, 9 tissue classes.
We subsample (stratified) train->15k / test->3k, upscale 28->224 (LANCZOS).
Constrained class = TUM (8, colorectal adenocarcinoma epithelium) -> the
"cap tumor-positive predictions to pathology-review capacity" story.
Group = synthetic balanced 2-split (no real site metadata in MedMNIST).

Run on the server in data/pathmnist/ after pathmnist.npz is downloaded:
    python prepare_pathmnist.py
"""
import os
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
NPZ = os.path.join(DATA_DIR, 'pathmnist.npz')

TARGET = 224
N_TRAIN = 15000
N_TEST = 3000
SEED = 43  # slice_1

CLASS_NAMES = {0: 'ADI', 1: 'BACK', 2: 'DEB', 3: 'LYM', 4: 'MUC',
               5: 'MUS', 6: 'NORM', 7: 'STR', 8: 'TUM'}
CONSTRAINED = 8  # TUM


def subsample(images, labels, n, seed):
    if n >= len(labels):
        return images, labels
    idx = np.arange(len(labels))
    keep, _ = train_test_split(idx, train_size=n, stratify=labels,
                               random_state=seed)
    return images[keep], labels[keep]


def upscale(images_28):
    n = len(images_28)
    arr = np.empty((n, 3, TARGET, TARGET), dtype=np.float32)
    for i in range(n):
        img = Image.fromarray(images_28[i]).convert('RGB').resize(
            (TARGET, TARGET), Image.LANCZOS)
        arr[i] = np.asarray(img, dtype=np.float32).transpose(2, 0, 1) / 255.0
        if (i + 1) % 3000 == 0:
            print(f"    {i + 1}/{n}")
    return arr


def main():
    d = np.load(NPZ)
    rng = np.random.RandomState(SEED)
    out = os.path.join(DATA_DIR, 'slice_1')
    os.makedirs(out, exist_ok=True)

    for name, n in (('train', N_TRAIN), ('test', N_TEST)):
        X = d[f'{name}_images']
        y = d[f'{name}_labels'].reshape(-1).astype(np.int64)
        X, y = subsample(X, y, n, SEED)
        print(f"\n{name}: {len(y)} images -> upscaling 28->{TARGET}")
        for c in range(9):
            cnt = int((y == c).sum())
            print(f"  {c} ({CLASS_NAMES[c]:>4}): {cnt}")
        images = upscale(X)
        groups = rng.randint(0, 2, size=len(y)).astype(np.int64)
        np.save(os.path.join(out, f'{name}_images.npy'), images)
        np.save(os.path.join(out, f'{name}_labels.npy'), y)
        import pandas as pd
        pd.DataFrame({'label': y,
                      'class_name': [CLASS_NAMES[int(v)] for v in y],
                      'synth_group': groups}).to_csv(
            os.path.join(out, f'{name}_meta.csv'), index=False)
        print(f"  saved {name}_images {images.shape} {images.dtype} "
              f"range[{images.min():.2f},{images.max():.2f}]")

    print(f"\nDone -> {out}/  (constrained class TUM={CONSTRAINED}, "
          f"group col 'synth_group')")


if __name__ == '__main__':
    main()
