"""Prepare ISIC 2019 into the slice format used by the pipeline loader.

Mirrors data/dermmnist/{download_data,create_slices}.py:
  - images saved as (N, 3, 224, 224) float32 in [0,1], channels-first
  - labels saved as (N,) int64
  - meta.csv columns: label, class_name, site_group

ISIC 2019: 25,331 dermoscopy images, 8 diagnostic classes.
Constrained class = MEL (melanoma) -> biopsy/specialist-review capacity story.
Group = anatomical site (torso/extremity/head_neck/other) -> per-region local caps.

Run on the server in data/isic2019/ after the 3 source files are downloaded:
    python prepare_isic.py
"""
import os
import zipfile
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import StratifiedShuffleSplit

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
ZIP = os.path.join(DATA_DIR, 'ISIC_2019_Training_Input.zip')
GT = os.path.join(DATA_DIR, 'ISIC_2019_Training_GroundTruth.csv')
META = os.path.join(DATA_DIR, 'ISIC_2019_Training_Metadata.csv')
IMG_DIR = os.path.join(DATA_DIR, 'ISIC_2019_Training_Input')

TARGET = 224
TEST_FRACTION = 0.2
SEED = 43  # slice_1 (derm uses BASE_SEED+1=43 for slice_1)

# Class order; MEL index 0 = constrained class
CLASS_COLS = ['MEL', 'NV', 'BCC', 'AK', 'BKL', 'DF', 'VASC', 'SCC']
CLASS_NAMES = {i: c for i, c in enumerate(CLASS_COLS)}

# anatom_site_general -> coarse region group
SITE_MAP = {
    'anterior torso': 'torso', 'posterior torso': 'torso', 'lateral torso': 'torso',
    'upper extremity': 'extremity', 'lower extremity': 'extremity',
    'head/neck': 'head_neck',
    'palms/soles': 'other', 'oral/genital': 'other',
}
SITE_ENCODE = {'torso': 0, 'extremity': 1, 'head_neck': 2, 'other': 3}


def find_image_dir():
    """Locate the dir containing ISIC_*.jpg (zip may nest one level)."""
    for root, _dirs, files in os.walk(DATA_DIR):
        if any(f.startswith('ISIC_') and f.lower().endswith('.jpg') for f in files):
            return root
    raise FileNotFoundError("No ISIC_*.jpg found; did the unzip succeed?")


def unzip_if_needed():
    has_jpg = os.path.isdir(IMG_DIR) and any(
        f.lower().endswith('.jpg') for f in os.listdir(IMG_DIR))
    if has_jpg:
        print(f"Images already extracted: {IMG_DIR}")
        return
    print(f"Unzipping {ZIP} ...")
    with zipfile.ZipFile(ZIP) as z:
        z.extractall(DATA_DIR)
    print("Unzip done.")


def build_table():
    gt = pd.read_csv(GT)
    meta = pd.read_csv(META)
    # label = argmax over the 8 one-hot class columns
    gt['label'] = gt[CLASS_COLS].values.argmax(axis=1).astype(np.int64)
    df = gt[['image', 'label']].merge(meta[['image', 'anatom_site_general']],
                                      on='image', how='left')
    site = df['anatom_site_general'].str.lower().map(SITE_MAP).fillna('other')
    df['site_group'] = site.map(SITE_ENCODE).astype(np.int64)
    df['class_name'] = df['label'].map(CLASS_NAMES)
    return df.reset_index(drop=True)


def load_resize(image_id, img_dir):
    p = os.path.join(img_dir, image_id + '.jpg')
    img = Image.open(p).convert('RGB').resize((TARGET, TARGET), Image.LANCZOS)
    # (H, W, 3) uint8 -> (3, H, W) float32 [0,1]
    return np.asarray(img, dtype=np.float32).transpose(2, 0, 1) / 255.0


def fill_split(df_split, img_dir):
    n = len(df_split)
    arr = np.empty((n, 3, TARGET, TARGET), dtype=np.float32)
    for i, image_id in enumerate(df_split['image'].values):
        arr[i] = load_resize(image_id, img_dir)
        if (i + 1) % 2000 == 0:
            print(f"    {i + 1}/{n}")
    return arr


def main():
    for f in (ZIP, GT, META):
        if not os.path.exists(f):
            raise FileNotFoundError(f"missing source file: {f}")
    unzip_if_needed()
    img_dir = find_image_dir()
    df = build_table()
    print(f"Total: {len(df)} images")
    for c in range(len(CLASS_COLS)):
        n = (df['label'] == c).sum()
        print(f"  {c} ({CLASS_NAMES[c]:>4}): {n:>5} ({n / len(df) * 100:4.1f}%)")
    print("Site groups:", df['site_group'].value_counts().to_dict())

    sss = StratifiedShuffleSplit(n_splits=1, test_size=TEST_FRACTION, random_state=SEED)
    train_idx, test_idx = next(sss.split(df['image'].values, df['label'].values))
    out = os.path.join(DATA_DIR, 'slice_1')
    os.makedirs(out, exist_ok=True)

    for name, idx in (('train', train_idx), ('test', test_idx)):
        sub = df.iloc[idx].reset_index(drop=True)
        print(f"\n{name}: {len(sub)} images -> resizing")
        images = fill_split(sub, img_dir)
        labels = sub['label'].values.astype(np.int64)
        np.save(os.path.join(out, f'{name}_images.npy'), images)
        np.save(os.path.join(out, f'{name}_labels.npy'), labels)
        sub[['label', 'class_name', 'site_group']].to_csv(
            os.path.join(out, f'{name}_meta.csv'), index=False)
        print(f"  saved {name}_images {images.shape} {images.dtype} "
              f"range[{images.min():.2f},{images.max():.2f}]")

    print(f"\nDone -> {out}/  (constrained class MEL=0, group col 'site_group')")


if __name__ == '__main__':
    main()
