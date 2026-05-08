"""Download So2Sat LCZ42 (validation split) and prepare arrays.

So2Sat LCZ42: Sentinel-1 + Sentinel-2 patches, 17 Local Climate Zones (LCZ).
Source: zhu-xlab/So2Sat-LCZ42 on Hugging Face (CC-BY-4.0).
Validation split = 10 western-half cities (~24K samples), real geo backstory.

Strategy:
  1. Download v4/validation.h5.gz (1.05 GB) + validation_geo.h5 from HF.
  2. Decompress, read SEN2 only (10 bands), extract RGB = bands [B4, B3, B2]
     = indices [2, 1, 0] in the dataset's B2..B12 ordering.
  3. Resize 32x32 -> 224x224 for ImageNet pretrained backbones.
  4. Argmax one-hot LCZ label -> int (17 classes).
  5. validation_geo.h5 stores explicit `city` bytes per sample (10 cities).
     Map to int city_id; the column is REAL geographic group, not synthetic.
  6. Stratified 80/20 train/test split jointly on (label, city_id).
  7. Save *.npy + *_meta.csv.

Class names follow Stewart-Oke 2012 LCZ scheme:
  0 LCZ1  Compact high-rise
  1 LCZ2  Compact mid-rise
  2 LCZ3  Compact low-rise
  3 LCZ4  Open high-rise
  4 LCZ5  Open mid-rise
  5 LCZ6  Open low-rise
  6 LCZ7  Lightweight low-rise
  7 LCZ8  Large low-rise
  8 LCZ9  Sparsely built
  9 LCZA  Heavy industry
 10 LCZB  Dense trees
 11 LCZC  Scattered trees
 12 LCZD  Bush/scrub
 13 LCZE  Low plants
 14 LCZF  Bare rock or paved
 15 LCZG  Bare soil or sand
 16 LCZH  Water

Usage:
    python download_data.py
"""

import gzip
import os
import shutil
import urllib.request

import h5py
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split

CLASS_NAMES = {
    0: 'LCZ1', 1: 'LCZ2', 2: 'LCZ3', 3: 'LCZ4', 4: 'LCZ5',
    5: 'LCZ6', 6: 'LCZ7', 7: 'LCZ8', 8: 'LCZ9', 9: 'LCZA',
    10: 'LCZB', 11: 'LCZC', 12: 'LCZD', 13: 'LCZE',
    14: 'LCZF', 15: 'LCZG', 16: 'LCZH',
}
CLASS_FULL_NAMES = {
    0: 'Compact high-rise', 1: 'Compact mid-rise', 2: 'Compact low-rise',
    3: 'Open high-rise', 4: 'Open mid-rise', 5: 'Open low-rise',
    6: 'Lightweight low-rise', 7: 'Large low-rise', 8: 'Sparsely built',
    9: 'Heavy industry', 10: 'Dense trees', 11: 'Scattered trees',
    12: 'Bush/scrub', 13: 'Low plants', 14: 'Bare rock or paved',
    15: 'Bare soil or sand', 16: 'Water',
}

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
H5_GZ = os.path.join(DATA_DIR, 'validation.h5.gz')
H5 = os.path.join(DATA_DIR, 'validation.h5')
GEO_H5 = os.path.join(DATA_DIR, 'validation_geo.h5')
URL_BASE = 'https://huggingface.co/datasets/zhu-xlab/So2Sat-LCZ42/resolve/main/v4'

TARGET_SIZE = 224
N_CITIES = 10


def _wget(url, dst):
    if os.path.exists(dst):
        print(f"exists {dst}")
        return
    print(f"download {url} -> {dst}")
    urllib.request.urlretrieve(url, dst)
    print(f"  ok {os.path.getsize(dst)/1024**2:.1f} MB")


def download_if_needed():
    _wget(f"{URL_BASE}/validation.h5.gz", H5_GZ)
    _wget(f"{URL_BASE}/validation_geo.h5", GEO_H5)
    if not os.path.exists(H5):
        print(f"decompress {H5_GZ}")
        with gzip.open(H5_GZ, 'rb') as fi, open(H5, 'wb') as fo:
            shutil.copyfileobj(fi, fo, length=64 * 1024 * 1024)
        print(f"  ok {os.path.getsize(H5)/1024**2:.1f} MB")


def load_raw():
    with h5py.File(H5, 'r') as f:
        print(f"keys={list(f.keys())}")
        sen2 = np.array(f['sen2'])               # (N, 32, 32, 10) float
        label_oh = np.array(f['label'])          # (N, 17) one-hot
    labels = label_oh.argmax(axis=1).astype(np.int64)
    with h5py.File(GEO_H5, 'r') as f:
        cities_bytes = np.array(f['city'])       # (N,) bytestrings
    cities = np.array([c.decode('utf-8').strip() for c in cities_bytes])
    print(f"sen2={sen2.shape} dtype={sen2.dtype} range=[{sen2.min():.3f},{sen2.max():.3f}]")
    print(f"labels={labels.shape} unique={np.unique(labels).tolist()}")
    print(f"cities={np.unique(cities).tolist()}")
    return sen2, labels, cities


def normalize_rgb(sen2):
    # Sentinel-2 band order in So2Sat: B2 B3 B4 B5 B6 B7 B8 B8a B11 B12.
    # RGB = (B4, B3, B2) -> indices (2, 1, 0).
    rgb = sen2[..., [2, 1, 0]]                   # (N, 32, 32, 3)
    # Sentinel-2 reflectance is float [~0, ~0.5]; clip + percentile-stretch to [0, 1].
    p2, p98 = np.percentile(rgb, [2, 98])
    print(f"rgb percentile 2/98: {p2:.4f} / {p98:.4f}")
    rgb = np.clip((rgb - p2) / (p98 - p2 + 1e-8), 0.0, 1.0).astype(np.float32)
    return rgb


def resize_chw(rgb_nhwc, target=TARGET_SIZE):
    n = len(rgb_nhwc)
    out = np.empty((n, 3, target, target), dtype=np.float32)
    for i in range(n):
        img = (rgb_nhwc[i] * 255.0).astype(np.uint8)        # (32,32,3) uint8
        pil = Image.fromarray(img, mode='RGB').resize(
            (target, target), Image.BILINEAR)
        arr = np.asarray(pil, dtype=np.float32) / 255.0     # (224,224,3)
        out[i] = arr.transpose(2, 0, 1)                     # CHW
        if (i + 1) % 2000 == 0:
            print(f"  resized {i+1:>6}/{n}", end='\r')
    print()
    return out


def encode_cities(cities):
    unique_cities = sorted(np.unique(cities).tolist())
    name_to_id = {name: i for i, name in enumerate(unique_cities)}
    cid = np.array([name_to_id[c] for c in cities], dtype=np.int64)
    print(f"city map ({len(unique_cities)} cities):")
    for name, i in name_to_id.items():
        n = int((cid == i).sum())
        print(f"  {i:>2} {name:>15}  n={n}")
    return cid, name_to_id


def stratified_split(labels, city_ids, test_size=0.2, seed=42):
    # Joint stratification (label, city) preserves both class and city ratios.
    n_cities = int(city_ids.max()) + 1
    strata = labels.astype(np.int64) * (n_cities + 1) + city_ids.astype(np.int64)
    # Some (class, city) cells may have <2 samples; fall back to label-only stratification.
    _, cnt = np.unique(strata, return_counts=True)
    if cnt.min() < 2:
        print(f"WARN: smallest (label,city) cell has {cnt.min()} samples; "
              f"falling back to label-only stratification")
        return train_test_split(
            np.arange(len(labels)), test_size=test_size,
            random_state=seed, stratify=labels)
    return train_test_split(
        np.arange(len(labels)),
        test_size=test_size,
        random_state=seed,
        stratify=strata,
    )


def save_split(images_chw, labels, city_ids, split):
    np.save(os.path.join(DATA_DIR, f'{split}_images.npy'), images_chw)
    np.save(os.path.join(DATA_DIR, f'{split}_labels.npy'), labels)
    meta = pd.DataFrame({
        'label': labels,
        'class_name': [CLASS_NAMES[int(l)] for l in labels],
        'city_id': city_ids,
    })
    meta.to_csv(os.path.join(DATA_DIR, f'{split}_meta.csv'), index=False)
    print(f"saved {split}: images={images_chw.shape} "
          f"labels={labels.shape} groups={len(np.unique(city_ids))}")


def main():
    download_if_needed()
    sen2, labels, cities = load_raw()
    rgb = normalize_rgb(sen2)
    del sen2
    images_chw = resize_chw(rgb)
    del rgb
    city_ids, name_map = encode_cities(cities)
    pd.DataFrame({'city_id': list(name_map.values()), 'city_name': list(name_map.keys())}
                 ).to_csv(os.path.join(DATA_DIR, 'city_map.csv'), index=False)
    train_idx, test_idx = stratified_split(labels, city_ids)
    save_split(images_chw[train_idx], labels[train_idx], city_ids[train_idx], 'train')
    save_split(images_chw[test_idx], labels[test_idx], city_ids[test_idx], 'test')
    # class distribution summary
    print("\nclass distribution (full):")
    _, counts = np.unique(labels, return_counts=True)
    for c in range(17):
        n = int(counts[c]) if c < len(counts) else 0
        pct = n / len(labels) * 100
        print(f"  {c} {CLASS_NAMES[c]:>4} ({CLASS_FULL_NAMES[c][:25]:>25}): "
              f"{n:>5} ({pct:5.1f}%)")
    minority = np.argsort(counts)[:3]
    print("\nconstrained-class candidates (smallest):")
    for c in minority:
        print(f"  {c} {CLASS_NAMES[c]} ({CLASS_FULL_NAMES[c]}): {counts[c]}")


if __name__ == '__main__':
    main()
