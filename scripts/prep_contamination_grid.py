"""Generate uniform contamination grid for 3 datasets x 3 sigmas.

Adds Gaussian sensor noise to BOTH train and test. uint8 storage.
Naming: data/{dataset}_sigma{NN}/slice_1/

Skip if already present (idempotent).
"""
import io
import os
import shutil

import numpy as np

DATASETS = ["tissuemnist", "dermmnist", "aider"]
SIGMAS = [0.10, 0.20, 0.30]
RNG = np.random.default_rng(42)


def _to_uint8(X_float01):
    return (np.clip(X_float01, 0, 1) * 255).astype(np.uint8)


def gauss_noise(X, sigma):
    return _to_uint8(X + RNG.normal(0, sigma, X.shape).astype(X.dtype))


def save_variant(dst, Xtn, Xsn, src):
    os.makedirs(dst, exist_ok=True)
    np.save(f"{dst}/train_images.npy", Xtn)
    np.save(f"{dst}/test_images.npy", Xsn)
    for f in ("train_labels.npy", "test_labels.npy",
              "train_meta.csv", "test_meta.csv"):
        shutil.copy(f"{src}/{f}", f"{dst}/{f}")
    print(f"  saved {dst}  size={(Xtn.nbytes+Xsn.nbytes)/1e6:.0f}MB")


def main():
    for ds in DATASETS:
        src = f"data/{ds}/slice_1"
        if not os.path.isdir(src):
            print(f"!! missing {src}, skipping")
            continue
        Xt = np.load(f"{src}/train_images.npy")
        Xs = np.load(f"{src}/test_images.npy")
        print(f"\n{ds}: train {Xt.shape}  test {Xs.shape}")
        for sigma in SIGMAS:
            tag = f"sigma{int(sigma*100):02d}"
            dst = f"data/{ds}_{tag}/slice_1"
            if os.path.isfile(f"{dst}/train_images.npy"):
                print(f"  exists {dst}, skipping")
                continue
            save_variant(dst, gauss_noise(Xt, sigma), gauss_noise(Xs, sigma), src)
    print("\ndone")


if __name__ == "__main__":
    main()
