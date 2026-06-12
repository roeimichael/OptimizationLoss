"""Generate 5 corrupted DermMNIST variants for the derm-cripple experiment.

Saves as uint8 (0-255) to fit disk; data_loader.py auto-converts to
float32 [0,1] on load. Each variant lives in
data/dermmnist_<name>/slice_1/ alongside copied labels/meta.

Corruptions:
  noise:   Gaussian sensor noise (sigma=0.15)
  blur:    Diagonal motion blur (kernel 11px, 45 deg)
  jpeg:    JPEG quality=15 - compression artifacts
  color:   HSV jitter (hue=0.15, sat=0.4, bri=0.25)
  defocus: Gaussian (out-of-focus) blur sigma=2.5

Run once on the server (~5min total).
"""
import io
import os
import shutil

import numpy as np
from PIL import Image
from scipy.ndimage import convolve, gaussian_filter

SRC = "data/dermmnist/slice_1"
RNG = np.random.default_rng(42)


def _to_uint8(X_float01):
    return (np.clip(X_float01, 0, 1) * 255).astype(np.uint8)


def gauss_noise(X, sigma=0.15):
    out = X + RNG.normal(0, sigma, X.shape).astype(X.dtype)
    return _to_uint8(out)


def motion_blur(X, ksize=11, angle_deg=45):
    k = np.zeros((ksize, ksize), dtype=np.float32)
    cx = cy = ksize // 2
    th = np.deg2rad(angle_deg)
    for i in range(-cx, cx + 1):
        x = int(round(cx + i * np.cos(th)))
        y = int(round(cy + i * np.sin(th)))
        if 0 <= x < ksize and 0 <= y < ksize:
            k[y, x] = 1.0
    k /= k.sum()
    out = np.zeros_like(X)
    for n in range(X.shape[0]):
        for c in range(3):
            out[n, c] = convolve(X[n, c], k, mode="reflect")
    return _to_uint8(out)


def jpeg_compress(X, quality=15):
    out = np.zeros_like(X, dtype=np.uint8)
    for n in range(X.shape[0]):
        img_u8 = (np.transpose(X[n], (1, 2, 0)) * 255).astype(np.uint8)
        buf = io.BytesIO()
        Image.fromarray(img_u8).save(buf, format="JPEG", quality=quality)
        buf.seek(0)
        comp = np.array(Image.open(buf), dtype=np.uint8)
        out[n] = np.transpose(comp, (2, 0, 1))
    return out


def color_jitter(X, hue=0.15, sat=0.4, bri=0.25):
    out = np.zeros_like(X, dtype=np.uint8)
    for n in range(X.shape[0]):
        img_u8 = (np.transpose(X[n], (1, 2, 0)) * 255).astype(np.uint8)
        im = Image.fromarray(img_u8).convert("HSV")
        h, s, v = im.split()
        h = np.array(h, dtype=np.float32)
        s = np.array(s, dtype=np.float32)
        v = np.array(v, dtype=np.float32)
        h = (h + RNG.uniform(-hue, hue) * 180) % 256
        s = np.clip(s * (1 + RNG.uniform(-sat, sat)), 0, 255)
        v = np.clip(v * (1 + RNG.uniform(-bri, bri)), 0, 255)
        merged = Image.merge("HSV", (
            Image.fromarray(h.astype(np.uint8)),
            Image.fromarray(s.astype(np.uint8)),
            Image.fromarray(v.astype(np.uint8)),
        )).convert("RGB")
        arr = np.array(merged, dtype=np.uint8)
        out[n] = np.transpose(arr, (2, 0, 1))
    return out


def defocus_blur(X, sigma=2.5):
    out = np.zeros_like(X, dtype=np.float32)
    for n in range(X.shape[0]):
        for c in range(3):
            out[n, c] = gaussian_filter(X[n, c], sigma=sigma)
    return _to_uint8(out)


def save_variant(name, Xtn, Xsn):
    dst = f"data/dermmnist_{name}/slice_1"
    os.makedirs(dst, exist_ok=True)
    np.save(f"{dst}/train_images.npy", Xtn)
    np.save(f"{dst}/test_images.npy", Xsn)
    for f in ("train_labels.npy", "test_labels.npy",
              "train_meta.csv", "test_meta.csv"):
        shutil.copy(f"{SRC}/{f}", f"{dst}/{f}")
    sz_mb = (Xtn.nbytes + Xsn.nbytes) / 1e6
    print(f"  saved {dst}  size={sz_mb:.0f}MB  dtype={Xtn.dtype}")


def main():
    Xt = np.load(f"{SRC}/train_images.npy")
    Xs = np.load(f"{SRC}/test_images.npy")
    print(f"derm train: {Xt.shape} {Xt.dtype} range={Xt.min():.3f}-{Xt.max():.3f}")
    print(f"derm test:  {Xs.shape} {Xs.dtype}")

    for name, fn in [
        ("noise",   gauss_noise),
        ("blur",    motion_blur),
        ("jpeg",    jpeg_compress),
        ("color",   color_jitter),
        ("defocus", defocus_blur),
    ]:
        print(f"\n{name}...")
        save_variant(name, fn(Xt), fn(Xs))
    print("\ndone")


if __name__ == "__main__":
    main()
