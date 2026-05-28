"""Download BloodMNIST (28x28 RGB) via the medmnist package and stage as npz.

After this finishes, run prepare_bloodmnist.py to build slice_1/.
"""
import os
import shutil

from medmnist import INFO, BloodMNIST

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
NPZ_LOCAL = os.path.join(DATA_DIR, 'bloodmnist.npz')


def main():
    if os.path.exists(NPZ_LOCAL):
        print(f"Already present: {NPZ_LOCAL}")
        return
    # Force download into DATA_DIR.
    print(f"Downloading BloodMNIST 28x28 npz into {DATA_DIR} ...")
    _ = BloodMNIST(split='train', download=True, root=DATA_DIR)
    # medmnist saves the npz under root/{flag}.npz
    src = os.path.join(DATA_DIR, INFO['bloodmnist']['python_class'].lower() + '.npz')
    if os.path.exists(src) and src != NPZ_LOCAL:
        shutil.move(src, NPZ_LOCAL)
    print(f"OK: {NPZ_LOCAL} ({os.path.getsize(NPZ_LOCAL)/1e6:.1f} MB)")


if __name__ == '__main__':
    main()
