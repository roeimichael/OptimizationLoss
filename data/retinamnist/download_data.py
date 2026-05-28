"""Download RetinaMNIST (28x28 RGB) via the medmnist package and stage as npz.

After this finishes, run prepare_retinamnist.py to build slice_1/.
"""
import os
import shutil

from medmnist import INFO, RetinaMNIST

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
NPZ_LOCAL = os.path.join(DATA_DIR, 'retinamnist.npz')


def main():
    if os.path.exists(NPZ_LOCAL):
        print(f"Already present: {NPZ_LOCAL}")
        return
    print(f"Downloading RetinaMNIST 28x28 npz into {DATA_DIR} ...")
    _ = RetinaMNIST(split='train', download=True, root=DATA_DIR)
    src = os.path.join(DATA_DIR, INFO['retinamnist']['python_class'].lower() + '.npz')
    if os.path.exists(src) and src != NPZ_LOCAL:
        shutil.move(src, NPZ_LOCAL)
    print(f"OK: {NPZ_LOCAL} ({os.path.getsize(NPZ_LOCAL)/1e6:.1f} MB)")


if __name__ == '__main__':
    main()
