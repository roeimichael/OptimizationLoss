import numpy as np, pandas as pd
d="data/octmnist/slice_1"
Xtr=np.load(f"{d}/train_images.npy"); ytr=np.load(f"{d}/train_labels.npy")
Xte=np.load(f"{d}/test_images.npy"); yte=np.load(f"{d}/test_labels.npy")
print("train images", Xtr.shape, Xtr.dtype, "labels", ytr.shape)
print("test  images", Xte.shape, "labels", yte.shape)
print("train class counts", np.bincount(ytr).tolist())
print("test  class counts", np.bincount(yte).tolist())
m=pd.read_csv(f"{d}/test_meta.csv")
print("test_meta cols:", list(m.columns))
print("synth_group vals:", sorted(m.synth_group.unique().tolist()), "counts", np.bincount(m.synth_group.values).tolist())
