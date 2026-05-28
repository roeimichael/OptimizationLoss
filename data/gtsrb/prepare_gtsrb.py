"""Download GTSRB via torchvision, resize to 224, write slice_1 in our format.

Output (matches aider / dermmnist conventions):
  data/gtsrb/slice_1/{train,test}_images.npy  uint8 NHWC (N, 224, 224, 3)
  data/gtsrb/slice_1/{train,test}_labels.npy  int64 (N,)
  data/gtsrb/slice_1/{train,test}_meta.csv    columns: label, class_name, filename, synth_group

Story for the paper:
  Constrained class = STOP sign (class 14). Real-world driving safety: a
  deployed traffic-sign classifier should not OVER-predict STOP from noisy
  frames (an over-aggressive system causes unnecessary braking / rear-end
  collisions). The constraint caps STOP predictions to match the deployment
  prior, and 'synth_group' partitions signs by category so the local
  constraint enforces per-category caps:
      group 0 = regulatory  (speed limits + prohibitory)
      group 1 = warning     (triangular warning signs)
      group 2 = mandatory   (priority, yield, stop, direction, end restrictions)
"""
import os
import numpy as np
import pandas as pd
from PIL import Image
import torchvision.datasets as tvd

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
SLICE_DIR = os.path.join(DATA_DIR, "slice_1")
os.makedirs(SLICE_DIR, exist_ok=True)
IMG_SIZE = 224

CLASS_NAMES = {
    0: "Speed_20",   1: "Speed_30",   2: "Speed_50",   3: "Speed_60",
    4: "Speed_70",   5: "Speed_80",   6: "End_Speed_80", 7: "Speed_100",
    8: "Speed_120",  9: "No_Passing", 10: "No_Passing_Trucks",
    11: "Right_of_Way_Intersection", 12: "Priority_Road",
    13: "Yield",     14: "STOP",      15: "No_Vehicles",
    16: "No_Trucks", 17: "No_Entry",  18: "General_Caution",
    19: "Dangerous_Curve_L",  20: "Dangerous_Curve_R",
    21: "Double_Curve",       22: "Bumpy_Road",
    23: "Slippery_Road",      24: "Road_Narrows_R",
    25: "Road_Work",          26: "Traffic_Signals",
    27: "Pedestrians",        28: "Children_Crossing",
    29: "Bicycles_Crossing",  30: "Beware_Ice_Snow",
    31: "Wild_Animals",       32: "End_All_Restrictions",
    33: "Turn_Right",         34: "Turn_Left",
    35: "Ahead_Only",         36: "Straight_or_Right",
    37: "Straight_or_Left",   38: "Keep_Right",
    39: "Keep_Left",          40: "Roundabout",
    41: "End_No_Passing",     42: "End_No_Passing_Trucks",
}

# 3 sign-category groups for local constraints
REGULATORY = {0,1,2,3,4,5,6,7,8,9,10,15,16,17}
WARNING    = {11,18,19,20,21,22,23,24,25,26,27,28,29,30,31}
MANDATORY  = {12,13,14,32,33,34,35,36,37,38,39,40,41,42}

def to_group(label: int) -> int:
    if label in REGULATORY: return 0
    if label in WARNING:    return 1
    if label in MANDATORY:  return 2
    raise ValueError(f"label {label} not categorised")


def _dump(split_name: str, dataset):
    n = len(dataset)
    images = np.empty((n, IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
    labels = np.empty(n, dtype=np.int64)
    rows = []
    for i in range(n):
        pil, lbl = dataset[i]
        if pil.mode != "RGB":
            pil = pil.convert("RGB")
        pil_r = pil.resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
        images[i] = np.asarray(pil_r)
        labels[i] = int(lbl)
        rows.append({
            "label": int(lbl),
            "class_name": CLASS_NAMES[int(lbl)],
            "filename": f"{split_name}_{i:06d}.png",
            "synth_group": to_group(int(lbl)),
        })
        if (i + 1) % 2000 == 0:
            print(f"  [{split_name}] {i+1}/{n}")
    np.save(os.path.join(SLICE_DIR, f"{split_name}_images.npy"), images)
    np.save(os.path.join(SLICE_DIR, f"{split_name}_labels.npy"), labels)
    pd.DataFrame(rows).to_csv(
        os.path.join(SLICE_DIR, f"{split_name}_meta.csv"), index=False)
    print(f"{split_name}: {n} images, classes={len(np.unique(labels))}, "
          f"STOP count={int((labels==14).sum())}")


def main():
    print("Downloading GTSRB via torchvision (~250 MB)...")
    train_ds = tvd.GTSRB(root=DATA_DIR, split="train", download=True)
    test_ds  = tvd.GTSRB(root=DATA_DIR, split="test",  download=True)
    print(f"GTSRB downloaded: train={len(train_ds)} test={len(test_ds)}")
    _dump("train", train_ds)
    _dump("test",  test_ds)
    print(f"Done. Slice at: {SLICE_DIR}")


if __name__ == "__main__":
    main()
