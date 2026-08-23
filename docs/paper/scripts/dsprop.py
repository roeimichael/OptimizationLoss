"""What does the cap actually ASK of the model, per dataset?

Measured on the plain-CE control (headroom_b30 `heuristic`, warmup 30 /
constraint 0 -- the same architecture, seed and epoch budget with the
constraint switched off).

For each (dataset, backbone, cap, seed):
  S      = soft count  = sum_i P[i,c]           (what the penalty actually sees)
  H      = hard count  = #argmax == c
  drop   = H - K                                 samples whose argmax must move
  dropfrac_pool = drop / |pool|                  fraction of the pool to flip
  dS/S   = (S - K) / S                           relative soft-count reduction asked
  margin = P[i,c] - max_{j != c} P[i,j] averaged over the DROP SET
           (ranks K..H by P[:,c]) -- how far those samples have to be pushed
  droptrue = fraction of the drop set that is genuinely class c
           (if high, CE actively fights the constraint on those samples)
  p_at_K   = P[:,c] at rank K  (the cut threshold)
  p_at_H   = P[:,c] at rank H
Also: CE optimizer steps per epoch = ceil(n_train / batch) versus ONE clipped
constraint step per epoch, per dataset.
"""
import argparse
import glob
import json
import math
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.getcwd())
from src.training.constraints import compute_global_constraints  # noqa: E402
from src.utils.constants import UNLIMITED  # noqa: E402

CELL = ["dataset", "model", "cap"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", default="results/headroom/headroom_b30")
    args = ap.parse_args()

    rows = []
    for cfgp in glob.glob(args.clip + "/**/config.json", recursive=True):
        cfg = json.load(open(cfgp))
        if cfg.get("methodology") != "heuristic":
            continue
        d = os.path.dirname(cfgp)
        f = os.path.join(d, "final_predictions_raw.csv")
        if not os.path.exists(f):
            continue
        t = pd.read_csv(f)
        cols = sorted((c for c in t.columns if c.startswith("Prob_Class_")),
                      key=lambda c: int(c.rsplit("_", 1)[1]))
        P = t[cols].to_numpy(float)
        y = t["True_Label"].to_numpy(int)
        raw = t["Predicted_Label"].to_numpy(int)
        dc = cfg.get("dataset_config") or {}
        c = dc.get("constrained_class")
        c = int(c[0] if isinstance(c, (list, tuple)) else c)
        lp, gp = cfg["constraint"]
        gcol = t["Group_ID"].to_numpy(int) if "Group_ID" in t.columns else None
        df = pd.DataFrame({"label": y, "grp": gcol if gcol is not None else 0})
        G = compute_global_constraints(df, "label", gp, constrained_class=[c],
                                       num_classes=P.shape[1])
        if G[c] >= UNLIMITED:
            continue
        K = int(G[c])
        pc = P[:, c]
        other = P.copy()
        other[:, c] = -np.inf
        margin_all = pc - other.max(axis=1)
        order = np.argsort(-pc)
        H = int((raw == c).sum())
        S = float(pc.sum())
        drop = order[K:H] if H > K else np.array([], dtype=int)
        rows.append({
            "dataset": cfg.get("dataset_mode"), "model": cfg.get("model_name"),
            "cap": cfg.get("constraint_tag"), "seed": (cfg.get("hyperparams") or {}).get("seed"),
            "pool": len(y), "n_true": int((y == c).sum()),
            "prev": (y == c).mean(), "K": K, "H": H, "S": S,
            "K_over_H": K / max(H, 1), "dS_over_S": (S - K) / max(S, 1e-9),
            "drop": len(drop), "dropfrac_pool": len(drop) / float(len(y)),
            "drop_margin": float(margin_all[drop].mean()) if len(drop) else np.nan,
            "drop_pc": float(pc[drop].mean()) if len(drop) else np.nan,
            "drop_true": float((y[drop] == c).mean()) if len(drop) else np.nan,
            "p_at_K": float(pc[order[K - 1]]), "p_at_H": float(pc[order[H - 1]]) if H else np.nan,
            "keep_margin": float(margin_all[order[:K]].mean()),
        })
    t = pd.DataFrame(rows)
    pd.set_option("display.width", 250)
    print("=" * 128)
    print("WHAT THE CAP ASKS -- measured on the constraint-OFF control (heuristic, warmup 30)")
    print("=" * 128)
    agg = t.groupby(CELL).mean(numeric_only=True).reset_index()
    print(agg.drop(columns=["seed"]).to_string(index=False,
                                               float_format=lambda x: "%.4f" % x))

    print()
    print("=" * 128)
    print("CE steps per epoch vs ONE clipped constraint step per epoch")
    print("=" * 128)
    for ds in sorted(t.dataset.unique()):
        # locate the train csv the loader uses
        sub = [p for p in glob.glob(args.clip + "/**/config.json", recursive=True)
               if json.load(open(p)).get("dataset_mode") == ds]
        cfg = json.load(open(sub[0]))
        ddir = (cfg.get("dataset_config") or {}).get("data_dir")
        n_tr = None
        for cand in ("train.csv", "train_labels.csv", "labels_train.csv"):
            p = os.path.join(ddir, cand)
            if os.path.exists(p):
                n_tr = len(pd.read_csv(p))
                break
        if n_tr is None:
            files = sorted(glob.glob(os.path.join(ddir, "*.csv")))
            info = {os.path.basename(x): len(pd.read_csv(x)) for x in files}
            print("  %-12s data_dir=%s  csvs=%s" % (ds, ddir, info))
            continue
        bs = (cfg.get("hyperparams") or {}).get("batch_size", 64)
        print("  %-12s n_train=%d batch=%d -> %d CE steps/epoch vs 1 constraint step  (%dx)"
              % (ds, n_tr, bs, math.ceil(n_tr / bs), math.ceil(n_tr / bs)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
