"""Where does the suppressed mass GO?  Is there a cheap escape class?

On the constraint-OFF control (heuristic, warmup 30), for every pool sample
whose argmax is the constrained class c, record the runner-up class and the
margin P[c] - P[runner-up].  A class that is confusable with one dominant
neighbour can be erased by a small logit shift; a class with no such neighbour
cannot.

Then check where the mass actually went in the runs that DID collapse:
compare the predicted-label histogram of hounie_rcl vs the CE control.
"""
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.getcwd())

CELL = ["dataset", "model", "cap"]


def read(d):
    t = pd.read_csv(os.path.join(d, "final_predictions_raw.csv"))
    cols = sorted((c for c in t.columns if c.startswith("Prob_Class_")),
                  key=lambda x: int(x.rsplit("_", 1)[1]))
    return t[cols].to_numpy(float), t["True_Label"].to_numpy(int), \
        t["Predicted_Label"].to_numpy(int)


def cls_of(cfg):
    c = (cfg.get("dataset_config") or {}).get("constrained_class")
    return int(c[0] if isinstance(c, (list, tuple)) else c)


def main():
    print("=" * 116)
    print("(A) ESCAPE ROUTE on the constraint-OFF control (heuristic, warmup 30, seed 1)")
    print("    over pool samples predicted class c: which class is runner-up, and by how much")
    print("=" * 116)
    seen = set()
    for cfgp in sorted(glob.glob("results/headroom/headroom_b30/**/config.json",
                                 recursive=True)):
        cfg = json.load(open(cfgp))
        if cfg.get("methodology") != "heuristic":
            continue
        hp = cfg.get("hyperparams") or {}
        if hp.get("seed") != 1 or cfg.get("constraint_tag") != "L30_G30":
            continue
        k = (cfg.get("dataset_mode"), cfg.get("model_name"))
        if k in seen:
            continue
        seen.add(k)
        d = os.path.dirname(cfgp)
        if not os.path.exists(os.path.join(d, "final_predictions_raw.csv")):
            continue
        P, y, raw = read(d)
        c = cls_of(cfg)
        sel = raw == c
        Q = P[sel].copy()
        Q[:, c] = -np.inf
        run = Q.argmax(axis=1)
        marg = P[sel][:, c] - Q.max(axis=1)
        prev = np.bincount(y, minlength=P.shape[1]) / len(y)
        hist = pd.Series(run).value_counts(normalize=True).sort_values(ascending=False)
        top = hist.index[0]
        print("\n  %-12s %-13s  c=%d  n_pred_c=%d" % (k[0], k[1], c, sel.sum()))
        print("     class prevalence in pool: %s"
              % ", ".join("%d:%.3f" % (i, v) for i, v in enumerate(prev)))
        print("     runner-up share:          %s"
              % ", ".join("%d:%.3f" % (i, v) for i, v in hist.items()))
        print("     TOP escape class %d (share %.3f, pool prevalence %.3f); "
              "mean margin to it %.3f, median %.3f"
              % (top, hist.iloc[0], prev[top], marg.mean(), np.median(marg)))

    print()
    print("=" * 116)
    print("(B) WHERE THE MASS WENT in the runs that collapsed (seed 1, L30_G30):")
    print("    predicted-label histogram, CE control vs hounie_rcl vs tralo")
    print("=" * 116)
    for ds in ["dermmnist", "octmnist"]:
        for mo in ["MobileNetV3", "RegNetY400MF"]:
            got = {}
            for root, meths in [("results/headroom/headroom_b30", ["heuristic"]),
                                ("results/headroom/headroom_b30_lrc0.0001_noceskip",
                                 ["hounie_rcl", "fioretto_ldf", "tralo"])]:
                for cfgp in glob.glob(root + "/**/config.json", recursive=True):
                    cfg = json.load(open(cfgp))
                    hp = cfg.get("hyperparams") or {}
                    if (cfg.get("dataset_mode") != ds or cfg.get("model_name") != mo
                            or cfg.get("constraint_tag") != "L30_G30"
                            or hp.get("seed") != 1
                            or cfg.get("methodology") not in meths):
                        continue
                    d = os.path.dirname(cfgp)
                    if not os.path.exists(os.path.join(d, "final_predictions_raw.csv")):
                        continue
                    P, y, raw = read(d)
                    got[cfg["methodology"]] = np.bincount(raw, minlength=P.shape[1])
                    ncls = P.shape[1]
            if not got:
                continue
            print("\n  --- %s %s (seed 1, L30_G30)" % (ds, mo))
            hdr = "    %-14s " % "method" + " ".join("cls%d" % i for i in range(ncls))
            print(hdr)
            for m in ["heuristic", "tralo", "fioretto_ldf", "hounie_rcl"]:
                if m in got:
                    print("    %-14s " % m + " ".join("%4d" % v for v in got[m]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
