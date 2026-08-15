"""Why did a run end well or badly? Attribute outcomes to training dynamics.

The comparison scripts say WHICH configuration wins. This says WHY, by joining
each run's final score to features extracted from its own training_log.csv, so
that a delta between two variants can be traced to the behaviour that produced
it rather than assumed.

Two questions it is built to answer:

  COLLAPSE. 21 of 240 runs in the matched-LR campaign end with the model
  predicting the constrained class for almost nobody (count_raw < K/3), and all
  21 are on DermMNIST. Because satisfaction is tested one-sidedly
  (train.py:216 flags only count > K), such a run is recorded as "satisfied"
  and its checkpoint is kept. What distinguishes a run that collapses from one
  that does not?

  GAIN. Where a variant beats the incumbent, which dynamic feature moves with
  the gain? That is the difference between "this change helped" and knowing
  which component of it helped, which is what makes the next change informed
  rather than another guess.

Features are read from the log, never assumed: first satisfied epoch, epochs
spent satisfied, whether cross-entropy was zeroed by the saturation gate and for
how long, the lambda trajectory, and how far the constrained-class count travels
from its starting value and from K.

    python paper/scripts/attribute.py --trained results/headroom/headroom_b30_lrc0.0001
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.getcwd())
sys.path.insert(0, "paper/scripts")
import analyze_headroom as A  # noqa: E402


def log_features(run_dir, cls):
    p = os.path.join(run_dir, "training_log.csv")
    if not os.path.exists(p):
        return None
    try:
        d = pd.read_csv(p)
    except Exception:
        return None
    if not len(d):
        return None
    f = {"epochs": len(d)}
    hard = "Hard_Class%d" % cls
    lim = "Limit_Class%d" % cls
    if hard in d.columns:
        h = d[hard].to_numpy(float)
        f["count_first"] = h[0]
        f["count_last"] = h[-1]
        f["count_min"] = h.min()
        f["count_drop_frac"] = (h[0] - h[-1]) / max(h[0], 1.0)
        if lim in d.columns and d[lim].iloc[0] > 0:
            K = float(d[lim].iloc[0])
            f["K"] = K
            f["undershoot_frac"] = max(0.0, (K - h[-1]) / K)
            # Time spent BELOW the cap. The penalty does not charge this at all,
            # so it is the region where the objective gives no signal and the
            # one-sided satisfaction test still reads "satisfied".
            f["frac_epochs_below_K"] = float((h < K).mean())
    if "L_CE" in d.columns:
        ce = d["L_CE"].to_numpy(float)
        f["ce_first"] = ce[0]
        f["ce_last"] = ce[-1]
        f["frac_epochs_ce_zero"] = float((ce == 0).mean())
        nz = np.where(ce == 0)[0]
        f["first_ce_zero_epoch"] = int(nz[0]) if len(nz) else -1
    if "Train_Acc" in d.columns:
        f["acc_last"] = float(d["Train_Acc"].iloc[-1])
    if "Global_Satisfied" in d.columns:
        s = d["Global_Satisfied"].to_numpy()
        f["frac_epochs_satisfied"] = float(np.mean(s == 1))
        w = np.where(s == 1)[0]
        f["first_sat_epoch"] = int(w[0]) if len(w) else -1
        f["epochs_after_first_sat"] = (len(d) - int(w[0])) if len(w) else 0
    if "Lambda_Global" in d.columns:
        lam = d["Lambda_Global"].to_numpy(float)
        f["lam_last"] = lam[-1]
        f["lam_max"] = lam.max()
    return f


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trained", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    scored = A.rows_for(args.trained)
    if scored.empty:
        print("no scorable runs")
        return 1

    rows = []
    for cfg_path in glob.glob(args.trained + "/**/config.json", recursive=True):
        try:
            cfg = json.load(open(cfg_path))
        except Exception:
            continue
        dc = cfg.get("dataset_config", {}) or {}
        c = dc.get("constrained_class")
        if c is None:
            continue
        cls = int(c[0] if isinstance(c, (list, tuple)) else c)
        f = log_features(os.path.dirname(cfg_path), cls)
        if not f:
            continue
        hp = cfg.get("hyperparams") or {}
        f.update({"dataset": cfg.get("dataset_mode"), "cap": cfg.get("constraint_tag"),
                  "model": cfg.get("model_name"), "seed": hp.get("seed"),
                  "method": cfg.get("methodology")})
        rows.append(f)
    feats = pd.DataFrame(rows)
    if feats.empty:
        print("no training logs parsed")
        return 1

    d = scored.merge(feats, on=["dataset", "cap", "model", "seed", "method"],
                     how="inner", suffixes=("", "_log"))
    d["collapsed"] = d["count_raw"] < (d["K"] / 3.0)
    print("joined %d runs (%d collapsed)" % (len(d), int(d.collapsed.sum())))

    print("\n" + "=" * 76)
    print("COLLAPSED vs HEALTHY -- what separates them")
    print("=" * 76)
    cols = [c for c in ["frac_epochs_ce_zero", "first_ce_zero_epoch", "acc_last",
                        "frac_epochs_satisfied", "first_sat_epoch",
                        "epochs_after_first_sat", "frac_epochs_below_K",
                        "undershoot_frac", "count_drop_frac", "lam_max",
                        "ce_last", "epochs"] if c in d.columns]
    if d.collapsed.any() and (~d.collapsed).any():
        cmp = d.groupby("collapsed")[cols].mean().T
        cmp.columns = ["healthy", "COLLAPSED"]
        cmp["delta"] = cmp["COLLAPSED"] - cmp["healthy"]
        print(cmp.round(3).to_string())
    else:
        print("  (only one class present in this campaign)")

    print("\n" + "=" * 76)
    print("CORRELATION of each dynamic feature with the PRIMARY metric (ccF1eq)")
    print("=" * 76)
    for ds, g in d.groupby("dataset"):
        if len(g) < 8:
            continue
        cc = {}
        for c in cols:
            v = g[c].astype(float)
            if v.nunique() > 2:
                cc[c] = v.corr(g["ccF1eq"])
        if cc:
            s = pd.Series(cc).sort_values()
            print("\n%s (n=%d)" % (ds, len(g)))
            print(s.round(3).to_string())

    if args.out:
        d.to_csv(args.out, index=False)
        print("\nwrote %s" % args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
