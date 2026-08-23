"""Dose-response: is the AP deficit caused by the constraint STEP?

The sibling campaigns differ in `lr_constraint` while sharing everything else
(same warm-up cache, same 1+29 epoch budget, same seeds, same caps).  That is
an already-run 20x dose ladder on the size of the constraint update:

    headroom_b30                        lr_constraint = 5e-06
    headroom_b30_lrc5e-05               lr_constraint = 5e-05
    headroom_b30_lrc0.0001              lr_constraint = 1e-04
    headroom_b30_lrc0.0001_noceskip     lr_constraint = 1e-04, ce-skip off

The pure-CE reference (heuristic/danits_lp inside headroom_b30) is the SAME
model for every campaign, because AP is read off raw probabilities and those
arms never take a constraint step.  So dAP is comparable across the ladder.

If dAP scales with lr_constraint, the damage is done BY the constraint update.
If it does not, the damage is not the update.

Reuses the loaders in ap_damage2.py.

    python paper/scripts/ap_damage3.py
"""
import glob
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, os.getcwd())
sys.path.insert(0, "paper/scripts")
import ap_damage2 as A2                                              # noqa: E402

CAMPAIGNS = [
    ("b30_lrc5e-06", "results/headroom/headroom_b30"),
    ("lrc5e-05", "results/headroom/headroom_b30_lrc5e-05"),
    ("lrc1e-04_ceskip", "results/headroom/headroom_b30_lrc0.0001"),
    ("lrc1e-04_noskip", "results/headroom/headroom_b30_lrc0.0001_noceskip"),
]
TRAINED = A2.TRAINED
GRP = ["dataset", "model", "cap", "method"]


def knobs(root):
    out = []
    for p in sorted(glob.glob(root + "/**/config.json", recursive=True)):
        c = json.load(open(p))
        if c.get("methodology") not in TRAINED + A2.CLIP:
            continue
        hp = c.get("hyperparams") or {}
        out.append({"method": c["methodology"], "lr": hp.get("lr"),
                    "lr_constraint": hp.get("lr_constraint"),
                    "warmup": hp.get("warmup_epochs"),
                    "cepochs": hp.get("constraint_epochs"),
                    "ce_skip": hp.get("enable_ce_skip"),
                    "stable_thr": hp.get("stable_count_threshold")})
    return pd.DataFrame(out)


def main():
    print("=" * 96)
    print("0. CONFIRM THE KNOBS (read out of every config.json, not assumed)")
    print("=" * 96)
    for name, root in CAMPAIGNS:
        k = knobs(root)
        if k.empty:
            print("  %-18s  NO RUNS" % name)
            continue
        g = (k.groupby("method")
             .agg(lambda s: sorted(set(map(str, s.tolist())))))
        print("\n  --- %s  (%s)  n=%d" % (name, root, len(k)))
        print(g.to_string())

    cl = A2.scan("results/headroom/headroom_b30", A2.CLIP)
    key = ["dataset", "model", "seed"]
    ref = cl.groupby(key)[["AP", "precAtK"]].mean()
    ref.columns = ["AP_ce", "precAtK_ce"]

    frames = []
    for name, root in CAMPAIGNS:
        tr = A2.scan(root, TRAINED)
        if tr.empty:
            continue
        tr["campaign"] = name
        tr["lr_c"] = tr["path"].map(
            lambda p: (json.load(open(os.path.join(p, "config.json")))
                       .get("hyperparams", {}) or {}).get("lr_constraint"))
        frames.append(tr.merge(ref.reset_index(), on=key, how="left"))
    d = pd.concat(frames, ignore_index=True)
    d["dAP"] = d["AP"] - d["AP_ce"]
    d["dPrecK"] = d["precAtK"] - d["precAtK_ce"]
    d["over_unspent"] = np.maximum(0.0, d["unspent"])       # under-predicting only
    d["abs_unspent"] = d["unspent"].abs()
    d.to_csv("paper/scripts/out_ap_damage3.csv", index=False)
    print("\nscored %d trained runs across %d campaigns; wrote out_ap_damage3.csv"
          % (len(d), d.campaign.nunique()))

    print("\n" + "=" * 96)
    print("1. DOSE RESPONSE.  mean dAP per (dataset, method) at each lr_constraint.")
    print("   Cells are means of 4 seeds x 2 backbones x 2 caps = 16 runs; the")
    print("   per-cell version is below it so nothing is hidden by the average.")
    print("=" * 96)
    order = [c[0] for c in CAMPAIGNS]
    t = d.pivot_table(index=["dataset", "method"], columns="campaign", values="dAP")
    print(t[[c for c in order if c in t.columns]].round(4).to_string())
    print("\n  n runs per (dataset, campaign):")
    print(d.pivot_table(index="dataset", columns="campaign", values="dAP",
                        aggfunc="count").to_string())

    print("\n  DERM, per cell (never pooled across backbone/cap):")
    td = d[d.dataset == "dermmnist"].pivot_table(
        index=["model", "cap", "method"], columns="campaign", values="dAP")
    print(td[[c for c in order if c in td.columns]].round(4).to_string())

    print("\n  Spearman(lr_constraint, dAP) within each (dataset,backbone,cap,method):")
    print("  %-12s %6s %6s %8s" % ("dataset", "neg", "pos", "median_rho"))
    for ds, gds in d.groupby("dataset"):
        rs = []
        for _, g in gds.groupby(GRP):
            x = g["lr_c"].to_numpy(float)
            y = g["dAP"].to_numpy(float)
            m = np.isfinite(x) & np.isfinite(y)
            if m.sum() < 6 or len(set(x[m])) < 2:
                continue
            rs.append(spearmanr(x[m], y[m])[0])
        rs = np.array([r for r in rs if np.isfinite(r)])
        print("  %-12s %6d %6d %8s" % (ds, int((rs < 0).sum()), int((rs > 0).sum()),
                                       "%.3f" % np.median(rs) if len(rs) else "-"))

    print("\n" + "=" * 96)
    print("2. THE COUNT-COLLAPSE HYPOTHESIS.  Damage vs how far the model's OWN raw")
    print("   count ends below the cap.  unspent = (K - count_raw)/K;  >0 means the")
    print("   model predicts FEWER positives than it is allowed.")
    print("=" * 96)
    cm = d.groupby(["campaign"] + GRP)[["dAP", "dPrecK", "unspent", "over_unspent",
                                       "abs_unspent", "dose_active", "n_sat_total",
                                       "eval_c", "peak_lambda"]].mean().reset_index()
    for lbl, col in [("unspent (signed)", "unspent"),
                     ("max(0,unspent) = undershoot only", "over_unspent"),
                     ("|unspent|", "abs_unspent"),
                     ("dose_active", "dose_active"),
                     ("n_sat_total", "n_sat_total"),
                     ("eval_c", "eval_c")]:
        r, p = spearmanr(cm[col], cm["dAP"])
        print("  %-34s rho(dAP) = %+.3f   p = %.2e   (n=%d cells)"
              % (lbl, r, p, len(cm)))
    print("\n  per dataset (cells only from that dataset):")
    for ds, g in cm.groupby("dataset"):
        r1 = spearmanr(g["over_unspent"], g["dAP"])
        r2 = spearmanr(g["dose_active"], g["dAP"])
        print("    %-12s n=%2d cells   undershoot rho=%+.3f (p=%.3f)   dose rho=%+.3f (p=%.3f)"
              % (ds, len(g), r1[0], r1[1], r2[0], r2[1]))

    print("\n  binned: mean dAP by undershoot bucket (cells, all campaigns)")
    cm["bucket"] = pd.cut(cm["unspent"], [-99, -0.2, -0.02, 0.02, 0.2, 0.5, 99],
                          labels=["over>20%", "over 2-20%", "at cap +-2%",
                                  "under 2-20%", "under 20-50%", "under >50%"])
    print(cm.groupby("bucket")[["dAP", "dPrecK"]].agg(["mean", "count"]).round(4).to_string())

    print("\n" + "=" * 96)
    print("3. TraLO ONLY -- the count trajectory is logged for TraLO, so we can ask")
    print("   whether the DEEPEST dive below the cap (not the endpoint) is the")
    print("   predictor.  traj_min_over_K = min logged hard count / K up to eval_c.")
    print("=" * 96)
    tt = d[(d.method == "tralo") & np.isfinite(d.traj_min_over_K)]
    ct = tt.groupby(["campaign", "dataset", "model", "cap"])[
        ["dAP", "traj_min_over_K", "traj_max_over_K", "traj_range_K",
         "unspent", "dose_active"]].mean().reset_index()
    for col in ["traj_min_over_K", "traj_max_over_K", "traj_range_K", "unspent",
                "dose_active"]:
        r, p = spearmanr(ct[col], ct["dAP"])
        print("  %-20s rho(dAP) = %+.3f  p = %.2e   (n=%d TraLO cells)"
              % (col, r, p, len(ct)))
    print()
    print(ct.sort_values("dAP").to_string(index=False, float_format=lambda x: "%.3f" % x))

    print("\n" + "=" * 96)
    print("4. EXISTENCE PROOF, all campaigns.  Cells where the cap BINDS and")
    print("   constraint training cost no ranking quality (mean dAP >= -0.005).")
    print("=" * 96)
    ok = cm[cm.dAP >= -0.005].copy()
    print("  %d of %d (campaign x cell) combinations show no AP damage"
          % (len(ok), len(cm)))
    print(ok.sort_values("dAP", ascending=False)
          .to_string(index=False, float_format=lambda x: "%.4f" % x))
    print("\n  by method:", ok.method.value_counts().to_dict())
    print("  by dataset:", ok.dataset.value_counts().to_dict())
    return 0


if __name__ == "__main__":
    sys.exit(main())
