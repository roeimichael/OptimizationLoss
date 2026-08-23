"""Is the derm/oct split a COLLAPSE story?

(a) Across every trained run, relate the model's own raw count of the
    constrained class to its ranking quality (AP), within dataset.
(b) TraLO instability: mean |delta count| between consecutive LOGGED epochs,
    normalised by K -- does TraLO thrash more on one dataset than the other?
(c) The CE-saturation probe: headroom_b30_lrc0.0001 is byte-identical to
    _noceskip except tralo's enable_ce_skip. Report where the gate fired
    (train acc >= 0.995 twice) and what it did.
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, os.getcwd())
sys.path.insert(0, "paper/scripts")
import analyze_headroom as A  # noqa: E402

TRAINED = ["tralo", "fioretto_ldf", "hounie_rcl"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="results/headroom/headroom_b30_lrc0.0001_noceskip")
    ap.add_argument("--traj", default="paper/scripts/out_traj_epochs.csv")
    args = ap.parse_args()
    pd.set_option("display.width", 250)

    d = A.rows_for(args.root)
    d = d[d.method.isin(TRAINED)].copy()
    d["fill"] = d["count_raw"] / d["K"]

    print("=" * 118)
    print("(a) COLLAPSE vs RANKING. One row per trained run (n=48 per dataset).")
    print("    fill = model's own raw count / K.  rho = Spearman(fill, AP) over those runs.")
    print("=" * 118)
    for ds, g in d.groupby("dataset"):
        r = spearmanr(g["fill"], g["AP"])
        print("\n  %s   n=%d   Spearman(fill, AP) = %+.3f   p=%.2g"
              % (ds, len(g), r.correlation, r.pvalue))
        b = pd.cut(g["fill"], [0, 0.25, 0.5, 0.75, 1.0, 1.5, 10],
                   labels=["<0.25", "0.25-0.5", "0.5-0.75", "0.75-1.0", "1.0-1.5", ">1.5"])
        t = g.groupby(b, observed=False).agg(n=("AP", "size"), AP=("AP", "mean"),
                                             ccF1eq=("ccF1eq", "mean")).reset_index()
        print(t.to_string(index=False, float_format=lambda x: "%.4f" % x))
        print("    per-method mean fill:  " + "  ".join(
            "%s %.2f" % (m, g[g.method == m]["fill"].mean()) for m in TRAINED))

    print()
    print("=" * 118)
    print("(b) TraLO count instability across logged epochs")
    print("    swing = mean |count_t - count_{t-1}| / K over consecutive LOGGED rows")
    print("    cv    = sd(count)/mean(count) over logged epochs >= 10")
    print("=" * 118)
    if os.path.exists(args.traj):
        J = pd.read_csv(args.traj)
        rows = []
        for (ds, mo, cap, sd), g in J.groupby(["dataset", "model", "cap", "seed"]):
            g = g.sort_values("ep")
            K = g["K"].iloc[0]
            h = g["hard"].to_numpy(float)
            late = g[g.ep >= 10]["hard"].to_numpy(float)
            rows.append({"dataset": ds, "model": mo, "cap": cap, "seed": sd,
                         "swing": np.abs(np.diff(h)).mean() / K if len(h) > 1 else np.nan,
                         "cv": late.std() / late.mean() if len(late) > 1 else np.nan,
                         "above": float((h > K).mean()), "K": K})
        t = pd.DataFrame(rows)
        agg = t.groupby(["dataset", "model", "cap"]).agg(
            swing=("swing", "mean"), cv=("cv", "mean"),
            frac_epochs_above_K=("above", "mean")).reset_index()
        print(agg.to_string(index=False, float_format=lambda x: "%.3f" % x))
        print()
        print("  by dataset: " + " | ".join(
            "%s swing %.3f cv %.3f above %.2f" % (ds, g.swing.mean(), g.cv.mean(),
                                                  g["above"].mean())
            for ds, g in t.groupby("dataset")))
    else:
        print("  (missing %s -- run traj_ds.py --dumptraj first)" % args.traj)

    print()
    print("=" * 118)
    print("(c) CE-SATURATION GATE probe: _noceskip (gate OFF) vs lrc0.0001 (gate ON).")
    print("    Only tralo differs between the two campaigns; the dual runs are the same draws.")
    print("=" * 118)
    a = A.rows_for(args.root)
    b = A.rows_for("results/headroom/headroom_b30_lrc0.0001")
    key = ["dataset", "model", "cap", "seed", "method"]
    m = a.merge(b, on=key, suffixes=("_off", "_on"))
    m = m[m.method == "tralo"]
    g = m.groupby(["dataset", "model", "cap"]).agg(
        AP_off=("AP_off", "mean"), AP_on=("AP_on", "mean"),
        cc_off=("ccF1eq_off", "mean"), cc_on=("ccF1eq_on", "mean"),
        raw_off=("count_raw_off", "mean"), raw_on=("count_raw_on", "mean"),
        K=("K_off", "mean"),
        identical=("AP_off", "size")).reset_index()
    g["dAP"] = g["AP_on"] - g["AP_off"]
    g["dcc"] = g["cc_on"] - g["cc_off"]
    g["same_run"] = (g["AP_on"] - g["AP_off"]).abs() < 1e-12
    print(g.to_string(index=False, float_format=lambda x: "%.4f" % x))
    return 0


if __name__ == "__main__":
    sys.exit(main())
