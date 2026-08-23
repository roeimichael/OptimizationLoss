"""Backbone interaction, step 1: read the existing fact base only. No rebuild.

Prints the 12 atomic cells of headroom_b30_lrc0.0001_noceskip with the overshoot
of the UNCONSTRAINED model (the post-hoc clipper's own raw count) against K, and
TraLO's paired advantage over the best dual, then correlates the two.
"""
import sys

import numpy as np
import pandas as pd

FB = "paper/scripts/out_factbase.csv"
CAMP = "lrc0.0001_noceskip"


def main():
    fb = pd.read_csv(FB)
    print("campaigns in fact base:", sorted(fb.campaign.unique()))
    d = fb[fb.campaign == CAMP].copy()
    print("rows for %s: %d   methods: %s" % (CAMP, len(d), sorted(d.method.unique())))

    # one row per cell, taken off the tralo row (d_vs_* are cell-level, repeated)
    t = d[d.method == "tralo"].copy()
    t["overshoot_ratio"] = t["clip_raw"] / t["K"]
    t["overshoot_abs"] = t["clip_raw"] - t["K"]
    t["cut_frac"] = 1.0 - t["K"] / t["clip_raw"]
    cols = ["dataset", "model", "cap", "K", "n_pool", "n_true_cls", "clip_raw",
            "clip_raw_min", "clip_raw_max", "overshoot_ratio", "cut_frac",
            "d_vs_bestdual", "d_vs_clip", "ccF1eq", "count_raw", "AP", "macroEq",
            "n_collapsed"]
    print("\n" + "=" * 120)
    print("12 CELLS (tralo row; d_vs_* are cell-level paired means)")
    print("=" * 120)
    print(t[cols].sort_values(["dataset", "model", "cap"])
          .to_string(index=False, float_format=lambda x: "%.4f" % x))

    print("\n" + "=" * 120)
    print("ALL METHODS PER CELL: ccF1eq / AP / count_raw / sat / epochs")
    print("=" * 120)
    piv = d.pivot_table(index=["dataset", "model", "cap"], columns="method",
                        values=["ccF1eq", "AP", "count_raw", "sat",
                                "constraint_epochs_run", "n_collapsed"])
    for m in ["ccF1eq", "AP", "count_raw", "sat", "constraint_epochs_run",
              "n_collapsed"]:
        print("\n--- %s ---" % m)
        print(piv[m].to_string(float_format=lambda x: "%.4f" % x))

    print("\n" + "=" * 120)
    print("CORRELATION over all 12 cells: does overshoot predict TraLO's edge?")
    print("=" * 120)
    for xc in ["overshoot_ratio", "overshoot_abs", "cut_frac", "K_over_clip_raw",
               "natural_rate", "clip_raw", "K"]:
        x = t[xc].to_numpy(float)
        for yc in ["d_vs_bestdual", "d_vs_clip"]:
            y = t[yc].to_numpy(float)
            ok = ~(np.isnan(x) | np.isnan(y))
            if ok.sum() < 3:
                continue
            r = np.corrcoef(x[ok], y[ok])[0, 1]
            rs = pd.Series(x[ok]).corr(pd.Series(y[ok]), method="spearman")
            print("  %-16s vs %-14s  n=%2d  pearson r=%+0.3f  r2=%.3f  spearman=%+0.3f"
                  % (xc, yc, ok.sum(), r, r * r, rs))

    print("\n  within-dataset (n=4 each), overshoot_ratio vs d_vs_bestdual:")
    for ds, g in t.groupby("dataset"):
        x, y = g["overshoot_ratio"].to_numpy(float), g["d_vs_bestdual"].to_numpy(float)
        print("    %-12s r=%+0.3f   overshoot %s   edge %s"
              % (ds, np.corrcoef(x, y)[0, 1],
                 np.round(x, 3).tolist(), np.round(y, 4).tolist()))

    print("\n" + "=" * 120)
    print("BACKBONE SPLIT on every dataset (d_vs_bestdual per cell)")
    print("=" * 120)
    b = t.pivot_table(index=["dataset", "cap"], columns="model",
                      values="d_vs_bestdual")
    b["MNV3_minus_RegNet"] = b["MobileNetV3"] - b["RegNetY400MF"]
    print(b.to_string(float_format=lambda x: "%+0.4f" % x))
    print("\n  cells won by TraLO (d_vs_bestdual > 0.005), by backbone:")
    for mo, g in t.groupby("model"):
        print("    %-14s  W %d / L %d / T %d  of %d   mean %+0.4f"
              % (mo, int((g.d_vs_bestdual > 0.005).sum()),
                 int((g.d_vs_bestdual < -0.005).sum()),
                 int((g.d_vs_bestdual.abs() <= 0.005).sum()), len(g),
                 g.d_vs_bestdual.mean()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
