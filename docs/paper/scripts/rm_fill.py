"""Does the "no fill asymmetry" evidence survive per-cell stratification?

The claim: "mean count_adj 85.1-85.8 vs K=86 for all five methods; exactly-K
fraction 0.771-0.833, so there is no fill asymmetry for the metric to expose."

K is not 86 anywhere. It is 51/67/75/86/112/125 across the six (dataset,cap)
combinations. 86 is what you get by AVERAGING those six K's -- the exact
operation the project forbids.

    python paper/scripts/rm_fill.py
"""
import sys

import numpy as np
import pandas as pd

D = "paper/scripts/"
M5 = ["tralo", "fioretto_ldf", "hounie_rcl", "heuristic", "danits_lp"]


def main():
    b30 = pd.read_csv(D + "rm_headroom_b30.csv")
    nce = pd.read_csv(D + "rm_headroom_b30_lrc0.0001_noceskip.csv")

    print("=" * 100)
    print("A. WHAT IS 'K=86'?  the K's actually present in headroom_b30")
    print("=" * 100)
    t = b30.groupby(["dataset", "cap"])["K"].agg(["mean", "nunique"])
    print(t.to_string())
    ks = sorted(b30.groupby(["dataset", "cap"]).K.mean().values)
    print("\n  distinct cell K's: %s" % [int(k) for k in ks])
    print("  unweighted mean of the six cell K's = %.1f" % np.mean(ks))
    print("  run-weighted mean K over the pooled 240 runs = %.2f" % b30.K.mean())
    print("  -> the quoted 'K=86' is the POOLED mean of caps spanning 51..125;")
    print("     no cell in the study has K=86 except tissuemnist/L50.")

    print()
    print("=" * 100)
    print("B. POOLED (the claim's view)  vs  PER-CELL (the project's rule)")
    print("=" * 100)
    print("\n  POOLED over all 6 cells x 4 seeds:")
    g = b30.groupby("method").agg(count_adj=("count_adj", "mean"),
                                  K=("K", "mean"),
                                  exactK=("deficit", lambda s: float((s == 0).mean())),
                                  fill=("count_adj", "sum"))
    g["fill_frac_pooled"] = [b30[b30.method == m].count_adj.sum() /
                             b30[b30.method == m].K.sum() for m in g.index]
    print(g.reindex(M5).round(4).to_string())

    print("\n  PER CELL -- mean count_adj / K (1.0 = budget fully spent):")
    rows = []
    for (ds, mo, cap), gg in b30.groupby(["dataset", "model", "cap"]):
        r = {"dataset": ds, "model": mo, "cap": cap, "K": int(gg.K.iloc[0])}
        for m in M5:
            s = gg[gg.method == m]
            r[m] = s.count_adj.mean() / s.K.mean() if len(s) else np.nan
        r["spread_pp"] = 100 * (max(r[m] for m in M5) - min(r[m] for m in M5))
        rows.append(r)
    t = pd.DataFrame(rows).sort_values(["dataset", "cap", "model"])
    print(t.to_string(index=False, float_format=lambda x: "%.3f" % x))
    print("\n  worst per-cell spread between methods: %.1f percentage points"
          % t.spread_pp.max())
    print("  pooled spread between methods: %.1f percentage points"
          % (100 * (g.fill_frac_pooled.max() - g.fill_frac_pooled.min())))

    print("\n  PER CELL -- exactly-K fraction (deficit == 0):")
    rows = []
    for (ds, mo, cap), gg in b30.groupby(["dataset", "model", "cap"]):
        r = {"dataset": ds, "model": mo, "cap": cap, "K": int(gg.K.iloc[0])}
        for m in M5:
            s = gg[gg.method == m]
            r[m] = float((s.deficit == 0).mean()) if len(s) else np.nan
        rows.append(r)
    t2 = pd.DataFrame(rows).sort_values(["dataset", "cap", "model"])
    print(t2.to_string(index=False, float_format=lambda x: "%.2f" % x))

    print()
    print("=" * 100)
    print("C. THE DEFICIT IS ONE-SIDED (count_adj can only fall short of K)")
    print("=" * 100)
    print("  runs with count_adj > K : %d of %d"
          % (int((b30.deficit < 0).sum()), len(b30)))
    print("  runs with count_adj < K : %d of %d"
          % (int((b30.deficit > 0).sum()), len(b30)))
    print("  mean deficit by method (counts, and as %% of that run's K):")
    b30["defpct"] = 100.0 * b30.deficit / b30.K
    print(b30.groupby("method")[["deficit", "defpct"]].mean()
          .reindex(M5).round(3).to_string())
    print("\n  per-cell mean deficit as %% of K:")
    p = b30.pivot_table(index=["dataset", "model", "cap"], columns="method",
                        values="defpct", aggfunc="mean")
    print(p[M5].round(2).to_string())

    print()
    print("=" * 100)
    print("D. IS ccF1eq ACTUALLY BUDGET-EQUALIZED?  count_eq vs K")
    print("=" * 100)
    for name, d in [("headroom_b30", b30),
                    ("noceskip", pd.read_csv(D + "rm_headroom_b30_lrc0.0001_noceskip.csv"))]:
        bad = d[d.count_eq != d.K]
        print("  %-14s runs where count_eq != K: %d of %d" % (name, len(bad), len(d)))
        if len(bad):
            print(bad.groupby("method")[["K", "count_eq"]].mean().to_string())


if __name__ == "__main__":
    main()
    sys.exit(0)
