"""Per-cell fill in the campaign actually under study (noceskip trained arms vs
the b30 post-hoc arms), plus the provenance of the quoted pooled numbers.

    python paper/scripts/rm_final.py
"""
import sys

import numpy as np
import pandas as pd

D = "paper/scripts/"
M5 = ["tralo", "fioretto_ldf", "hounie_rcl", "heuristic", "danits_lp"]
CELL = ["dataset", "model", "cap"]


def main():
    b30 = pd.read_csv(D + "rm_headroom_b30.csv")
    nce = pd.read_csv(D + "rm_headroom_b30_lrc0.0001_noceskip.csv")
    d = pd.concat([nce, b30[b30.method.isin(["heuristic", "danits_lp"])]],
                  ignore_index=True)

    print("=" * 112)
    print("1. PER-CELL BUDGET FILL, campaign under study "
          "(noceskip trained + b30 post-hoc).  cell mean of count_adj / K")
    print("=" * 112)
    rows = []
    for key, g in d.groupby(CELL):
        r = dict(zip(CELL, key))
        r["K"] = int(g.K.iloc[0])
        for m in M5:
            s = g[g.method == m]
            r[m] = s.count_adj.mean() / s.K.mean() if len(s) else np.nan
        r["spread_pp"] = 100 * (max(r[m] for m in M5) - min(r[m] for m in M5))
        r["worst_seed"] = g.groupby("method").count_adj.min().min() / r["K"]
        rows.append(r)
    t = pd.DataFrame(rows).sort_values(["dataset", "cap", "model"])
    print(t.to_string(index=False, float_format=lambda x: "%.3f" % x))
    print("\n  pooled over all 6 cells (the claim's view): spread %.1f pp"
          % (100 * (d.groupby("method").apply(
              lambda s: s.count_adj.sum() / s.K.sum()).max()
              - d.groupby("method").apply(
                  lambda s: s.count_adj.sum() / s.K.sum()).min())))
    print("  worst SINGLE CELL spread: %.1f pp   (cell %s)"
          % (t.spread_pp.max(),
             t.loc[t.spread_pp.idxmax(), CELL].to_dict()))

    print("\n  per-run budget shortfall, worst runs in the study:")
    d["fill"] = d.count_adj / d.K
    w = d.nsmallest(8, "fill")[["dataset", "model", "cap", "method", "seed",
                                "K", "count_raw", "count_adj", "fill"]]
    print(w.to_string(index=False, float_format=lambda x: "%.3f" % x))

    print()
    print("=" * 112)
    print("2. THE METRIC CHANGE IS A NO-OP FOR THE CLIPPER AND NOT FOR ANYONE "
          "ELSE")
    print("=" * 112)
    for lbl, x in [("headroom_b30", b30), ("noceskip + b30 clip", d)]:
        x = x.copy()
        x["gap"] = x.ccF1eq - x.ccF1adj
        g = x.groupby("method")["gap"].agg(
            runs="count", changed=lambda s: int((s.abs() > 1e-12).sum()),
            mean="mean", worst=lambda s: s.abs().max())
        print("\n  %s" % lbl)
        print(g.reindex([m for m in M5 if m in g.index]).round(4).to_string())
    print("\n  heuristic: ccF1eq == ccF1adj in every single run, so switching "
          "metric cannot move it at all.")
    print("  Every other arm moves. A metric that is identically inert for one "
          "side of a comparison and")
    print("  live for the other is by definition capable of changing that "
          "comparison.")

    print()
    print("=" * 112)
    print("3. PROVENANCE OF THE QUOTED '85.1-85.8 / 0.771-0.833'")
    print("=" * 112)
    blend = pd.concat([nce, b30], ignore_index=True)
    g = blend.groupby("method").agg(count_adj=("count_adj", "mean"),
                                    exactK=("deficit", lambda s: float((s == 0).mean())))
    print(g.reindex(M5).round(4).to_string())
    print("\n  -> matches the quoted range exactly. It is the CONCATENATION of "
          "the noceskip campaign with the")
    print("     lrc=5e-6 ce-skip-ON campaign, i.e. each trained arm's fill is "
          "the average of two different")
    print("     campaigns, only one of which is the campaign whose ccF1 numbers "
          "the evidence is defending.")
    print("  quoted 85.1 = tralo %.4f ;  quoted 0.771 = fioretto_ldf %.4f"
          % (g.loc["tralo", "count_adj"], g.loc["fioretto_ldf", "exactK"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
