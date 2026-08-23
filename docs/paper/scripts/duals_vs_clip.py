"""Dual methods vs post-hoc clippers, per atomic cell, on BOTH metrics.

Never pools. Atomic cell = (dataset, model, cap); seeds are the paired unit
inside a cell; cells are COUNTED, not averaged.

    python paper/scripts/duals_vs_clip.py --csv paper/scripts/out_paperfinal.csv
"""
import argparse
import sys

import pandas as pd

DUAL = ["fioretto_ldf", "hounie_rcl"]
CLIP = ["heuristic", "danits_lp"]
CELL = ["dataset", "model", "cap"]
TOL = 0.005


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--csv2", default=None, help="second file to concat (clip arm)")
    ap.add_argument("--metrics", default="ccF1adj,ccF1eq,AP,macroAdj,macroEq")
    ap.add_argument("--filter-cap", default=None)
    ap.add_argument("--per-cell", action="store_true")
    args = ap.parse_args()

    d = pd.read_csv(args.csv)
    if args.csv2:
        d = pd.concat([d, pd.read_csv(args.csv2)], ignore_index=True)
    if args.filter_cap:
        d = d[d.cap.isin(args.filter_cap.split(","))]
    d = d[d.method.isin(DUAL + CLIP + ["tralo"])]
    metrics = args.metrics.split(",")

    print("=" * 108)
    print("FILE %s   runs=%d" % (args.csv, len(d)))
    print("=" * 108)
    print("methods x n:", d.method.value_counts().to_dict())
    print("caps:", sorted(d.cap.dropna().unique())[:20])
    print("models:", sorted(d.model.dropna().unique()))
    print()
    print("---- budget usage: mean count_raw / count_adj / K by method ----")
    print(d.groupby("method")[["count_raw", "count_adj", "K"]].mean().round(1).to_string())
    print("  fraction of runs whose SHIPPED count == K exactly:")
    print((d.assign(exact=(d.count_adj == d.K)).groupby("method")["exact"].mean()
           .round(3)).to_string())

    rows = []
    for (ds, mo, cap), g in d.groupby(CELL):
        r = {"dataset": ds, "model": mo, "cap": cap, "K": g.K.iloc[0]}
        for m in DUAL + CLIP + ["tralo"]:
            sub = g[g.method == m]
            r["n_" + m] = len(sub)
        for M in metrics:
            piv = g.pivot_table(index="seed", columns="method", values=M)
            hd = [m for m in DUAL if m in piv.columns]
            hc = [m for m in CLIP if m in piv.columns]
            if not hd or not hc:
                continue
            s = piv.dropna(subset=hd + hc)
            if s.empty:
                continue
            delta = s[hd].max(axis=1) - s[hc].max(axis=1)
            r[M + "_dual"] = s[hd].max(axis=1).mean()
            r[M + "_clip"] = s[hc].max(axis=1).mean()
            r[M] = delta.mean()
            r[M + "_ns"] = len(delta)
        rows.append(r)
    t = pd.DataFrame(rows).sort_values(CELL)

    print()
    print("=" * 108)
    print("bestDUAL minus bestCLIP, per cell.  COUNT the cells.")
    print("=" * 108)
    for M in metrics:
        if M not in t.columns:
            continue
        v = t[M].dropna()
        w = int((v > TOL).sum())
        l = int((v < -TOL).sum())
        print("  %-9s  DUAL wins %2d cells | CLIP wins %2d | tie %2d  (of %d)   "
              "mean delta %+0.4f   median %+0.4f" %
              (M, w, l, len(v) - w - l, len(v), v.mean(), v.median()))

    print()
    print("  the same, split by dataset:")
    for ds, g in t.groupby("dataset"):
        line = "    %-12s" % ds
        for M in metrics:
            if M not in g.columns:
                continue
            v = g[M].dropna()
            if not len(v):
                continue
            line += "  %s %+0.4f (%d/%d dualwin)" % (M, v.mean(),
                                                     int((v > TOL).sum()), len(v))
        print(line)

    if args.per_cell:
        cols = CELL + ["K"] + [c for M in metrics for c in
                               (M + "_dual", M + "_clip", M) if c in t.columns]
        print()
        print(t[cols].to_string(index=False, float_format=lambda x: "%.4f" % x))
    return 0


if __name__ == "__main__":
    sys.exit(main())
