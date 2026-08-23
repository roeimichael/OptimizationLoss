"""In the regime where TraLO works, what does a good run look like in the log?

The fill floor answered this for warm-up 1 and does not transfer: at warm-up 50
essentially nothing lands in the damaged zone, so it has no support there. That
leaves the question open in exactly the regime the paper reports, which is the
regime worth understanding.

Every candidate below is something visible while training is still running --
the point is to find a signal that says "this run is going badly" before the
metrics are computed, not to post-hoc rationalise the score.

Discipline carried over from check_fill.py: scores are centered WITHIN CELL
(dataset, model, cap) before any correlation. Cells differ hugely in absolute
cc-F1, so a pooled correlation finds structure even when the feature is
irrelevant -- it would just be reading dataset identity. Features are ranked by
the within-cell number and the raw one is printed alongside so the gap is
visible.
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

CELL = ["dataset", "model", "cap"]


def features(path):
    """Everything derivable from one run's training log, during training."""
    cfg = json.load(open(os.path.join(path, "config.json")))
    log = os.path.join(path, "training_log.csv")
    if not os.path.exists(log):
        return None
    try:
        t = pd.read_csv(log)
    except Exception:
        return None
    if "Epoch" not in t.columns or t.empty:
        return None
    cc = cfg["dataset_config"]["constrained_class"]
    cc = int(cc[0] if isinstance(cc, (list, tuple)) else cc)
    hard, lim = "Hard_Class%d" % cc, "Limit_Class%d" % cc
    if hard not in t.columns:
        return None
    warm = cfg["hyperparams"].get("warmup_epochs", 0)
    c = t[t.Epoch >= warm]
    if len(c) < 3:
        return None

    h = pd.to_numeric(c[hard], errors="coerce").dropna()
    K = pd.to_numeric(c[lim], errors="coerce").replace([np.inf], np.nan).dropna()
    K = float(K.iloc[0]) if len(K) else np.nan
    sat = pd.to_numeric(c.get("Global_Satisfied", 0), errors="coerce").fillna(0) > 0
    lam = pd.to_numeric(c.get("Lambda_Global", 0), errors="coerce").fillna(0)
    ce = pd.to_numeric(c.get("L_CE", np.nan), errors="coerce")
    acc = pd.to_numeric(c.get("Train_Acc", np.nan), errors="coerce")

    firsts = c.Epoch[sat.values]
    # how often satisfaction is won and then lost again
    flips = int((sat.astype(int).diff().fillna(0) != 0).sum())

    return {
        "path": path,
        "first_sat": float(firsts.iloc[0] - warm) if len(firsts) else np.nan,
        "sat_fraction": float(sat.mean()),
        "sat_flips": flips,
        "lam_max": float(lam.max()),
        "fill_final": float(h.iloc[-1]) / K if K == K and K else np.nan,
        "fill_min": float(h.min()) / K if K == K and K else np.nan,
        # volatility of the count, scaled by the cap so it compares across cells
        "count_cv": float(h.std() / K) if K == K and K else np.nan,
        "count_drop": float((h.iloc[0] - h.min()) / K) if K == K and K else np.nan,
        "ce_final": float(ce.dropna().iloc[-1]) if ce.notna().any() else np.nan,
        "ce_rise": (float(ce.dropna().iloc[-1] - ce.dropna().min())
                    if ce.notna().sum() > 1 else np.nan),
        "acc_final": float(acc.dropna().iloc[-1]) if acc.notna().any() else np.nan,
        "n_logged": len(c),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--warmup-min", type=int, default=50)
    ap.add_argument("--method", default="tralo")
    args = ap.parse_args()

    out = A.rows_for(args.root)
    if out.empty or "path" not in out.columns:
        print("rows_for returned nothing usable (needs the `path` column)")
        return 1
    out = out[(out.method == args.method) & (out.warmup >= args.warmup_min)]
    if out.empty:
        print("no %s runs at warmup >= %d" % (args.method, args.warmup_min))
        return 1

    feats = [f for f in (features(p) for p in out.path) if f]
    if not feats:
        print("no readable training logs")
        return 1
    d = out.merge(pd.DataFrame(feats), on="path", how="inner")
    print("%d %s runs, warm-up >= %d, under %s"
          % (len(d), args.method, args.warmup_min, args.root))

    cands = ["first_sat", "sat_fraction", "sat_flips", "lam_max", "fill_final",
             "fill_min", "count_cv", "count_drop", "ce_final", "ce_rise",
             "acc_final"]
    for m in ["ccF1eq", "AP"]:
        d["d_" + m] = d[m] - d.groupby(CELL)[m].transform("mean")

    print()
    print("=" * 94)
    print("WHAT PREDICTS A GOOD RUN?   spearman, ranked by |within-cell| on cc-F1")
    print("=" * 94)
    rows = []
    for f in cands:
        s = d[[f, "d_ccF1eq", "d_AP", "ccF1eq"]].dropna()
        if len(s) < 15 or s[f].nunique() < 3:
            rows.append((f, np.nan, np.nan, np.nan, len(s)))
            continue
        rows.append((f,
                     s[[f, "d_ccF1eq"]].corr(method="spearman").iloc[0, 1],
                     s[[f, "d_AP"]].corr(method="spearman").iloc[0, 1],
                     s[[f, "ccF1eq"]].corr(method="spearman").iloc[0, 1],
                     len(s)))
    rows.sort(key=lambda r: -abs(r[1]) if r[1] == r[1] else 0)
    print("%-14s %12s %12s %12s %6s" % ("feature", "cc-F1 (cell)", "AP (cell)", "cc-F1 (raw)", "n"))
    for f, a, b, c, n in rows:
        if a != a:
            print("%-14s %12s %12s %12s %6d   too little variation" % (f, "-", "-", "-", n))
        else:
            print("%-14s %+12.3f %+12.3f %+12.3f %6d" % (f, a, b, c, n))
    print()
    print("  cc-F1 (cell) is the honest column. Where |raw| is much larger than")
    print("  |cell|, the feature was tracking dataset identity, not run quality.")

    top = [r[0] for r in rows if r[1] == r[1]][:1]
    if top:
        f = top[0]
        print()
        print("=" * 94)
        print("TOP FEATURE %r, binned by quartile (deviation from own cell mean)" % f)
        print("=" * 94)
        s = d[[f, "d_ccF1eq", "d_AP"]].dropna().copy()
        try:
            s["q"] = pd.qcut(s[f], 4, duplicates="drop")
            print(s.groupby("q", observed=True).agg(
                n=(f, "size"), value=(f, "mean"),
                d_ccF1eq=("d_ccF1eq", "mean"), d_AP=("d_AP", "mean")).round(4).to_string())
        except ValueError:
            print("  not enough distinct values to quartile")
    return 0


if __name__ == "__main__":
    sys.exit(main())
