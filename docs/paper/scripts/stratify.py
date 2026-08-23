"""Never pool. One row per (dataset, backbone, cap) -- the atomic cell.

compare_all.py reports one number per dataset, which averages L30 and L50 across
two backbones: sixteen paired comparisons collapsed into a single figure. That
hides exactly what we need to find -- the regions where the method wins.

Also reports whether the cap BINDS, because a comparison in a cell where the
unconstrained model already satisfies the cap is not a test of anything: every
method converges to the same predictions and the ordering is noise.

    python paper/scripts/stratify.py --trained results/headroom/<campaign>
"""
import argparse
import os
import sys

import pandas as pd

sys.path.insert(0, os.getcwd())
sys.path.insert(0, "paper/scripts")
import analyze_headroom as A  # noqa: E402

CELL = ["dataset", "model", "cap"]
TRAINED = ["tralo", "fioretto_ldf", "hounie_rcl"]
CLIP = ["heuristic", "danits_lp"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trained", required=True)
    ap.add_argument("--clip", default="results/headroom/headroom_b30")
    ap.add_argument("--metric", default="ccF1eq")
    args = ap.parse_args()

    tr = A.rows_for(args.trained)
    tr = tr[tr.method.isin(TRAINED)]
    cl = A.rows_for(args.clip)
    cl = cl[cl.method.isin(CLIP)]
    d = pd.concat([tr, cl], ignore_index=True)
    if d.empty:
        print("no runs")
        return 1

    M = args.metric
    print("=" * 100)
    print("PER-CELL (never pooled).  campaign=%s  metric=%s" % (args.trained, M))
    print("=" * 100)

    rows = []
    for (ds, mo, cap), g in d.groupby(CELL):
        piv = g.pivot_table(index="seed", columns="method", values=M)
        raw = g.pivot_table(index="seed", columns="method", values="count_raw")
        K = float(g["K"].iloc[0])
        r = {"dataset": ds, "model": mo, "cap": cap, "K": K, "n": len(piv)}
        # Does the cap bind? The clipper's raw count is the unconstrained
        # model's own rate: if it is already <= K, nothing is being constrained.
        if CLIP[0] in raw.columns:
            r["clip_raw"] = raw[CLIP[0]].mean()
            r["binds"] = "yes" if r["clip_raw"] > K else "NO"
        for m in TRAINED + CLIP:
            r[m] = piv[m].mean() if m in piv.columns else float("nan")
            if m in raw.columns:
                r[m + "_raw"] = raw[m].mean()
        if "tralo" in piv.columns:
            for lbl, ref in [("vBest", TRAINED[1:]), ("vClip", CLIP)]:
                have = [x for x in ref if x in piv.columns]
                if have:
                    delta = piv["tralo"] - piv[have].max(axis=1)
                    r[lbl] = delta.mean()
                    r[lbl + "_w"] = int((delta > 0).sum())
        rows.append(r)

    t = pd.DataFrame(rows).sort_values(["dataset", "cap", "model"])
    show = ["dataset", "model", "cap", "K", "clip_raw", "binds",
            "tralo", "fioretto_ldf", "hounie_rcl", "heuristic",
            "vBest", "vBest_w", "vClip", "vClip_w", "n"]
    show = [c for c in show if c in t.columns]
    print(t[show].to_string(index=False, float_format=lambda x: "%.4f" % x))

    print()
    print("=" * 100)
    print("COUNT THE CELLS, do not average them")
    print("=" * 100)
    for lbl in ["vBest", "vClip"]:
        if lbl not in t.columns:
            continue
        w = int((t[lbl] > 0.005).sum())
        l = int((t[lbl] < -0.005).sum())
        print("  tralo %-6s  WIN %d cells   LOSS %d cells   TIE %d cells   (of %d)"
              % (lbl, w, l, len(t) - w - l, len(t)))

    print()
    print("=" * 100)
    print("SPLIT BY CAP (is the effect a tight-cap effect?)")
    print("=" * 100)
    for cap, g in t.groupby("cap"):
        line = "  %-10s" % cap
        for lbl in ["vBest", "vClip"]:
            if lbl in g.columns:
                line += "  %s %+0.4f (%d/%d cells >0)" % (lbl, g[lbl].mean(),
                                                          int((g[lbl] > 0).sum()), len(g))
        print(line)

    print()
    print("=" * 100)
    print("SPLIT BY BACKBONE")
    print("=" * 100)
    for mo, g in t.groupby("model"):
        line = "  %-14s" % mo
        for lbl in ["vBest", "vClip"]:
            if lbl in g.columns:
                line += "  %s %+0.4f (%d/%d cells >0)" % (lbl, g[lbl].mean(),
                                                          int((g[lbl] > 0).sum()), len(g))
        print(line)
    return 0


if __name__ == "__main__":
    sys.exit(main())
