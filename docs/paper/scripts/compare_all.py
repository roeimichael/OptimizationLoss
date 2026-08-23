"""Full pairwise comparison: TraLO against EVERY baseline, not just the clipper.

The claim the paper needs is an ORDERING -- TraLO above the post-hoc clippers
and above the dual methods -- so a comparison against one family is not enough.
Reporting only "beats the clipper" hides the case where a dual is quietly above
TraLO, which is exactly what happens on OctMNIST.

Two comparators are reported for the duals and they answer different questions,
so both are shown rather than whichever is kinder:

  vs EACH method     TraLO against fioretto_ldf and against hounie_rcl
                     separately. This is the ordering question: is TraLO on top?
  vs BEST-OF         TraLO against max(duals) chosen per cell. This is the
                     paper's existing adjudication rule and it is deliberately
                     harsh -- it lets the baselines pick their best arm in every
                     cell, which no single deployed method could do.

A method can beat both duals individually on the mean and still lose to the
per-cell best-of-two. That is not a contradiction; it is what the two questions
mean, and conflating them is how a comparison becomes misleading.

COLLAPSE FLAG. A verified defect in the checkpoint selector (one-sided
satisfaction at train.py:216, plus an unconditional overwrite at :371) lets a
model that predicts the constrained class for almost nobody be recorded as
"satisfied" and shipped. Runs whose realized count fell below K/3 are counted
and reported separately rather than silently averaged in, because their scores
measure the selector rather than the objective.

    python paper/scripts/compare_all.py --trained results/headroom/headroom_b30_lrc0.0001
    python paper/scripts/compare_all.py --trained results/headroom/headroom_b30_lrc0.0001_noceskip
"""
import argparse
import os
import sys

import pandas as pd

sys.path.insert(0, os.getcwd())
sys.path.insert(0, "paper/scripts")
import analyze_headroom as A  # noqa: E402

CELL = ["dataset", "cap", "model", "seed"]
TRAINED = ["tralo", "fioretto_ldf", "hounie_rcl"]
CLIP = ["heuristic", "danits_lp"]
METRICS = [("ccF1eq", "PRIMARY -- constrained-class F1 at equal budget"),
           ("AP", "diagnostic -- allocation-free ranking"),
           ("macroEq", "guard -- overall quality at equal budget")]


def load(trained_root, clip_root):
    tr = A.rows_for(trained_root)
    tr = tr[tr.method.isin(TRAINED)]
    cl = A.rows_for(clip_root)
    cl = cl[cl.method.isin(CLIP)]
    return pd.concat([tr, cl], ignore_index=True)


def pair_vs(d, metric, ref_methods, best_of):
    """TraLO minus a reference, paired within cell.

    best_of=False compares against a single named method. best_of=True lets the
    reference take its best arm in each cell.
    """
    piv = d.pivot_table(index=CELL, columns="method", values=metric)
    have = [m for m in ref_methods if m in piv.columns]
    if "tralo" not in piv.columns or not have:
        return None
    s = piv.dropna(subset=["tralo"]).copy()
    s["ref"] = s[have].max(axis=1) if best_of else s[have[0]]
    s = s.dropna(subset=["ref"])
    if s.empty:
        return None
    g = s["tralo"] - s["ref"]
    return {"mean": float(g.mean()), "won": int((g > 0).sum()), "n": len(g)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trained", required=True)
    ap.add_argument("--clip", default="results/headroom/headroom_b30")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    d = load(args.trained, args.clip)
    if d.empty:
        print("no runs")
        return 1
    # Collapse must be measured on the model's OWN count, before post-hoc
    # adjustment. Post-hoc fills up to K, so the released `count` sits at ~K
    # even for a network that has stopped predicting the class entirely.
    d["collapsed"] = d["count_raw"] < (d["K"] / 3.0)

    print("=" * 78)
    print("TRAINED: %s" % args.trained)
    print("CLIPPER: %s   (unaffected by the LR trap: constraint_epochs=0)" % args.clip)
    print("=" * 78)
    nc = int(d.collapsed.sum())
    print("\nruns with a COLLAPSED constrained-class count (< K/3): %d of %d" % (nc, len(d)))
    if nc:
        print(d[d.collapsed].groupby(["dataset", "method"]).size().to_string())
        print("  -> these measure the checkpoint-selector defect, not the objective.")

    for ds, g in d.groupby("dataset"):
        print("\n" + "-" * 78)
        print("%s   (%d runs, %d collapsed)" % (ds, len(g), int(g.collapsed.sum())))
        print("-" * 78)
        t = (g.groupby("method")[["ccF1eq", "AP", "macroEq",
                                  "count_raw", "count", "K"]]
             .mean().reindex(TRAINED + CLIP).round(4))
        # Rank on the primary metric so the ordering is visible at a glance.
        # Ties produce half-ranks, so this stays float rather than Int64.
        t["rank"] = t["ccF1eq"].rank(ascending=False)
        print(t.to_string())

        order = t["ccF1eq"].dropna().sort_values(ascending=False)
        if len(order):
            print("  ordering on ccF1eq: " + "  >  ".join(order.index))

        for metric, note in METRICS:
            print("\n  %s   [%s]" % (metric, note))
            for label, refs, best in [
                    ("vs fioretto_ldf", ["fioretto_ldf"], False),
                    ("vs hounie_rcl", ["hounie_rcl"], False),
                    ("vs BEST dual (per cell)", ["fioretto_ldf", "hounie_rcl"], True),
                    ("vs clipper", CLIP, True)]:
                r = pair_vs(g, metric, refs, best)
                if r:
                    flag = "WIN " if r["mean"] > 0.005 else (
                           "loss" if r["mean"] < -0.005 else "tie ")
                    print("    %-26s %s %+0.4f   %d/%d seeds"
                          % (label, flag, r["mean"], r["won"], r["n"]))

    if args.out:
        d.to_csv(args.out, index=False)
        print("\nwrote %s" % args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
