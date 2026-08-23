"""Does TraLO's advantage live in the CE-headroom regime? RUN ON THE SERVER.

The budget-equalized control killed the tight-cap constrained-class claim at
warm-up 50. The hypothesis under test here is that warm-up 50 is the wrong
regime to have measured in: the CE-saturation gate fires at train accuracy
0.995, so during the constraint phase nothing is learning any more and every
method can only re-threshold a frozen score vector. Re-thresholding optimally is
exactly what the post-hoc clipper does, so the clipper wins that regime by
construction and no trained method can do better than tie it.

At short warm-up the representation is still plastic, so the constraint term
shapes what is learned rather than only where the cut sits. If TraLO has a real
advantage, this is the only place it can live, and it has to show up in a
quantity the clipper cannot fake.

Reported per warm-up level, always paired within (campaign, dataset, model, cap,
seed) so no comparison crosses a campaign boundary:

  cc-F1 EQUALIZED  every arm filled to exactly K, so budget is not a free
                   variable. This is the metric the released numbers got wrong.
  macro-F1 EQ      overall quality at matched budget: what clipping's rewrites
                   cost the other classes.
  AP               allocation-free ranking on the constrained class. No budget,
                   no threshold. A clipper cannot improve it, so an edge here is
                   an edge in the model rather than in the allocation.

    python paper/scripts/headroom_test.py --eq ~/budget_equalized_full.csv
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

CELL = ["campaign", "dataset", "model", "cap", "seed"]
DUALS = ["fioretto_ldf", "hounie_rcl"]
CLIP = ["heuristic", "danits_lp"]


def add_ap(d, roots):
    """Attach allocation-free average precision by re-reading each run's stored
    probabilities. AP never picks a threshold or spends a budget, so it is the
    one number in this file that quota utilization cannot touch."""
    want = {}
    for _, r in d.iterrows():
        want[r.config_path] = None
    hit = 0
    for cfg_path in list(want):
        raw = os.path.join(os.path.dirname(cfg_path), "final_predictions_raw.csv")
        if not os.path.exists(raw):
            continue
        try:
            t = pd.read_csv(raw)
            cols = sorted((c for c in t.columns if c.startswith("Prob_Class_")),
                          key=lambda c: int(c.rsplit("_", 1)[1]))
            y = t["True_Label"].to_numpy(int)
            cls = int(d.loc[d.config_path == cfg_path, "cls"].iloc[0])
            want[cfg_path] = average_precision_score((y == cls).astype(int),
                                                     t[cols[cls]].to_numpy(float))
            hit += 1
        except Exception:
            continue
    d["ap"] = d.config_path.map(want)
    print("   AP attached for %d/%d runs" % (hit, len(want)), file=sys.stderr)
    return d


def paired(d, metric, ref_methods, label):
    """Paired TraLO-minus-reference within each matched cell.

    The reference is the best of `ref_methods` in that same cell, which is the
    comparator the paper already uses: it gives the baselines their best shot
    per cell rather than picking one arm globally.
    """
    piv = d.pivot_table(index=CELL, columns="method", values=metric)
    have = [m for m in ref_methods if m in piv.columns]
    if "tralo" not in piv.columns or not have:
        return None
    sub = piv.dropna(subset=["tralo"]).copy()
    sub["ref"] = sub[have].max(axis=1)
    sub = sub.dropna(subset=["ref"])
    if sub.empty:
        return None
    g = sub["tralo"] - sub["ref"]
    return {"cmp": label, "n": len(g), "mean": g.mean(),
            "win": int((g > 0).sum()), "median": g.median()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eq", default=os.path.expanduser("~/budget_equalized_full.csv"))
    ap.add_argument("--roots", nargs="*", default=["results/pending_runs",
                                                   "results/track_b", "results/r4"])
    ap.add_argument("--datasets", nargs="*",
                    default=["octmnist", "dermmnist", "tissuemnist"])
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    d = pd.read_csv(args.eq)
    d = d[d.dataset.isin(args.datasets)]
    # b1 sweeps warm-up as its own variable; keeping it is the point here, but
    # its imbalanced-baseline arms are a different comparison and are dropped.
    d = d[~d.method.isin(["focal", "class_balanced", "logit_adjust"])]
    print("rows after filter: %d" % len(d), file=sys.stderr)

    d = add_ap(d, args.roots)

    rows = []
    for w, grp in d.groupby("warmup"):
        n_tralo = int((grp.method == "tralo").sum())
        if n_tralo < 4:
            continue
        for metric, mlabel in [("cc_f1_equalized", "cc-F1 eq"),
                               ("f1_macro_equalized", "macro-F1 eq"),
                               ("ap", "AP")]:
            g = grp.dropna(subset=[metric])
            for ref, rlabel in [(DUALS, "vs best dual"), (CLIP, "vs clipper")]:
                r = paired(g, metric, ref, rlabel)
                if r:
                    r.update({"warmup": w, "metric": mlabel})
                    rows.append(r)

    o = pd.DataFrame(rows)
    if o.empty:
        print("no paired cells found")
        return 1
    for metric in ["cc-F1 eq", "macro-F1 eq", "AP"]:
        print("\n" + "=" * 74)
        print("%s   (paired within campaign/dataset/model/cap/seed)" % metric)
        print("=" * 74)
        t = o[o.metric == metric].pivot_table(index="warmup", columns="cmp",
                                              values=["mean", "win", "n"])
        print(t.round(4).to_string())

    if args.out:
        o.to_csv(args.out, index=False)
        print("\nwrote %s" % args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
