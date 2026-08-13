"""Do the six headline cells survive at n=10 seeds?

With four seeds a within-cell paired Wilcoxon floors at p=0.125, so no headline
cell can reach individual significance whatever the effect size. The r2 campaign
adds seeds 5-10 for all seven methods on the same six OctMNIST tight-cap cells.

ON POOLING ACROSS CAMPAIGNS. Seeds 1-4 are the frozen `paper_final` sweep; seeds
5-10 are a new campaign with fresh warmups. Absolute metrics are NOT comparable
across campaigns -- re-running one configuration elsewhere moves cc-F1 by ~0.025.
PAIRED DIFFERENCES are, because a campaign-level shift that moves every method in
a cell equally cancels in the difference. That is an assumption, not a
guarantee, so this script does not assume it: it reports seeds 1-4 and 5-10
separately, checks they agree, and only then pools.

Run:  python paper/scripts/analyze_seeds10.py
"""
import os

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
CORPUS = os.path.join(ROOT, "data", "corpus", "corpus_final.csv")
R2 = os.path.join(ROOT, "data", "corpus", "r2_seeds10.csv")

CELL = ["dataset", "model", "constraint_tag"]
TIGHT = ["L30_G30", "L40_G40"]
DUALS = ["fioretto_ldf", "hounie_rcl"]


def paired(df, metric="cc_f1"):
    """TraLO minus the per-seed best dual, one row per (cell, seed)."""
    p = df.pivot_table(index=CELL + ["seed"], columns="method", values=metric)
    have = [d for d in DUALS if d in p.columns]
    p = p.dropna(subset=["tralo"] + have)
    p["gap"] = p["tralo"] - p[have].max(axis=1)
    return p.reset_index()[CELL + ["seed", "gap"]]


def summarise(g, label):
    rows = []
    for key, s in g.groupby(CELL):
        v = s.gap.to_numpy(float)
        if len(v) < 2:
            continue
        try:
            p = wilcoxon(v).pvalue
        except ValueError:            # all-zero differences
            p = 1.0
        rows.append(dict(cell="%s/%s/%s" % key, n=len(v), mean=v.mean(),
                         wins=int((v > 0).sum()), p=p))
    out = pd.DataFrame(rows)
    print("\n--- %s ---" % label)
    if out.empty:
        print("  no complete cells yet")
        return out
    print(out.round(4).to_string(index=False))
    print("  cells with p < 0.05: %d of %d" % ((out.p < 0.05).sum(), len(out)))
    return out


def main():
    old = pd.read_csv(CORPUS)
    old = old[(old.sweep == "paper_final") & (old.dataset == "octmnist")
              & old.constraint_tag.isin(TIGHT)]
    a = paired(old)

    if not os.path.exists(R2):
        print("r2 results not pulled yet (%s missing)" % R2)
        summarise(a, "seeds 1-4 (paper_final)")
        return
    new = pd.read_csv(R2)
    new = new[new.constraint_tag.isin(TIGHT)]
    b = paired(new)

    sa = summarise(a, "seeds 1-4 (paper_final)")
    sb = summarise(b, "seeds 5-10 (r2 campaign)")

    if not sa.empty and not sb.empty:
        m = sa.merge(sb, on="cell", suffixes=("_old", "_new"))
        d = (m.mean_new - m.mean_old).abs()
        print("\nCONSISTENCY CHECK (do the two campaigns agree on the paired gap?)")
        print(m[["cell", "mean_old", "mean_new"]].round(4).to_string(index=False))
        print("  max |difference| between campaigns: %.4f" % d.max())
        if d.max() > 0.02:
            print("  WARNING: campaigns disagree by more than the drift we expect.")
            print("  Report them separately; do NOT pool.")
            return
        print("  within expected drift -- pooling is defensible")

    summarise(pd.concat([a, b], ignore_index=True), "POOLED n=10")


if __name__ == "__main__":
    main()
