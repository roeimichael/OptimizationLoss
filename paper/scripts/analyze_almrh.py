"""Does the reset+hinge graft transfer to ALM as it did to the other duals?

Blind review round 1's sharpest structural point: the paper argues the optimizer
reset and the undershoot hinge carry the constrained-class effect, and ALM -- the
comparator it calls strongest -- is the one baseline that never received them.
Beating an un-upgraded opponent proves less than it appears to.

This measures three things on the six OctMNIST tight-cap cells:

  1. graft lift        alm_rh - alm          does it help the host at all?
  2. residual gap      tralo  - alm_rh       is anything left after grafting?
  3. recovery          fraction of TraLO's margin over ALM the graft closes

All arms descend from the same frozen paper_final configs and share warmup
caches, so pairing is by (dataset, model, cap, seed) with no cross-campaign
mixing -- see extract_alm_results.py for why that matters.

Run:  python paper/scripts/analyze_almrh.py
"""
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
CORPUS = os.path.join(ROOT, "data", "corpus", "corpus_final.csv")
ALM = os.path.join(ROOT, "data", "corpus", "alm_results.csv")

CELLKEY = ["dataset", "model", "constraint_tag"]
KEY = CELLKEY + ["seed"]
TIGHT = ["L30_G30", "L40_G40"]


def load(metric="cc_f1"):
    a = pd.read_csv(ALM)
    alm = a[a.method == "fioretto_alm"][KEY + [metric]].rename(columns={metric: "alm"})
    rh = a[a.method == "alm_rh"][KEY + [metric]].rename(columns={metric: "alm_rh"})

    tr = pd.read_csv(CORPUS)
    tr = tr[(tr.sweep == "paper_final") & (tr.method == "tralo")]
    tr = tr[KEY + [metric]].rename(columns={metric: "tralo"})

    df = rh.merge(alm, on=KEY).merge(tr, on=KEY)
    return df[df.constraint_tag.isin(TIGHT)]


def main():
    df = load()
    if df.empty:
        print("no alm_rh runs paired yet -- campaign still in flight")
        return
    n_cells = df.groupby(CELLKEY).seed.nunique()
    print("paired runs: %d over %d cells (%d complete at 4 seeds)"
          % (len(df), len(n_cells), (n_cells == 4).sum()))
    if (n_cells < 4).any():
        print("INCOMPLETE cells present -- treat the numbers below as provisional")

    df = df.copy()
    df["lift"] = df.alm_rh - df.alm          # graft helps the host?
    df["resid"] = df.tralo - df.alm_rh       # anything left for TraLO?
    df["margin"] = df.tralo - df.alm         # what there was to close

    cell = df.groupby(CELLKEY).agg(
        lift=("lift", "mean"), resid=("resid", "mean"), margin=("margin", "mean"),
        n=("lift", "size"),
        resid_wins=("resid", lambda s: int((s > 0).sum())))
    print()
    print(cell.round(4).to_string())

    print("\nGRAFT LIFT (alm_rh - ALM): mean %+.4f, helps in %d of %d cells"
          % (cell.lift.mean(), (cell.lift > 0).sum(), len(cell)))
    print("RESIDUAL   (TraLO - alm_rh): mean %+.4f, TraLO leads %d of %d cells"
          % (cell.resid.mean(), (cell.resid > 0.005).sum(), len(cell)))
    print("           seed-level: TraLO wins %d of %d paired comparisons"
          % (int((df.resid > 0).sum()), len(df)))

    # Recovery: how much of TraLO's margin over the host the graft closes.
    # Guard the degenerate case where the host was already ahead.
    ok = cell.margin.abs() > 1e-9
    rec = (cell.lift[ok] / cell.margin[ok]).median()
    print("RECOVERY   median %.0f%% of TraLO's margin over ALM closed by the graft"
          % (100 * rec))

    tied = cell.resid.abs() <= 0.005
    print("\nVERDICT: after grafting, %d of %d cells are within the 0.005 tie band"
          % (tied.sum(), len(cell)))


if __name__ == "__main__":
    main()
