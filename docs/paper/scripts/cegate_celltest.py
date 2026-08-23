"""Cell-COUNTING test of "the CE gate is the only dynamic quantity that tracks
the dataset reversal".  Read-only.

Never averages across cells.  For each of the 12 atomic cells it asks:
  (a) what is the sign of tralo - best(duals) on ccF1eq,
  (b) which dual actually SETS that comparison,
  (c) did the CE gate fire for THAT dual, and
  (d) do other dynamic quantities from the same logs separate the cells at
      least as well.

    python paper/scripts/cegate_celltest.py
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.getcwd())
sys.path.insert(0, "paper/scripts")
import analyze_headroom as A  # noqa: E402

CELL = ["dataset", "model", "cap"]
DUALS = ["fioretto_ldf", "hounie_rcl"]
NOCE = "results/headroom/headroom_b30_lrc0.0001_noceskip"


def main():
    pd.set_option("display.width", 260)
    pd.set_option("display.max_columns", 60)

    gate = pd.read_csv("paper/scripts/out_refute_cegate_perrun.csv")
    tr = A.rows_for(NOCE)
    tr = tr[tr.method.isin(["tralo"] + DUALS)]

    rows = []
    for (ds, mo, cap), g in tr.groupby(CELL):
        piv = g.pivot_table(index="seed", columns="method", values="ccF1eq")
        have = [m for m in DUALS if m in piv.columns]
        best_per_seed = piv[have].idxmax(axis=1)
        delta = (piv["tralo"] - piv[have].max(axis=1)).mean()
        # the dual that sets the comparison in this cell (majority over seeds)
        setter = best_per_seed.value_counts().idxmax()
        gg = gate[(gate.dataset == ds) & (gate.model == mo) & (gate.cap == cap)]
        q = {}
        for m in DUALS:
            s = gg[gg.method == m]
            q[m + "_fired"] = int((s.n_ce_off > 0).sum())
            q[m + "_nceoff"] = s.n_ce_off.mean()
            q[m + "_eprun"] = s.epochs_run.mean()
        t = gg[gg.method == "tralo"]
        rows.append(dict(dataset=ds, model=mo, cap=cap, vBest=delta,
                         sign="WIN" if delta > 0 else "LOSS",
                         best_dual=setter,
                         setter_fired=q[setter + "_fired"],
                         setter_nceoff=q[setter + "_nceoff"],
                         hou_fired=q["hounie_rcl_fired"],
                         fio_fired=q["fioretto_ldf_fired"],
                         dual_eprun=np.mean([q["hounie_rcl_eprun"],
                                             q["fioretto_ldf_eprun"]]),
                         tralo_trainacc=t.train_acc_max.mean()))
    t = pd.DataFrame(rows).sort_values(["dataset", "model", "cap"])

    print("=" * 150)
    print("PER-CELL: outcome vs the CE-gate quantity (12 atomic cells, never pooled)")
    print("=" * 150)
    print(t.to_string(index=False, float_format=lambda x: "%.4f" % x))

    print()
    print("=" * 150)
    print("CELL-COUNTING TEST: does 'the gate fired' predict the sign of vBest?")
    print("=" * 150)

    def score(name, pred):
        ok = int((pred == (t["sign"] == "WIN")).sum())
        print("  %-46s correct in %2d/12 cells   (majority-class baseline 6/12)"
              % (name, ok))
        for ds, g in t.groupby("dataset"):
            p = pred[g.index]
            print("      %-12s correct %d/%d   (outcome %s)"
                  % (ds, int((p == (g["sign"] == "WIN")).sum()), len(g),
                     "/".join(g["sign"])))

    score("gate fired for the dual that SETS vBest", t.setter_fired > 0)
    score("gate fired for hounie_rcl (the cited arm)", t.hou_fired > 0)
    print()
    print("  COMPETING dynamic quantities from the SAME logs:")
    score("duals exited early (mean epochs_run < 29)", t.dual_eprun < 29)
    score("tralo's own train_acc_max >= 0.995", t.tralo_trainacc >= 0.995)

    print()
    print("=" * 150)
    print("WITHIN-DATASET VARIANCE of the gate quantity (can it explain the "
          "BACKBONE split that is 4 of the 12 cells?)")
    print("=" * 150)
    for ds, g in t.groupby("dataset"):
        print("  %-12s hounie seeds-fired per cell = %s   | outcomes = %s"
              % (ds, list(g.hou_fired), list(g.sign)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
