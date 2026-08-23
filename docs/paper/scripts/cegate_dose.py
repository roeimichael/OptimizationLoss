"""Dose-response test for the CE-gate mechanism.  Read-only.

If losing CE gradient is what hands TraLO the DermMNIST cells, then WITHIN
DermMNIST the runs that lost the most CE epochs should be the runs TraLO beats
by the most.  Paired per seed against the dual that actually sets the
comparison (fioretto_ldf in all four Derm cells).

    python paper/scripts/cegate_dose.py
"""
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, os.getcwd())
sys.path.insert(0, "paper/scripts")
import analyze_headroom as A  # noqa: E402

NOCE = "results/headroom/headroom_b30_lrc0.0001_noceskip"
CELL = ["dataset", "model", "cap"]


def main():
    pd.set_option("display.width", 220)
    gate = pd.read_csv("paper/scripts/out_refute_cegate_perrun.csv")
    d = A.rows_for(NOCE)

    out = []
    for m in ["fioretto_ldf", "hounie_rcl"]:
        piv = d[d.method.isin(["tralo", m])].pivot_table(
            index=CELL + ["seed"], columns="method", values="ccF1eq").reset_index()
        g = gate[gate.method == m][CELL + ["seed", "n_ce_off", "epochs_run"]]
        j = piv.merge(g, on=CELL + ["seed"])
        j["gap"] = j["tralo"] - j[m]
        j["dual"] = m
        out.append(j[CELL + ["seed", "dual", "n_ce_off", "epochs_run", "gap"]])
    j = pd.concat(out, ignore_index=True)

    for ds in ["dermmnist", "octmnist"]:
        s = j[(j.dataset == ds)]
        print("=" * 110)
        print("%s -- per-seed CE-off epochs vs the paired TraLO gap" % ds)
        print("=" * 110)
        for m, gm in s.groupby("dual"):
            if gm.n_ce_off.nunique() < 2:
                print("  %-13s n_ce_off is constant (%s) -- no dose to respond to"
                      % (m, sorted(gm.n_ce_off.unique())))
                continue
            r, p = spearmanr(gm.n_ce_off, gm.gap)
            print("  %-13s n=%d  spearman(n_ce_off, gap) = %+0.3f  p=%.3f   "
                  "[range of dose %d-%d epochs]"
                  % (m, len(gm), r, p, gm.n_ce_off.min(), gm.n_ce_off.max()))
            # within-cell, so the correlation cannot ride on cell identity
            for c, gc in gm.groupby(CELL):
                if gc.n_ce_off.nunique() > 1:
                    rr, _ = spearmanr(gc.n_ce_off, gc.gap)
                    print("        %-38s n=%d rho=%+0.2f  dose %s  gap %s"
                          % ("/".join(c[1:]), len(gc), rr, list(gc.n_ce_off),
                             [round(x, 3) for x in gc.gap]))
        print()

    print("=" * 110)
    print("EXTREME runs: the two Derm fioretto seeds that never satisfied and "
          "lost 15-16 CE epochs")
    print("=" * 110)
    ex = j[(j.dataset == "dermmnist") & (j.dual == "fioretto_ldf")]
    print(ex.sort_values("n_ce_off", ascending=False)
          .to_string(index=False, float_format=lambda x: "%.4f" % x))
    return 0


if __name__ == "__main__":
    sys.exit(main())
