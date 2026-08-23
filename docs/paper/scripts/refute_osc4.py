"""Part 4: the causal payload only.
  'This is why oct/tissue never converge and derm overshoots.'
Tests:
  (a) oct @ lrc1e-4 natural experiment (8 gated / 8 not) -- Fisher exact on convergence
  (b) is tissue's 'satisfaction' real or an epoch-0 transient?
  (c) does the CE-ON phase already do the work? where is the count when the gate fires
      relative to K, and where does satisfaction happen?
  (d) oct @ lrc5e-5 -- same transition, does it converge / overshoot?
"""
import glob
import json
import os
import sys
from math import comb

import numpy as np
import pandas as pd

sys.path.insert(0, os.getcwd())
sys.path.insert(0, "paper/scripts")
import hounie_dyn as H  # noqa: E402

ROOTS = {"lrc1e-4": "results/headroom/headroom_b30_lrc0.0001_noceskip",
         "lrc5e-5": "results/headroom/headroom_b30_lrc5e-05"}


def fisher(a, b, c, d):
    """two-sided Fisher exact p for [[a,b],[c,d]]"""
    n = a + b + c + d
    r1, c1 = a + b, a + c
    def pr(x):
        return comb(r1, x) * comb(n - r1, c1 - x) / comb(n, c1)
    p0 = pr(a)
    lo = max(0, c1 - (n - r1))
    hi = min(r1, c1)
    return sum(pr(x) for x in range(lo, hi + 1) if pr(x) <= p0 + 1e-12)


def main():
    rows = []
    for tag, root in ROOTS.items():
        for p in sorted(glob.glob(root + "/**/config.json", recursive=True)):
            cfg = json.load(open(p))
            if cfg.get("methodology") != "hounie_rcl":
                continue
            r = H.load_run(p)
            if r is None:
                continue
            T = H.hounie_traj(r)
            log = r["log"]
            ep = log["epoch"].to_numpy(int)
            ce = log["ce_loss"].to_numpy(float)
            exc = log["total_excess"].to_numpy(float)
            sat = log["all_satisfied"].to_numpy(int)
            soft = T["soft"]
            off = np.isnan(ce)
            gate = int(ep[off][0]) if off.any() else None
            d = dict(campaign=tag, dataset=r["dataset"], model=r["model"], cap=r["cap"],
                     seed=r["seed"], K=r["K"], gated=int(off.any()), gate_ep=gate,
                     ever_sat=int(sat.any()),
                     first_sat=(int(ep[sat == 1][0]) if sat.any() else np.nan),
                     ends_sat=int(exc[-1] == 0), exc_final=float(exc[-1]),
                     n_sat=int(sat.sum()), raw_count=r["raw_count"],
                     soft_final=float(soft[-1]))
            if gate is not None:
                m = ep == gate
                d["soft_at_gate"] = float(soft[m][0])
                d["exc_at_gate"] = float(exc[m][0])
                d["sat_before_gate"] = int(sat[ep < gate].any())
                d["sat_after_gate"] = int(sat[ep >= gate].any())
            rows.append(d)
    d = pd.DataFrame(rows)
    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 40)
    f = lambda x: "%.3f" % x  # noqa: E731

    print("=" * 112)
    print("(a) octmnist @ lrc1e-4 NATURAL EXPERIMENT: 8 runs gated, 8 not, matched cells")
    print("=" * 112)
    o = d[(d.campaign == "lrc1e-4") & (d.dataset == "octmnist")]
    for lbl in ["ever_sat", "ends_sat"]:
        a = int(o[(o.gated == 1)][lbl].sum()); b = 8 - a
        c = int(o[(o.gated == 0)][lbl].sum()); e = 8 - c
        print("  %-9s : gated %d/8   not-gated %d/8   Fisher two-sided p = %.3f"
              % (lbl, a, c, fisher(a, b, c, e)))
    print("  mean final excess : gated %.1f   not-gated %.1f"
          % (o[o.gated == 1].exc_final.mean(), o[o.gated == 0].exc_final.mean()))
    print("  mean raw count    : gated %.1f   not-gated %.1f   (K = %.1f)"
          % (o[o.gated == 1].raw_count.mean(), o[o.gated == 0].raw_count.mean(), o.K.mean()))

    print()
    print("=" * 112)
    print("(b) IS TISSUE'S 'SATISFACTION' REAL?  per-run first_sat / n_sat / final excess")
    print("=" * 112)
    t = d[(d.dataset == "tissuemnist")]
    print(t[t.ever_sat == 1][["campaign", "model", "cap", "seed", "first_sat", "n_sat",
                              "exc_final", "raw_count", "K"]]
          .to_string(index=False, float_format=f))
    print("  tissue runs that END satisfied: %d / %d" % (int(t.ends_sat.sum()), len(t)))

    print()
    print("=" * 112)
    print("(c) WHERE IS THE COUNT WHEN THE GATE FIRES?  (gated runs)")
    print("    if the CE-ON phase already brought it to the cap, the CE-off descent is overshoot")
    print("=" * 112)
    g = d[d.gated == 1]
    print(g.groupby(["campaign", "dataset"]).agg(
        runs=("seed", "size"), K=("K", "mean"), gate_ep=("gate_ep", "mean"),
        soft_at_gate=("soft_at_gate", "mean"), exc_at_gate=("exc_at_gate", "mean"),
        first_sat=("first_sat", "mean"), soft_final=("soft_final", "mean"),
        raw_count=("raw_count", "mean"),
        sat_before_gate=("sat_before_gate", "sum"),
        sat_after_gate=("sat_after_gate", "sum")).to_string(float_format=f))

    print()
    print("=" * 112)
    print("(d) THE SAME TRANSITION IN OCT @ lrc5e-5 (15/16 gated, ratio_off = 1.000)")
    print("=" * 112)
    print(d.groupby(["campaign", "dataset"]).agg(
        runs=("seed", "size"), gated=("gated", "sum"), ever_sat=("ever_sat", "sum"),
        ends_sat=("ends_sat", "sum"), K=("K", "mean"), raw_count=("raw_count", "mean"),
        exc_final=("exc_final", "mean"), soft_final=("soft_final", "mean"))
        .to_string(float_format=f))
    print()
    print("  'overshoot' = final raw count strictly below the cap:")
    d["undershoots_cap"] = (d.raw_count < d.K).astype(int)
    print(d.groupby(["campaign", "dataset"]).agg(
        runs=("seed", "size"), gated=("gated", "sum"),
        runs_below_cap=("undershoots_cap", "sum")).to_string(float_format=f))
    d.to_csv("paper/scripts/out_refute_osc4.csv", index=False)
    print("\nwrote paper/scripts/out_refute_osc4.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
