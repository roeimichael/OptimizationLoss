"""Within-CELL test of the CE-gate mechanism for hounie_rcl.

The between-dataset split is confounded (on derm the CE gate fires in 16/16
runs, on tissue in 0/16).  The half-constraint-LR campaign breaks the confound:
there the gate fires on OCTMNIST too, and it fires for some seeds and not
others inside the SAME (dataset, backbone, cap).  So the number of
constraint-only epochs can be regressed against the outcome with dataset,
backbone, cap and learning rate all held fixed.

    python paper/scripts/hounie_within.py
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.getcwd())
sys.path.insert(0, "paper/scripts")
import analyze_headroom as A   # noqa: E402

CELL = ["dataset", "model", "cap"]
CAMPS = {"lrc1e-4": ("results/headroom/headroom_b30_lrc0.0001_noceskip",
                     "paper/scripts/out_hounie_dyn.csv"),
         "lrc5e-5": ("results/headroom/headroom_b30_lrc5e-05",
                     "paper/scripts/out_dyn_headroom_b30_lrc5e-05.csv"),
         "fullbudget": ("results/headroom/headroom_b30_lrc0.0001_noceskip_full",
                        "paper/scripts/out_dyn_headroom_b30_lrc0.0001_noceskip_full.csv")}


def spear(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 3 or len(set(a[m])) < 2 or len(set(b[m])) < 2:
        return np.nan
    return float(np.corrcoef(pd.Series(a[m]).rank(), pd.Series(b[m]).rank())[0, 1])


def load(tag):
    root, dynp = CAMPS[tag]
    dyn = pd.read_csv(dynp)
    dyn = dyn[dyn.method == "hounie_rcl"].copy()
    sc = A.rows_for(root)
    sc = sc[sc.method == "hounie_rcl"]
    d = dyn.merge(sc[CELL + ["seed", "ccF1eq", "AP", "macroEq", "count_raw"]],
                  on=CELL + ["seed"], how="inner")
    d["util"] = d["count_raw"] / d["K"]
    d["campaign"] = tag
    return d


def main():
    pd.set_option("display.width", 240)
    pd.set_option("display.max_columns", 40)
    all_d = pd.concat([load(t) for t in CAMPS], ignore_index=True)

    print("=" * 112)
    print("A. WITHIN-CELL: does seed-to-seed variation in constraint-only epochs")
    print("   predict budget utilisation and ranking quality?  (4 seeds per row)")
    print("=" * 112)
    rows = []
    for (camp, ds, mo, cap), g in all_d.groupby(["campaign"] + CELL):
        rows.append(dict(campaign=camp, dataset=ds, model=mo, cap=cap, n=len(g),
                         n_ce_off=list(np.sort(g.n_ce_off.values))[::-1],
                         rho_util=spear(g.n_ce_off, g.util),
                         rho_AP=spear(g.n_ce_off, g.AP),
                         rho_F1=spear(g.n_ce_off, g.ccF1eq),
                         mean_util=g.util.mean(), mean_AP=g.AP.mean()))
    t = pd.DataFrame(rows)
    t = t[t.n_ce_off.apply(lambda v: len(set(v)) > 1)]
    print(t.to_string(index=False, float_format=lambda x: "%.3f" % x))
    for c in ["rho_util", "rho_AP", "rho_F1"]:
        v = t[c].dropna()
        print("  %-9s  median %+0.2f   negative in %d of %d cells with variation"
              % (c, v.median(), int((v < 0).sum()), len(v)))

    print("\n" + "=" * 112)
    print("B. OCTMNIST ONLY, across the two LRs: the gate flips, the verdict flips")
    print("=" * 112)
    o = all_d[all_d.dataset == "octmnist"]
    print(o.groupby(["campaign"] + CELL).agg(
        n=("seed", "size"), n_ce_off=("n_ce_off", "mean"),
        seeds_gate_fired=("n_ce_off", lambda s: int((s >= 4).sum())),
        first_sat=("first_sat", "mean"), lam_max=("lam_max", "mean"),
        count_raw=("count_raw", "mean"), util=("util", "mean"),
        AP=("AP", "mean"), ccF1eq=("ccF1eq", "mean")
    ).to_string(float_format=lambda x: "%.3f" % x))

    print("\n" + "=" * 112)
    print("C. DERMMNIST, early-stop vs full budget: MORE constraint-only epochs,")
    print("   DEEPER collapse (dose-response, cells never pooled)")
    print("=" * 112)
    dm = all_d[(all_d.dataset == "dermmnist")
               & all_d.campaign.isin(["lrc1e-4", "fullbudget"])]
    print(dm.groupby(CELL + ["campaign"]).agg(
        n=("seed", "size"), epochs=("epochs", "mean"),
        n_ce_off=("n_ce_off", "mean"), first_sat=("first_sat", "mean"),
        soft_drop_after_sat=("soft_drop_after_sat", "mean"),
        soft_final_over_K=("soft_final_over_K", "mean"),
        count_raw=("count_raw", "mean"), util=("util", "mean"),
        AP=("AP", "mean"), ccF1eq=("ccF1eq", "mean")
    ).to_string(float_format=lambda x: "%.3f" % x))

    print("\n" + "=" * 112)
    print("D. IS IT THE DUAL STEP?  peak lambda and the resilience slack, by dataset")
    print("   N*u = how many extra predictions RCL's resilience term buys")
    print("=" * 112)
    a = all_d[all_d.campaign == "lrc1e-4"].copy()
    a["Nu"] = a["N"] * a["u_max"]
    print(a.groupby(CELL).agg(
        N=("N", "mean"), K=("K", "mean"), lam_max=("lam_max", "mean"),
        u_max=("u_max", "mean"), relax_predictions=("Nu", "mean"),
        n_ce_off=("n_ce_off", "mean"), util=("util", "mean"), AP=("AP", "mean")
    ).to_string(float_format=lambda x: "%.4f" % x))
    print("\n  Spearman(lam_max, util) over the 48 lrc1e-4 hounie runs: %+.3f"
          % spear(a.lam_max, a.util))
    print("  Spearman(n_ce_off, util) over the same 48 runs:            %+.3f"
          % spear(a.n_ce_off, a.util))
    return 0


if __name__ == "__main__":
    sys.exit(main())
