"""Three cross-checks on the derm-wins / oct-loses split.

1. FEASIBILITY: do the duals actually reach total_excess == 0, and do they
   early-stop when they do?  (their own training_log, per seed)
2. COMPARATOR INFLATION: "best of two duals" is a max over two noisy
   comparators, which is biased upward.  Report tralo vs EACH dual separately
   and the size of the max-vs-mean inflation, per cell.
3. REPLICATION: same per-cell delta in the sibling campaigns
   (lrc0.0001 = CE-skip ON, lrc5e-05 = 20x lower constraint LR).
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
DUALS = ["fioretto_ldf", "hounie_rcl"]


def dual_feas(root):
    rows = []
    for cfgp in glob.glob(root + "/**/config.json", recursive=True):
        cfg = json.load(open(cfgp))
        m = cfg.get("methodology")
        if m not in DUALS:
            continue
        d = os.path.dirname(cfgp)
        p = os.path.join(d, "training_log.csv")
        if not os.path.exists(p):
            continue
        lg = pd.read_csv(p)
        e = pd.to_numeric(lg["epoch"], errors="coerce")
        lg = lg[e.notna()]
        exc = pd.to_numeric(lg["total_excess"], errors="coerce").to_numpy(float)
        sat = pd.to_numeric(lg["all_satisfied"], errors="coerce").to_numpy(float)
        ep = pd.to_numeric(lg["epoch"], errors="coerce").to_numpy(float)
        rows.append({
            "dataset": cfg.get("dataset_mode"), "model": cfg.get("model_name"),
            "cap": cfg.get("constraint_tag"), "method": m,
            "seed": (cfg.get("hyperparams") or {}).get("seed"),
            "epochs_run": int(ep.max()) + 1,
            "ever_feasible": int((exc == 0).any()),
            "first_feas_ep": float(ep[exc == 0][0]) if (exc == 0).any() else np.nan,
            "n_feas_ep": int((exc == 0).sum()),
            "exc_last": float(exc[-1]), "exc_min": float(exc.min()),
            "n_sat_ep": int((sat == 1).sum()),
        })
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="results/headroom/headroom_b30_lrc0.0001_noceskip")
    ap.add_argument("--sibs", nargs="*",
                    default=["results/headroom/headroom_b30_lrc0.0001",
                             "results/headroom/headroom_b30_lrc5e-05"])
    args = ap.parse_args()
    pd.set_option("display.width", 250)

    print("=" * 122)
    print("1. DUAL FEASIBILITY (their own logs).  epochs_run: they early-stop on")
    print("   their own convergence rule; TraLO runs 29 constraint epochs.")
    print("=" * 122)
    f = dual_feas(args.root)
    agg = f.groupby(CELL + ["method"]).agg(
        n=("seed", "count"), epochs_run=("epochs_run", "mean"),
        seeds_feasible=("ever_feasible", "sum"),
        first_feas=("first_feas_ep", "mean"), n_feas_ep=("n_feas_ep", "mean"),
        exc_min=("exc_min", "mean"), exc_last=("exc_last", "mean")).reset_index()
    print(agg.to_string(index=False, float_format=lambda x: "%.3f" % x))
    print()
    print("  ROLLUP by dataset: how many of the 16 dual runs per dataset ever hit excess 0")
    for ds, g in f.groupby("dataset"):
        print("    %-12s %2d/%2d runs feasible   mean epochs_run %.1f   mean final excess %.1f"
              % (ds, g.ever_feasible.sum(), len(g), g.epochs_run.mean(), g.exc_last.mean()))

    print()
    print("=" * 122)
    print("2. COMPARATOR INFLATION. tralo vs each dual alone, and vs max(both).")
    print("   inflate = mean_seeds[ max(fior,houn) - mean(fior,houn) ]")
    print("=" * 122)
    d = A.rows_for(args.root)
    d = d[d.method.isin(["tralo"] + DUALS)]
    piv = d.pivot_table(index=CELL + ["seed"], columns="method", values="ccF1eq")
    piv = piv.dropna()
    piv = piv.reset_index()
    out = []
    for (ds, mo, cap), g in piv.groupby(CELL):
        mx = g[DUALS].max(axis=1)
        mn = g[DUALS].mean(axis=1)
        out.append({
            "dataset": ds, "model": mo, "cap": cap, "n": len(g),
            "tralo": g.tralo.mean(), "fior": g.fioretto_ldf.mean(),
            "houn": g.hounie_rcl.mean(),
            "vFior": (g.tralo - g.fioretto_ldf).mean(),
            "wFior": int((g.tralo > g.fioretto_ldf).sum()),
            "vHoun": (g.tralo - g.hounie_rcl).mean(),
            "wHoun": int((g.tralo > g.hounie_rcl).sum()),
            "vMax": (g.tralo - mx).mean(), "wMax": int((g.tralo > mx).sum()),
            "vMean": (g.tralo - mn).mean(), "wMean": int((g.tralo > mn).sum()),
            "inflate": (mx - mn).mean(),
            "sd_tralo": g.tralo.std(), "sd_fior": g.fioretto_ldf.std(),
            "sd_houn": g.hounie_rcl.std(),
        })
    o = pd.DataFrame(out).sort_values(CELL)
    print(o.to_string(index=False, float_format=lambda x: "%.4f" % x))
    print()
    for ds, g in o.groupby("dataset"):
        print("  %-12s  vs FIORETTO alone: %d/4 cells>0 (mean %+.4f, %d/16 seeds)   "
              "vs HOUNIE alone: %d/4 cells>0 (mean %+.4f, %d/16 seeds)   "
              "vs MAX: %d/4 cells>0 (%d/16 seeds)   inflation %+.4f"
              % (ds, (g.vFior > 0).sum(), g.vFior.mean(), g.wFior.sum(),
                 (g.vHoun > 0).sum(), g.vHoun.mean(), g.wHoun.sum(),
                 (g.vMax > 0).sum(), g.wMax.sum(), g.inflate.mean()))

    print()
    print("=" * 122)
    print("3. REPLICATION in sibling campaigns (metric ccF1eq, tralo - max(duals))")
    print("=" * 122)
    for root in [args.root] + list(args.sibs):
        dd = A.rows_for(root)
        dd = dd[dd.method.isin(["tralo"] + DUALS)]
        if dd.empty:
            print("  %-52s  (no runs)" % root)
            continue
        pv = dd.pivot_table(index=CELL + ["seed"], columns="method",
                            values="ccF1eq").dropna().reset_index()
        res = []
        for (ds, mo, cap), g in pv.groupby(CELL):
            res.append({"dataset": ds, "model": mo, "cap": cap, "n": len(g),
                        "vMax": (g.tralo - g[DUALS].max(axis=1)).mean(),
                        "vFior": (g.tralo - g.fioretto_ldf).mean(),
                        "vHoun": (g.tralo - g.hounie_rcl).mean()})
        r = pd.DataFrame(res).sort_values(CELL)
        print("\n  --- %s" % root)
        print(r.to_string(index=False, float_format=lambda x: "%+.4f" % x))
    return 0


if __name__ == "__main__":
    sys.exit(main())
