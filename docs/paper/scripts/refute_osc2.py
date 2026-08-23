"""Part 2: does the claimed phase transition EXPLAIN anything?

  (A) length-matched tail placebo -- |drift|/step is a length-biased statistic
  (B) within-dataset natural experiment: octmnist @ lrc1e-4 has 8 gated / 8 not
  (C) does gating predict convergence at all?
  (D) is derm's overshoot caused by the gate, or by a non-hinged penalty?
"""
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.getcwd())
sys.path.insert(0, "paper/scripts")
import hounie_dyn as H  # noqa: E402

ROOTS = {"lrc1e-4": "results/headroom/headroom_b30_lrc0.0001_noceskip",
         "lrc5e-5": "results/headroom/headroom_b30_lrc5e-05"}
TAILS = [4, 5, 7, 9, 11, 13]


def seg(s):
    s = np.asarray(s, float)
    s = s[np.isfinite(s)]
    if len(s) < 3:
        return None
    d = np.diff(s)
    step = float(np.mean(np.abs(d)))
    drift = float((s[-1] - s[0]) / len(d))
    return step, drift, abs(drift) / step if step > 0 else np.nan, len(s)


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
            lam = T["lam"]
            soft = T["soft"]
            off = np.isnan(ce)
            d = dict(campaign=tag, dataset=r["dataset"], model=r["model"], cap=r["cap"],
                     seed=r["seed"], K=r["K"], gated=int(off.any()),
                     gate_ep=(int(ep[off][0]) if off.any() else np.nan),
                     n_off=int(off.sum()), n_rows=len(ep),
                     satisfied=int(sat.any()),
                     first_sat=(int(ep[sat == 1][0]) if sat.any() else np.nan),
                     n_sat=int(sat.sum()), exc_final=float(exc[-1]),
                     exc_min=float(exc.min()),
                     soft_final=float(soft[-1]) if np.isfinite(soft[-1]) else np.nan,
                     soft_min=float(np.nanmin(soft)),
                     raw_count=r["raw_count"],
                     lam_peak_ep=int(ep[int(np.argmax(lam))]), lam_max=float(lam.max()),
                     lam_final=float(lam[-1]), lam_still_rising=int(np.argmax(lam) == len(lam) - 1))
            d["overshoot_final"] = (r["K"] - d["soft_final"]) / r["K"]
            d["undercount_raw"] = (r["K"] - d["raw_count"]) / r["K"]
            # (A) length-matched tails on the FULL trajectory (gate-agnostic)
            for L in TAILS:
                v = seg(soft[-L:])
                d["tail%d" % L] = v[2] if v else np.nan
            # ratio over the segment after lam's peak (a gate-free alternative
            # explanation: the dual is simply relaxing)
            pk = int(np.argmax(lam))
            v = seg(soft[pk:])
            d["ratio_after_lampeak"] = v[2] if v else np.nan
            d["npts_after_lampeak"] = v[3] if v else np.nan
            v = seg(soft[:pk + 1])
            d["ratio_before_lampeak"] = v[2] if v else np.nan
            rows.append(d)
    d = pd.DataFrame(rows)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 60)
    f = lambda x: "%.3f" % x  # noqa: E731

    print("=" * 120)
    print("(A) LENGTH-MATCHED TAIL PLACEBO -- |drift|/step over the LAST L epochs of EVERY run,")
    print("    gate-agnostic.  The claim's CE-off segments are L~5.5 (oct) and L~11-13 (derm).")
    print("=" * 120)
    print(d.groupby(["campaign", "dataset"])[["tail%d" % L for L in TAILS]]
          .mean().to_string(float_format=f))

    print()
    print("=" * 120)
    print("(A2) SAME, but restricted to runs that NEVER gated (CE ran every epoch)")
    print("=" * 120)
    ng = d[d.gated == 0]
    print(ng.groupby(["campaign", "dataset"]).agg(
        n=("seed", "size"), **{("tail%d" % L): ("tail%d" % L, "mean") for L in TAILS})
        .to_string(float_format=f))

    print()
    print("=" * 120)
    print("(B) ALTERNATIVE EXPLANATION: the dual is simply past its peak (integral wind-down).")
    print("    ratio before/after lam's peak, for ALL runs incl. the never-gated ones.")
    print("=" * 120)
    print(d.groupby(["campaign", "dataset"]).agg(
        n=("seed", "size"), gated=("gated", "sum"),
        lam_peak_ep=("lam_peak_ep", "mean"),
        lam_still_rising_at_end=("lam_still_rising", "sum"),
        gate_ep=("gate_ep", "mean"),
        r_before_peak=("ratio_before_lampeak", "mean"),
        r_after_peak=("ratio_after_lampeak", "mean"),
        npts_after=("npts_after_lampeak", "mean")).to_string(float_format=f))

    print()
    print("=" * 120)
    print("(C) DOES GATING PREDICT CONVERGENCE?  octmnist @ lrc1e-4 is a natural experiment:")
    print("    8 runs gated, 8 did not, same dataset/backbones/caps.")
    print("=" * 120)
    for camp in ["lrc1e-4", "lrc5e-5"]:
        print("\n  --- %s ---" % camp)
        g = d[d.campaign == camp]
        print(g.groupby(["dataset", "gated"]).agg(
            runs=("seed", "size"), satisfied=("satisfied", "sum"),
            mean_first_sat=("first_sat", "mean"), n_sat_epochs=("n_sat", "mean"),
            exc_final=("exc_final", "mean"), exc_min=("exc_min", "mean"),
            raw_count=("raw_count", "mean"), K=("K", "mean")).to_string(float_format=f))

    print()
    print("=" * 120)
    print("(C2) oct @ lrc1e-4 ONLY -- per cell, gated vs not")
    print("=" * 120)
    o = d[(d.campaign == "lrc1e-4") & (d.dataset == "octmnist")]
    print(o[["model", "cap", "seed", "gated", "gate_ep", "n_off", "satisfied",
             "first_sat", "exc_final", "raw_count", "K", "soft_final"]]
          .sort_values(["model", "cap", "seed"]).to_string(index=False, float_format=f))

    print()
    print("=" * 120)
    print("(D) THE CLAIM SAYS THE MONOTONE DESCENT IS WHY DERM OVERSHOOTS.")
    print("    oct @ lrc5e-5 gets the SAME monotone descent (15/16 gated, ratio 1.000).")
    print("    Does it also overshoot?  Does tissue (never gated) fail to converge?")
    print("=" * 120)
    print(d.groupby(["campaign", "dataset"]).agg(
        runs=("seed", "size"), gated=("gated", "sum"), satisfied=("satisfied", "sum"),
        K=("K", "mean"), raw_count=("raw_count", "mean"),
        soft_final=("soft_final", "mean"), overshoot_final=("overshoot_final", "mean"),
        undercount_raw=("undercount_raw", "mean"),
        exc_final=("exc_final", "mean")).to_string(float_format=f))

    d.to_csv("paper/scripts/out_refute_osc2.csv", index=False)
    print("\nwrote paper/scripts/out_refute_osc2.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
