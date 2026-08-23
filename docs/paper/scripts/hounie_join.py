"""Join the reconstructed dual dynamics (hounie_dyn.py) to the per-run scores
(analyze_headroom.rows_for) and test, run by run, which dynamic quantity
predicts hounie_rcl's dataset reversal.

Candidate discriminators, all measured per run:
  n_ce_off        epochs in which the CE batch loop did NOT run (ce_loss = NaN)
  lam_max         peak global multiplier   (tests "the dual step drives it")
  first_sat       first hard-satisfied epoch
  soft_final/K    where the soft count ended relative to the cap

Scores come from analyze_headroom.rows_for so the metric definitions are the
project's own (ccF1eq at equal budget, allocation-free AP).

    python paper/scripts/hounie_join.py
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.getcwd())
sys.path.insert(0, "paper/scripts")
import analyze_headroom as A   # noqa: E402

KEY = ["dataset", "model", "cap", "seed", "method"]


def spearman(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 4:
        return np.nan
    ra, rb = pd.Series(a[m]).rank(), pd.Series(b[m]).rank()
    return float(np.corrcoef(ra, rb)[0, 1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="results/headroom/headroom_b30_lrc0.0001_noceskip")
    ap.add_argument("--dyn", default="paper/scripts/out_hounie_dyn.csv")
    args = ap.parse_args()

    dyn = pd.read_csv(args.dyn)
    sc = A.rows_for(args.root)
    sc["seed"] = sc["seed"].astype(int)
    dyn["seed"] = dyn["seed"].astype(int)
    d = dyn.merge(sc[KEY + ["ccF1eq", "AP", "macroEq", "count_raw", "K"]],
                  on=KEY, how="inner", suffixes=("", "_sc"))
    print("joined %d runs (%s)" % (len(d), d.method.value_counts().to_dict()))

    # reference: the per-seed best of the OTHER trained arms
    piv_ap = sc.pivot_table(index=["dataset", "model", "cap", "seed"],
                            columns="method", values="AP")
    piv_f1 = sc.pivot_table(index=["dataset", "model", "cap", "seed"],
                            columns="method", values="ccF1eq")
    ref = pd.DataFrame({
        "AP_ref": piv_ap[["tralo", "fioretto_ldf"]].max(axis=1),
        "F1_ref": piv_f1[["tralo", "fioretto_ldf"]].max(axis=1)}).reset_index()
    h = d[d.method == "hounie_rcl"].merge(
        ref, on=["dataset", "model", "cap", "seed"], how="left")
    h["dAP"] = h["AP"] - h["AP_ref"]
    h["dF1"] = h["ccF1eq"] - h["F1_ref"]
    h["softfin_over_K"] = h["soft_final"] / h["K"]
    h["util"] = h["count_raw"] / h["K"]

    pd.set_option("display.width", 260)
    pd.set_option("display.max_columns", 50)

    print("\n" + "=" * 118)
    print("hounie_rcl per RUN: dynamics vs score gap to the best of {tralo, fioretto}")
    print("=" * 118)
    cols = ["dataset", "model", "cap", "seed", "epochs", "ce_off_ep", "n_ce_off",
            "first_sat", "lam_max", "u_max", "soft_final", "softfin_over_K",
            "count_raw", "K", "util", "AP", "dAP", "ccF1eq", "dF1"]
    print(h[cols].sort_values(["dataset", "model", "cap", "seed"])
          .to_string(index=False, float_format=lambda x: "%.4g" % x))

    print("\n" + "=" * 118)
    print("WHICH DYNAMIC QUANTITY PREDICTS THE GAP?  Spearman over all %d hounie runs" % len(h))
    print("=" * 118)
    for v in ["n_ce_off", "lam_max", "u_max", "epochs", "first_sat",
              "softfin_over_K", "util"]:
        print("  %-16s vs dAP  %+.3f     vs dF1  %+.3f     vs AP %+.3f" %
              (v, spearman(h[v], h["dAP"]), spearman(h[v], h["dF1"]),
               spearman(h[v], h["AP"])))

    print("\n" + "=" * 118)
    print("SPLIT ON THE CE GATE (n_ce_off >= 5 constraint-only epochs vs not)")
    print("=" * 118)
    h["ce_gate"] = np.where(h["n_ce_off"] >= 5, "CE OFF >=5 epochs", "CE on ~throughout")
    g = h.groupby("ce_gate").agg(
        n=("seed", "size"),
        datasets=("dataset", lambda s: dict(s.value_counts())),
        mean_n_ce_off=("n_ce_off", "mean"),
        mean_lam_max=("lam_max", "mean"),
        mean_util=("util", "mean"),
        mean_AP=("AP", "mean"), mean_dAP=("dAP", "mean"),
        mean_ccF1eq=("ccF1eq", "mean"), mean_dF1=("dF1", "mean"),
        n_dAP_negative=("dAP", lambda s: int((s < 0).sum())))
    print(g.to_string())

    print("\n" + "=" * 118)
    print("SAME SPLIT, WITHIN DATASET (does it survive holding the dataset fixed?)")
    print("=" * 118)
    print(h.groupby(["dataset", "ce_gate"]).agg(
        n=("seed", "size"), mean_n_ce_off=("n_ce_off", "mean"),
        mean_util=("util", "mean"), mean_AP=("AP", "mean"),
        mean_dAP=("dAP", "mean"), mean_dF1=("dF1", "mean")
    ).to_string(float_format=lambda x: "%.4g" % x))

    print("\n" + "=" * 118)
    print("fioretto_ldf gets the SAME CE gate on derm -- does it collapse the same way?")
    print("=" * 118)
    f = d[d.method == "fioretto_ldf"].copy()
    f["util"] = f["count_raw"] / f["K"]
    both = pd.concat([h.assign(m="hounie"), f.assign(m="fioretto")])
    print(both.groupby(["dataset", "m"]).agg(
        n=("seed", "size"), epochs=("epochs", "mean"),
        n_ce_off=("n_ce_off", "mean"), first_sat=("first_sat", "mean"),
        count_raw=("count_raw", "mean"), util=("util", "mean"),
        AP=("AP", "mean"), ccF1eq=("ccF1eq", "mean")
    ).to_string(float_format=lambda x: "%.4g" % x))
    return 0


if __name__ == "__main__":
    sys.exit(main())
