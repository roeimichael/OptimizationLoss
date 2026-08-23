"""Third pass -- the two decisive tests.

(A) Is the CE / train-acc separation a TraLO-TRAJECTORY property, or a dataset
    constant?  If fioretto_ldf and hounie_rcl show the same derm<oct CE ordering
    on the same cells, CE cannot be what locates a split that only TraLO shows.
    (derm duals write literal `nan` into ce_loss once satisfied -> use the last
    FINITE value, not iloc[-1].)

(B) Does ANY trajectory quantity track the outcome?  Two falsification tests:
      B1  within-cell across seeds: Spearman(trajectory quantity, per-seed
          ccF1eq margin vs best dual), pooled over cells after centring.
      B2  the tissuemnist cross-check: tissue splits by BACKBONE (MNV3 wins,
          RegNet loses).  Any quantity that "locates the dataset split" must
          also reproduce that backbone split.  Quantities that fail this are
          not diagnostic.

(C) Per-run rank tests (16 vs 16) on the quantities that separated at cell level.
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, spearmanr

sys.path.insert(0, os.getcwd())
sys.path.insert(0, "paper/scripts")
import analyze_headroom as A  # noqa: E402

CELL = ["dataset", "model", "cap"]
ROOT = "results/headroom/headroom_b30_lrc0.0001_noceskip"


def num(df, col):
    return pd.to_numeric(df[col], errors="coerce")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=ROOT)
    args = ap.parse_args()
    pd.set_option("display.width", 250)

    # ---------- (A) duals' CE, nan-safe ----------
    rows = []
    for cfgp in sorted(glob.glob(args.root + "/**/config.json", recursive=True)):
        d = os.path.dirname(cfgp)
        cfg = json.load(open(cfgp))
        m = cfg.get("methodology")
        lgp = os.path.join(d, "training_log.csv")
        if not os.path.exists(lgp) or m not in ("fioretto_ldf", "hounie_rcl"):
            continue
        lg = pd.read_csv(lgp)
        e = num(lg, "epoch")
        lg = lg[e.notna()].copy()
        lg["ep"] = e[e.notna()].astype(float)
        lg = lg.sort_values("ep")
        ce = num(lg, "ce_loss").to_numpy(float)
        fin = ce[np.isfinite(ce)]
        rows.append(dict(dataset=cfg["dataset_mode"], model=cfg["model_name"],
                         cap=cfg["constraint_tag"], method=m,
                         seed=(cfg.get("hyperparams") or {}).get("seed"),
                         last_ep=float(lg["ep"].max()),
                         n_nan=int((~np.isfinite(ce)).sum()),
                         ce_lastfinite=float(fin[-1]) if len(fin) else np.nan,
                         ce_min=float(fin.min()) if len(fin) else np.nan))
    du = pd.DataFrame(rows)
    print("=" * 120)
    print("(A) FINAL CE -- is derm<oct a TraLO thing or a dataset constant?")
    print("    duals: last FINITE ce_loss (derm duals write nan after satisfaction)")
    print("=" * 120)
    g = du.groupby(["method", "dataset"]).agg(
        n=("seed", "size"), ce=("ce_lastfinite", "mean"),
        ce_lo=("ce_lastfinite", "min"), ce_hi=("ce_lastfinite", "max"),
        ce_min_seen=("ce_min", "mean"), nan_rows=("n_nan", "mean"),
        last_ep=("last_ep", "mean")).reset_index()
    print(g.to_string(index=False, float_format=lambda x: "%.4f" % x))

    tr = pd.read_csv("paper/scripts/out_refute_traj.csv")
    print("\n  tralo (never nan): " + "  ".join(
        "%s %.4f [%.4f,%.4f]" % (ds, gg.ce_last.mean(), gg.ce_last.min(), gg.ce_last.max())
        for ds, gg in tr.groupby("dataset")))
    for meth in ["fioretto_ldf", "hounie_rcl"]:
        s = g[g.method == meth].set_index("dataset")["ce"]
        if "dermmnist" in s and "octmnist" in s:
            print("  %-12s derm %.4f vs oct %.4f  ->  %s" % (
                meth, s["dermmnist"], s["octmnist"],
                "SAME ordering as tralo (derm<oct)" if s["dermmnist"] < s["octmnist"]
                else "OPPOSITE ordering"))

    # ---------- (B) does the trajectory track the outcome? ----------
    d = A.rows_for(args.root)
    key = ["dataset", "model", "cap", "seed"]
    piv = d[d.method.isin(["tralo", "fioretto_ldf", "hounie_rcl"])].pivot_table(
        index=key, columns="method", values="ccF1eq").reset_index()
    piv["margin"] = piv["tralo"] - piv[["fioretto_ldf", "hounie_rcl"]].max(axis=1)
    tr["seed"] = tr["seed"].astype(int)
    piv["seed"] = piv["seed"].astype(int)
    j = tr.merge(piv[key + ["margin", "tralo"]], on=key, how="inner")
    print("\n  joined runs: %d" % len(j))

    QS = ["n_sat", "ever_sat", "first_sat", "r_last", "r_min", "cross",
          "cross_grid", "swing_grid", "above_grid", "lam_last", "ce_last",
          "unspent", "raw_cnt", "flips"]
    print("\n" + "=" * 120)
    print("(B1) WITHIN-CELL: centre each quantity and the margin inside its cell,")
    print("     then Spearman over all 48 runs.  If the trajectory carried the")
    print("     outcome, something here would be strongly non-zero.")
    print("=" * 120)
    jj = j.copy()
    for q in QS + ["margin"]:
        jj[q + "_c"] = jj.groupby(CELL)[q].transform(lambda s: s - s.mean())
    for q in QS:
        a_ = jj[q + "_c"].to_numpy(float)
        b_ = jj["margin_c"].to_numpy(float)
        ok = np.isfinite(a_) & np.isfinite(b_)
        if ok.sum() < 8 or np.nanstd(a_[ok]) == 0:
            print("  %-12s (degenerate)" % q)
            continue
        r = spearmanr(a_[ok], b_[ok])
        print("  %-12s n=%2d  rho=%+.3f  p=%.3f" % (q, ok.sum(), r.correlation, r.pvalue))

    print("\n" + "=" * 120)
    print("(B2) TISSUEMNIST CROSS-CHECK.  tissue splits by BACKBONE: MNV3 wins")
    print("     (+0.0203,+0.0175), RegNet loses (-0.0225,-0.0097).  Any quantity")
    print("     that 'locates' the derm/oct split must also reproduce THIS split.")
    print("=" * 120)
    cell = j.groupby(CELL).agg(**{q: (q, "mean") for q in QS},
                               margin=("margin", "mean")).reset_index()
    print(cell.to_string(index=False, float_format=lambda x: "%.3f" % x))
    print("\n  Spearman(quantity, cell margin) over all 12 cells:")
    for q in QS:
        a_ = cell[q].to_numpy(float)
        b_ = cell["margin"].to_numpy(float)
        ok = np.isfinite(a_) & np.isfinite(b_)
        if ok.sum() < 5 or np.nanstd(a_[ok]) == 0:
            print("    %-12s (degenerate)" % q)
            continue
        r = spearmanr(a_[ok], b_[ok])
        print("    %-12s n=%2d  rho=%+.3f  p=%.3f" % (q, ok.sum(), r.correlation, r.pvalue))

    print("\n" + "=" * 120)
    print("(C) PER-RUN rank tests, derm(16) vs oct(16) -- no cell averaging")
    print("=" * 120)
    for q in QS + ["acc_last" if "acc_last" in tr.columns else "n_sat"]:
        a_ = tr[tr.dataset == "dermmnist"][q].to_numpy(float)
        b_ = tr[tr.dataset == "octmnist"][q].to_numpy(float)
        a_, b_ = a_[np.isfinite(a_)], b_[np.isfinite(b_)]
        if len(a_) < 3 or len(b_) < 3:
            print("  %-12s (too few: derm %d oct %d)" % (q, len(a_), len(b_)))
            continue
        try:
            u = mannwhitneyu(a_, b_, alternative="two-sided")
            p = u.pvalue
        except Exception:
            p = np.nan
        print("  %-12s derm n=%2d med %.3f [%.3f,%.3f]   oct n=%2d med %.3f [%.3f,%.3f]   MWU p=%.4f%s"
              % (q, len(a_), np.median(a_), a_.min(), a_.max(),
                 len(b_), np.median(b_), b_.min(), b_.max(), p,
                 "   <<< p<0.05" if p < 0.05 else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
