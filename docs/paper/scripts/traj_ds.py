"""Per-run training dynamics for the headroom campaign, stratified by cell.

Answers: what does TraLO's count trajectory look like against K, when/how often
does it satisfy, does it oscillate across the cap, what do lambda and CE do,
and how much budget does it leave unspent -- compared between dermmnist (where
TraLO wins 4/4 cells) and octmnist (where it loses 4/4).

SCHEMA NOTES (do not "simplify" these away):
  * TraLO's training_log.csv is SPARSE: a row is written iff
    (epoch+1) % 5 == 0  OR  is_satisfied  OR  epoch == warmup_epochs.
    So EVERY fully-satisfied epoch IS logged -- satisfied-epoch counts read off
    the log are exact.  Unsatisfied epochs are sampled every 5.
    len(df) is meaningless; use df["Epoch"].max().
  * Column case differs: tralo -> "Epoch", fioretto_ldf/hounie_rcl -> "epoch".
  * The duals log NO per-class counts. They log `total_excess` = global excess
    + sum of local group excesses (verified against tralo's own counts at the
    shared first constraint epoch). Counts are NOT recoverable exactly.
  * Headers can repeat mid-file -> to_numeric(errors="coerce").dropna().
  * constrained class = config["dataset_config"]["constrained_class"] ([0] if list)

    python paper/scripts/traj_ds.py --root results/headroom/headroom_b30_lrc0.0001_noceskip
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

CELL = ["dataset", "model", "cap"]


def num(df, col):
    return pd.to_numeric(df[col], errors="coerce")


def load_cfg(d):
    with open(os.path.join(d, "config.json")) as f:
        return json.load(f)


def cls_of(cfg):
    c = (cfg.get("dataset_config") or {}).get("constrained_class")
    return int(c[0] if isinstance(c, (list, tuple)) else c)


def evalmetrics(d):
    p = os.path.join(d, "evaluation_metrics.csv")
    if not os.path.exists(p):
        return {}
    t = pd.read_csv(p)
    return dict(zip(t["Metric"], t["Value"]))


def raw_count(d, cls):
    p = os.path.join(d, "final_predictions_raw.csv")
    if not os.path.exists(p):
        return np.nan
    return int((pd.read_csv(p)["Predicted_Label"].to_numpy(int) == cls).sum())


def tralo_row(d, cfg):
    c = cls_of(cfg)
    lg = pd.read_csv(os.path.join(d, "training_log.csv"))
    ep = num(lg, "Epoch")
    keep = ep.notna()
    lg = lg[keep].copy()
    lg["ep"] = ep[keep]
    hard = num(lg, "Hard_Class%d" % c)
    soft = num(lg, "Soft_Class%d" % c)
    K = num(lg, "Limit_Class%d" % c).iloc[0]
    gs = num(lg, "Global_Satisfied")
    ls = num(lg, "Local_Satisfied")
    sat = ((gs == 1) & (ls == 1))
    lam = num(lg, "Lambda_Global")
    ce = num(lg, "L_CE")
    acc = num(lg, "Train_Acc")
    Lg = num(lg, "L_Global")
    e = lg["ep"].to_numpy(float)
    h = hard.to_numpy(float)
    sgn = np.sign(h - K)
    # crossings among LOGGED epochs only (undercount: unsatisfied epochs sampled /5)
    nz = sgn[sgn != 0]
    cross = int((np.diff(nz) != 0).sum()) if len(nz) > 1 else 0
    ev = evalmetrics(d)

    def g(k):
        v = ev.get(k, "")
        try:
            return float(v)
        except Exception:
            return np.nan

    return {
        "K": float(K), "last_epoch": float(e.max()), "n_rows": len(lg),
        "n_sat_logged": int(sat.sum()),                # exact: sat epochs always logged
        "first_sat_ep": float(e[sat.to_numpy()][0]) if sat.any() else np.nan,
        "n_globsat_logged": int((gs == 1).sum()),
        "count_first": float(h[0]), "count_last": float(h[-1]),
        "count_min": float(h.min()), "count_max": float(h.max()),
        "ratio_first": float(h[0] / K), "ratio_last": float(h[-1] / K),
        "ratio_min": float(h.min() / K), "ratio_max": float(h.max() / K),
        "cross": cross,
        "soft_last": float(soft.to_numpy(float)[-1]),
        "softhard_gap_last": float(soft.to_numpy(float)[-1] - h[-1]),
        "lam_first": float(lam.iloc[0]), "lam_last": float(lam.iloc[-1]),
        "ce_first": float(ce.iloc[0]), "ce_last": float(ce.iloc[-1]),
        "acc_first": float(acc.iloc[0]), "acc_last": float(acc.iloc[-1]),
        "acc_max": float(acc.max()),
        "Lg_first": float(Lg.iloc[0]), "Lg_last": float(Lg.iloc[-1]),
        "raw_count": raw_count(d, c),
        "sat_epoch_ev": g("Satisfaction Epoch"),
        "best_sat_ep_ev": g("Best Satisfied Epoch"),
        "min_exc_ep_ev": g("Min Excess Epoch"),
        "min_exc_ev": g("Min Total Excess"),
        "restored_ep": g("Restored From Epoch"),
        "restore_kind": ev.get("Restore Kind", ""),
        "raw_excess": g("Raw Total Excess"),
        "flips": g("Flips Required"),
    }


def dual_row(d, cfg):
    c = cls_of(cfg)
    lg = pd.read_csv(os.path.join(d, "training_log.csv"))
    ep = num(lg, "epoch")
    lg = lg[ep.notna()].copy()
    lg["ep"] = ep[ep.notna()]
    exc = num(lg, "total_excess").to_numpy(float)
    satc = num(lg, "all_satisfied").to_numpy(float)
    ce = num(lg, "ce_loss").to_numpy(float)
    lamcol = "max_lambda_g" if "max_lambda_g" in lg.columns else "max_lam_g"
    lam = num(lg, lamcol).to_numpy(float)
    e = lg["ep"].to_numpy(float)
    ev = evalmetrics(d)

    def g(k):
        try:
            return float(ev.get(k, ""))
        except Exception:
            return np.nan

    return {
        "K": np.nan, "last_epoch": float(e.max()), "n_rows": len(lg),
        "n_sat_logged": int((satc == 1).sum()),
        "first_sat_ep": float(e[satc == 1][0]) if (satc == 1).any() else np.nan,
        "exc_first": float(exc[0]), "exc_last": float(exc[-1]),
        "exc_min": float(exc.min()),
        "ce_first": float(ce[0]), "ce_last": float(ce[-1]),
        "lam_first": float(lam[0]), "lam_last": float(lam[-1]),
        "raw_count": raw_count(d, c),
        "raw_excess": g("Raw Total Excess"), "flips": g("Flips Required"),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--dumptraj", default=None,
                    help="also write every logged tralo epoch to this csv")
    args = ap.parse_args()

    rows, traj = [], []
    for cfgp in glob.glob(args.root + "/**/config.json", recursive=True):
        d = os.path.dirname(cfgp)
        if not os.path.exists(os.path.join(d, "training_log.csv")):
            continue
        cfg = json.load(open(cfgp))
        m = cfg.get("methodology")
        hp = cfg.get("hyperparams") or {}
        base = {"dataset": cfg.get("dataset_mode"), "model": cfg.get("model_name"),
                "cap": cfg.get("constraint_tag"), "method": m,
                "seed": hp.get("seed"), "dir": d}
        try:
            if m == "tralo":
                base.update(tralo_row(d, cfg))
                c = cls_of(cfg)
                lg = pd.read_csv(os.path.join(d, "training_log.csv"))
                ep = num(lg, "Epoch")
                lg = lg[ep.notna()]
                for i in range(len(lg)):
                    traj.append({
                        "dataset": base["dataset"], "model": base["model"],
                        "cap": base["cap"], "seed": base["seed"],
                        "ep": float(num(lg, "Epoch").iloc[i]),
                        "hard": float(num(lg, "Hard_Class%d" % c).iloc[i]),
                        "soft": float(num(lg, "Soft_Class%d" % c).iloc[i]),
                        "K": float(num(lg, "Limit_Class%d" % c).iloc[i]),
                        "gsat": float(num(lg, "Global_Satisfied").iloc[i]),
                        "lsat": float(num(lg, "Local_Satisfied").iloc[i]),
                        "lam": float(num(lg, "Lambda_Global").iloc[i]),
                        "ce": float(num(lg, "L_CE").iloc[i]),
                        "acc": float(num(lg, "Train_Acc").iloc[i]),
                        "Lg": float(num(lg, "L_Global").iloc[i]),
                        "Ll": float(num(lg, "L_Local").iloc[i]),
                    })
            elif m in ("fioretto_ldf", "hounie_rcl"):
                base.update(dual_row(d, cfg))
            else:
                continue
        except Exception as e:  # noqa: BLE001
            base["err"] = repr(e)
        rows.append(base)

    t = pd.DataFrame(rows)
    if args.out:
        t.to_csv(args.out, index=False)
    if args.dumptraj and traj:
        pd.DataFrame(traj).to_csv(args.dumptraj, index=False)
    pd.set_option("display.width", 260)

    tr = t[t.method == "tralo"]
    print("=" * 130)
    print("TraLO per-cell dynamics (mean over seeds).  root=%s" % args.root)
    print("  ratio_* = hard count / K.  cross = sign changes of (count-K) among LOGGED epochs (lower bound).")
    print("  n_sat = # fully-satisfied epochs (EXACT: satisfied epochs are always logged).")
    print("=" * 130)
    agg = tr.groupby(CELL).agg(
        K=("K", "mean"), n=("seed", "count"),
        last_ep=("last_epoch", "mean"),
        n_sat=("n_sat_logged", "mean"), seeds_ever_sat=("first_sat_ep", lambda s: s.notna().sum()),
        first_sat=("first_sat_ep", "mean"),
        r_first=("ratio_first", "mean"), r_min=("ratio_min", "mean"),
        r_max=("ratio_max", "mean"), r_last=("ratio_last", "mean"),
        cross=("cross", "mean"),
        raw_cnt=("raw_count", "mean"),
        lam_last=("lam_last", "mean"),
        ce_first=("ce_first", "mean"), ce_last=("ce_last", "mean"),
        acc_first=("acc_first", "mean"), acc_last=("acc_last", "mean"),
        sh_gap=("softhard_gap_last", "mean"),
        min_exc_ep=("min_exc_ep_ev", "mean"), min_exc=("min_exc_ev", "mean"),
        raw_exc=("raw_excess", "mean"), flips=("flips", "mean"),
        n_restored=("restored_ep", lambda s: s.notna().sum()),
    ).reset_index()
    agg["budget_left"] = agg["K"] - agg["raw_cnt"]
    print(agg.to_string(index=False, float_format=lambda x: "%.3f" % x))

    print()
    print("=" * 130)
    print("TraLO per-SEED count/K trajectory (logged epochs only)")
    print("=" * 130)
    if traj:
        J = pd.DataFrame(traj)
        for (ds, mo, cap), g in J.groupby(CELL):
            print("\n--- %s %s %s   K=%d" % (ds, mo, cap, g["K"].iloc[0]))
            for sd, gs in g.groupby("seed"):
                gs = gs.sort_values("ep")
                s = "  seed %s: " % sd
                s += " ".join("E%d:%d%s" % (e, h, "*" if (a == 1 and b == 1) else "")
                              for e, h, a, b in zip(gs.ep, gs.hard, gs.gsat, gs.lsat))
                print(s)

    du = t[t.method.isin(["fioretto_ldf", "hounie_rcl"])]
    print()
    print("=" * 130)
    print("DUALS per-cell (total_excess = global excess + sum local group excess; counts NOT logged)")
    print("=" * 130)
    dagg = du.groupby(CELL + ["method"]).agg(
        n=("seed", "count"), last_ep=("last_epoch", "mean"),
        n_sat=("n_sat_logged", "mean"),
        seeds_ever_sat=("first_sat_ep", lambda s: s.notna().sum()),
        first_sat=("first_sat_ep", "mean"),
        exc_first=("exc_first", "mean"), exc_min=("exc_min", "mean"),
        exc_last=("exc_last", "mean"),
        ce_first=("ce_first", "mean"), ce_last=("ce_last", "mean"),
        lam_last=("lam_last", "mean"), raw_cnt=("raw_count", "mean"),
        raw_exc=("raw_excess", "mean"), flips=("flips", "mean"),
    ).reset_index()
    print(dagg.to_string(index=False, float_format=lambda x: "%.3f" % x))
    return 0


if __name__ == "__main__":
    sys.exit(main())
