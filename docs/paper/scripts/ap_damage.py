"""WHEN does constraint training cost ranking quality (AP)?

The post-hoc arms in `headroom_b30` (heuristic, danits_lp) are pure-CE models:
warmup_epochs=30, constraint_epochs=0.  Their average precision on the
constrained class is therefore the UNDAMAGED reference for the same
(dataset, backbone, seed) -- AP is computed from raw probabilities, so it is
identical for heuristic and danits_lp and identical across caps (asserted
below, not assumed).

For every trained run we compute  dAP = AP(run) - AP(matched pure-CE)  and
regress it on features read out of the run's own training_log.csv and
evaluation_metrics.csv.

SCHEMA NOTES (all verified against the files, see docstrings inline):
  * TraLO writes "Epoch" (1-indexed) and logs a row iff
    (epoch+1)%5==0 or is_satisfied or epoch==warmup_epochs
    (src/methodologies/tralo/train.py:430).  Because EVERY satisfied epoch is
    logged, the satisfied/violated indicator is EXACTLY recoverable for all
    29 constraint epochs: an epoch absent from the log was violated.
    len(df) is NOT the epoch count -- use df["Epoch"].max().
  * fioretto_ldf / hounie_rcl write "epoch" (0-indexed) densely, with
    `total_excess` (sum over global+local constraints of max(0, count-cap),
    src/methodologies/fioretto_ldf/train.py:196) and `all_satisfied`.
    They do NOT write Hard_Class*/Soft_Class*, so their per-epoch predicted
    COUNT is not recorded -- only the one-sided excess.  Undershoot is
    invisible in their logs.  This is stated in the output, not papered over.
  * Headers repeat mid-file -> to_numeric(errors="coerce").dropna().

    python paper/scripts/ap_damage.py
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

sys.path.insert(0, os.getcwd())
sys.path.insert(0, "paper/scripts")
from src.utils.constants import UNLIMITED                            # noqa: E402
from src.training.constraints import (compute_global_constraints,    # noqa: E402
                                      compute_local_constraints)

TRAINED = ["tralo", "fioretto_ldf", "hounie_rcl"]
CLIP = ["heuristic", "danits_lp"]


def num(s):
    return pd.to_numeric(s, errors="coerce")


def read_eval_metrics(d):
    p = os.path.join(d, "evaluation_metrics.csv")
    if not os.path.exists(p):
        return {}
    t = pd.read_csv(p)
    return dict(zip(t["Metric"].astype(str), t["Value"].astype(str)))


def f(x):
    try:
        v = float(x)
        return v
    except (TypeError, ValueError):
        return np.nan


def tralo_features(d, cls, K, warmup, n_cepochs):
    """Exact satisfied/violated vector + sparse count trajectory."""
    t = pd.read_csv(os.path.join(d, "training_log.csv"))
    ep = num(t["Epoch"])
    gs = num(t["Global_Satisfied"])
    ls = num(t["Local_Satisfied"])
    hard = num(t["Hard_Class%d" % cls])
    lam = num(t["Lambda_Global"])
    laml = num(t["Lambda_Local"])
    ok = ep.notna() & hard.notna()
    ep, gs, ls, hard, lam, laml = [v[ok].to_numpy() for v in (ep, gs, ls, hard, lam, laml)]
    last = int(ep.max())
    # local excess, if group columns exist
    grp_h = sorted([c for c in t.columns
                    if c.startswith("Group") and c.endswith("_Hard_Class%d" % cls)])
    exc = np.maximum(0.0, hard - K)
    for gh in grp_h:
        gl = gh.replace("_Hard_", "_Limit_")
        if gl not in t.columns:
            continue
        h = num(t[gh])[ok].to_numpy()
        L = num(t[gl])[ok].to_numpy()
        L = np.where(L >= UNLIMITED, np.inf, L)
        exc = exc + np.maximum(0.0, h - L)
    sat = ((gs == 1) & (ls == 1)).astype(int)
    sat_epochs = set(ep[sat == 1].astype(int).tolist())
    # constraint phase = epochs warmup+1 .. last (1-indexed).  every satisfied
    # epoch is logged, so absence from the log implies violated.
    phase = list(range(int(warmup) + 1, last + 1))
    sat_vec = np.array([1 if e in sat_epochs else 0 for e in phase])
    return dict(
        last_epoch=last,
        n_constraint_epochs=len(phase),
        n_satisfied=int(sat_vec.sum()),
        n_active=int((1 - sat_vec).sum()),
        frac_active=float((1 - sat_vec).mean()) if len(phase) else np.nan,
        n_transitions=int((np.abs(np.diff(sat_vec)) > 0).sum()) if len(phase) > 1 else 0,
        first_sat_epoch=(min(sat_epochs) - warmup if sat_epochs else np.nan),
        peak_lambda=float(np.nanmax(np.concatenate([lam, laml]))),
        peak_lambda_g=float(np.nanmax(lam)),
        # amplitude of the PREDICTED COUNT.  sparse -> lower bound on the truth
        osc_range_K=float((hard.max() - hard.min()) / K),
        osc_std_K=float(hard.std(ddof=0) / K),
        osc_step_K=float(np.mean(np.abs(np.diff(hard))) / K) if len(hard) > 1 else np.nan,
        # how deep below the cap the logged satisfied epochs sit
        undershoot_mean=float(np.mean((K - hard[sat == 1]) / K)) if sat.sum() else np.nan,
        undershoot_max=float(np.max((K - hard[sat == 1]) / K)) if sat.sum() else np.nan,
        n_logged_below_K=int((hard < K).sum()),
        excess_mean_logged=float(exc.mean() / K),
        excess_max_logged=float(exc.max() / K),
        final_ce=float(num(t["L_CE"])[ok].to_numpy()[-1]),
        final_train_acc=float(num(t["Train_Acc"])[ok].to_numpy()[-1]),
        n_log_rows=int(len(ep)),
    )


def dual_features(d, K, warmup, lam_col):
    t = pd.read_csv(os.path.join(d, "training_log.csv"))
    ep = num(t["epoch"])
    exc = num(t["total_excess"])
    sat = num(t["all_satisfied"])
    lam = num(t[lam_col]) if lam_col in t.columns else pd.Series([np.nan] * len(t))
    ce = num(t["ce_loss"])
    ok = ep.notna() & exc.notna()
    ep, exc, sat, lam, ce = [v[ok].to_numpy() for v in (ep, exc, sat, lam, ce)]
    last = int(ep.max()) + 1          # 0-indexed in file
    sat_vec = (sat == 1).astype(int)
    ce_ok = ce[~np.isnan(ce)]
    return dict(
        last_epoch=last,
        n_constraint_epochs=len(ep),
        n_satisfied=int(sat_vec.sum()),
        n_active=int((1 - sat_vec).sum()),
        frac_active=float((1 - sat_vec).mean()),
        n_transitions=int((np.abs(np.diff(sat_vec)) > 0).sum()) if len(sat_vec) > 1 else 0,
        first_sat_epoch=(float(ep[sat_vec == 1].min() + 1 - warmup)
                         if sat_vec.sum() else np.nan),
        peak_lambda=float(np.nanmax(lam)) if np.isfinite(lam).any() else np.nan,
        peak_lambda_g=float(np.nanmax(lam)) if np.isfinite(lam).any() else np.nan,
        # the duals do not log the predicted count, only the one-sided excess
        osc_range_K=float((exc.max() - exc.min()) / K),
        osc_std_K=float(exc.std(ddof=0) / K),
        osc_step_K=float(np.mean(np.abs(np.diff(exc))) / K) if len(exc) > 1 else np.nan,
        undershoot_mean=np.nan,
        undershoot_max=np.nan,
        n_logged_below_K=int((exc == 0).sum()),
        excess_mean_logged=float(exc.mean() / K),
        excess_max_logged=float(exc.max() / K),
        final_ce=float(ce_ok[-1]) if len(ce_ok) else np.nan,
        final_train_acc=np.nan,
        n_log_rows=int(len(ep)),
    )


def scan(root):
    out = []
    for cfg_path in sorted(glob.glob(root + "/**/config.json", recursive=True)):
        try:
            cfg = json.load(open(cfg_path))
        except Exception:
            continue
        d = os.path.dirname(cfg_path)
        raw = os.path.join(d, "final_predictions_raw.csv")
        fin = os.path.join(d, "final_predictions.csv")
        if not (os.path.exists(raw) and os.path.exists(fin)):
            continue
        t = pd.read_csv(raw)
        cols = sorted((c for c in t.columns if c.startswith("Prob_Class_")),
                      key=lambda c: int(c.rsplit("_", 1)[1]))
        if not cols:
            continue
        P = t[cols].to_numpy(float)
        y = t["True_Label"].to_numpy(int)
        rawp = t["Predicted_Label"].to_numpy(int)
        g = t["Group_ID"].to_numpy(int) if "Group_ID" in t.columns else None
        dc = cfg.get("dataset_config", {}) or {}
        cls = dc.get("constrained_class")
        cls = int(cls[0] if isinstance(cls, (list, tuple)) else cls)
        lp, gp = cfg["constraint"]
        df = pd.DataFrame({"label": y, "grp": g if g is not None else 0})
        G = compute_global_constraints(df, "label", gp, constrained_class=[cls],
                                       num_classes=P.shape[1])
        if G[cls] >= UNLIMITED:
            continue
        K = int(G[cls])
        hp = cfg.get("hyperparams") or {}
        meth = cfg.get("methodology")
        r = {
            "path": d, "dataset": cfg.get("dataset_mode"),
            "cap": cfg.get("constraint_tag"), "model": cfg.get("model_name"),
            "seed": hp.get("seed"), "method": meth,
            "warmup": hp.get("warmup_epochs"), "cepochs": hp.get("constraint_epochs"),
            "K": K, "cls": cls,
            "count_raw": int((rawp == cls).sum()),
            "count_adj": int((pd.read_csv(fin)["Predicted_Label"].to_numpy(int) == cls).sum()),
            "AP": average_precision_score((y == cls).astype(int), P[:, cls]),
        }
        em = read_eval_metrics(d)
        r["sat_epoch"] = f(em.get("Satisfaction Epoch"))
        r["best_sat_epoch"] = f(em.get("Best Satisfied Epoch"))
        r["min_excess_epoch"] = f(em.get("Min Excess Epoch"))
        r["restored_from_epoch"] = f(em.get("Restored From Epoch"))
        r["restore_kind"] = em.get("Restore Kind", "") or ""
        r["flips"] = f(em.get("Flips Required"))
        r["raw_total_excess"] = f(em.get("Raw Total Excess"))
        if meth in TRAINED:
            try:
                lc = {"fioretto_ldf": "max_lambda_g", "hounie_rcl": "max_lam_g"}
                if meth == "tralo":
                    r.update(tralo_features(d, cls, K, r["warmup"], r["cepochs"]))
                else:
                    r.update(dual_features(d, K, r["warmup"], lc[meth]))
            except Exception as e:                                   # noqa: BLE001
                r["log_error"] = "%s: %s" % (type(e).__name__, e)
        out.append(r)
    return pd.DataFrame(out)


def spearman(x, y):
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 4 or len(set(x[m])) < 2 or len(set(y[m])) < 2:
        return np.nan, np.nan, int(m.sum())
    from scipy.stats import spearmanr
    r, p = spearmanr(x[m], y[m])
    return float(r), float(p), int(m.sum())


FEATS = ["n_active", "frac_active", "n_satisfied", "n_transitions",
         "first_sat_epoch", "peak_lambda", "osc_range_K", "osc_std_K",
         "osc_step_K", "unspent_final", "n_logged_below_K", "undershoot_mean",
         "undershoot_max", "excess_mean_logged", "eval_epoch",
         "n_constraint_epochs"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trained",
                    default="results/headroom/headroom_b30_lrc0.0001_noceskip")
    ap.add_argument("--clip", default="results/headroom/headroom_b30")
    ap.add_argument("--out", default="paper/scripts/out_ap_damage.csv")
    args = ap.parse_args()

    tr = scan(args.trained)
    tr = tr[tr.method.isin(TRAINED)].copy()
    cl = scan(args.clip)
    cl = cl[cl.method.isin(CLIP)].copy()
    print("trained runs %d   clip runs %d" % (len(tr), len(cl)))

    # ---- 1. validate the pure-CE reference ------------------------------
    print("\n" + "=" * 92)
    print("STEP 1  is the pure-CE AP reference well defined?")
    print("=" * 92)
    key = ["dataset", "model", "seed"]
    agg = cl.groupby(key)["AP"].agg(["mean", "std", "min", "max", "count"])
    print("  AP spread WITHIN (dataset,model,seed) across {heuristic,danits_lp} x {L30,L50}:")
    print("    max std = %.3e   max range = %.3e   n groups = %d  (each n=%s)"
          % (agg["std"].max(), (agg["max"] - agg["min"]).max(), len(agg),
             sorted(agg["count"].unique())))
    print("  -> if ~0 the four post-hoc rows share one model, so AP_CE is a")
    print("     property of (dataset,model,seed) alone, which is what we need.")
    ref = cl.groupby(key)["AP"].mean().rename("AP_ce").reset_index()
    print(ref.to_string(index=False, float_format=lambda x: "%.4f" % x))

    # ---- 2. deficit -----------------------------------------------------
    d = tr.merge(ref, on=key, how="left")
    if d["AP_ce"].isna().any():
        print("\n!! %d trained runs have no pure-CE match" % int(d["AP_ce"].isna().sum()))
    d["dAP"] = d["AP"] - d["AP_ce"]
    d["unspent_final"] = (d["K"] - d["count_raw"]) / d["K"]
    d["eval_epoch"] = np.where(np.isfinite(d["restored_from_epoch"]),
                               d["restored_from_epoch"], d["last_epoch"])
    d.to_csv(args.out, index=False)
    print("\nwrote %s" % args.out)

    print("\n" + "=" * 92)
    print("STEP 2  AP deficit vs the matched pure-CE model, per CELL (never pooled)")
    print("=" * 92)
    piv = d.pivot_table(index=["dataset", "model", "cap"], columns="method",
                        values="dAP", aggfunc="mean")
    cnt = d.pivot_table(index=["dataset", "model", "cap"], columns="method",
                        values="dAP", aggfunc="count")
    apc = d.pivot_table(index=["dataset", "model", "cap"], values="AP_ce", aggfunc="mean")
    show = piv.join(apc)
    print(show.round(4).to_string())
    print("\n  seeds per cell/method: %s" % sorted(pd.unique(cnt.to_numpy().ravel())))

    print("\n  cells with NO AP damage (mean dAP >= 0), by method:")
    any_pos = False
    for m in TRAINED:
        if m not in piv.columns:
            continue
        pos = piv[piv[m] >= 0]
        for idx, row in pos.iterrows():
            any_pos = True
            print("    %-12s %-12s %-14s %-9s  dAP = %+.4f" % (m, idx[0], idx[1], idx[2], row[m]))
    if not any_pos:
        print("    NONE")
    print("\n  cells within noise (|mean dAP| < 0.005):")
    for m in TRAINED:
        if m not in piv.columns:
            continue
        for idx, v in piv[m].items():
            if abs(v) < 0.005:
                print("    %-12s %-12s %-14s %-9s  dAP = %+.4f" % (m, idx[0], idx[1], idx[2], v))

    # per-run sign census
    print("\n  per-RUN census (144 runs): dAP>0 %d   dAP<0 %d"
          % (int((d.dAP > 0).sum()), int((d.dAP < 0).sum())))
    print(d.groupby(["method"])["dAP"].agg(["mean", "min", "max", "count"]).round(4).to_string())

    # ---- 3. correlations ------------------------------------------------
    print("\n" + "=" * 92)
    print("STEP 3  which log feature predicts the deficit?  Spearman rho (all 144 runs)")
    print("        NOTE: pooled across cells, so cell-level confounds are live;")
    print("        the within-cell version below is the controlled one.")
    print("=" * 92)
    print("  %-20s %8s %10s %6s" % ("feature", "rho", "p", "n"))
    glob_rows = []
    for c in FEATS:
        if c not in d.columns:
            continue
        r, p, n = spearman(d[c].to_numpy(float), d["dAP"].to_numpy(float))
        glob_rows.append((c, r, p, n))
        print("  %-20s %8.3f %10.2e %6d" % (c, r, p, n))

    print("\n  within METHOD (48 runs each):")
    print("  %-20s %18s %18s %18s" % ("feature", "tralo", "fioretto_ldf", "hounie_rcl"))
    for c in FEATS:
        if c not in d.columns:
            continue
        line = "  %-20s" % c
        for m in TRAINED:
            s = d[d.method == m]
            r, p, n = spearman(s[c].to_numpy(float), s["dAP"].to_numpy(float))
            line += "  %7.3f (p%6.3f)" % (r, p) if np.isfinite(r) else "  %16s" % "-"
        print(line)

    # ---- 4. within-cell (the controlled test) ---------------------------
    print("\n" + "=" * 92)
    print("STEP 4  WITHIN-CELL correlation.  One (dataset,backbone,cap) at a time,")
    print("        12 runs each (3 methods x 4 seeds).  Cells are COUNTED, not averaged.")
    print("=" * 92)
    cells = list(d.groupby(["dataset", "model", "cap"]))
    print("  %-20s %6s %6s %6s %8s" % ("feature", "neg", "pos", "n/a", "median_rho"))
    for c in FEATS:
        if c not in d.columns:
            continue
        rs = []
        for _, g in cells:
            r, p, n = spearman(g[c].to_numpy(float), g["dAP"].to_numpy(float))
            rs.append(r)
        rs = np.array(rs, float)
        fin = rs[np.isfinite(rs)]
        print("  %-20s %6d %6d %6d %8s"
              % (c, int((fin < 0).sum()), int((fin > 0).sum()),
                 int((~np.isfinite(rs)).sum()),
                 "%.3f" % np.median(fin) if len(fin) else "-"))

    # within-cell, TraLO only (the count-trajectory features are TraLO-only)
    print("\n  same, TraLO runs only (4 seeds per cell -- weak, reported as a tally):")
    print("  %-20s %6s %6s %8s" % ("feature", "neg", "pos", "median_rho"))
    for c in FEATS:
        if c not in d.columns:
            continue
        rs = []
        for _, g in d[d.method == "tralo"].groupby(["dataset", "model", "cap"]):
            r, p, n = spearman(g[c].to_numpy(float), g["dAP"].to_numpy(float))
            rs.append(r)
        rs = np.array(rs, float)
        fin = rs[np.isfinite(rs)]
        print("  %-20s %6d %6d %8s" % (c, int((fin < 0).sum()), int((fin > 0).sum()),
                                       "%.3f" % np.median(fin) if len(fin) else "-"))

    # ---- 5. feature table by method x dataset ---------------------------
    print("\n" + "=" * 92)
    print("STEP 5  the features themselves (mean over 4 seeds), per cell")
    print("=" * 92)
    cols = ["dAP", "AP", "AP_ce", "n_active", "n_satisfied", "n_transitions",
            "first_sat_epoch", "peak_lambda", "osc_range_K", "unspent_final",
            "eval_epoch", "last_epoch"]
    cols = [c for c in cols if c in d.columns]
    t = d.groupby(["dataset", "model", "cap", "method"])[cols].mean()
    print(t.round(3).to_string())

    print("\n" + "=" * 92)
    print("STEP 6  checkpoint restore -- which epoch's model is actually evaluated?")
    print("=" * 92)
    d["restored"] = np.isfinite(d["restored_from_epoch"])
    print(d.groupby(["method", "restore_kind"]).size().to_string())
    print()
    print(d.groupby(["method", "restored"])["dAP"].agg(["mean", "count"]).round(4).to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
