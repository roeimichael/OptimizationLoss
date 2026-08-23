"""Controlled follow-up to ap_damage.py.

Three things the pooled Spearman table cannot do:
  1. Put every method on ONE epoch axis.  TraLO's "Epoch" is a GLOBAL index
     (warmup included, 1-indexed: Epoch 2 = first constraint epoch, verified by
     L_CE matching fioretto's epoch 0 to 6 d.p.).  fioretto/hounie write a
     0-indexed CONSTRAINT-phase index.  `satisfaction_epoch` etc. are epoch+1
     in each method's own frame.  Everything below uses c = constraint-epoch,
     1..29, for all three methods.
  2. Control for the cell AND the method.  dAP is demeaned inside each of the
     36 (dataset, backbone, cap, method) groups, so what is left is pure
     seed-to-seed variation.  That is a fixed-effects test.
  3. Ask WHERE in the ranking the AP is lost -- precision@K is the part of the
     ranking the cap actually consumes.

The dose variable: the constraint gradient is clipped to max_norm=1.0 and is
only taken on VIOLATED epochs, so the mechanistically correct "amount of
constraint applied to the evaluated checkpoint" is the number of violated
constraint-epochs at or before the epoch the evaluated checkpoint came from.

    python paper/scripts/ap_damage2.py
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score

sys.path.insert(0, os.getcwd())
sys.path.insert(0, "paper/scripts")
from src.utils.constants import UNLIMITED                            # noqa: E402
from src.training.constraints import compute_global_constraints      # noqa: E402

TRAINED = ["tralo", "fioretto_ldf", "hounie_rcl"]
CLIP = ["heuristic", "danits_lp"]
GRP = ["dataset", "model", "cap", "method"]


def num(s):
    return pd.to_numeric(s, errors="coerce")


def evalmet(d):
    p = os.path.join(d, "evaluation_metrics.csv")
    if not os.path.exists(p):
        return {}
    t = pd.read_csv(p)
    return dict(zip(t["Metric"].astype(str), t["Value"].astype(str)))


def ff(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return np.nan


def sat_series(d, meth, cls, warmup):
    """Return (c_index array 1..29, satisfied 0/1, excess or nan, lambda).

    TraLO: sparse.  Every satisfied epoch IS logged
    (src/methodologies/tralo/train.py:430 -> `or is_satisfied`), so an epoch
    missing from the log is a VIOLATED epoch.  The satisfied vector is
    therefore exact for all 29 epochs; the excess/count trajectory is not.
    """
    t = pd.read_csv(os.path.join(d, "training_log.csv"))
    if meth == "tralo":
        ep = num(t["Epoch"])
        gs, ls = num(t["Global_Satisfied"]), num(t["Local_Satisfied"])
        ok = ep.notna() & gs.notna()
        ep, gs, ls = ep[ok].to_numpy(), gs[ok].to_numpy(), ls[ok].to_numpy()
        c_logged = ep.astype(int) - int(warmup)          # -> 1..29
        last_c = int(c_logged.max())
        satset = set(c_logged[(gs == 1) & (ls == 1)].tolist())
        c = np.arange(1, last_c + 1)
        sat = np.array([1 if x in satset else 0 for x in c])
        hard = num(t["Hard_Class%d" % cls])[ok].to_numpy()
        lam = np.nanmax(num(t["Lambda_Global"])[ok].to_numpy())
        return c, sat, c_logged, hard, float(lam), True
    ep = num(t["epoch"])
    exc = num(t["total_excess"])
    st = num(t["all_satisfied"])
    lcol = "max_lambda_g" if "max_lambda_g" in t.columns else "max_lam_g"
    lam = num(t[lcol])
    ok = ep.notna() & exc.notna()
    ep, exc, st, lam = [v[ok].to_numpy() for v in (ep, exc, st, lam)]
    c = ep.astype(int) + 1                                # -> 1..29
    return c, (st == 1).astype(int), c, exc, float(np.nanmax(lam)), False


def scan(root, want):
    rows = []
    for cfg_path in sorted(glob.glob(root + "/**/config.json", recursive=True)):
        try:
            cfg = json.load(open(cfg_path))
        except Exception:
            continue
        meth = cfg.get("methodology")
        if meth not in want:
            continue
        d = os.path.dirname(cfg_path)
        raw = os.path.join(d, "final_predictions_raw.csv")
        if not os.path.exists(raw):
            continue
        t = pd.read_csv(raw)
        cols = sorted((c for c in t.columns if c.startswith("Prob_Class_")),
                      key=lambda c: int(c.rsplit("_", 1)[1]))
        P = t[cols].to_numpy(float)
        y = t["True_Label"].to_numpy(int)
        rawp = t["Predicted_Label"].to_numpy(int)
        g = t["Group_ID"].to_numpy(int) if "Group_ID" in t.columns else None
        dc = cfg.get("dataset_config", {}) or {}
        cls = dc.get("constrained_class")
        cls = int(cls[0] if isinstance(cls, (list, tuple)) else cls)
        lp, gp = cfg["constraint"]
        G = compute_global_constraints(pd.DataFrame({"label": y, "grp": g if g is not None else 0}),
                                       "label", gp, constrained_class=[cls],
                                       num_classes=P.shape[1])
        if G[cls] >= UNLIMITED:
            continue
        K = int(G[cls])
        s = P[:, cls]
        pos = (y == cls).astype(int)
        order = np.argsort(-s)
        topK = order[:K]
        hp = cfg.get("hyperparams") or {}
        em = evalmet(d)
        r = {"path": d, "dataset": cfg.get("dataset_mode"), "cap": cfg.get("constraint_tag"),
             "model": cfg.get("model_name"), "seed": hp.get("seed"), "method": meth,
             "warmup": hp.get("warmup_epochs"), "K": K, "cls": cls,
             "count_raw": int((rawp == cls).sum()),
             "AP": average_precision_score(pos, s),
             "precAtK": float(pos[topK].mean()),
             "recAtK": float(pos[topK].sum() / max(1, pos.sum())),
             # score-degeneracy diagnostics: if the constrained-class score
             # collapses to a constant, AP falls for a reason that has nothing
             # to do with ranking the hard cases.
             "n_uniq_score": int(len(np.unique(np.round(s, 9)))),
             "frac_score_below_1e6": float((s < 1e-6).mean()),
             "max_score": float(s.max()),
             "sat_epoch_raw": ff(em.get("Satisfaction Epoch")),
             "restored_from_raw": ff(em.get("Restored From Epoch")),
             "restore_kind": em.get("Restore Kind", "") or "",
             "min_excess_raw": ff(em.get("Min Excess Epoch")),
             "min_total_excess": ff(em.get("Min Total Excess"))}
        if meth in TRAINED:
            c, sat, c_log, traj, lam, is_tralo = sat_series(d, meth, cls, r["warmup"])
            last_c = int(c.max())
            # unify epoch frames -> constraint-epoch index
            ev_c = (r["restored_from_raw"] - r["warmup"]) if is_tralo else r["restored_from_raw"]
            if not np.isfinite(ev_c):
                ev_c = last_c
            ev_c = int(min(max(ev_c, 1), last_c))
            first_sat_c = ((r["sat_epoch_raw"] - r["warmup"]) if is_tralo
                           else r["sat_epoch_raw"])
            keep = c <= ev_c
            r.update({
                "last_c": last_c, "eval_c": ev_c,
                "n_active_total": int((sat == 0).sum()),
                "n_sat_total": int(sat.sum()),
                "dose_active": int((sat[keep] == 0).sum()),
                "dose_sat": int(sat[keep].sum()),
                "dose_frac": float((sat[keep] == 0).mean()),
                "n_transitions": int((np.abs(np.diff(sat)) > 0).sum()) if last_c > 1 else 0,
                "first_sat_c": first_sat_c,
                "peak_lambda": lam,
                "restored": int(np.isfinite(r["restored_from_raw"])),
                "unspent": (K - r["count_raw"]) / K,
                "is_tralo": int(is_tralo),
            })
            k2 = c_log <= ev_c
            if is_tralo:
                h = traj[k2]
                r["traj_min_over_K"] = float(h.min() / K) if len(h) else np.nan
                r["traj_max_over_K"] = float(h.max() / K) if len(h) else np.nan
                r["traj_range_K"] = (float((h.max() - h.min()) / K) if len(h) else np.nan)
                r["cum_excess_K"] = float(np.maximum(0, h - K).sum() / K) if len(h) else np.nan
            else:
                e = traj[k2]
                r["traj_min_over_K"] = np.nan
                r["traj_max_over_K"] = float((e.max() + K) / K) if len(e) else np.nan
                r["traj_range_K"] = float((e.max() - e.min()) / K) if len(e) else np.nan
                r["cum_excess_K"] = float(e.sum() / K) if len(e) else np.nan
        rows.append(r)
    return pd.DataFrame(rows)


def fe_corr(d, feat, target="dAP", groups=GRP):
    """Fixed-effects Spearman: rank within group, then pool the residuals."""
    sub = d[np.isfinite(d[feat]) & np.isfinite(d[target])].copy()
    if len(sub) < 8:
        return np.nan, np.nan, 0, 0
    x, yv, ng = [], [], 0
    for _, g in sub.groupby(groups):
        if len(g) < 3 or g[feat].nunique() < 2:
            continue
        ng += 1
        x.append(g[feat].rank().to_numpy() - g[feat].rank().mean())
        yv.append(g[target].rank().to_numpy() - g[target].rank().mean())
    if ng < 2:
        return np.nan, np.nan, 0, ng
    x, yv = np.concatenate(x), np.concatenate(yv)
    r, p = spearmanr(x, yv)
    return float(r), float(p), len(x), ng


FEATS = ["dose_active", "dose_frac", "dose_sat", "n_active_total", "n_sat_total",
         "eval_c", "first_sat_c", "n_transitions", "peak_lambda", "unspent",
         "cum_excess_K", "traj_range_K", "restored", "n_uniq_score",
         "frac_score_below_1e6"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trained", default="results/headroom/headroom_b30_lrc0.0001_noceskip")
    ap.add_argument("--clip", default="results/headroom/headroom_b30")
    ap.add_argument("--out", default="paper/scripts/out_ap_damage2.csv")
    args = ap.parse_args()

    tr = scan(args.trained, TRAINED)
    cl = scan(args.clip, CLIP)
    key = ["dataset", "model", "seed"]
    ref = cl.groupby(key)[["AP", "precAtK", "recAtK", "count_raw", "n_uniq_score"]].mean()
    ref.columns = [c + "_ce" for c in ref.columns]
    d = tr.merge(ref.reset_index(), on=key, how="left")
    d["dAP"] = d["AP"] - d["AP_ce"]
    d["dPrecK"] = d["precAtK"] - d["precAtK_ce"]
    d.to_csv(args.out, index=False)
    print("trained %d   clip %d   wrote %s" % (len(tr), len(cl), args.out))

    print("\n" + "=" * 96)
    print("A. WHERE in the ranking is the AP lost?  AP deficit vs precision@K deficit")
    print("   precision@K = precision of the top-K scored items, i.e. exactly the")
    print("   part of the ranking the cap consumes.  Per cell, mean of 4 seeds.")
    print("=" * 96)
    t = d.pivot_table(index=["dataset", "model", "cap"], columns="method",
                      values=["dAP", "dPrecK"], aggfunc="mean")
    print(t.round(4).to_string())
    print("\n  correlation of dAP with dPrecK over all 144 runs: rho=%.3f p=%.2e"
          % spearmanr(d.dAP, d.dPrecK)[:2])
    print("  mean dAP %+.4f   mean dPrecK %+.4f" % (d.dAP.mean(), d.dPrecK.mean()))

    print("\n" + "=" * 96)
    print("B. FIXED-EFFECTS correlation with dAP.  dAP and the feature are ranked")
    print("   INSIDE each (dataset, backbone, cap, method) group, so cell, backbone,")
    print("   cap and method are all held fixed and only seed variation remains.")
    print("=" * 96)
    print("  %-22s %8s %10s %6s %5s" % ("feature", "rho", "p", "n", "grps"))
    for c in FEATS:
        if c not in d.columns:
            continue
        r, p, n, ng = fe_corr(d, c)
        if np.isfinite(r):
            print("  %-22s %8.3f %10.2e %6d %5d" % (c, r, p, n, ng))
        else:
            print("  %-22s %8s" % (c, "-"))

    print("\n  same, DERMMNIST only (where the damage is):")
    dd = d[d.dataset == "dermmnist"]
    print("  %-22s %8s %10s %6s %5s" % ("feature", "rho", "p", "n", "grps"))
    for c in FEATS:
        if c not in d.columns:
            continue
        r, p, n, ng = fe_corr(dd, c)
        if np.isfinite(r):
            print("  %-22s %8.3f %10.2e %6d %5d" % (c, r, p, n, ng))

    print("\n  same, TraLO only, all datasets:")
    dt = d[d.method == "tralo"]
    for c in FEATS:
        if c not in d.columns:
            continue
        r, p, n, ng = fe_corr(dt, c, groups=["dataset", "model", "cap"])
        if np.isfinite(r):
            print("  %-22s %8.3f %10.2e %6d %5d" % (c, r, p, n, ng))

    print("\n" + "=" * 96)
    print("C. BETWEEN-METHOD-CELL: 36 cell means.  Which feature orders the cells?")
    print("=" * 96)
    m = d.groupby(GRP)[FEATS + ["dAP", "dPrecK"]].mean().reset_index()
    print("  %-22s %8s %10s" % ("feature", "rho", "p"))
    for c in FEATS:
        v = m[c].to_numpy(float)
        ok = np.isfinite(v)
        if ok.sum() < 6 or len(set(v[ok])) < 3:
            print("  %-22s %8s" % (c, "-"))
            continue
        r, p = spearmanr(v[ok], m["dAP"].to_numpy(float)[ok])
        print("  %-22s %8.3f %10.2e   (n=%d cells)" % (c, r, p, int(ok.sum())))

    print("\n" + "=" * 96)
    print("D. THE RESTORE NATURAL EXPERIMENT.  The evaluated model is a snapshot from")
    print("   epoch `eval_c`, not from epoch 29.  Runs whose final model already")
    print("   satisfied are NOT rolled back; runs that ended violating are.")
    print("=" * 96)
    for ds, g in d.groupby("dataset"):
        print("\n  --- %s ---" % ds)
        print(g.groupby(["method", "restored"])[["dAP", "eval_c", "dose_active", "unspent"]]
              .agg(["mean", "count"]).round(3).to_string())

    print("\n" + "=" * 96)
    print("E. dAP versus eval_c (which epoch's weights were scored), DERM, per method")
    print("=" * 96)
    for meth, g in dd.groupby("method"):
        print("\n  %s" % meth)
        print(g[["model", "cap", "seed", "eval_c", "last_c", "dose_active",
                 "n_sat_total", "unspent", "restore_kind", "dAP", "dPrecK"]]
              .sort_values(["model", "cap", "seed"])
              .to_string(index=False, float_format=lambda x: "%.3f" % x))

    print("\n" + "=" * 96)
    print("F. SCORE DEGENERACY -- is the AP drop a collapse of the score itself?")
    print("=" * 96)
    print(d.groupby(["dataset", "method"])[["n_uniq_score", "n_uniq_score_ce",
                                            "frac_score_below_1e6", "max_score",
                                            "dAP"]].mean().round(4).to_string())

    print("\n" + "=" * 96)
    print("G. EXISTENCE PROOF: cells where constraint training did NOT cost ranking")
    print("   (mean over 4 seeds of dAP, and of dPrecK)")
    print("=" * 96)
    cm = d.groupby(GRP)[["dAP", "dPrecK", "dose_active", "n_sat_total", "unspent",
                         "eval_c"]].mean().reset_index()
    cm["verdict"] = np.where(cm.dAP >= -0.005, "NO DAMAGE",
                             np.where(cm.dAP >= -0.02, "mild", "DAMAGED"))
    print(cm.sort_values("dAP", ascending=False)
          .to_string(index=False, float_format=lambda x: "%.4f" % x))
    print("\n  counts:", cm.verdict.value_counts().to_dict())
    print("\n  TraLO cells with NO DAMAGE and the cap actually binding:")
    print(cm[(cm.method == "tralo")].sort_values("dAP", ascending=False)
          .to_string(index=False, float_format=lambda x: "%.4f" % x))
    return 0


if __name__ == "__main__":
    sys.exit(main())
