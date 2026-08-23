"""Backbone interaction, step 2: trajectories + pre-treatment calibration.

Two questions the fact base cannot answer:

  (1) How far does each method actually MOVE the hard count during training, per
      backbone?  TraLO logs Hard_Class{c}/Soft_Class{c}/Global_Satisfied; the
      duals log only total_excess/all_satisfied, so they are read separately and
      never compared on a column they do not share.

  (2) Is the backbone difference a property of the UNCONSTRAINED model?  The
      post-hoc clipper arm in headroom_b30 is plain 30-epoch CE, so its
      final_predictions_raw.csv is the pre-treatment probability surface.  From
      it: soft count (what the loss sees) vs hard count (what satisfaction
      uses), and the margin distribution (how many argmax decisions sit close
      enough to the boundary to be cheap to flip).

Schema traps honoured: sparse TraLO log -> Epoch.max(), case differs by method,
repeated headers -> to_numeric+dropna, constrained class from config.
"""
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

CAMP = "results/headroom/headroom_b30_lrc0.0001_noceskip"
CLIP = "results/headroom/headroom_b30"


def cfg_of(d):
    return json.load(open(os.path.join(d, "config.json")))


def cls_of(cfg):
    c = (cfg.get("dataset_config") or {}).get("constrained_class")
    return int(c[0] if isinstance(c, (list, tuple)) else c)


def num(df, col):
    return pd.to_numeric(df[col], errors="coerce")


def traj_rows(root):
    out = []
    for cp in glob.glob(root + "/**/config.json", recursive=True):
        d = os.path.dirname(cp)
        lp = os.path.join(d, "training_log.csv")
        if not os.path.exists(lp):
            continue
        cfg = cfg_of(cp.replace("/config.json", "") if False else d)
        m = cfg.get("methodology")
        cls = cls_of(cfg)
        hp = cfg.get("hyperparams") or {}
        base = {"dataset": cfg.get("dataset_mode"), "model": cfg.get("model_name"),
                "cap": cfg.get("constraint_tag"), "seed": hp.get("seed"),
                "method": m, "dir": d}
        try:
            lg = pd.read_csv(lp)
        except Exception:
            continue
        if m == "tralo":
            need = ["Epoch", "Hard_Class%d" % cls, "Soft_Class%d" % cls,
                    "Global_Satisfied", "Lambda_Global", "Train_Acc", "L_CE"]
            if any(c not in lg.columns for c in need):
                continue
            t = pd.DataFrame({c: num(lg, c) for c in need}).dropna(subset=["Epoch"])
            t = t.sort_values("Epoch")
            # TraLO Epoch is 1-indexed over warm-up(1)+constraint(29)
            t["cep"] = t["Epoch"] - 1
            t = t[t.cep >= 1]
            if t.empty:
                continue
            hard, soft = t["Hard_Class%d" % cls], t["Soft_Class%d" % cls]
            satm = t["Global_Satisfied"] > 0
            base.update({
                "n_log_rows": len(t), "max_cep": float(t.cep.max()),
                "hard_first": float(hard.iloc[0]), "hard_last": float(hard.iloc[-1]),
                "hard_min": float(hard.min()),
                "soft_first": float(soft.iloc[0]), "soft_last": float(soft.iloc[-1]),
                "soft_minus_hard_last": float(soft.iloc[-1] - hard.iloc[-1]),
                "first_sat_cep": float(t.cep[satm].min()) if satm.any() else np.nan,
                "n_sat_rows": int(satm.sum()),
                "lam_last": float(t["Lambda_Global"].iloc[-1]),
                "lam_max": float(t["Lambda_Global"].max()),
                "tracc_last": float(t["Train_Acc"].iloc[-1]),
                "tracc_first": float(t["Train_Acc"].iloc[0]),
                "ce_last": float(t["L_CE"].iloc[-1]),
            })
        elif m in ("fioretto_ldf", "hounie_rcl"):
            if "epoch" not in lg.columns or "total_excess" not in lg.columns:
                continue
            t = pd.DataFrame({c: num(lg, c) for c in
                              ["epoch", "total_excess", "all_satisfied",
                               "ce_loss", "max_lambda_g"]
                              if c in lg.columns}).dropna(subset=["epoch"])
            t = t.sort_values("epoch")
            t["cep"] = t["epoch"] + 1          # duals are 0-indexed, constraint only
            satm = t["all_satisfied"] > 0
            base.update({
                "n_log_rows": len(t), "max_cep": float(t.cep.max()),
                "excess_first": float(t.total_excess.iloc[0]),
                "excess_last": float(t.total_excess.iloc[-1]),
                "excess_min": float(t.total_excess.min()),
                "first_sat_cep": float(t.cep[satm].min()) if satm.any() else np.nan,
                "n_sat_rows": int(satm.sum()),
                "lam_last": float(t.max_lambda_g.iloc[-1]) if "max_lambda_g" in t else np.nan,
                "ce_last": float(t.ce_loss.iloc[-1]) if "ce_loss" in t else np.nan,
            })
        else:
            continue
        out.append(base)
    return pd.DataFrame(out)


def surface_rows(root, methods):
    """Pre/post-treatment probability surface from final_predictions_raw.csv."""
    out = []
    for cp in glob.glob(root + "/**/config.json", recursive=True):
        d = os.path.dirname(cp)
        raw = os.path.join(d, "final_predictions_raw.csv")
        if not os.path.exists(raw):
            continue
        cfg = cfg_of(d)
        if cfg.get("methodology") not in methods:
            continue
        cls = cls_of(cfg)
        t = pd.read_csv(raw)
        cols = sorted((c for c in t.columns if c.startswith("Prob_Class_")),
                      key=lambda c: int(c.rsplit("_", 1)[1]))
        P = t[cols].to_numpy(float)
        y = t["True_Label"].to_numpy(int)
        pred = t["Predicted_Label"].to_numpy(int)
        pc = P[:, cls]
        other = P.copy(); other[:, cls] = -np.inf
        best_other = other.max(axis=1)
        margin = pc - best_other          # >0 <=> argmax is the constrained class
        hp = cfg.get("hyperparams") or {}
        hard = int((pred == cls).sum())
        soft = float(pc.sum())
        # how many decisions sit within eps of the boundary (cheap to flip)
        near = {("near%.2f" % e): int((np.abs(margin) < e).sum())
                for e in (0.05, 0.10, 0.20)}
        r = {"dataset": cfg.get("dataset_mode"), "model": cfg.get("model_name"),
             "cap": cfg.get("constraint_tag"), "seed": hp.get("seed"),
             "method": cfg.get("methodology"),
             "hard": hard, "soft": soft, "soft_minus_hard": soft - hard,
             "n_pool": len(y), "n_true": int((y == cls).sum()),
             "pc_mean": float(pc.mean()), "pc_max": float(pc.max()),
             "maxprob_mean": float(P.max(axis=1).mean()),
             "margin_pos_mean": float(margin[margin > 0].mean()) if (margin > 0).any() else np.nan,
             }
        r.update(near)
        # how much probability mass sits above the K-th ranked score, and how
        # steep the score curve is there: the local slope says how many samples
        # a small score perturbation moves across the cut.
        out.append(r)
    return pd.DataFrame(out)


def main():
    CELL = ["dataset", "model", "cap"]
    tj = traj_rows(CAMP)
    print("trajectory rows: %d   methods: %s" % (len(tj), sorted(tj.method.unique())))

    tr = tj[tj.method == "tralo"]
    print("\n" + "=" * 118)
    print("TraLO count trajectory, per cell (mean over 4 seeds).")
    print("hard_first = hard count at the FIRST logged constraint epoch;")
    print("hard_last  = at the last.  soft_minus_hard_last = calibration gap the")
    print("loss sees vs what satisfaction uses.")
    print("=" * 118)
    g = tr.groupby(CELL).agg(
        n=("seed", "size"), hard_first=("hard_first", "mean"),
        hard_last=("hard_last", "mean"), hard_min=("hard_min", "mean"),
        soft_first=("soft_first", "mean"), soft_last=("soft_last", "mean"),
        soft_gap=("soft_minus_hard_last", "mean"),
        first_sat=("first_sat_cep", "mean"), nsat=("n_sat_rows", "mean"),
        lam_last=("lam_last", "mean"), lam_max=("lam_max", "mean"),
        tracc=("tracc_last", "mean"), maxcep=("max_cep", "mean"),
    ).reset_index()
    print(g.to_string(index=False, float_format=lambda x: "%.3f" % x))

    print("\n" + "=" * 118)
    print("DUALS: total_excess trajectory + first satisfied constraint epoch")
    print("=" * 118)
    du = tj[tj.method.isin(["fioretto_ldf", "hounie_rcl"])]
    gd = du.groupby(CELL + ["method"]).agg(
        n=("seed", "size"), exc_first=("excess_first", "mean"),
        exc_last=("excess_last", "mean"), exc_min=("excess_min", "mean"),
        first_sat=("first_sat_cep", "mean"), nsat=("n_sat_rows", "mean"),
        maxcep=("max_cep", "mean"),
    ).reset_index()
    print(gd.to_string(index=False, float_format=lambda x: "%.3f" % x))

    print("\n" + "=" * 118)
    print("PRE-TREATMENT SURFACE: plain-CE clipper arm (headroom_b30, heuristic)")
    print("soft = sum_i P(i,cls) = what a count penalty actually differentiates;")
    print("hard = argmax count.  nearX = |P_cls - best_other| < X, i.e. decisions")
    print("cheap enough to flip.")
    print("=" * 118)
    sf = surface_rows(CLIP, {"heuristic"})
    gs = sf.groupby(["dataset", "model"]).agg(
        n=("seed", "size"), hard=("hard", "mean"), soft=("soft", "mean"),
        soft_minus_hard=("soft_minus_hard", "mean"),
        pc_mean=("pc_mean", "mean"), maxprob=("maxprob_mean", "mean"),
        near05=("near0.05", "mean"), near10=("near0.10", "mean"),
        near20=("near0.20", "mean"), n_pool=("n_pool", "first"),
    ).reset_index()
    gs["near10_frac_of_hard"] = gs["near10"] / gs["hard"]
    print(gs.to_string(index=False, float_format=lambda x: "%.3f" % x))
    sf.to_csv("paper/scripts/out_bb_surface_clip.csv", index=False)
    tj.to_csv("paper/scripts/out_bb_traj.csv", index=False)
    print("\nwrote paper/scripts/out_bb_traj.csv, out_bb_surface_clip.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
