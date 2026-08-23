"""The two-force test, read off training_log.csv.

The premise of TraLO is that the cap pushes the constrained count DOWN while the
base loss keeps training the model toward quality. Those are two claims about a
trajectory, and the training log can test both without waiting for a scorecard:

  HOLD   does the hard count sit AT the cap, or oscillate across it?
         Reported as the fraction of constraint epochs satisfied, and the count
         path itself.
  LEARN  does L_CE keep FALLING while the cap is held? If CE is flat or rising
         once the cap binds, the two forces are fighting, not cooperating --
         which is the alternating scheme's predicted failure and the whole
         reason `joint` exists.

Usage:  python two_force.py <campaign_dir> [<campaign_dir> ...]
"""
import glob
import json
import os
import sys

import numpy as np
import pandas as pd


def cls_of(cfg):
    dc = cfg.get("dataset_config", {}) or {}
    c = dc.get("constrained_class")
    return int(c[0] if isinstance(c, (list, tuple)) else c)


def arm_of(cfg, d):
    hp = cfg.get("hyperparams") or {}
    for k in ("joint_objective", "ortho_project", "constraint_as_reweight"):
        if k in hp:
            return "%s=%s" % (k, hp[k])
    for k in ("base_loss", "rank_weight", "constraint_clip_norm"):
        if k in hp:
            return "%s=%s" % (k, hp[k])
    return os.path.basename(os.path.dirname(d))


def rows(campaign):
    out = []
    for p in sorted(glob.glob(campaign + "/**/training_log.csv", recursive=True)):
        d = os.path.dirname(p)
        try:
            cfg = json.load(open(os.path.join(d, "config.json")))
        except Exception:
            continue
        hp = cfg.get("hyperparams") or {}
        try:
            df = pd.read_csv(p)
        except Exception:
            continue
        if not len(df) or "Epoch" not in df:
            continue
        w = hp.get("warmup_epochs", 0)
        c = cls_of(cfg)
        hard = "Hard_Class%d" % c
        lim = "Limit_Class%d" % c
        if hard not in df or lim not in df:
            continue
        # constraint-phase rows only. TraLO logs ABSOLUTE Epoch, and it logs
        # SPARSELY, so never use row count as an epoch count.
        cp = df[df.Epoch > w]
        if len(cp) < 2:
            continue
        K = float(cp[lim].iloc[0])
        cnt = cp[hard].to_numpy(float)
        ce = cp["L_CE"].to_numpy(float)
        sat = ((cp.get("Global_Satisfied", 0).astype(float) > 0)
               & (cp.get("Local_Satisfied", 1).astype(float) > 0))
        # does CE still fall while the cap is being held?
        held = sat.to_numpy()
        ce_held = ce[held]
        # DENOMINATOR. TraLO logs sparsely -- train.py:584
        #     if (epoch + 1) % 5 == 0 or is_satisfied or epoch == warmup_epochs
        # -- so EVERY satisfied epoch is written, but unsatisfied ones only
        # every 5th. Dividing satisfied rows by LOGGED rows therefore measures
        # satisfaction on a sample deliberately enriched with satisfied epochs,
        # and it inflates arms that rarely satisfy (they log ~7 of 29 rows)
        # far more than arms that usually do (they log ~28 of 29). Because
        # every satisfied epoch is logged, the satisfied COUNT is exact, so
        # divide it by the epoch BUDGET instead.
        budget = hp.get("constraint_epochs") or len(cp)
        out.append({
            "arm": arm_of(cfg, p),
            "ds": cfg.get("dataset_mode"), "model": cfg.get("model_name"),
            "cap": cfg.get("constraint_tag"),
            "seed": hp.get("seed"),
            "K": K,
            "sat_frac": float(held.sum()) / float(budget),
            "logged_of_budget": "%d/%d" % (len(cp), budget),
            "count_first": cnt[0], "count_last": cnt[-1],
            "count_over_K_last": cnt[-1] / K if K else np.nan,
            "count_cv": float(np.std(cnt) / max(1e-9, np.mean(cnt))),
            "cross": int(np.sum(np.diff((cnt <= K).astype(int)) != 0)),
            "lam_max": float(cp["Lambda_Global"].max()),
            "ce_first": ce[0], "ce_last": ce[-1],
            "ce_fell_while_held": (float(ce_held[0] - ce_held[-1])
                                   if len(ce_held) >= 2 else np.nan),
            "n_logged": len(cp),
        })
    return out


def main():
    allr = []
    for c in sys.argv[1:]:
        r = rows(c)
        for x in r:
            x["campaign"] = os.path.basename(c.rstrip("/"))
        allr += r
    if not allr:
        sys.exit("no usable training logs")
    df = pd.DataFrame(allr)
    pd.set_option("display.width", 220)
    # The atomic cell is (dataset, backbone, cap). Pooling across cells hides
    # sign flips, so group by the cell and never average over it.
    g = df.groupby(["ds", "model", "cap", "campaign", "arm"]).agg(
        n=("seed", "count"),
        sat_frac=("sat_frac", "mean"),
        cross=("cross", "mean"),
        cnt_over_K=("count_over_K_last", "mean"),
        cnt_cv=("count_cv", "mean"),
        lam_max=("lam_max", "mean"),
        ce_first=("ce_first", "mean"),
        ce_last=("ce_last", "mean"),
        ce_fell_held=("ce_fell_while_held", "mean"),
    ).round(4)
    print(g.to_string())
    print()
    print("sat_frac      satisfied epochs / constraint-epoch BUDGET. Exact, not a")
    print("              sample: the logger writes every satisfied epoch (train.py:584),")
    print("              so the satisfied count is complete even though the log is sparse.")
    print("n_logged      rows written / budget. A low number is itself informative --")
    print("              an arm that rarely satisfies only gets its every-5th-epoch rows.")
    print("cross         how many times the count crossed the cap (0 = held, high = oscillating)")
    print("cnt_over_K    realized count / cap at the last logged epoch")
    print("lam_max       peak lambda_global; if it stays at its initial value the")
    print("              ratchet never had to escalate, i.e. the cap held on its own")
    print("ce_fell_held  L_CE(first held epoch) - L_CE(last held epoch). POSITIVE means")
    print("              CE kept improving WHILE the cap was held -- the two forces")
    print("              cooperating. <= 0 means holding the cap stalled learning.")


if __name__ == "__main__":
    main()
