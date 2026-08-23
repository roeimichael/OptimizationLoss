"""Is the win LOCAL to the cap, or global?

The panel says AP and AUROC tie while TP-at-budget-K wins significantly. Those
two facts can only both be true if the constraint reorders the score list
*near the cap* and leaves the rest of the ordering alone -- AP integrates over
every threshold, so a change confined to one of them washes out.

So: sweep the budget. Count true positives in the top-k for k spanning an
order of magnitude around K. If the advantage peaks at k=K and decays either
side, the constraint sharpens the boundary exactly where it is imposed. If it
is flat across k, this is an ordinary ranking win that AP simply lacked the
power to see, and the local story is wrong.

Reported as TP fraction (TP/k) so the columns are comparable across budgets.
"""
import argparse
import glob
import importlib.util as ilu
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

_HERE = os.path.dirname(os.path.abspath(__file__))
_spec = ilu.spec_from_file_location("_sa", os.path.join(_HERE, "score_arm.py"))
_sa = ilu.module_from_spec(_spec)
_spec.loader.exec_module(_sa)

MULT = [0.125, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 4.0, 8.0]


def row(run_dir, cfg):
    raw = os.path.join(run_dir, "final_predictions_raw.csv")
    if not os.path.exists(raw):
        return None
    t = pd.read_csv(raw)
    cols = sorted((c for c in t.columns if c.startswith("Prob_Class_")),
                  key=lambda c: int(c.rsplit("_", 1)[1]))
    if not cols:
        return None
    P = t[cols].to_numpy(float)
    P = P / np.clip(P.sum(axis=1, keepdims=True), 1e-12, None)
    y = t["True_Label"].to_numpy(int)
    dc = cfg.get("dataset_config", {}) or {}
    cls = dc.get("constrained_class")
    cls = int(cls[0] if isinstance(cls, (list, tuple)) else cls)
    lp, gp = cfg["constraint"]
    g = t["Group_ID"].to_numpy(int) if "Group_ID" in t.columns else None
    d = pd.DataFrame({"label": y, "grp": g if g is not None else 0})
    G = _sa.compute_global_constraints(d, "label", gp, constrained_class=[cls],
                                       num_classes=P.shape[1])
    if G[cls] >= _sa.UNLIMITED:
        return None
    K = int(G[cls])

    # Rank by the constrained-class score, exactly what the cap thresholds.
    order = np.argsort(-P[:, cls])
    hit = (y[order] == cls).astype(int)
    cum = np.cumsum(hit)

    out = {"dataset": cfg.get("dataset_mode"), "model": cfg.get("model_name"),
           "cap": cfg.get("constraint_tag"),
           "seed": (cfg.get("hyperparams") or {}).get("seed"),
           "arm": cfg.get("arm"), "K": K, "npos": int((y == cls).sum())}
    for m in MULT:
        k = int(round(m * K))
        k = max(1, min(k, len(y)))
        out["p@%gK" % m] = cum[k - 1] / float(k)
    return out


def main():
    a = argparse.ArgumentParser()
    a.add_argument("--campaign", required=True, nargs="+")
    a.add_argument("--control", required=True)
    a.add_argument("--arm", required=True)
    args = a.parse_args()

    rows = []
    for camp in args.campaign:
        for p in glob.glob(camp + "/**/config.json", recursive=True):
            try:
                cfg = json.load(open(p))
            except Exception:
                continue
            r = row(os.path.dirname(p), cfg)
            if r:
                rows.append(r)
    if not rows:
        sys.exit("nothing scorable")
    df = pd.DataFrame(rows)
    key = ["dataset", "model", "cap", "seed"]

    print("K per cell:", df.groupby(["dataset", "model", "cap"]).K.first().to_dict())
    print("n_pos     :", df.groupby(["dataset", "model", "cap"]).npos.first().to_dict())
    print()
    print("%s  minus  %s   -- true-positive fraction in the top-k" % (args.arm, args.control))
    print("%-8s %10s %10s %10s %8s %9s" % ("budget", "control", args.arm, "delta", "cells", "wilcoxon"))
    for m in MULT:
        c_ = "p@%gK" % m
        q = df.pivot_table(index=key, columns="arm", values=c_).dropna()
        if args.control not in q or args.arm not in q:
            continue
        c, t = q[args.control], q[args.arm]
        d = t - c
        try:
            pv = stats.wilcoxon(t, c)[1]
        except Exception:
            pv = np.nan
        star = " <<<" if pv < 0.05 and d.mean() > 0 else (" LOSS" if pv < 0.05 else "")
        print("%-8s %10.4f %10.4f %+10.4f %5d/%-3d %9.4f%s"
              % (c_, c.mean(), t.mean(), d.mean(), int((d > 0).sum()), len(d), pv, star))


if __name__ == "__main__":
    main()
