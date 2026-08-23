"""When does CE saturate, RELATIVE to the start of the constraint phase?

The question that matters is not "does CE saturate" but "how many constraint
epochs are left once it has". At warm-up 1 the constraint phase starts at epoch
2, so if CE saturates at epoch 12 of 30 then 10 constraint epochs ran on a live
representation and 18 ran on a frozen one -- the warm-up-50 pathology recurring
from INSIDE the constraint phase, just later.

Saturation is read three ways because they disagree and the disagreement is the
point:
    acc995   first epoch with Train_Acc >= 0.995   (what enable_ce_skip watches)
    acc99    first epoch with Train_Acc >= 0.99    (a looser, earlier mark)
    ce_flat  first epoch after which L_CE never again falls by more than 10%
             of its value at that epoch -- i.e. CE has stopped making progress,
             which can happen well before accuracy pins at 1.0

Also reports, per run, whether the constraint was EVER satisfied and at which
epoch, so "saturated before it satisfied" is visible directly.
"""
import argparse
import glob
import json
import os

import numpy as np
import pandas as pd


def first_ge(ep, v, thr):
    m = np.where(v >= thr)[0]
    return int(ep[m[0]]) if len(m) else None


def ce_flat_epoch(ep, ce):
    """First epoch after which CE never drops >10% below its value there."""
    for i in range(len(ce) - 1):
        if ce[i] <= 0:
            continue
        if np.min(ce[i + 1:]) > 0.9 * ce[i]:
            return int(ep[i])
    return None


def run(cfg_path):
    cfg = json.load(open(cfg_path))
    d = os.path.dirname(cfg_path)
    f = os.path.join(d, "training_log.csv")
    if not os.path.exists(f):
        return None
    df = pd.read_csv(f)
    if df.empty or "Train_Acc" not in df:
        return None
    hp = cfg.get("hyperparams") or {}
    wu = int(hp.get("warmup_epochs", 0))
    ce_ep = int(hp.get("constraint_epochs", 0))
    ep = df["Epoch"].to_numpy()
    acc = df["Train_Acc"].to_numpy()
    ce = df["L_CE"].to_numpy() if "L_CE" in df else np.zeros(len(df))

    gs = df["Global_Satisfied"].to_numpy() if "Global_Satisfied" in df else None
    ls = df["Local_Satisfied"].to_numpy() if "Local_Satisfied" in df else None
    sat = None
    if gs is not None:
        both = (gs > 0) & (ls > 0) if ls is not None else (gs > 0)
        w = np.where(both)[0]
        sat = int(ep[w[0]]) if len(w) else None

    return {"arm": cfg.get("arm"), "dataset": cfg.get("dataset_mode"),
            "cap": cfg.get("constraint_tag"), "seed": hp.get("seed"),
            "warmup": wu, "con_epochs": ce_ep, "last_epoch": int(ep.max()),
            "acc995": first_ge(ep, acc, 0.995), "acc99": first_ge(ep, acc, 0.99),
            "ce_flat": ce_flat_epoch(ep, ce),
            "acc_final": float(acc[-1]), "ce_final": float(ce[-1]),
            "sat_epoch": sat, "ever_sat": sat is not None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--campaign", nargs="+", required=True)
    a = ap.parse_args()
    rows = []
    for camp in a.campaign:
        for p in glob.glob(camp + "/**/config.json", recursive=True):
            r = run(p)
            if r:
                rows.append(r)
    df = pd.DataFrame(rows)
    if df.empty:
        print("no logs")
        return
    trained = df[df.con_epochs > 0].copy()
    print("runs with a constraint phase: %d of %d" % (len(trained), len(df)))
    if trained.empty:
        return

    # constraint phase starts the epoch after warm-up
    trained["con_start"] = trained.warmup + 1
    for c in ("acc995", "acc99", "ce_flat"):
        trained["left_" + c] = trained[c] - trained.con_start

    print("\n%-14s %-10s %5s | %6s %6s %7s | %s"
          % ("arm", "dataset", "cap", "acc995", "acc99", "ce_flat",
             "constraint epochs left after saturation"))
    for (arm, ds, cap), g in trained.groupby(["arm", "dataset", "cap"]):
        def m(c):
            v = g[c].dropna()
            return ("%5.1f" % v.mean()) if len(v) else "   --"

        def left(c):
            v = g["left_" + c].dropna()
            if not len(v):
                return "  --"
            tot = g.con_epochs.iloc[0]
            return "%5.1f of %d (%.0f%% frozen)" % (
                v.mean(), tot, 100.0 * (1 - v.mean() / tot))
        print("%-14s %-10s %5s | %s  %s   %s | acc995: %s"
              % (arm, ds, cap, m("acc995"), m("acc99"), m("ce_flat"), left("acc995")))

    print("\nsatisfaction (constraint EVER satisfied at any epoch):")
    for (arm, ds, cap), g in trained.groupby(["arm", "dataset", "cap"]):
        n = len(g)
        k = int(g.ever_sat.sum())
        se = g.sat_epoch.dropna()
        print("   %-14s %-10s %-5s  %d/%d ever  first-sat mean %s  acc_final %.4f  ce_final %.4f"
              % (arm, ds, cap, k, n,
                 ("%.1f" % se.mean()) if len(se) else "never",
                 g.acc_final.mean(), g.ce_final.mean()))


if __name__ == "__main__":
    main()
