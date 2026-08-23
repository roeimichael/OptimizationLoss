"""How many epochs did each arm actually run a live CE loop, and at what lr?

The dual trainers disable the CE minibatch loop once train acc >= 0.995 twice
(src/methodologies/fioretto_ldf/train.py:131). When that fires, `ce_losses` is
empty and np.mean([]) writes NaN into training_log.csv, so the NaN is a reliable
marker of the epoch the CE loop went dark.

TraLO writes a different schema (capital Epoch, L_CE, Train_Acc) and its log is
SPARSE -- rows only every 5th epoch / on satisfaction -- so for TraLO the first
dark epoch can only be bracketed, and that is what is reported.

    python paper/scripts/ce_liveness.py --root results/headroom/headroom_b30_lrc0.0001
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

DUAL = ["fioretto_ldf", "hounie_rcl"]


def scan_one(d, cfg):
    log = os.path.join(d, "training_log.csv")
    hp = cfg.get("hyperparams") or {}
    r = {"method": cfg.get("methodology"), "dataset": cfg.get("dataset_mode"),
         "model": cfg.get("model_name"), "cap": cfg.get("constraint_tag"),
         "seed": hp.get("seed"), "warmup": hp.get("warmup_epochs"),
         "cepochs_budget": hp.get("constraint_epochs"),
         "lr": hp.get("lr"), "lr_c": hp.get("lr_constraint"),
         "ce_skip_in_config": hp.get("enable_ce_skip", "ABSENT(default True)")}
    if not os.path.exists(log):
        r["log"] = "none (post-hoc arm: warm-up only)"
        r["ce_epochs_at_lrc"] = 0
        r["last_epoch"] = 0
        return r
    t = pd.read_csv(log)
    lower = {c.lower(): c for c in t.columns}
    ecol = lower.get("epoch")
    if ecol is None:
        r["log"] = "no epoch column"
        return r
    ep = pd.to_numeric(t[ecol], errors="coerce")
    t = t[ep.notna()].copy()
    t[ecol] = ep[ep.notna()]
    r["last_epoch"] = float(t[ecol].max())
    r["n_rows"] = len(t)
    if "ce_loss" in t.columns:                      # dual schema, dense log
        ce = pd.to_numeric(t["ce_loss"], errors="coerce")
        live = ce.notna()
        r["ce_epochs_at_lrc"] = int(live.sum())
        dark = t.loc[~live.values, ecol]
        r["first_dark_epoch"] = float(dark.min()) if len(dark) else np.nan
        r["schema"] = "dual"
    elif "L_CE" in t.columns:                       # tralo schema, SPARSE log
        ce = pd.to_numeric(t["L_CE"], errors="coerce")
        dark = t.loc[(ce == 0).values, ecol]
        r["first_dark_epoch"] = float(dark.min()) if len(dark) else np.nan
        r["ce_epochs_at_lrc"] = (float(dark.min()) - 1 if len(dark)
                                 else r["last_epoch"])
        r["schema"] = "tralo(sparse: bracketed)"
    if "all_satisfied" in t.columns:
        s = pd.to_numeric(t["all_satisfied"], errors="coerce")
        sat = t.loc[(s == 1).values, ecol]
        r["first_sat_epoch"] = float(sat.min()) if len(sat) else np.nan
    elif "Global_Satisfied" in t.columns:
        s = pd.to_numeric(t["Global_Satisfied"], errors="coerce")
        sat = t.loc[(s == 1).values, ecol]
        r["first_sat_epoch"] = float(sat.min()) if len(sat) else np.nan
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, nargs="+")
    args = ap.parse_args()
    for root in args.root:
        rows = []
        for cfg_path in glob.glob(root + "/**/config.json", recursive=True):
            try:
                cfg = json.load(open(cfg_path))
            except Exception:
                continue
            rows.append(scan_one(os.path.dirname(cfg_path), cfg))
        t = pd.DataFrame(rows)
        if t.empty:
            print("%s: nothing" % root)
            continue
        print("=" * 104)
        print("ROOT %s   n=%d" % (root, len(t)))
        print("=" * 104)
        cols = [c for c in ["warmup", "cepochs_budget", "lr", "lr_c",
                            "ce_epochs_at_lrc", "first_dark_epoch",
                            "first_sat_epoch", "last_epoch"] if c in t.columns]
        print(t.groupby("method")[cols].mean().round(2).to_string())
        print("\n  enable_ce_skip as written in config.json:")
        print(t.groupby("method")["ce_skip_in_config"].agg(
            lambda s: ",".join(sorted({str(x) for x in s}))).to_string())
        if "first_dark_epoch" in t.columns:
            print("\n  runs whose CE loop went dark before the budget ended:")
            for m, g in t.groupby("method"):
                n = int(g.first_dark_epoch.notna().sum())
                print("    %-14s %d/%d   (median dark epoch %s of %s)"
                      % (m, n, len(g),
                         g.first_dark_epoch.median(), g.last_epoch.median()))
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
