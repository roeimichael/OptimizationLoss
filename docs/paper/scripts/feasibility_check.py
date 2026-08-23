"""Do the SHIPPED predictions actually satisfy the caps they were adjusted for?

`audit_findings.md` carries an open warning that the post-hoc local pass can
re-violate a global cap it had already satisfied: phase 3 enforces per-group
limits AFTER phases 1-2 have balanced the global counts, and moving a sample
into a class to satisfy a local cap can push that class back over its global
budget. There is a re-verify step, but whether it always closes is untested.

That warning has never been checked against real output. If it fires, then some
runs ship infeasible predictions -- and every metric computed from them is
describing a solution that does not satisfy the problem.

Checks each run's final_predictions.csv against the caps that run enforced,
reconstructed with the repo's own `compute_global_constraints` /
`compute_local_constraints` (verified elsewhere to reproduce the runs' logged
`Limit_Class*` exactly).

Reports violations separately for global and local, because they fail for
different reasons, and prints the worst offenders rather than a bare count.
"""
import argparse
import glob
import importlib.util as ilu
import json
import os
import sys

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
_spec = ilu.spec_from_file_location("_sa", os.path.join(_HERE, "score_arm.py"))
_sa = ilu.module_from_spec(_spec)
_spec.loader.exec_module(_sa)


def check(cfg_path):
    cfg = json.load(open(cfg_path))
    d = os.path.dirname(cfg_path)
    f = os.path.join(d, "final_predictions.csv")
    if not os.path.exists(f):
        return None
    dc = cfg.get("dataset_config") or {}
    raw = dc.get("constrained_class")
    if raw is None:
        return None
    classes = [int(c) for c in raw] if isinstance(raw, (list, tuple)) else [int(raw)]
    t = pd.read_csv(f)
    y = t["True_Label"].to_numpy(int)
    pred = t["Predicted_Label"].to_numpy(int)
    g = t["Group_ID"].to_numpy(int) if "Group_ID" in t.columns else np.zeros(len(y), int)
    n_cls = len([c for c in t.columns if c.startswith("Prob_Class_")])
    lp, gp = cfg["constraint"]
    df = pd.DataFrame({"label": y, "grp": g})
    G = _sa.compute_global_constraints(df, "label", gp, constrained_class=classes,
                                       num_classes=n_cls)
    L = _sa.compute_local_constraints(df, "label", lp, "grp",
                                      constrained_class=classes, num_classes=n_cls)
    gv, lv = 0, 0
    worst_g, worst_l = 0, 0
    for c in classes:
        if G[c] >= _sa.UNLIMITED:
            continue
        over = int((pred == c).sum()) - int(G[c])
        if over > 0:
            gv += 1
            worst_g = max(worst_g, over)
    for gid, lim in (L or {}).items():
        for c in classes:
            if c < len(lim) and lim[c] < _sa.UNLIMITED:
                over = int(((pred == c) & (g == gid)).sum()) - int(lim[c])
                if over > 0:
                    lv += 1
                    worst_l = max(worst_l, over)
    return {"arm": cfg.get("arm"), "dataset": cfg.get("dataset_mode"),
            "cap": cfg.get("constraint_tag"),
            "seed": (cfg.get("hyperparams") or {}).get("seed"),
            "n_classes_capped": len(classes),
            "global_violations": gv, "worst_global_over": worst_g,
            "local_violations": lv, "worst_local_over": worst_l,
            "run": d}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--campaign", nargs="+", required=True)
    a = ap.parse_args()
    rows = []
    for camp in a.campaign:
        for p in glob.glob(camp + "/**/config.json", recursive=True):
            r = check(p)
            if r:
                rows.append(r)
    df = pd.DataFrame(rows)
    if df.empty:
        print("no runs")
        return
    print("runs checked: %d" % len(df))
    gbad = df[df.global_violations > 0]
    lbad = df[df.local_violations > 0]
    print("  GLOBAL cap violated in %d / %d runs" % (len(gbad), len(df)))
    print("  LOCAL  cap violated in %d / %d runs" % (len(lbad), len(df)))
    for name, bad, col in (("GLOBAL", gbad, "worst_global_over"),
                           ("LOCAL", lbad, "worst_local_over")):
        if len(bad):
            print("\n  worst %s offenders:" % name)
            cols = ["arm", "dataset", "cap", "seed", "n_classes_capped", col]
            print(bad.sort_values(col, ascending=False)[cols]
                  .head(10).to_string(index=False))
            print("\n  by arm: %s" % bad.groupby("arm").size().to_dict())


if __name__ == "__main__":
    main()
