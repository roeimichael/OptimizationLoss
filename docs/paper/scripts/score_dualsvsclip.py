"""Score a campaign on BOTH metrics at once, so metric and regime can be separated.

analyze_headroom.rows_for gives the budget-equalized numbers only. The question
here is why the dual methods outranked the clippers in the old corpus and do not
in the new one, and one candidate is the metric itself: the old tables used the
constrained-class F1 of the SHIPPED predictions (final_predictions.csv, i.e.
after post-hoc adjustment), which lets a method that fills more of its quota
score higher without ranking anything better.

So for every run we record, per run:
  ccF1adj  -- constrained-class F1 on final_predictions.csv   (the OLD metric)
  ccF1eq   -- constrained-class F1 after re-allocating exactly K by score  (NEW)
  AP       -- allocation-free ranking quality
  macroAdj / macroEq
  count_raw (model's own count), count_adj (shipped count), K

    python paper/scripts/score_dualsvsclip.py --root results/pending_runs/paper_final \
        --out paper/scripts/out_paperfinal.csv
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score

sys.path.insert(0, os.getcwd())
sys.path.insert(0, "paper/scripts")
import analyze_headroom as A                                        # noqa: E402
from src.utils.constants import UNLIMITED                           # noqa: E402
from src.training.constraints import (compute_global_constraints,   # noqa: E402
                                      compute_local_constraints)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    out = []
    skipped = {"no_preds": 0, "unlimited": 0, "bad_cfg": 0}
    for cfg_path in glob.glob(args.root + "/**/config.json", recursive=True):
        try:
            cfg = json.load(open(cfg_path))
        except Exception:
            skipped["bad_cfg"] += 1
            continue
        d = os.path.dirname(cfg_path)
        raw = os.path.join(d, "final_predictions_raw.csv")
        fin = os.path.join(d, "final_predictions.csv")
        if not (os.path.exists(raw) and os.path.exists(fin)):
            skipped["no_preds"] += 1
            continue
        t = pd.read_csv(raw)
        cols = sorted((c for c in t.columns if c.startswith("Prob_Class_")),
                      key=lambda c: int(c.rsplit("_", 1)[1]))
        if not cols:
            skipped["no_preds"] += 1
            continue
        P = t[cols].to_numpy(float)
        y = t["True_Label"].to_numpy(int)
        rawp = t["Predicted_Label"].to_numpy(int)
        g = t["Group_ID"].to_numpy(int) if "Group_ID" in t.columns else None
        dc = cfg.get("dataset_config", {}) or {}
        cls = dc.get("constrained_class")
        if cls is None:
            skipped["bad_cfg"] += 1
            continue
        cls = int(cls[0] if isinstance(cls, (list, tuple)) else cls)
        lp, gp = cfg["constraint"]
        df = pd.DataFrame({"label": y, "grp": g if g is not None else 0})
        G = compute_global_constraints(df, "label", gp, constrained_class=[cls],
                                       num_classes=P.shape[1])
        L = compute_local_constraints(df, "label", lp, "grp",
                                      constrained_class=[cls],
                                      num_classes=P.shape[1])
        if G[cls] >= UNLIMITED:
            skipped["unlimited"] += 1
            continue
        rel = pd.read_csv(fin)["Predicted_Label"].to_numpy(int)
        eq = A.equalize(P, g, G, L, cls)
        hp = cfg.get("hyperparams") or {}
        out.append({
            "campaign": args.root,
            "dataset": cfg.get("dataset_mode"), "cap": cfg.get("constraint_tag"),
            "model": cfg.get("model_name"), "seed": hp.get("seed"),
            "method": cfg.get("methodology"), "sweep": cfg.get("sweep"),
            "arm": cfg.get("arm"),
            "warmup": hp.get("warmup_epochs"), "cepochs": hp.get("constraint_epochs"),
            "lr": hp.get("lr"), "lr_c": hp.get("lr_constraint"),
            "ce_skip": hp.get("enable_ce_skip"),
            "K": int(G[cls]),
            "count_raw": int((rawp == cls).sum()),
            "count_adj": int((rel == cls).sum()),
            "sat": int((rawp != rel).sum() == 0),
            "AP": average_precision_score((y == cls).astype(int), P[:, cls]),
            "ccF1adj": f1_score(y, rel, labels=[cls], average="macro", zero_division=0),
            "ccF1eq": f1_score(y, eq, labels=[cls], average="macro", zero_division=0),
            "macroAdj": f1_score(y, rel, average="macro", zero_division=0),
            "macroEq": f1_score(y, eq, average="macro", zero_division=0),
            "path": d,
        })
    o = pd.DataFrame(out)
    o.to_csv(args.out, index=False)
    print("scored %d runs from %s -> %s   skipped=%s" %
          (len(o), args.root, args.out, skipped))
    if len(o):
        print(o.groupby("method")[["warmup", "cepochs", "lr", "lr_c"]].mean().to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
