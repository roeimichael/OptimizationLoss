"""Re-score every run at EQUAL realized count. RUN ON THE SERVER.

The paper's constrained-class comparison is not budget-matched: the post-hoc
clippers allocate exactly K by construction, while the trained duals finish
below it and simply stop short. Part of any cc-F1 gap is therefore quota
utilization rather than better ranking, and the control that separates the two
is to fill every arm up to exactly K and re-score.

No retraining is needed. Each run directory stores `final_predictions_raw.csv`
with the per-class probabilities and the group id, which is everything
`targeted_correction` consumes, so this is a pure re-evaluation of models that
have already been trained.

For each run it emits the as-released numbers and the budget-equalized ones side
by side, plus the realized constrained-class count before and after, so the size
of the confound is visible per cell rather than argued about.

    python paper/scripts/budget_equalize.py -o /tmp/budget_equalized.csv \
        [--roots results/pending_runs ...] [--datasets octmnist] [--caps L30_G30 L40_G40]

Then copy to paper/data/corpus/budget_equalized.csv.
"""
import argparse
import csv
import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score

sys.path.insert(0, os.getcwd())
from src.utils.constants import UNLIMITED                      # noqa: E402
from src.utils.posthoc_adjustment import targeted_correction   # noqa: E402
from src.training.constraints import (compute_global_constraints,   # noqa: E402
                                      compute_local_constraints)

FIELDS = ["campaign", "dataset", "model", "method", "cap", "seed", "warmup",
          "cls", "K_global",
          "count_raw", "count_released", "count_equalized",
          "cc_f1_released", "cc_f1_equalized",
          "cc_prec_released", "cc_prec_equalized",
          "cc_rec_released", "cc_rec_equalized",
          "f1_macro_released", "f1_macro_equalized",
          "flips_released", "flips_equalized", "config_path"]


def limits_from(cfg, y_true, gids, n_classes):
    """Rebuild the exact caps the run was trained and verified against.

    The config stores percentages, not counts -- `constraint` is
    [local_percent, global_percent] and the pipeline turns those into integer
    budgets from the TEST-set class counts (src/utils/data_loader.py). The raw
    predictions file carries the same true labels and group ids, so calling the
    pipeline's own two functions here reproduces the run's caps exactly rather
    than re-deriving them and hoping the rounding agrees.
    """
    dc = cfg.get("dataset_config", {}) or {}
    cls = dc.get("constrained_class")
    if cls is None:
        return None, None, None
    classes = cls if isinstance(cls, (list, tuple)) else [cls]
    local_pct, global_pct = cfg["constraint"]
    df = pd.DataFrame({"label": y_true,
                       "grp": gids if gids is not None else 0})
    glob = compute_global_constraints(df, "label", global_pct,
                                      constrained_class=classes,
                                      num_classes=n_classes)
    loc = compute_local_constraints(df, "label", local_pct, "grp",
                                    constrained_class=classes,
                                    num_classes=n_classes)
    return glob, loc, classes


def equalize(y_proba, gids, glob, loc, cls):
    """Spend exactly the budget: label the K best-scoring samples class `cls`,
    subject to each group's own cap, and give every other sample its best
    remaining class.

    This deliberately does NOT reuse the pipeline's verification step. That step
    is greedy-then-LP and, at symmetric caps, the two disagree: the per-group
    caps sum to one more than the global cap (they are rounded independently),
    so the local fill pushes the count one over, the LP fallback fires, and the
    run lands back where it started -- which is why `force_exact=True` leaves an
    under-target model under target.

    The rule here is the one the post-hoc clippers already follow, so applying it
    to every arm is what makes the comparison budget-matched: the arms then
    differ only in the probabilities they produce, not in how much of the quota
    they were allowed to spend. Greedy by score is optimal for this structure --
    a cardinality bound intersected with a partition matroid on the groups.
    """
    n = len(y_proba)
    K = int(glob[cls])
    order = np.argsort(-y_proba[:, cls])
    room = {}
    if gids is not None and loc:
        for g, lim in loc.items():
            room[int(g)] = int(lim[cls])
    chosen = np.zeros(n, dtype=bool)
    taken = 0
    for i in order:
        if taken >= K:
            break
        if room:
            g = int(gids[i])
            if room.get(g, 0) <= 0:
                continue
            room[g] -= 1
        chosen[i] = True
        taken += 1
    other = y_proba.copy()
    other[:, cls] = -np.inf
    y = np.argmax(other, axis=1)
    y[chosen] = cls
    return y, taken


def score(y_true, y_pred, cls):
    return (float(f1_score(y_true, y_pred, labels=[cls], average="macro", zero_division=0)),
            float(precision_score(y_true, y_pred, labels=[cls], average="macro", zero_division=0)),
            float(recall_score(y_true, y_pred, labels=[cls], average="macro", zero_division=0)),
            float(f1_score(y_true, y_pred, average="macro", zero_division=0)))


def campaign_of(path, root):
    rel = os.path.relpath(path, root).replace("\\", "/").split("/")
    for part in rel:
        if part.lower().startswith("lane"):
            continue
        return part
    return os.path.basename(root)


def row_for(cfg_path, root, args):
    d = os.path.dirname(cfg_path)
    raw = os.path.join(d, "final_predictions_raw.csv")
    fin = os.path.join(d, "final_predictions.csv")
    if not (os.path.exists(raw) and os.path.exists(fin)):
        return None
    try:
        cfg = json.load(open(cfg_path))
    except Exception:
        return None
    if args.datasets and cfg.get("dataset_mode") not in args.datasets:
        return None
    if args.caps and cfg.get("constraint_tag") not in args.caps:
        return None
    if args.methods and cfg.get("methodology") not in args.methods:
        return None

    r = pd.read_csv(raw)
    prob_cols = sorted((c for c in r.columns if c.startswith("Prob_Class_")),
                       key=lambda c: int(c.rsplit("_", 1)[1]))
    if not prob_cols:
        return None
    y_proba = r[prob_cols].to_numpy(dtype=float)
    y_true = r["True_Label"].to_numpy(dtype=int)
    raw_pred = r["Predicted_Label"].to_numpy(dtype=int)
    gids = r["Group_ID"].to_numpy(dtype=int) if "Group_ID" in r.columns else None

    glob, loc, classes = limits_from(cfg, y_true, gids, y_proba.shape[1])
    if not classes:
        return None
    cls = int(classes[0])
    if glob[cls] >= UNLIMITED:
        return None

    released = pd.read_csv(fin)["Predicted_Label"].to_numpy(dtype=int)
    eq, _taken = equalize(y_proba, gids, glob, loc, cls)
    flips_eq = int((raw_pred != eq).sum())

    f1r, pr, rr, mr = score(y_true, released, cls)
    f1e, pe, re_, me = score(y_true, eq, cls)
    hp = cfg.get("hyperparams", {}) or {}
    return {
        "campaign": campaign_of(cfg_path, root), "dataset": cfg.get("dataset_mode"),
        "model": cfg.get("model_name"), "method": cfg.get("methodology"),
        "cap": cfg.get("constraint_tag"), "seed": hp.get("seed"),
        "warmup": hp.get("warmup_epochs"), "cls": cls,
        "K_global": int(glob[cls]),
        "count_raw": int((raw_pred == cls).sum()),
        "count_released": int((released == cls).sum()),
        "count_equalized": int((eq == cls).sum()),
        "cc_f1_released": f1r, "cc_f1_equalized": f1e,
        "cc_prec_released": pr, "cc_prec_equalized": pe,
        "cc_rec_released": rr, "cc_rec_equalized": re_,
        "f1_macro_released": mr, "f1_macro_equalized": me,
        "flips_released": int((raw_pred != released).sum()),
        "flips_equalized": int(flips_eq),
        "config_path": cfg_path.replace("\\", "/"),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-o", "--out", required=True)
    ap.add_argument("--roots", nargs="*", default=["results/pending_runs",
                                                   "results/track_b"])
    ap.add_argument("--datasets", nargs="*", default=None)
    ap.add_argument("--caps", nargs="*", default=None)
    ap.add_argument("--methods", nargs="*", default=None)
    args = ap.parse_args()

    rows, seen = [], 0
    for root in args.roots:
        if not os.path.isdir(root):
            print("  (skip) %s" % root, file=sys.stderr)
            continue
        for dirpath, _dn, fn in os.walk(root):
            if "config.json" not in fn:
                continue
            seen += 1
            r = row_for(os.path.join(dirpath, "config.json"), root, args)
            if r:
                rows.append(r)
            if seen % 200 == 0:
                print("  scanned %d dirs, kept %d" % (seen, len(rows)),
                      file=sys.stderr)

    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    print("\nwrote %s (%d runs re-scored at equal count)" % (args.out, len(rows)),
          file=sys.stderr)

    if rows:
        d = pd.DataFrame(rows)
        d["short"] = d.count_released < d.K_global
        print("\nruns finishing BELOW the cap as released: %d of %d"
              % (int(d.short.sum()), len(d)), file=sys.stderr)
        print(d.groupby("method")[["count_released", "count_equalized", "K_global",
                                   "cc_f1_released", "cc_f1_equalized"]]
              .mean().round(3).to_string(), file=sys.stderr)


if __name__ == "__main__":
    main()
