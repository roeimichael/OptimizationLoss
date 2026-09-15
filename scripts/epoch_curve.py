"""Score every epoch's stored probabilities, so the constraint's effect is a CURVE.

WHY. Every campaign so far fixed a budget, ran it to the end, and scored the
final model. That number is a SUM over epochs: it cannot separate the epochs
where the constraint improved the boundary from the epochs where it wrecked one
cross-entropy was still building. The standing suspicion is that the sign flips
at CE saturation -- while CE is still moving the boundary the constraint rides
along, and once the training set is memorised the constraint is kicking a dead
surface. Adding those two halves together is exactly what hides the effect.

`src/training/epoch_trace.py` writes one probability snapshot per epoch, with no
labels anywhere near the training path. This script joins them to the labels
afterwards and reports, per epoch:

  train_acc   from the run's own training log, so CE saturation is locatable on
              the same axis as the damage
  cc_f1       after the REAL allocation, using the maintained allocator on the
              campaign's own frozen quotas -- not a re-derived cap
  gap         mean per-group average precision, ALLOCATION-FREE: it reads the
              probability column and never a predicted label, so a change in it
              is the model moving rather than the allocator reshuffling

and then, per epoch, the paired difference between an arm and its
zero-constraint twin. That difference IS the constraint's contribution at that
epoch, which is the quantity no end-of-run number has ever shown.

🛑 AN EPOCH CHOSEN ON THIS CURVE IS AN ORACLE, NOT A METHOD. The curve is scored
against the test set. "Stop at the best epoch" therefore reads the answer off
the evaluation data, and reporting it as a method would be selection on the test
set. It is still worth measuring, because it bounds what ANY stopping rule could
win -- if the oracle maximum is barely above the final-epoch value, no stopping
rule is worth building, and that is a cheap way to close the direction. The
`--oracle` column is printed with that label attached for exactly that reason. A
deployable rule needs a held-out split that is not the test set.
"""
import argparse
import collections
import csv
import glob
import json
import os
import sys

import numpy as np
from sklearn.metrics import average_precision_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.methodologies.heuristic.train import (           # noqa: E402
    _build_hierarchy, apply_allocation_heuristic,
)
from src.pipeline.campaign import campaign_for_config     # noqa: E402
from src.training.metrics import compute_metrics          # noqa: E402


def quotas_for(run_dir):
    """The campaign's OWN frozen quotas for this run, never re-derived."""
    config_path = os.path.join(run_dir, "config.json")
    root, manifest = campaign_for_config(config_path)
    rel = os.path.relpath(os.path.abspath(config_path), str(root)).replace(os.sep, "/")
    frozen = manifest["data"][manifest["runs"][rel]["data_id"]]
    local = {int(k): v for k, v in frozen["quotas"]["local"].items()}
    return frozen["quotas"]["global"], local


def group_ap(proba, y, groups, classes):
    """Allocation-free: reads the probability column, never a predicted label."""
    scores = []
    for c in classes:
        p = proba[:, c]
        for gid in np.unique(groups):
            m = groups == gid
            truth = (y[m] == c).astype(int)
            if 0 < truth.sum() < truth.size:        # AP undefined otherwise
                scores.append(average_precision_score(truth, p[m]))
    return float(np.mean(scores)) if scores else float("nan")


def train_acc_by_epoch(run_dir):
    out = {}
    path = os.path.join(run_dir, "training_log.csv")
    if not os.path.exists(path):
        return out
    with open(path, encoding="utf-8") as handle:
        for r in csv.DictReader(handle):
            try:
                out[int(float(r["Epoch"])) + 1] = float(r["Train_Acc"])
            except (KeyError, ValueError, TypeError):
                pass
    return out


def score_run(run_dir):
    """[(epoch, train_acc, cc_f1, gap)] for one run, or [] if it has no snapshots."""
    index = os.path.join(run_dir, "epoch_trace.csv")
    raw = os.path.join(run_dir, "final_predictions_raw.csv")
    if not (os.path.exists(index) and os.path.exists(raw)):
        return []

    import pandas as pd
    table = pd.read_csv(raw)
    y = table["True_Label"].to_numpy(int)
    groups = table["Group_ID"].to_numpy()

    with open(os.path.join(run_dir, "config.json"), encoding="utf-8") as handle:
        cfg = json.load(handle)
    classes = cfg["dataset_config"]["constrained_class"]
    classes = classes if isinstance(classes, list) else [classes]
    n_classes = cfg["dataset_config"]["num_classes"]
    global_con, local_con = quotas_for(run_dir)
    hierarchy = _build_hierarchy(n_classes, global_con, classes)
    accs = train_acc_by_epoch(run_dir)

    rows = []
    with open(index, encoding="utf-8") as handle:
        for r in csv.DictReader(handle):
            if not r.get("probs_file"):
                continue
            proba = np.load(os.path.join(run_dir, r["probs_file"])).astype(float)
            proba = proba / proba.sum(axis=1, keepdims=True)
            y_pred, _ = apply_allocation_heuristic(
                proba, groups, hierarchy, global_con, local_con, n_classes)
            m = compute_metrics(y, y_pred, constrained_classes=classes)
            epoch = int(r["epoch_absolute_1based"])
            rows.append((epoch,
                         accs.get(epoch, float(r["train_acc"] or "nan")),
                         float(m["cc_f1"]),
                         group_ap(proba, y, groups, classes)))
    return sorted(rows)


def collect(root):
    """(cap, arm, seed) -> per-epoch rows, for every run carrying snapshots."""
    out = {}
    for cfg_path in glob.glob(os.path.join(root, "**", "config.json"), recursive=True):
        run_dir = os.path.dirname(cfg_path)
        with open(cfg_path, encoding="utf-8") as handle:
            cfg = json.load(handle)
        rows = score_run(run_dir)
        if rows:
            key = (cfg.get("constraint_tag"), cfg.get("arm"),
                   (cfg.get("hyperparams") or {}).get("seed"))
            out[key] = rows
    return out


def report(data, pairs):
    caps = sorted({k[0] for k in data})
    for cap in caps:
        for arm, control in pairs:
            seeds = sorted({k[2] for k in data
                            if k[0] == cap and k[1] in (arm, control)})
            paired = collections.defaultdict(list)
            accs = collections.defaultdict(list)
            for seed in seeds:
                a = dict((e, (t, f, g)) for e, t, f, g in
                         data.get((cap, arm, seed), []))
                b = dict((e, (t, f, g)) for e, t, f, g in
                         data.get((cap, control, seed), []))
                for epoch in sorted(set(a) & set(b)):
                    paired[epoch].append((a[epoch][1] - b[epoch][1],
                                          a[epoch][2] - b[epoch][2]))
                    accs[epoch].append(a[epoch][0])
            if not paired:
                continue
            print("")
            print("%s   %s minus %s   (n=%d seeds)"
                  % (cap, arm, control, len(seeds)))
            print("  the constraint's OWN contribution at each epoch; "
                  "train_acc locates CE saturation on the same axis")
            print("  %5s %10s %12s %12s" % ("epoch", "train_acc", "d cc-F1", "d gAP"))
            best, best_epoch = None, None
            for epoch in sorted(paired):
                d_f1 = float(np.mean([p[0] for p in paired[epoch]]))
                d_ap = float(np.mean([p[1] for p in paired[epoch]]))
                acc = float(np.mean(accs[epoch])) if accs[epoch] else float("nan")
                mark = ""
                if acc >= 0.95:
                    mark = "  <- train acc >= 0.95 (saturated)"
                if best is None or d_f1 > best:
                    best, best_epoch = d_f1, epoch
                print("  %5d %10.4f %+12.5f %+12.5f%s"
                      % (epoch, acc, d_f1, d_ap, mark))
            final = sorted(paired)[-1]
            final_d = float(np.mean([p[0] for p in paired[final]]))
            print("  final epoch %d: %+.5f    ORACLE best epoch %d: %+.5f"
                  % (final, final_d, best_epoch, best))
            print("  ORACLE is an upper bound chosen on the TEST set, not a "
                  "method. Headroom over the final epoch = %+.5f."
                  % (best - final_d))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("root", help="campaign results root")
    ap.add_argument("--pair", action="append", default=["tralo:tralo_null"],
                    metavar="ARM:CONTROL",
                    help="an arm and its zero-constraint twin (repeatable)")
    args = ap.parse_args()
    if not os.path.isdir(args.root):
        print("no such campaign root: %s" % args.root)
        return 2
    pairs = []
    for spec in args.pair:
        if ":" not in spec:
            ap.error("--pair wants ARM:CONTROL, got %r" % spec)
        pairs.append(tuple(spec.split(":", 1)))
    data = collect(args.root)
    if not data:
        print("no epoch snapshots under %s -- NOTHING SCORED" % args.root)
        return 2
    report(data, pairs)
    return 0


if __name__ == "__main__":
    sys.exit(main())
