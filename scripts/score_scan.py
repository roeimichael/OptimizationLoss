"""Score a dose/shape scan on QUALITY, because the counts are not a metric.

The scan prints count trajectories, and a count trajectory cannot say whether a
setting is good. Post-hoc allocation drives every capped class to its budget for
free, so "the count came down" is a statement about the allocator, not the
method -- this project has mistaken the two often enough that `flips`, raw count
over K, and proximity to feasibility are all banned as headline numbers.

What a scan CAN answer is whether the constraint phase left a better classifier
behind. Three views, and they disagree in a way that is itself the finding:

  ALLOC   metrics on final_predictions.csv, after post-hoc allocation. The
          deployable output, and the most flattering: allocation repairs a lot.
  RAW     metrics on final_predictions_raw.csv, plain argmax. Measured before,
          tralo sits 0.0245 BELOW a plain clipper here while tying after
          allocation -- i.e. the constraint phase made the classifier worse and
          equalisation hid it. Any shape that reverses that shows up in RAW.
  AUROC   computed from the probabilities, which are IDENTICAL in both files.
          Allocation reorders labels, never scores, so this column is the one
          thing in the scan no allocator can flatter. If AUROC does not move,
          the constraint did not improve the model, whatever the counts did.

Read every row against the `null` row (CE only), never against zero.

    python -m scripts.score_scan /tmp/shape_scan
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score, average_precision_score

def capped_classes(run_dir):
    """The capped classes, from the run's own config.

    Hardcoding them is how the granularity read went wrong: a swept dimension
    that lives only in a directory name is a dimension that gets pooled away.
    """
    import json
    try:
        c = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
        v = c.get("dataset_config", {}).get("constrained_class", [])
        return tuple(v) if isinstance(v, (list, tuple)) else (int(v),)
    except Exception:
        return ()


def read(path):
    d = pd.read_csv(path)
    y = d["True_Label"].to_numpy()
    p = d["Predicted_Label"].to_numpy()
    prob = d[[c for c in d.columns if c.startswith("Prob_Class_")]].to_numpy()
    return y, p, prob


def row(y, pred, CAPPED):
    f1 = f1_score(y, pred, average=None, labels=sorted(set(y.tolist())),
                  zero_division=0)
    lab = sorted(set(y.tolist()))
    per = dict(zip(lab, f1))
    unc = [per[c] for c in lab if c not in CAPPED]
    return {
        "acc": float((y == pred).mean()),
        "macroF1": float(np.mean(f1)),
        "F1_cap": float(np.mean([per[c] for c in lab if c in CAPPED]))
                  if any(c in CAPPED for c in lab) else float("nan"),
        "F1_unc": float(np.mean(unc)) if unc else float("nan"),
    }


def main():
    a = argparse.ArgumentParser(description=__doc__)
    a.add_argument("root")
    args = a.parse_args()

    root = Path(args.root)
    dirs = sorted(d for d in root.iterdir()
                  if d.is_dir() and (d / "final_predictions.csv").exists())
    if not dirs:
        # Nested campaign layout (<root>/<model>/<data>/<cap>/<arm>/<seed>/).
        dirs = sorted(f.parent for f in root.rglob("final_predictions.csv"))
    if not dirs:
        print("no scored runs under %s" % args.root)
        return 1

    out = []
    for d in dirs:
        y, pa, prob = read(d / "final_predictions.csv")
        _, pr, _ = read(d / "final_predictions_raw.csv")
        cap = capped_classes(d)
        al, rw = row(y, pa, cap), row(y, pr, cap)
        present = sorted(set(y.tolist()))
        oh = np.eye(prob.shape[1])[y][:, present]
        out.append({
            "run": (d.name if d.parent == root
                    else "/".join(d.relative_to(root).parts[-2:])),
            "auroc": float(roc_auc_score(oh, prob[:, present], average="macro")),
            "ap": float(average_precision_score(oh, prob[:, present],
                                                average="macro")),
            "alloc": al, "raw": rw,
            "cap": cap,
            "n_c2": int((pa == 2).sum()), "n_c4": int((pa == 4).sum()),
            "n_c2_raw": int((pr == 2).sum()), "n_c4_raw": int((pr == 4).sum()),
        })

    # null first: it is the CE-only counterfactual and the only row that
    # isolates the constraint. clip is a fallback so a campaign without a null
    # is still read against its bar rather than against zero.
    base = (next((o for o in out if o["run"].split("/")[0] == "null"), None)
            or next((o for o in out if o["run"].split("/")[0] == "clip"), None))
    if base is not None:
        print("\nbaseline row for the deltas: %s" % base["run"])

    def delta(v, b):
        return "" if b is None else " (%+.4f)" % (v - b)

    print("\nALLOCATION-FREE -- probabilities only, no allocator can move these")
    print("%-24s %9s %9s" % ("run", "AUROC", "AP"))
    print("-" * 46)
    for o in out:
        print("%-24s %9.4f%s %9.4f%s" % (
            o["run"], o["auroc"], delta(o["auroc"], base and base["auroc"]),
            o["ap"], delta(o["ap"], base and base["ap"])))

    for view in ("raw", "alloc"):
        tag = ("RAW argmax -- did the constraint leave a better classifier?"
               if view == "raw" else
               "AFTER ALLOCATION -- the deployable output, and the flattering one")
        print("\n%s" % tag)
        print("%-24s %8s %18s %8s %8s   %s" % (
            "run", "acc", "macroF1", "F1_cap", "F1_unc", "pred c2/c4"))
        print("-" * 88)
        for o in out:
            m = o[view]
            b = base[view] if base else None
            n2 = o["n_c2_raw"] if view == "raw" else o["n_c2"]
            n4 = o["n_c4_raw"] if view == "raw" else o["n_c4"]
            print("%-24s %8.4f %8.4f%-10s %8.4f %8.4f   %d/%d" % (
                o["run"], m["acc"], m["macroF1"],
                delta(m["macroF1"], b and b["macroF1"]),
                m["F1_cap"], m["F1_unc"], n2, n4))

    print("\nn=1, four epochs, and dermmnist's test set shares lesion_ids with")
    print("its training set -- so no absolute number here is quotable. These")
    print("rows pick a shape to take to a real campaign. Nothing else.")
    if base is None:
        print("\nNO `null` ROW. Re-run the scan with --with-null: without the")
        print("CE-only control none of this is attributable to the constraint.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
