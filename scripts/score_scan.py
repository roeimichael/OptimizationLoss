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


def budgets(run_dir):
    """The per-class budgets, read from the run's own log.

    `Limit_Class<c>` is written every epoch and is inf for an uncapped class,
    so the finite ones are the budgets the allocator actually enforced.
    """
    try:
        t = pd.read_csv(run_dir / "training_log.csv")
    except Exception:
        return {}
    out = {}
    for col in t.columns:
        if col.startswith("Limit_Class"):
            v = pd.to_numeric(t[col], errors="coerce")
            v = v[v < 1e9]
            if len(v):
                out[int(col[len("Limit_Class"):])] = int(v.iloc[-1])
    return out


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
    flat = bool(dirs)
    if not flat:
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
        # What the cap actually consumes: rank by p_c, take the top K_c.
        # AUROC is a GLOBAL ranking measure and hides this completely -- a run
        # can replace two thirds of the selected items and leave AUROC, and
        # prec@K, exactly where they were.
        topk, prec = {}, {}
        for c, K in sorted(budgets(d).items()):
            order = np.argsort(-prob[:, c])[:K]
            topk[c] = set(order.tolist())
            prec[c] = float((y[order] == c).mean())
        parts = d.relative_to(root).parts
        # THE CELL IS THE KEY. Two runs that differ in cap level are two
        # different cells and must never share a baseline row -- this scorer
        # first reported `clip/seed_1` twice, once at L30_G30 and once at
        # L50_G30, because it labelled runs by the last two path parts. That is
        # the same mistake the granularity read made: a swept dimension that
        # lives only in a directory name is one that gets pooled away.
        cell = "/".join(parts[:-2]) if len(parts) > 2 else ""
        out.append({
            "cell": cell,
            "arm": parts[-2] if len(parts) > 1 else parts[-1],
            "run": "/".join(parts[-2:]) if len(parts) > 1 else parts[-1],
            "auroc": float(roc_auc_score(oh, prob[:, present], average="macro")),
            "ap": float(average_precision_score(oh, prob[:, present],
                                                average="macro")),
            "alloc": al, "raw": rw, "cap": cap,
            "topk": topk, "prec": prec,
            "n_c2": int((pa == 2).sum()), "n_c4": int((pa == 4).sum()),
            "n_c2_raw": int((pr == 2).sum()), "n_c4_raw": int((pr == 4).sum()),
        })

    for cell in sorted({o["cell"] for o in out}):
        rows = [o for o in out if o["cell"] == cell]
        # null first: it is the CE-only counterfactual and the only row that
        # isolates the constraint. clip is the fallback bar.
        base = (next((o for o in rows if o["arm"] == "null"), None)
                or next((o for o in rows if o["arm"] == "clip"), None))
        print("\n" + "=" * 88)
        print("CELL: %s   (%d runs, capped classes %s)"
              % (cell or root.name, len(rows),
                 ",".join(str(c) for c in rows[0]["cap"]) or "?"))
        print("baseline for deltas: %s"
              % (base["run"] if base else "NONE -- deltas suppressed"))
        print("=" * 88)

        def delta(v, b):
            return "" if b is None else " (%+.4f)" % (v - b)

        print("\nALLOCATION-FREE -- probabilities only, no allocator moves these")
        print("%-30s %9s %19s" % ("run", "AUROC", "AP"))
        print("-" * 62)
        for o in rows:
            print("%-30s %9.4f%-11s %8.4f%s" % (
                o["run"], o["auroc"], delta(o["auroc"], base and base["auroc"]),
                o["ap"], delta(o["ap"], base and base["ap"])))

        if any(o["prec"] for o in rows):
            print("\nAT THE OPERATING POINT -- top K_c by p_c, which is all a cap uses")
            print("%-30s %6s %9s %9s %s" % (
                "run", "class", "prec@K", "K", "Jaccard vs baseline"))
            print("-" * 82)
            for o in rows:
                for c in sorted(o["prec"]):
                    j = ""
                    if base is not None and c in base["topk"] and o is not base:
                        a, b = base["topk"][c], o["topk"][c]
                        j = "%.3f" % (len(a & b) / max(1, len(a | b)))
                    print("%-30s %6d %9.4f %9d %s" % (
                        o["run"], c, o["prec"][c], len(o["topk"][c]), j))
            print("\nA low Jaccard with an unchanged prec@K is CHURN: the run"
                  "\nreplaced the selected items and gained nothing where the"
                  "\ncap binds. Measured 2026-08-20: Jaccard 0.29-0.42 with"
                  "\nprec@K identical to the control on both capped classes.")

        for view in ("raw", "alloc"):
            tag = ("RAW argmax -- did the constraint leave a better classifier?"
                   if view == "raw" else
                   "AFTER ALLOCATION -- deployable, and the flattering view")
            print("\n%s" % tag)
            print("%-30s %8s %18s %8s %8s   %s" % (
                "run", "acc", "macroF1", "F1_cap", "F1_unc", "pred c2/c4"))
            print("-" * 90)
            for o in rows:
                m = o[view]
                b = base[view] if base else None
                n2 = o["n_c2_raw"] if view == "raw" else o["n_c2"]
                n4 = o["n_c4_raw"] if view == "raw" else o["n_c4"]
                print("%-30s %8.4f %8.4f%-10s %8.4f %8.4f   %d/%d" % (
                    o["run"], m["acc"], m["macroF1"],
                    delta(m["macroF1"], b and b["macroF1"]),
                    m["F1_cap"], m["F1_unc"], n2, n4))

    print("\nn=1 unless a cell shows several seeds, and dermmnist's test set")
    print("shares lesion_ids with its training set -- no absolute number here")
    print("is quotable. These rows pick a setting to take to a real campaign.")
    if not any(o["arm"] == "null" for o in out):
        print("\nNO `null` ROW anywhere. Re-run the scan with --with-null:")
        print("without the CE-only control nothing here is attributable to")
        print("the constraint rather than to CE still training.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
