"""Do the shipped predictions actually satisfy the caps they claim?

`docs/FRAMEWORK.md` lists "post-hoc local adjustment never re-violates the global
cap -- 199 runs, zero violations" under what we know works, and retires an open
warning on the strength of it. The script that produced that number lived at
`paper/scripts/feasibility_check.py` and was lost with the rest of that
directory, so the claim outlived its receipt. Unlike `build_corpus.py`, whose
absence PROVENANCE.md flags explicitly, this one was asserted as settled fact.

Rebuilt here, and it is the same check: reconstruct each run's caps from its own
config and its own test labels -- exactly as the pipeline does, via
`compute_global_constraints` -- then count the shipped `final_predictions.csv`.

    python -m scripts.feasibility_check <root> [<root> ...]

WHAT THIS CAN AND CANNOT SEE. The archived predictions carry `True_Label` and
`Predicted_Label` but NOT the group column, which lives in the dataset's
`test_meta.csv`. So against the evidence tarballs alone this verifies GLOBAL
caps only. Point `--data-root` at the datasets (i.e. run it on the server) and
it verifies the LOCAL caps too, which is the direction the original warning was
actually about. It says which mode it ran in rather than implying full coverage.
"""
import argparse
import glob
import json
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.training.constraints import (compute_global_constraints,   # noqa: E402
                                      compute_local_constraints)
from src.utils.constants import UNLIMITED                           # noqa: E402


def _groups_for(cfg, data_root):
    """The test-set group ids, if the dataset is reachable."""
    dc = cfg.get("dataset_config", {})
    d = dc.get("data_dir")
    col = dc.get("group_column")
    if not d or not col:
        return None, None
    path = os.path.join(data_root, os.path.basename(os.path.dirname(d)),
                        os.path.basename(d), "test_meta.csv")
    if not os.path.exists(path):
        path = os.path.join(data_root, d, "test_meta.csv")
    if not os.path.exists(path):
        return None, col
    meta = pd.read_csv(path)
    return (meta[col].to_numpy() if col in meta.columns else None), col


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("roots", nargs="+")
    ap.add_argument("--data-root", default=".",
                    help="where the dataset slices live; without them only the "
                         "GLOBAL caps can be checked")
    args = ap.parse_args(argv)

    checked = viol = no_local = 0
    bad = []
    for root in args.roots:
        for cpath in glob.glob(os.path.join(root, "**", "config.json"),
                               recursive=True):
            run = os.path.dirname(cpath)
            fin = os.path.join(run, "final_predictions.csv")
            if not os.path.exists(fin):
                continue
            try:
                cfg = json.load(open(cpath))
            except Exception:
                continue
            if cfg.get("status") != "completed":
                continue
            dc = cfg.get("dataset_config", {})
            capped = dc.get("constrained_class")
            capped = ([capped] if isinstance(capped, int)
                      else list(capped or []))
            con = cfg.get("constraint") or []
            if not capped or len(con) < 2:
                continue
            local_pct, global_pct = float(con[0]), float(con[1])
            n_classes = int(dc.get("num_classes", 0)) or None
            if not n_classes:
                continue

            df = pd.read_csv(fin)
            y_true = df["True_Label"].to_numpy(int)
            y_pred = df["Predicted_Label"].to_numpy(int)
            # Caps derive from the TEST labels, transductively -- the same
            # source the pipeline uses, so this is not a re-derivation from a
            # different quantity.
            frame = pd.DataFrame({"label": y_true})
            gcon = compute_global_constraints(frame, "label", global_pct,
                                              constrained_class=capped,
                                              num_classes=n_classes)
            checked += 1
            here = []
            for c in capped:
                if gcon[c] < UNLIMITED and int((y_pred == c).sum()) > int(gcon[c]):
                    here.append("GLOBAL c%d %d>%d"
                                % (c, int((y_pred == c).sum()), int(gcon[c])))

            groups, col = _groups_for(cfg, args.data_root)
            if groups is None or len(groups) != len(y_pred):
                no_local += 1
            else:
                frame[col] = groups
                lcon = compute_local_constraints(frame, "label", local_pct, col,
                                                 constrained_class=capped,
                                                 num_classes=n_classes)
                for g, bounds in lcon.items():
                    m = groups == g
                    for c in capped:
                        if bounds[c] < UNLIMITED and int((y_pred[m] == c).sum()) > int(bounds[c]):
                            here.append("LOCAL g%s c%d %d>%d"
                                        % (g, c, int((y_pred[m] == c).sum()),
                                           int(bounds[c])))
            if here:
                viol += 1
                bad.append((run, here[:3]))

    print("checked %d completed run(s) with shipped predictions" % checked)
    if no_local:
        print("  %d of them had NO reachable group column, so only their GLOBAL"
              % no_local)
        print("  caps were verified. The archived predictions do not carry the")
        print("  group ids; pass --data-root pointing at the dataset slices")
        print("  (i.e. run this on the server) to check the local caps too --")
        print("  which is the direction the original open warning was about.")
    if not checked:
        print("nothing to check: no completed run with a final_predictions.csv")
        return 2
    if viol:
        print("\n%d run(s) SHIP INFEASIBLE PREDICTIONS:" % viol)
        for run, why in bad[:20]:
            print("  %s\n     %s" % (run, "; ".join(why)))
        return 1
    print("\nzero violations.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
