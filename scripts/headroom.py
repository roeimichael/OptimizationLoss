"""How much is there to win here, in F1 and in ITEMS? Ask before running.

Every number this prints went into docs/FRAMEWORK.md section 4 on 2026-08-21
from a throwaway script. That violates this project's own standard -- section
1a exists because protocol numbers with no reproduction path had to be
re-derived under doubt -- so it lives here now.

IT SCORES THE WAY THE SCORER SCORES. `achieved` is `equalize_multi` over the
real global AND local budgets, the same call `full_panel` makes, so it is the
control's `_eq` number rather than an approximation of it. Two nearby
quantities are NOT that number, and both were tried here first:

  - the RAW argmax ignores the budget entirely (it emits 97 class-1 items
    against K=31), so scoring the control on it beats the ceiling outright and
    prints a NEGATIVE headroom.
  - the STORED final_predictions.csv is the RUNTIME allocator's output, which
    lands on K-1 about a third of the time. That is a different allocation from
    the scorer's, and at L50_G50 it scores HIGHER, not lower.

FOUR THINGS, all read off stored predictions, no GPU:

  CEILING     2K/(K+n): recall <= K/n and precision <= 1, so no allocator can
              beat it. An upper bound -- local caps can put it out of reach.
              HEADROOM is that minus what the control achieved.
  ITEMS       F1 = 2TP/(K+n) is linear in TP, so items = dF1*(K+n)/2. Convert
              before believing a delta: the paired seed sd is worth a couple of
              items on its own.
  EXCESS      how far over budget the model starts. A tight cap gives the
              constraint MORE to move and LESS to win; the cap level is that
              trade-off and neither end is free.
  SHORTFALL   how often the runtime allocator emits fewer than K. Never OVER,
              so no cap is violated, and the scorer re-equalizes, so it reaches
              nothing scored -- it reaches anything read off the stored
              predictions.

    python -m scripts.headroom <campaign-root> [--control clip]
"""

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.full_panel import equalize_multi
from src.training.constraints import (compute_global_constraints,
                                      compute_local_constraints,
                                      normalize_constrained_classes)
from src.utils.constants import UNLIMITED


def load(d):
    """(y, group ids, probabilities, classes, global caps, local caps) or None.

    Deliberately the same construction as the loader in full_panel. If these
    two ever disagree about what a run's budget is, the headroom is measured
    against a bar that nothing is scored on.
    """
    cfg = json.loads((d / "config.json").read_text(encoding="utf-8"))
    t = pd.read_csv(d / "final_predictions_raw.csv")
    cols = sorted((int(c[len("Prob_Class_"):]), c) for c in t.columns
                  if c.startswith("Prob_Class_"))
    P = t[[c for _, c in cols]].to_numpy(dtype=float)
    if not np.isfinite(P).all():
        return None
    P = P / np.clip(P.sum(axis=1, keepdims=True), 1e-12, None)
    y = t["True_Label"].to_numpy(int)
    g = t["Group_ID"].to_numpy(int)
    classes = normalize_constrained_classes(
        (cfg.get("dataset_config") or {}).get("constrained_class"))
    lp, gp = cfg["constraint"]
    df = pd.DataFrame({"label": y, "grp": g})
    G = compute_global_constraints(df, "label", gp, constrained_class=classes,
                                   num_classes=P.shape[1])
    L = compute_local_constraints(df, "label", lp, "grp",
                                  constrained_class=classes,
                                  num_classes=P.shape[1])
    classes = [c for c in classes
               if G[c] < UNLIMITED or any(b[c] < UNLIMITED for b in L.values())]
    return (y, g, P, classes, G, L) if classes else None


def f1(y, pred, c):
    tp = int(((pred == c) & (y == c)).sum())
    p = tp / max(1, int((pred == c).sum()))
    r = tp / max(1, int((y == c).sum()))
    return 2 * p * r / (p + r) if (p + r) else 0.0


def main():
    a = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    a.add_argument("root")
    a.add_argument("--control", default="clip",
                   help="arm whose achieved score sets the headroom. clip is "
                        "the stronger clipper and the honest bar.")
    args = a.parse_args()

    runs = sorted(f.parent
                  for f in Path(args.root).rglob("final_predictions_raw.csv"))
    if not runs:
        print("no runs with predictions under %s" % args.root)
        return 1

    cells, short_n, short_tot = {}, 0, 0
    for d in runs:
        got = load(d)
        if got is None:
            continue
        y, g, P, classes, G, L = got
        tag = next((p for p in d.parts if re.match(r"^L\d+_G\d+$", p)), "?")
        arm = d.parts[-2]
        pred = P.argmax(1)
        eq = equalize_multi(P, g, G, L, classes) if arm == args.control else None

        fp = d / "final_predictions.csv"
        stored = None
        if fp.exists():
            q = pd.read_csv(fp)
            if "Predicted_Label" in q.columns and len(q) == len(y):
                stored = q["Predicted_Label"].to_numpy(int)

        for c in sorted(classes):
            n = int((y == c).sum())
            k = int(G[c])
            if not n or k >= UNLIMITED:
                continue
            e = cells.setdefault((tag, c),
                                 {"n": n, "K": k, "hard": [], "ctrl": []})
            if eq is not None:
                # CONTROL ONLY, both columns. Averaging the raw count over
                # every arm in the tree mixes the clipper's starting excess
                # with a trained arm's post-constraint one and describes no
                # model at all.
                e["hard"].append(int((pred == c).sum()))
                e["ctrl"].append(f1(y, eq, c))
            if stored is not None:
                short_tot += 1
                short_n += int((stored == c).sum()) < k

    print("HEADROOM AND WHAT IT COSTS IN ITEMS   (control = %s)\n" % args.control)
    print("%-10s %5s %6s %6s %8s %9s %9s %9s %8s"
          % ("cap", "class", "n", "K", "ceiling", "achieved", "headroom",
             "= items", "excess"))
    print("-" * 82)
    per_cap = {}
    for (tag, c), e in sorted(cells.items()):
        n, k = e["n"], e["K"]
        ceil = 2.0 * k / (k + n)
        ach = float(np.mean(e["ctrl"])) if e["ctrl"] else float("nan")
        head = ceil - ach
        per_cap.setdefault(tag, []).append((ceil, ach))
        print("%-10s %5d %6d %6d %8.4f %9.4f %9.4f %9.1f %8.1f"
              % (tag, c, n, k, ceil, ach, head, head * (k + n) / 2,
                 float(np.mean(e["hard"])) - k))
    print()
    for tag, v in sorted(per_cap.items()):
        ceil = float(np.mean([x for x, _ in v]))
        ach = float(np.mean([x for _, x in v]))
        print("  %-10s macro ceiling %.4f  achieved %.4f  HEADROOM %.4f"
              % (tag, ceil, ach, ceil - ach))
    print()
    print("`= items` is the ENTIRE gap to a PERFECT allocator, not to a better")
    print("method. `items per 0.01 capF1` is (K+n)/200 per class, summed over")
    print("the capped classes because ccF1 is macro-averaged over them.")
    print()
    print("`excess` is how far over budget the model starts. A tight cap gives")
    print("the constraint more to move and less to win. Neither end is free.")
    if short_tot:
        print()
        print("ALLOCATOR: emitted fewer than K on %d of %d (run, class) pairs."
              % (short_n, short_tot))
        print("Never over, so no cap is violated, and the scorer re-equalizes to")
        print("exactly K -- so this reaches nothing it scores. It reaches any")
        print("number read straight off the stored predictions.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
