"""Does a cell's cap pose a QUESTION? One source of truth for the toolchain.

A count cap can only distinguish two methods where all three hold at once
(`scripts/task_window.py`, FRAMEWORK 2(z16) and 2(z17)):

    BINDS    the cap evicts >= 10 predictions the model would have made
    PRIZE    errors@K > 0, i.e. the top-K is not already perfect
    WIGGLE   p@K < 0.99, i.e. the cut is not buried in saturated scores

Outside that window a cell measures the ABSENCE of a question, and a null there
is not evidence about any method. Measured on all four backbones: 24 of 24
(backbone x class x cap) cells at L20/L30/L50 on iwildcam fail at least one
condition, and 8 of 8 at K/n = 0.90 pass.

WHY IT MATTERS TO THE SCORER AND NOT ONLY TO THE GENERATOR: which method looks
best CHANGES with the cell selection. On `equaldose1` `alm` is the best arm on
ccF1 in the non-task cells and the second worst in the task cells; on `dom1`
`tralo_uniform` is the best arm in one non-task cell and the worst in all four
task cells (2(z19), 2(z21)). Every historical ranking in this project pooled
cells without asking, so the scorers must be able to split them.

⚠️ THIS MODULE LIVES IN `configs/`, NOT `scripts/`, ON PURPOSE. `configs/` is
inside `TRAINING_PATHS` and `scripts/` is not, which is exactly why `scripts/`
can be deployed mid-campaign: nothing on the runner's import path reads it.
`gen_campaign` needs this logic, so the logic must live where `gen_campaign`
may import it from. The dependency runs `scripts` -> `configs` and must never
run the other way, or a scorer deploy starts splitting `code_version`.
"""
import os
import sys

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
WINDOWS_PATH = os.path.join(HERE, "task_windows.yml")


def cap_pair(tag):
    """'L30_G50' -> [0.30, 0.50]: (local, global), independent by construction.

    PER-CLASS LOCAL CAPS: 'L80-100_G95' -> [[0.80, 1.00], 0.95]. The list is
    read POSITIONALLY against `constrained_class`, so L80-100 with classes
    [2, 7] caps class 2 at 80% and class 7 at 100%.

    WHY (FRAMEWORK 2(z16), 2(z17)). The two capped classes' task windows DIFFER
    on every backbone and overlap at one point at most, so a single fraction
    cannot put both classes inside their windows. Per-class caps are required
    for the two-class setting to pose a question at all, not a refinement.

    A percentage ABOVE 100 is legal and is not degenerate: on iwildcam class 7
    the model predicts 478-498 against 456 true, so K/n = 1.00 still evicts
    22-42 predictions.
    """
    try:
        local, glob_ = tag.split("_")
        if local[0] != "L" or glob_[0] != "G":
            raise ValueError
        parts = local[1:].split("-")
        loc = ([int(x) / 100.0 for x in parts] if len(parts) > 1
               else int(parts[0]) / 100.0)
        return [loc, int(glob_[1:]) / 100.0]
    except (ValueError, IndexError):
        sys.exit("bad cap tag %r -- expected L<pct>_G<pct> (e.g. L30_G50) or "
                 "per-class L<pct>-<pct>_G<pct> (e.g. L80-100_G95)" % tag)


def load_windows(path=WINDOWS_PATH):
    """The MEASURED K/n windows in which a cap poses a question, or None."""
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def tolerance(TW):
    """Window edges are accurate to one step of the measurement grid."""
    return float((TW or {}).get("meta", {}).get("tolerance", 0.0))


def in_window(ratio, lo, hi, tol=0.0):
    """Is this K/n inside a measured task window? Pure, so it is testable with
    no dataset on disk."""
    return (lo - tol) <= ratio <= (hi + tol)


def effective_budgets(P, dataset, lp, gp):
    """{class: (K_eff, n_true)} for one cap, or None if the slice is absent.

    K_eff = min(global budget, SUM of the per-group local budgets) -- what the
    allocator can actually emit. Reading one scope alone has twice called a cap
    binding when the other scope was the one that bound.

    Returns None ONLY when the slice is genuinely not on this machine, because
    campaigns are generated on laptops as well as on the server. A slice that
    IS present and cannot be read RAISES: the first version of this read labels
    from `test_labels.npy`, which does not exist on the laptop where the meta
    does, and a bare `except` turned the whole task-window gate into a silent
    no-op that still printed nothing and refused nothing.
    """
    import numpy as np
    import pandas as pd
    from src.training.constraints import (compute_global_constraints,
                                          compute_local_constraints,
                                          normalize_constrained_classes)
    dc = P["datasets"][dataset]
    meta = os.path.join(dc["data_dir"], "test_meta.csv")
    if not os.path.exists(meta):
        return None
    te = pd.read_csv(meta)
    if "label" not in te.columns:
        npy = os.path.join(dc["data_dir"], "test_labels.npy")
        if not os.path.exists(npy):
            raise SystemExit(
                "REFUSED: %s has a test_meta.csv with no `label` column and no "
                "test_labels.npy beside it, so the task window cannot be "
                "checked. Fix the slice rather than proceeding ungated."
                % dataset)
        te["label"] = np.load(npy)
    y = te["label"].to_numpy()
    classes = normalize_constrained_classes(dc["constrained_class"])
    G = compute_global_constraints(te, "label", gp, constrained_class=classes,
                                   num_classes=dc["num_classes"])
    L = compute_local_constraints(te, "label", lp, dc["group_column"],
                                  constrained_class=classes,
                                  num_classes=dc["num_classes"])
    out = {}
    for c in classes:
        c = int(c)
        lsum = sum(int(L[g][c]) for g in L)
        out[c] = (min(int(G[c]), lsum), int((y == c).sum()))
    return out


def classify(P, TW, dataset, model, cap_tag):
    """Is (dataset, model, cap_tag) a TASK cell?

    Returns a dict with `status` in:
      "task"        every capped class sits inside its measured window
      "non_task"    at least one class is outside -- the cell measures nothing
      "no_window"   this (dataset, backbone) has never been measured
      "no_data"     the slice is not on this machine, so nothing can be said

    THE THREE NEGATIVES ARE NOT THE SAME and must never be collapsed. An
    unmeasured backbone is an unknown; a missing slice is a missing instrument;
    only "non_task" is a statement about the experiment.
    """
    if not TW:
        return dict(status="no_window", classes={})
    w = ((TW.get("windows") or {}).get(dataset) or {}).get(model)
    if not w:
        return dict(status="no_window", classes={})
    lp, gp = cap_pair(cap_tag)
    eff = effective_budgets(P, dataset, lp, gp)
    if eff is None:
        return dict(status="no_data", classes={})
    tol = tolerance(TW)
    per, ok_all = {}, True
    for c, (K, n) in sorted(eff.items()):
        ratio = (K / float(n)) if n else 0.0
        lo, hi = w["class"][c]
        ok = in_window(ratio, lo, hi, tol)
        ok_all = ok_all and ok
        # `margin` is how far OUTSIDE the window this ratio sits (0 when
        # strictly inside). A cell with a tiny positive margin is a task only
        # through the grid-snapping tolerance, and 46% of this project's
        # task-cell runs are in that position -- so the reader gets to see it
        # rather than having it folded into a boolean.
        margin = max(0.0, lo - ratio, ratio - hi)
        per[c] = dict(K=K, n=n, ratio=ratio, lo=lo, hi=hi, ok=ok,
                      margin=margin, snapped=bool(ok and margin > 0))
    return dict(status="task" if ok_all else "non_task", classes=per)


def self_test(out=sys.stdout):
    """Gate the predicate in BOTH directions, and gate that the three
    NEGATIVES stay distinguishable. Never claims a pass it did not run."""
    ok = True

    def check(name, cond):
        nonlocal ok
        ok = ok and cond
        print("  %-64s %s" % (name, "PASS" if cond else "FAIL"), file=out)

    TW = load_windows()
    check("configs/task_windows.yml loads", bool(TW))
    if not TW:
        return 1
    tol = tolerance(TW)
    iw = TW["windows"]["iwildcam"]
    check("every backbone the paper claims has a measured row",
          set(iw) >= {"ViTB16", "MobileNetV3", "MobileNetV2", "RegNetY400MF"})
    check("every row carries provenance naming the runs it came from",
          all(r.get("provenance") for r in iw.values()))
    check("the two capped classes' windows DIFFER on every backbone "
          "(else per-class caps would be pointless)",
          all(r["class"][2] != r["class"][7] for r in iw.values()))

    lo2, hi2 = iw["MobileNetV3"]["class"][2]
    lo7, hi7 = iw["MobileNetV3"]["class"][7]
    check("L30 class 2 (K/n=0.30) is OUTSIDE MobileNetV3 %.2f-%.2f"
          % (lo2, hi2), not in_window(0.30, lo2, hi2, tol))
    check("L20 class 7 (K/n=0.20) is OUTSIDE MobileNetV3 %.2f-%.2f"
          % (lo7, hi7), not in_window(0.20, lo7, hi7, tol))
    check("LIVENESS: taskwin1 class 2 (K/n=0.800) is INSIDE",
          in_window(0.800, lo2, hi2, tol))
    check("LIVENESS: taskwin1 class 7 (K/n=0.950) is INSIDE",
          in_window(0.950, lo7, hi7, tol))
    check("a window EDGE is inside, not excluded by float noise",
          in_window(hi2, lo2, hi2, tol) and in_window(lo7, lo7, hi7, tol))

    # THE TOLERANCE MUST ONLY EVER SNAP TO AN ALREADY-MEASURED GRID POINT.
    # It is load-bearing (46% of task-cell runs qualify through it), so the
    # relationship that makes it legitimate is asserted, not commented: it must
    # be far smaller than the grid step it is snapping onto. Raising it without
    # re-measuring on a finer grid fails here instead of silently widening
    # every window in the file.
    grid = TW["meta"]["fraction_grid"]
    step = min(round(b - a, 6) for a, b in zip(grid, grid[1:]))
    check("tolerance %.3f is at most a tenth of the %.2f measurement grid step"
          % (tol, step), tol <= step / 10.0)
    check("so the tolerance cannot rescue a ratio that is not already within "
          "rounding of a measured point", tol < step / 2.0)

    check("cap_pair parses the per-class form positionally",
          cap_pair("L80-100_G95") == [[0.80, 1.00], 0.95])
    check("cap_pair parses the single-fraction form",
          cap_pair("L30_G50") == [0.30, 0.50])

    # the three negatives must stay distinguishable
    check("an unmeasured backbone reports `no_window`, not `non_task`",
          classify({}, TW, "iwildcam", "NoSuchNet", "L30_G50")["status"]
          == "no_window")
    check("an empty window file reports `no_window`, not `task`",
          classify({}, None, "iwildcam", "ViTB16", "L30_G50")["status"]
          == "no_window")

    # end to end, only where the slice actually exists
    from configs.gen_campaign import load_protocol
    P = load_protocol()
    dc = P["datasets"]["iwildcam"]
    if not os.path.exists(os.path.join(dc["data_dir"], "test_meta.csv")):
        print("  %-64s %s" % ("end-to-end classify (needs the iwildcam slice)",
                              "SKIPPED -- slice not on this machine"), file=out)
    else:
        r30 = classify(P, TW, "iwildcam", "MobileNetV3", "L30_G50")
        r90 = classify(P, TW, "iwildcam", "MobileNetV3", "L90_G95")
        check("end to end: L30_G50 on MobileNetV3 is a NON-TASK",
              r30["status"] == "non_task")
        check("LIVENESS end to end: L90_G95 on MobileNetV3 IS a task",
              r90["status"] == "task")
        check("classify reports per-class K, n and K/n, not just a boolean",
              set(r90["classes"]) == {2, 7}
              and all({"K", "n", "ratio", "lo", "hi", "ok", "margin",
                       "snapped"} <= set(v) for v in r90["classes"].values()))
        # a strictly-inside cell must NOT be reported as snapped, or the flag
        # cannot single out the edge cases it exists for
        check("a strictly-inside ratio is not flagged as grid-snapped",
              not any(v["snapped"] for v in r90["classes"].values())
              or all(v["margin"] <= tol for v in r90["classes"].values()))

    print("", file=out)
    print("ALL PASS" if ok else "FAILURES ABOVE", file=out)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(HERE))
    sys.exit(self_test())
