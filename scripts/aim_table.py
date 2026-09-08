"""WHERE DOES THE CONSTRAINT'S PER-ITEM GRADIENT LAND, PER CLASS, AND IS THERE
ANYTHING THERE TO WIN?

The per-logit constraint gradient is `A_S * p(1-p)`. Every method in this
project -- TraLO, ALM, LDF, Hounie -- differs only in the SCOPE scalar `A_S`.
None varies the per-item factor `p(1-p)`, which peaks at p = 0.5 and vanishes
at both extremes. So whether a re-aiming idea can pay in a cell is decided by
two numbers that are both computable with no GPU:

    AIM     `p(1-p)` at the cut, as a fraction of its maximum 0.25
    PRIZE   errors inside K -- items the allocator emits and gets wrong

A cell only rewards re-aiming when the PRIZE is real AND the AIM is bad. This
tool prints both, per (campaign, backbone, cap, CLASS), and applies that rule.

## Why it exists, and why PER CLASS is the whole point

FRAMEWORK 2(z56) section 6 first asserted that every method aims at the items
least able to move the emitted set, quoting `p@K` 0.9948-0.9972 and `p(1-p)`
about 0.003. **That is a TIGHT-CAP number and it is false at the caps that are
actually run**, where the tight cells have a prize of exactly ZERO and cannot
be the subject of the claim. Corrected 2026-09-08 from 2(z15)'s own table:

    LOOSE L90_G95  class 2   p@K 0.38433   p(1-p) 0.23662   95% of max   28.5 errors
    LOOSE L90_G95  class 7   p@K 0.99253   p(1-p) 0.00741    3% of max   30.2 errors

Same cell, same campaign, comparable prizes -- and **32x** less per-item
gradient on class 7 (233x at L80_G95). The aim defect on iwildcam is
CLASS-asymmetric, not cap-asymmetric, and it had been stated the wrong way
round in this repo. A pooled or cap-level statistic cannot see it, which is why
the key here is (cell, CLASS) and the tool prints the RATIO between the capped
classes explicitly.

⚠️ It reads `final_predictions_raw.csv` -- the model's own probabilities BEFORE
the allocator. The deployed file is forced to exactly K, so every rank-K
quantity in it is an artefact of the allocator rather than of the model.

⚠️ AND IT DECIDES NOTHING ON ITS OWN. A bad aim beside a real prize says a
re-aiming mechanism is not ruled out by geometry. It does NOT say one exists,
that it survives `normalize`, or that it is absent from the rejected ledger --
`tralo_cut` had a 361x aim change and lost. Use this to REFUSE directions
cheaply, never to license one.
"""
import argparse
import collections
import csv
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.training.constraints import compute_global_constraints  # noqa: E402

# A prize this small is not worth a mechanism even if the aim is perfect.
MIN_PRIZE = 10.0
# `p(1-p)` at or below this is 4% of the maximum 0.25. Chosen to match the
# order of magnitude 2(z15) measures for class 7 (0.0007-0.0074) while sitting
# well below class 2's 0.165-0.237, so the two do not both fall on one side.
# 🛑 THERE ARE ALREADY THREE OTHER BARS ON THIS QUANTITY IN THIS REPO --
# `sensitivity_screen` 0.0099, `reachability` 0.040, `cut_gap` 0.005 -- and
# they differ 8x. SAY WHICH ONE YOU MEAN when quoting a number.
AIM_BAR = 0.01


def _load_raw(path):
    """(labels, {class: [probs]}, groups) from one final_predictions_raw.csv."""
    y, probs, groups = [], collections.defaultdict(list), []
    with open(path, "r", newline="") as fh:
        rd = csv.DictReader(fh)
        cols = [c for c in (rd.fieldnames or []) if c.startswith("Prob_Class_")]
        classes = sorted(int(c.rsplit("_", 1)[1]) for c in cols)
        has_g = "Group_ID" in (rd.fieldnames or [])
        for r in rd:
            y.append(int(float(r["True_Label"])))
            for c in classes:
                probs[c].append(float(r["Prob_Class_%d" % c]))
            groups.append(r["Group_ID"] if has_g else "_")
    return y, probs, groups


def aim_for(y, p, K, idx=None):
    """`p@K`, `p(1-p)` there, and errors inside the top K, over ONE scope.

    🛑 `idx` RESTRICTS THE RANKING TO A SINGLE GROUP, AND PASSING IT IS NOT
    OPTIONAL FOR A REAL CAMPAIGN. Every allocator in this project is PER-GROUP:
    it emits the top `K_gc` within each group, never a global top-K. Ranking
    globally answers a different question and this project has already paid for
    that exact confusion once -- the cap screen counted a global top-K and
    overstated the prize 4.25x. The default `idx=None` exists only for the
    self-test's single-group fixtures.
    """
    if idx is None:
        idx = list(range(len(p)))
    if K <= 0 or K > len(idx):
        return None
    order = sorted(idx, key=lambda i: -p[i])
    top = order[:K]
    pk = p[top[-1]]
    return dict(pK=pk, aim=pk * (1.0 - pk), errors=float(sum(1 for i in top if y[i] != _CLS)))


_CLS = None  # set per call; kept module-level so `aim_for` stays a pure read


def analyse(run_dirs):
    """One row per (backbone, dataset, cap, class), averaged over SEEDS only."""
    global _CLS
    acc = collections.defaultdict(lambda: collections.defaultdict(list))
    meta = {}
    for d in run_dirs:
        cfg_p = os.path.join(d, "config.json")
        raw_p = os.path.join(d, "final_predictions_raw.csv")
        if not (os.path.exists(cfg_p) and os.path.exists(raw_p)):
            continue
        cfg = json.load(open(cfg_p))
        if cfg.get("status") != "completed":
            continue
        y, probs, groups = _load_raw(raw_p)
        if not y:
            continue
        # the slice's own fields live under `dataset_config`; the top level
        # carries only the arm and the cap. Reading `cfg["constrained_class"]`
        # returns None and silently skips every run, which is how this tool
        # first reported "no completed runs" against 384 of them.
        dsc = cfg.get("dataset_config") or {}
        classes = dsc.get("constrained_class", cfg.get("constrained_class"))
        if classes is None:
            continue
        if isinstance(classes, int):
            classes = [classes]
        num_classes = dsc.get("num_classes") or cfg.get("num_classes") or (max(probs) + 1)
        gpct = cfg["constraint"][1]

        # the pipeline's OWN budget rule, imported rather than reimplemented,
        # so a change to the rounding cannot silently desynchronise this tool
        class _DF(object):
            def __init__(self, y):
                self._y = y

            def __getitem__(self, _):
                return _Col(self._y)

        class _Col(object):
            def __init__(self, y):
                self._y = y

            def __eq__(self, c):
                return _Mask([v == c for v in self._y])

        class _Mask(object):
            def __init__(self, m):
                self._m = m

            def sum(self):
                return sum(self._m)

        # 🔑 THE CUT IS PER (GROUP, CLASS). `local_pct` is cfg["constraint"][0].
        # The GLOBAL budget is deliberately not used: at the caps that are run,
        # G95 against a local sum well under it means the global scope does not
        # bind at all, so a global rank-K is not a cut anything reaches.
        lpct = cfg["constraint"][0]
        by_group = collections.defaultdict(list)
        for i, g in enumerate(groups):
            by_group[g].append(i)
        budgets = {}
        for g, idx in by_group.items():
            sub = [y[i] for i in idx]
            b = compute_global_constraints(
                _DF(sub), "label", lpct, constrained_class=classes,
                num_classes=num_classes)
            budgets[g] = b

        key = (cfg.get("model_name") or cfg.get("model"), cfg.get("dataset_mode"),
               cfg.get("constraint_tag") or _cap_from_path(d),
               cfg.get("arm") or cfg.get("methodology"))
        for c in classes:
            _CLS = c
            # one reading per GROUP, then combined. A group whose ceiling is
            # K=0 poses no cut at all and is excluded rather than scored as a
            # perfect one -- 7 of 14 iwildcam ceilings are K=0 and folding them
            # in as `aim 0` would manufacture starvation out of arithmetic.
            per_g, tot_err, wsum, wtot = [], 0.0, 0.0, 0.0
            for g, idx in by_group.items():
                K = int(budgets[g][c])
                if K <= 0 or K > len(idx):
                    continue
                r = aim_for(y, probs[c], K, idx=idx)
                if r is None:
                    continue
                per_g.append(r)
                tot_err += r["errors"]
                # weight the aim by the group's BUDGET: a 3-item ceiling and a
                # 200-item one are not two equal readings of one number.
                wsum += r["aim"] * K
                wtot += K
            if not per_g:
                continue
            acc[key][c].append(dict(
                pK=sum(r["pK"] * 1.0 for r in per_g) / len(per_g),
                aim=(wsum / wtot) if wtot else 0.0,
                errors=tot_err,
                K=int(wtot),
                n=float(sum(1 for v in y if v == c)),
                scopes=len(per_g),
                dead=len(by_group) - len(per_g)))
            meta[key] = len(groups)
    return acc


def _cap_from_path(d):
    parts = os.path.normpath(d).split(os.sep)
    for p in parts:
        if p.startswith("L") and "_G" in p:
            return p
    return "?"


def verdict(prize, aim):
    if prize < MIN_PRIZE:
        return "no prize    nothing inside K to win, aim is irrelevant"
    if aim <= AIM_BAR:
        return "STARVED     real prize, gradient at %.1f%% of max" % (100 * aim / 0.25)
    return "well aimed  %.0f%% of max -- re-aiming cannot pay here" % (100 * aim / 0.25)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--campaign", nargs="+", default=[],
                    help="campaign roots")
    ap.add_argument("--arms", nargs="+", default=None,
                    help="restrict to these arms; default reads whatever is "
                         "on disk and says so")
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args(argv)
    if a.self_test:
        return self_test()
    if not a.campaign:
        ap.error("--campaign is required (or --self-test)")

    runs = []
    for root in a.campaign:
        runs += glob.glob(os.path.join(root, "*", "*", "*", "*", "seed_*"))
    if a.arms:
        keep = set(a.arms)
        runs = [d for d in runs if os.path.basename(os.path.dirname(d)) in keep]
    acc = analyse(runs)
    if not acc:
        print("no completed runs with final_predictions_raw.csv under those roots")
        return 1

    print("")
    print("AIM vs PRIZE, per (cell, CLASS). `p(1-p)` is the per-item constraint")
    print("gradient at the cut; PRIZE is errors inside K. Averaged over SEEDS only.")
    print("Bars: prize >= %.0f items, aim <= %.4f (=%.0f%% of max 0.25)."
          % (MIN_PRIZE, AIM_BAR, 100 * AIM_BAR / 0.25))
    print("")
    print("%-14s %-8s %-12s %-8s %4s %6s %6s %9s %8s %8s  %s"
          % ("backbone", "dataset", "cap", "arm", "cls", "sumK", "n", "p@K",
             "p(1-p)", "errs<=K", "verdict"))
    print("-" * 132)

    starved = []
    for key in sorted(acc, key=lambda k: tuple(str(x) for x in k)):
        model, ds, cap, arm = key
        per = {}
        for c in sorted(acc[key]):
            rows = acc[key][c]
            pk = sum(r["pK"] for r in rows) / len(rows)
            aim = sum(r["aim"] for r in rows) / len(rows)
            err = sum(r["errors"] for r in rows) / len(rows)
            K = rows[0]["K"]
            n = rows[0]["n"]
            per[c] = (aim, err)
            v = verdict(err, aim)
            if v.startswith("STARVED"):
                starved.append((model, cap, arm, c, err, aim))
            print("%-14s %-8s %-12s %-8s %4d %6d %6.0f %9.5f %8.5f %8.1f  %s"
                  % (str(model)[:14], str(ds)[:8], str(cap)[:12], str(arm)[:8],
                     c, K, n, pk, aim, err, v))
        # 🔑 THE RATIO IS THE POINT: the metric macro-averages the capped
        # classes, so a large ratio means the loss and the score disagree about
        # which class matters.
        if len(per) == 2:
            (ca, (aa, ea)), (cb, (ab, eb)) = sorted(per.items())
            if min(aa, ab) > 0:
                hi, lo = (ca, cb) if aa > ab else (cb, ca)
                ratio = max(aa, ab) / min(aa, ab)
                pr = max(ea, eb) / max(min(ea, eb), 1e-9)
                print("%-14s %-8s %-12s %-8s  -> per-item gradient ratio c%d:c%d = "
                      "%7.1fx  while their PRIZES differ only %.1fx"
                      % ("", "", "", "", hi, lo, ratio, pr))
        print("")

    print("=" * 132)
    if starved:
        print("STARVED CELLS -- real prize, gradient at or below %.0f%% of max:"
              % (100 * AIM_BAR / 0.25))
        for model, cap, arm, c, err, aim in starved:
            print("    %-14s %-12s %-8s class %d: %.1f errors inside K at "
                  "p(1-p)=%.5f" % (model, cap, arm, c, err, aim))
        print("")
        print("  ⚠️  This says re-aiming is NOT RULED OUT by geometry in these")
        print("      cells. It does NOT say a mechanism exists, that it survives")
        print("      `normalize`, or that it is absent from the rejected ledger.")
        print("      `tralo_cut` moved the aim 361x and LOST. FRAMEWORK 2(z56) 6.")
    else:
        print("NO STARVED CELL. Wherever there is a prize the gradient is already")
        print("on it, so no within-scope re-aiming can pay on this evidence.")
    return 0


def self_test():
    """Gates the two verdicts AND the direction of the ratio.

    Negative controls, because a screen that has never refused has never been
    shown to work:
      (a) a PERFECT ranking must read `no prize`, whatever its aim -- otherwise
          the tool licenses a mechanism in the tight cells, which is exactly
          the error 2(z56) 6 made in prose;
      (b) a cell with a real prize AND a mid-range cut must read `well aimed`
          and NOT be reported as starved;
      (c) the starved verdict must actually fire when both conditions hold;
      (d) `aim_for` must reproduce 2(z15)'s published numbers from a
          constructed ranking, or the tool is measuring something else.
    """
    global _CLS
    fails = []

    def check(label, ok):
        print("  %-64s %s" % (label, "ok" if ok else "FAIL"))
        if not ok:
            fails.append(label)

    # (d) reproduce the published class-2 L90_G95 row: p@K 0.38433 -> 0.23662
    aim = 0.38433 * (1 - 0.38433)
    check("2(z15) class 2 L90: p(1-p) = 0.2366", abs(aim - 0.23662) < 1e-4)
    check("2(z15) class 2 L90 is 95% of max", abs(100 * aim / 0.25 - 94.6) < 1.0)
    aim7 = 0.99253 * (1 - 0.99253)
    check("2(z15) class 7 L90: p(1-p) = 0.0074", abs(aim7 - 0.00741) < 1e-4)
    check("...and the two differ by 32x", abs(aim / aim7 - 31.9) < 1.5)

    # (a) a PERFECT ranking: every item inside K is correct
    _CLS = 1
    y = [1] * 50 + [0] * 50
    p = [0.99 - 0.0001 * i for i in range(100)]
    r = aim_for(y, p, 50)
    check("perfect ranking: 0 errors inside K", r["errors"] == 0.0)
    check("perfect ranking -> `no prize`, NOT starved",
          verdict(r["errors"], r["aim"]).startswith("no prize"))

    # (b) real prize at a MID-RANGE cut -> well aimed, must not be starved
    y = [1] * 30 + [0] * 20 + [1] * 50
    p = [0.9 - 0.008 * i for i in range(100)]
    r = aim_for(y, p, 50)
    check("mid-range cut: prize is real", r["errors"] >= MIN_PRIZE)
    check("mid-range cut: aim is high", r["aim"] > 0.2)
    check("mid-range cut -> `well aimed`, NOT starved",
          verdict(r["errors"], r["aim"]).startswith("well aimed"))

    # (c) real prize at a SATURATED cut -> starved, and it must fire
    y = ([1] * 35 + [0] * 15) + [1] * 50
    p = [0.9999 - 1e-7 * i for i in range(50)] + [0.001] * 50
    r = aim_for(y, p, 50)
    check("saturated cut: prize is real", r["errors"] >= MIN_PRIZE)
    check("saturated cut: aim is below the bar", r["aim"] <= AIM_BAR)
    check("saturated cut -> STARVED fires",
          verdict(r["errors"], r["aim"]).startswith("STARVED"))

    # K out of range must refuse rather than invent a number
    check("K larger than the test set returns None", aim_for(y, p, 999) is None)
    check("K = 0 returns None", aim_for(y, p, 0) is None)

    print("")
    print("%d checks, %d failed" % (13, len(fails)))
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
