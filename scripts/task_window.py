"""IS THIS CAP AN ACTUAL TASK? Run this BEFORE choosing cap tags for a grid.

A count cap only poses a question when THREE things hold at once. Drop any one
and the cell cannot distinguish any two methods, however well aimed either is:

    BINDS    hard_count - K      how many items the cap FORCES OUT. Not a
             >= MIN_FORCED        boolean: at K/n=0.90 on iwildcam class 2 the
                                 model predicts 336 against K=333, so the cap
                                 evicts THREE items and constrains essentially
                                 nothing while still passing a `hard > K` test.
                                 If K is above the model's own count the
                                 constraint is free and every arm ties.
    PRIZE    errors@K > 0        the top-K is not already perfect. If every
                                 selected item is correct there is no swap that
                                 improves it, only swaps that damage it.
    WIGGLE   p@K < 0.99          the cut is not buried in saturated territory.
                                 At p@K = 1.0 the items either side of the cut
                                 are indistinguishable to any gradient.

\U0001f6d1 WHY THIS EXISTS (FRAMEWORK 2(z16), measured 2026-09-01). On iwildcam the
window is NARROW and it is DIFFERENT PER CLASS -- class 2 is a task only at
K/n 0.70-0.90, class 7 only at 0.90-1.00, and the sole overlap is exactly 0.90.
**Every L20 / L30 / L50 campaign this project ran tested a NON-TASK**: class 2
has 0.0-1.0 errors inside K there and class 7's cut sits at p >= 0.9999. Those
nulls are not evidence about a method; they are the absence of a question, and
they are the best single explanation on record for why so many arms tied.

⚠️ IT NEEDS A REFERENCE MODEL, so it is not a pre-GPU screen. Point it at a
finished run of an UNCONSTRAINED arm (`tralo_null`, or `clip`) from the same
dataset and backbone. `scripts/ceiling_screen` is the label-only cousin that
runs before any model exists; this one is sharper because it reads the model's
actual ranking and its actual unconstrained count.

⚠️ AND IT IS PER (BACKBONE, DATASET). The unconstrained hard count and the
ranking quality both move with the backbone, so a window measured on
MobileNetV3 does not transfer to ViTB16 -- re-measure rather than assume.
"""
import argparse
import csv
import glob
import os
import sys

import numpy as np

WIGGLE_MAX = 0.99          # p@K at or above this is saturated territory
MIN_FORCED = 10            # items the cap must evict to be more than nominal
FRACTIONS = (0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2)


def load(run_dir, cls, raw=True):
    """(true labels, prob of cls, model's UNCONSTRAINED hard count for cls)."""
    name = "final_predictions_raw.csv" if raw else "final_predictions.csv"
    path = os.path.join(run_dir, name)
    y, pr, hard = [], [], 0
    with open(path) as f:
        for r in csv.DictReader(f):
            y.append(int(r["True_Label"]))
            pr.append(float(r["Prob_Class_%d" % cls]))
            hard += int(int(r["Predicted_Label"]) == cls)
    return np.array(y), np.array(pr, dtype=float), hard


def verdict(errors, p_at_K, forced):
    """`forced` is the PER-SEED list of (hard_count - K), never a mean.

    🛑 THE MEAN IS THE WRONG STATISTIC HERE, and reading it cost this
    project two published cells. On iwildcam/MobileNetV3 the four seeds of one
    lambda=0 cell predict 278, 329, 354 and 383 for class 2 -- a spread of 105
    items. At K=333 the mean is 336, so `forced = 3` and the cap reads
    "barely binds"; in fact it evicts 50 items in one seed and is entirely
    SLACK in two others. No seed resembles the mean.

    ⛔ AND HERE IS WHAT `PARTIAL` DOES **NOT** MEAN, because the obvious
    reading is wrong and was tested. The tempting inference is "the cap is
    already satisfied in that seed, so the penalty is zero, so the treated arm
    IS its own null there and the seed dilutes the contrast to zero." Measured
    on dom1/MobileNetV3, 12 (cap, seed) pairs: `tralo` and `tralo_null` are
    md5-DISTINCT in **4 of 4 slack seeds**, not identical in any. Two reasons,
    both structural on iwildcam: the binding scope is the LOCAL per-group
    ceiling and 7 of its 14 are K=0, so a camera with any prediction of that
    class violates its ceiling however slack the class TOTAL is; and the
    penalty reads SOFT counts, which exceed K while the hard count does not.

    So `forced` is a statement about the CLASS TOTAL only. Read PARTIAL as
    "this cap does not pose the same question to every seed", never as "those
    seeds are free nulls". A cap is called a TASK only where it binds in EVERY
    seed; where it binds in some, the verdict names the fraction rather than
    rounding it to a boolean in either direction.
    """
    forced = list(forced)
    if not forced:
        return "no data"
    if max(forced) <= 0:
        return "cap slack"
    n_bind = sum(1 for f in forced if f >= MIN_FORCED)
    if n_bind == 0:
        return "barely binds"
    if errors <= 0:
        return "no prize"
    if p_at_K >= WIGGLE_MAX:
        return "saturated"
    if n_bind < len(forced):
        return "** PARTIAL %d/%d **" % (n_bind, len(forced))
    return "** TASK **"


def sweep(runs, cls, fractions=FRACTIONS):
    """One row per K/n: mean errors, p@K, whether the cap binds, verdict."""
    Y, PR, H, cells, unread = [], [], [], set(), []
    for rd in runs:
        try:
            y, pr, h = load(rd, cls)
        except (OSError, KeyError, ValueError) as e:
            # An unreadable run used to vanish uncounted, so a glob that
            # matched 12 directories and read 2 reported a 2-run window as
            # though it were 12.
            unread.append((rd, type(e).__name__))
            continue
        # <root>/<Backbone>/<dataset>/<cap>/<arm>/<seed>
        parts = os.path.normpath(rd).split(os.sep)
        if len(parts) >= 5:
            cells.add(tuple(parts[-5:-3]))
        Y.append(y)
        PR.append(pr)
        H.append(h)
    if unread:
        print("  !! %d run(s) in this glob could not be read and are NOT in "
              "the counts below: %s"
              % (len(unread), ", ".join("%s (%s)" % u for u in unread[:3])),
              file=sys.stderr)
    if len(cells) > 1:
        # 🛑 THIS FUNCTION WRITES POLICY. Its output goes into
        # `configs/task_windows.yml`, which `gen_campaign` enforces as a HARD
        # REFUSAL on every future campaign. Averaging p@K across backbones
        # pulls a saturated cell under WIGGLE_MAX and launders it into a
        # `** TASK **`, and that verdict then becomes permanent.
        raise SystemExit(
            "REFUSED: task_window is per (backbone, dataset) and this glob "
            "spans %s. Averaging p@K and the hard count across them turns a "
            "saturated cell into a task, and this tool's output becomes the "
            "generator's refusal policy. Run one glob per backbone."
            % sorted("/".join(c) for c in cells))
    if not Y:
        return None
    n_true = int((Y[0] == cls).sum())
    if n_true <= 0:
        raise SystemExit(
            "REFUSED: class %d has ZERO true instances in this test slice, so "
            "K = max(1, round(frac * 0)) = 1 at every fraction and every row "
            "would read as a task. That is a fake greenlight written into the "
            "policy file, not a measurement." % cls)
    hard = float(np.mean(H))
    rows = []
    for frac in fractions:
        K = max(1, int(round(frac * n_true)))
        errs, pks = [], []
        for y, pr in zip(Y, PR):
            o = np.argsort(-pr)
            errs.append(int((y[o[:K]] != cls).sum()))
            pks.append(float(pr[o[K - 1]]))
        e, pk = float(np.mean(errs)), float(np.mean(pks))
        per_seed = [h - K for h in H]          # NOT mean(H) - K; see verdict()
        forced = hard - K
        n_bind = sum(1 for f in per_seed if f >= MIN_FORCED)
        rows.append(dict(frac=frac, K=K, errors=e, p_at_K=pk,
                         forced=forced, forced_per_seed=per_seed,
                         forced_min=min(per_seed), forced_max=max(per_seed),
                         n_bind=n_bind, n_seeds=len(per_seed),
                         binds=forced > 0,
                         verdict=verdict(e, pk, per_seed)))
    return dict(cls=cls, n_true=n_true, hard=hard, n_seeds=len(Y), rows=rows)


def recommend(res):
    """The MIDDLE of the task window, which is the cap to pick."""
    ok = [r for r in res["rows"] if r["verdict"] == "** TASK **"]
    if not ok:
        return None
    return ok[len(ok) // 2]


def self_test(out=sys.stdout):
    """Synthetic rankings with a KNOWN verdict in each of the four states."""
    import tempfile
    ok = True

    def check(name, cond):
        nonlocal ok
        ok = ok and cond
        print("  %-64s %s" % (name, "PASS" if cond else "FAIL"), file=out)

    def write(d, y, pr, pred, C=3, cls=1):
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, "final_predictions_raw.csv"), "w",
                  newline="") as f:
            w = csv.writer(f)
            w.writerow(["True_Label", "Predicted_Label"]
                       + ["Prob_Class_%d" % k for k in range(C)])
            for i in range(len(y)):
                row = [0.0] * C
                row[cls] = pr[i]
                w.writerow([y[i], pred[i]] + row)

    n, cls = 400, 1
    base = tempfile.mkdtemp()

    # (a) PERFECT ranking: the first 100 items are the only true ones and they
    #     rank first. Any K <= 100 has zero errors -> "no prize".
    d = os.path.join(base, "perfect")
    y = np.array([cls] * 100 + [0] * (n - 100))
    pr = np.linspace(0.999, 0.001, n)
    write(d, y, pr, [cls] * 150 + [0] * (n - 150))
    r = sweep([d], cls, fractions=(0.5,))["rows"][0]
    check("PERFECT ranking at K/n=0.5 -> 'no prize'", r["verdict"] == "no prize")

    # (b) NEGATIVE CONTROL: a ranking with errors mixed in near the cut, an
    #     unsaturated p@K, and a binding cap MUST come back as a TASK -- or the
    #     gate can only ever say no and is worthless.
    d = os.path.join(base, "task")
    rng = np.random.default_rng(0)
    y = np.zeros(n, dtype=int)
    idx = rng.permutation(n)
    y[idx[:100]] = cls                       # true items scattered, not sorted
    pr = np.linspace(0.90, 0.10, n)          # unsaturated everywhere
    write(d, y, pr, [cls] * 150 + [0] * (n - 150))
    r = sweep([d], cls, fractions=(0.5,))["rows"][0]
    check("NEGATIVE CONTROL: errors + unsaturated + binding -> '** TASK **' "
          "(errors %.1f, p@K %.3f)" % (r["errors"], r["p_at_K"]),
          r["verdict"] == "** TASK **")

    # (c) cap that does not bind: K above the model's own hard count
    r = sweep([d], cls, fractions=(1.6,))["rows"][0]
    check("K above the model's unconstrained count -> 'cap slack'",
          r["verdict"] == "cap slack")

    # (d) saturated: errors exist but the cut sits at p ~ 1
    d = os.path.join(base, "sat")
    y = np.zeros(n, dtype=int)
    y[idx[:100]] = cls
    pr = np.concatenate([np.full(300, 0.999999), np.linspace(0.4, 0.01, n - 300)])
    write(d, y, pr, [cls] * 350 + [0] * (n - 350))
    r = sweep([d], cls, fractions=(0.5,))["rows"][0]
    check("errors present but p@K ~ 1 -> 'saturated' (p@K %.6f)" % r["p_at_K"],
          r["verdict"] == "saturated")

    # (e) BARELY BINDS: a cap that evicts only a couple of items must not be
    #     called a task, however many errors and however unsaturated it is.
    d2 = os.path.join(base, "barely")
    y = np.zeros(n, dtype=int)
    y[idx[:100]] = cls
    pr = np.linspace(0.90, 0.10, n)
    write(d2, y, pr, [cls] * 202 + [0] * (n - 202))   # hard=202 vs K=200
    r = sweep([d2], cls, fractions=(2.0,))["rows"][0]
    check("cap evicting only %d item(s) -> 'barely binds', not a task"
          % r["forced"], r["verdict"] == "barely binds")

    # (g) 🛑 THE ONE THIS FILE GOT WRONG. Four seeds whose MEAN count clears
    #     the cap while ONE of them does not must NOT read as a task: in that
    #     seed the penalty is identically zero and the treated arm is its own
    #     null, so the cell is a diluted measurement, not a clean one.
    #     Built to mirror the real numbers: hard counts 210, 260, 260, 260
    #     against K=200 give a mean of 247.5 (forced 47.5, comfortably over
    #     MIN_FORCED) while seed one evicts only 10 ... so make it 205 to sit
    #     under the threshold.
    partial = []
    for i, hardc in enumerate((205, 260, 260, 260)):
        d3 = os.path.join(base, "partial%d" % i)
        y = np.zeros(n, dtype=int)
        y[idx[:100]] = cls
        pr = np.linspace(0.90, 0.10, n)
        write(d3, y, pr, [cls] * hardc + [0] * (n - hardc))
        partial.append(d3)
    r = sweep(partial, cls, fractions=(2.0,))["rows"][0]
    check("one seed of four below MIN_FORCED -> PARTIAL, not '** TASK **' "
          "(mean forced %.1f, per seed %s)"
          % (r["forced"], r["forced_per_seed"]),
          r["verdict"] == "** PARTIAL 3/4 **")
    check("...and the mean ALONE would have said TASK, so the check is live",
          verdict(r["errors"], r["p_at_K"], [r["forced"]]) == "** TASK **")

    # (h) LIVENESS for (g): four seeds that ALL bind must still read TASK, or
    #     the new rule just refuses everything.
    allbind = []
    for i in range(4):
        d4 = os.path.join(base, "allbind%d" % i)
        y = np.zeros(n, dtype=int)
        y[idx[:100]] = cls
        pr = np.linspace(0.90, 0.10, n)
        write(d4, y, pr, [cls] * (255 + i) + [0] * (n - 255 - i))
        allbind.append(d4)
    r = sweep(allbind, cls, fractions=(2.0,))["rows"][0]
    check("LIVENESS: four seeds that ALL bind -> '** TASK **' (%d/%d)"
          % (r["n_bind"], r["n_seeds"]), r["verdict"] == "** TASK **")
    check("recommend() never returns a PARTIAL row",
          all(x["verdict"] == "** TASK **"
              for x in [recommend(sweep(partial, cls))] if x))

    # (i) 🛑 THE POLICY GUARD. This tool's output is pasted into
    #     configs/task_windows.yml, which gen_campaign enforces as a hard
    #     refusal, so a glob that spans two backbones writes a pooled window
    #     into permanent policy. Averaging p@K across a saturated backbone and
    #     an unsaturated one pulls the mean under WIGGLE_MAX.
    two = []
    for bb in ("BackboneA", "BackboneB"):
        d5 = os.path.join(base, bb, "iwildcam", "L90_G95", "tralo_null", "seed_1")
        y = np.zeros(n, dtype=int)
        y[idx[:100]] = cls
        write(d5, y, np.linspace(0.90, 0.10, n), [cls] * 260 + [0] * (n - 260))
        two.append(d5)
    try:
        sweep(two, cls, fractions=(2.0,))
        check("a glob spanning TWO backbones is REFUSED", False)
    except SystemExit:
        check("a glob spanning TWO backbones is REFUSED", True)
    # LIVENESS: one backbone, same shape, must still work.
    r = sweep(two[:1], cls, fractions=(2.0,))
    check("LIVENESS: the same glob restricted to ONE backbone still sweeps",
          r is not None and r["n_seeds"] == 1)

    # (j) a class with ZERO true instances would read as a task at every
    #     fraction, because K = max(1, round(frac * 0)) = 1. That is a fake
    #     greenlight written into the policy file.
    d6 = os.path.join(base, "BackboneC", "iwildcam", "L90_G95", "tralo_null",
                      "seed_1")
    write(d6, np.zeros(n, dtype=int), np.linspace(0.9, 0.1, n),
          [cls] * 200 + [0] * (n - 200))
    try:
        sweep([d6], cls, fractions=(0.5,))
        check("a class with ZERO true instances is REFUSED", False)
    except SystemExit:
        check("a class with ZERO true instances is REFUSED", True)

    # (f) the recommender returns something inside the window, or nothing
    res = sweep([os.path.join(base, "task")], cls)
    rec = recommend(res)
    check("recommend() lands inside the task window, or returns None",
          rec is None or rec["verdict"] == "** TASK **")

    print("\n%s" % ("ALL PASS" if ok else "FAILURES ABOVE"), file=out)
    return 0 if ok else 1


def main(argv=None):
    a = argparse.ArgumentParser()
    a.add_argument("--runs", nargs="+",
                   help="run dirs of an UNCONSTRAINED arm (tralo_null / clip)")
    a.add_argument("--glob", help="glob for the same")
    a.add_argument("--classes", nargs="+", type=int, default=[2, 7])
    a.add_argument("--self-test", action="store_true")
    args = a.parse_args(argv)
    if args.self_test:
        return self_test()

    runs = args.runs or sorted(glob.glob(args.glob or ""))
    if not runs:
        raise SystemExit("no runs -- pass --runs or --glob")

    print("TASK WINDOW -- is the cap posing a question at all?")
    print("BINDS forces out >= %d items | PRIZE errors@K > 0 | "
          "WIGGLE p@K < %.2f" % (MIN_FORCED, WIGGLE_MAX))
    print("'forced' = the model's unconstrained count minus K: how many "
          "predictions the cap actually evicts.")
    print("READ THE PER-SEED COLUMN, NOT THE MEAN. On iwildcam/MobileNetV3 the")
    print("four seeds spread 105 items, so a mean 'forced' of 3 can be 50 in one")
    print("seed and -55 in another. PARTIAL n/N means the cap poses its question")
    print("to only n seeds -- NOT that the other seeds are free nulls, which was")
    print("tested by md5 and refuted (see verdict.__doc__).")
    print("%d reference run(s)\n" % len(runs))
    any_task = False
    for cls in args.classes:
        res = sweep(runs, cls)
        if res is None:
            print("  class %d: no readable run" % cls)
            continue
        print("  class %d   n_true=%d   model predicts %.0f unconstrained   "
              "(%d seed(s))" % (cls, res["n_true"], res["hard"], res["n_seeds"]))
        print("    %6s %7s %8s %9s %8s %15s %6s   %s"
              % ("K/n", "K", "errors", "p_at_K", "forced",
                 "forced per seed", "binds", "verdict"))
        for r in res["rows"]:
            print("    %6.2f %7d %8.1f %9.5f %8.0f %15s %6s   %s"
                  % (r["frac"], r["K"], r["errors"], r["p_at_K"],
                     r["forced"],
                     "%.0f..%.0f" % (r["forced_min"], r["forced_max"]),
                     "%d/%d" % (r["n_bind"], r["n_seeds"]),
                     r["verdict"]))
        rec = recommend(res)
        if rec:
            any_task = True
            win = [r["frac"] for r in res["rows"] if r["verdict"] == "** TASK **"]
            print("    -> TASK WINDOW K/n %.2f to %.2f; pick %.2f (K=%d)"
                  % (min(win), max(win), rec["frac"], rec["K"]))
        else:
            print("    -> NO TASK AT ANY CAP. This class cannot test a method "
                  "on this model.")
        print("")

    if not any_task:
        print("!! NO CLASS HAS A TASK WINDOW. Do not run a grid here: every")
        print("!! cell would be measuring the absence of a question.")
        return 1
    print("Pick caps INSIDE every capped class's window. One fraction is")
    print("applied to all capped classes, so if the windows do not overlap the")
    print("protocol cannot express a correct cap -- see FRAMEWORK 2(z16).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
