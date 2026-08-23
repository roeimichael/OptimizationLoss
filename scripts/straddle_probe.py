"""How much of the headroom is REACHABLE by a step the size ours actually is?

WHY THIS EXISTS. `scripts/headroom.py` reports the gap from `clip` to a PERFECT
allocator: 1.9-9.9 items. That is an ORACLE quantity -- it assumes the ranking
can be rewritten arbitrarily. Ours cannot. FRAMEWORK 2(a3) measured that under
`constraint_grad_mode: normalize` the delivered displacement is exactly
`lr * clip` per step, so the constraint moves scores by a BOUNDED amount. An
item misranked by a wide margin is unreachable by any number of such steps, and
counting it as headroom overstates what any arm could ever collect.

So this file answers a different question from `headroom.py`: not "how much is
wrong" but "how much is wrong AND close enough to the cut to be fixed by a
change the size of the one we deliver".

WHERE THE IDEA COMES FROM, and what is new. Lorieul, Joly and Shasha,
"Classification Under Ambiguity: When Is Average-K Better Than Top-K?"
(arXiv:2112.08851, 2021) ask when an ADAPTIVE set-valued predictor -- one that
returns a variable number of labels averaging to K -- beats a fixed top-K, and
answer it from the AMBIGUITY PROFILE of the posteriors rather than from the
error rate. Their budget is on LABELS PER ITEM. Ours is the transpose, ITEMS PER
LABEL, and that transposition did not turn up in the survey.

⚠️ What is borrowed here is the IDEA that the structure AT THE CUT, not error in
general, decides whether anything is winnable. Verified from the abstract:
title, authors, year, and that the paper is about average-K versus top-K. This
file does not restate or depend on any of their propositions, and nothing below
is derived from them -- `reachable` is defined and gated on its own terms. Read
the paper before citing it for anything sharper than the idea.

THE QUANTITY. With exactly K predictions emitted for a capped class, improving
the endpoint means SWAPPING: a false positive above the cut leaves and a true
positive below it enters. One swap is worth one item. Within a displacement
budget `delta`,

    reachable(delta) = min( #FP in [t, t+delta], #TP in [t-delta, t) )

where `t` is the K-th largest score. That is exactly how many swaps a score
change of size `delta` can perform, in ITEMS, directly comparable to every other
effect in this project. `contested(delta)` -- how many items lie within `delta`
of the cut at all -- is the LABEL-free version, so it can be read on a test set
whose labels are not to be touched, or on a fresh unlabelled set under an
existing model. ⚠️ It is NOT model-free: without a model there is no ranking and
so no cut, and it cannot screen a candidate dataset before training. The screen
that runs before any GPU time is `dataset_screen`, from labels and metadata.

⚠️ It is an UPPER bound twice over. It assumes every near-cut item moves the
RIGHT way, and it ignores the per-group ceilings, which can forbid a swap the
global count allows. A method cannot beat this number; it can easily fall short.

THE SATURATION IDENTITY -- why this REFINES `headroom.py` rather than competing
with it. As `delta` grows, `fp_near -> K - tp_above` and
`tp_near -> n_pos - tp_above`, so

    reachable(inf) = min(K, n_pos) - tp_above = oracle,   exactly.

The two agree in the limit, by construction and not by luck. What the delta
ladder adds is the RATE of approach, which is the distance distribution of the
misranked items -- so `reachable` never contradicts a headroom figure, it says
how far the scores would have to move to collect it. It also follows that
`reachable <= oracle` at every delta, which is pinned as a regression: an
implementation returning a sum, or one side of the min, would break it.

WHERE `delta` COMES FROM. Assuming one is the whole difficulty, so the default
mode MEASURES it: given a treated run and its `_null` twin at the same seed --
same warm-up, same allocator, same RNG, lambda=0 -- the per-item difference in
the capped-class score IS the displacement the constraint delivered. `reachable`
at that measured delta is the ceiling for the constraint AS CONFIGURED, and the
campaign's own effect size can be read against it. `--sweep` falls back to
fractions of the score range when no twin exists, and says so.

⚠️ **The measured delta is the NET displacement over all 29 constraint steps,
not the path length.** For "what did this arm actually achieve" that is the
right quantity -- only the final ranking reaches the allocator, so an item that
moved out and back contributes nothing. For "what COULD a bounded-step method
achieve" it is a LOWER bound on the budget, because a non-monotone path can
cover more ground than its endpoints show. So a small `reachable` here says this
arm did not have the reach; it does not by itself prove no schedule could. Say
which of the two questions is being answered.

THE SHUFFLED CONTROL, and which way it points. Permuting the scores keeps their
DISTRIBUTION and destroys the ORDERING, so the swaps it leaves available depend
only on n, K and prevalence -- measured at 10.80 vs 11.60 items across two
regimes whose true error structures differ by 5x. That makes it a reference the
real number is read AGAINST, and the SIGN of the deviation is the diagnostic:

    real << shuffled   the ranking has already collected the easy swaps; what
                       is left at the cut is genuinely hard
    real ~= shuffled   the ranking carries no information at the cut, and this
                       statistic is reading the score distribution, not the
                       ordering -- report nothing
    real >> shuffled   true positives are parked BELOW the cut in numbers chance
                       does not explain. This is the one configuration in which
                       a cut-local method has something real to win.

⚠️ It does NOT collapse toward zero under shuffling; it RISES, because a random
top-K has false positives and true positives scattered on both sides of the cut.
Reading it as a must-collapse control inverts it.

THE GATE, via `--self-test`, over two regimes with known error geometry:
  * `matched`   -- clean labels, so residual errors sit AT the cut, where a
                   Bayes-optimal ranking still confuses items. Share must be
                   HIGH: few oracle items, nearly all of them reachable.
  * `tailnoise` -- true positives planted among the LOWEST-scoring items, far
                   below the cut. Share must be LOW: a large oracle gap that a
                   bounded step cannot touch.
The gate compares the SHARE across the whole delta ladder, not at the widest
band alone -- at a delta near the score range almost anything is reachable in
either regime, so a one-band gate would pass on a statistic that had stopped
discriminating. A statistic that cannot separate these two is reading the error
RATE rather than where the errors sit, and reports nothing.

    python -m scripts.straddle_probe --campaign results/iwc1
    python -m scripts.straddle_probe --self-test
"""
import argparse
import collections
import glob
import json
import os

import numpy as np

from scripts.frozen_head_probe import (allocate, budgets, load_real,
                                       make_synthetic)

SWEEP_FRACS = (0.001, 0.003, 0.01, 0.03, 0.1)
SWEEP_NAMES = ["%.3f x range" % f for f in SWEEP_FRACS]
MEASURED_NAMES = ["measured q95", "2x measured", "10x measured"]
CONTESTED_TARGETS = (20, 50, 100)
CONTESTED_NAMES = ["contested=%d" % n for n in CONTESTED_TARGETS]


def cut_score(scores, K):
    """The K-th largest score -- the threshold the allocator actually applies."""
    if K <= 0:
        return float("inf")
    if K >= len(scores):
        return float("-inf")
    return float(np.partition(scores, -K)[-K])


def straddle(scores, is_pos, K, deltas):
    """Swap counts at the cut per delta, plus the unbounded oracle gap."""
    scores = np.asarray(scores, float)
    is_pos = np.asarray(is_pos, bool)
    t = cut_score(scores, K)
    above = scores >= t
    tp_above = int((above & is_pos).sum())
    oracle = min(K, int(is_pos.sum())) - tp_above

    bands = []
    for d in deltas:
        fp_near = int((above & ~is_pos & (scores <= t + d)).sum())
        tp_near = int((~above & is_pos & (scores >= t - d)).sum())
        bands.append({"delta": float(d),
                      "contested": int((np.abs(scores - t) <= d).sum()),
                      "reachable": min(fp_near, tp_near)})
    return {"K": int(K), "n_pos": int(is_pos.sum()), "cut": t,
            "oracle": int(oracle), "bands": bands}


def delta_for_contested(scores, K, target):
    """The delta whose band holds ~`target` items around the cut.

    WHY THIS EXISTS. Sweeping delta as a FRACTION OF THE SCORE RANGE makes
    cross-cap comparison unreadable: at a looser cap the cut sits somewhere else
    in the score distribution, so the same fraction covers a different number of
    items. Measured on the stored evidence -- the reachable SHARE of the oracle
    gap falls as the cap loosens in 24 of 33 series (sign p=0.014), which looks
    like a geometry result until the control is read: `contested` falls too, in
    22 of 33 (p=0.080). Thinning density explains the same numbers, and that
    parameterisation cannot separate the two.

    Holding the CONTESTED MASS fixed removes the confound: the band always holds
    the same number of items, so a change in `reachable` is a change in how many
    of them are useful swaps, which is the geometry question.

    Bisection on delta, since `contested` is monotone non-decreasing in it.
    """
    scores = np.asarray(scores, float)
    t = cut_score(scores, K)
    if not np.isfinite(t):
        return 0.0
    hi = float(np.max(np.abs(scores - t))) or 1.0
    lo = 0.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if int((np.abs(scores - t) <= mid).sum()) < target:
            lo = mid
        else:
            hi = mid
    return hi


def emitted_K(data):
    """How many items the project's OWN endpoint emits per capped class.

    Not `G[c]`: the allocator answers to the local ceilings too, so the cut it
    really applies can sit below the global budget. `allocate` is imported, not
    reimplemented, so a change to the endpoint reaches this probe.
    """
    G, L = budgets(data.y, data.groups, data.classes, data.local_pct,
                   data.global_pct, data.n_classes)
    alloc = allocate(data.ref_probs, data.groups, G, L, data.classes)
    return {c: int((alloc == c).sum()) for c in data.classes}


def probe(data, deltas_for, rng):
    """One ProbeData -> per-class straddle, each beside its shuffled control.

    `deltas_for(c, scores)` returns this class's delta ladder, so the measured
    mode can give every class its OWN displacement rather than one pooled
    number -- the capped classes differ in prevalence and in how hard the cap
    bites, and pooling them would average a bound with a non-bound.
    """
    Ks = emitted_K(data)
    out = {}
    for c in data.classes:
        s = data.ref_probs[:, c].astype(float)
        deltas = deltas_for(c, s)
        row = straddle(s, data.y == c, Ks[c], deltas)
        # CONTROL: identical score DISTRIBUTION, ordering destroyed.
        row["shuffled"] = straddle(rng.permutation(s), data.y == c, Ks[c],
                                   deltas)
        out[c] = row
    return out


def sweep_deltas(_c, scores):
    span = float(scores.max() - scores.min()) or 1.0
    return [f * span for f in SWEEP_FRACS]


def matched_deltas_for(data):
    """A ladder that holds the CONTESTED MASS fixed instead of the delta.

    Returns a `deltas_for(c, scores)` closure, because the cut depends on the
    emitted K and that is per class.
    """
    Ks = emitted_K(data)

    def deltas_for(c, scores):
        return [delta_for_contested(scores, Ks[c], n) for n in CONTESTED_TARGETS]
    return deltas_for


# ------------------------------------------------- the measured displacement --

def _meta(run_dir):
    with open(os.path.join(run_dir, "config.json"), encoding="utf-8") as fh:
        cfg = json.load(fh)
    # The ARM is part of the cell. FRAMEWORK rule 4 fixes the atomic cell at
    # (dataset, backbone, cap, METHOD), and it belongs here for a reason
    # specific to this probe: the arm IS the ranking, and the ranking is the
    # whole object being measured. Averaging `clip` with `tralo_uniform` in one
    # row describes neither.
    # TWO keys, deliberately, because they answer different questions.
    # `cell` is for REPORTING and includes the ARM: rule 4 fixes the atomic
    # cell at (dataset, backbone, cap, METHOD), and here the arm IS the
    # ranking -- the whole object being measured -- so averaging `clip` with
    # `tralo_uniform` into one row describes neither.
    # `pair_cell` is for PAIRING and excludes it, because a treated arm and its
    # `_null` twin differ in exactly that field and must still find each other.
    base = (cfg.get("dataset_mode"), cfg.get("model_name"),
            cfg.get("constraint_tag"))
    return {"arm": cfg.get("arm"),
            "cell": base + (cfg.get("arm"),),
            "pair_cell": base,
            "seed": (cfg.get("hyperparams") or {}).get("seed")}


def pair_runs(run_dirs):
    """(treated, null) twins: same cell and seed, arm vs arm + '_null'.

    The null is the SAME warm-up, allocator and seed with lambda=0, so the
    per-item score difference is the constraint's doing and nothing else's.
    """
    by_key = {}
    for r in run_dirs:
        try:
            m = _meta(r)
        except (OSError, ValueError):
            continue
        by_key[(m["pair_cell"], m["seed"], m["arm"])] = r
    pairs = []
    for (cell, seed, arm), r in sorted(by_key.items(),
                                       key=lambda kv: str(kv[0])):
        if arm is None or arm.endswith("_null"):
            continue
        twin = by_key.get((cell, seed, arm + "_null"))
        if twin:
            pairs.append((arm, cell, seed, r, twin))
    return pairs


def measured_delta(treated, null, quantile=0.95):
    """How far the constraint moved the capped-class scores, in score units."""
    # probabilities only: this probe never touches `.features`, so runs
    # predating src/pipeline/features.py are legitimately probeable here.
    a, b = (load_real(treated, require_features=False),
            load_real(null, require_features=False))
    if len(a.y) != len(b.y) or not np.array_equal(a.y, b.y):
        raise SystemExit("%s and %s are not the same test set -- refusing to "
                         "difference them" % (treated, null))
    per_class = {}
    for c in a.classes:
        d = np.abs(a.ref_probs[:, c] - b.ref_probs[:, c])
        per_class[c] = {"q": float(np.quantile(d, quantile)),
                        "median": float(np.median(d)),
                        "max": float(d.max())}
    return a, per_class


# ------------------------------------------------------------------- report --

def collect(agg, rows, names, cell=None):
    """Accumulate one run into the aggregate, keyed by (CELL, class).

    The cell is part of the key, not decoration. `full_panel` groups by
    (dataset, model, cap, capped) because pooling across any of them is this
    project's most-repeated analysis error -- and the stored-evidence tree is
    exactly the shape that punishes it: 128 runs spanning THREE datasets and
    THREE cap levels, where "class 1" means a different class in each. Pooling
    them produced a confident row describing nothing.
    """
    for c, v in rows.items():
        key = (cell, c)
        a = agg.setdefault(key, {"oracle": [], "K": [], "n_pos": [],
                                 "bands": collections.defaultdict(list),
                                 "shuf": collections.defaultdict(list),
                                 "order": list(names)})
        a["oracle"].append(v["oracle"])
        a["K"].append(v["K"])
        a["n_pos"].append(v["n_pos"])
        for nm, b, sb in zip(names, v["bands"], v["shuffled"]["bands"]):
            a["bands"][nm].append(b)
            a["shuf"][nm].append(sb)


def reachable_share(agg, band):
    """Reachable items over oracle items at one delta, summed across classes.

    Summed, not averaged over classes: both numerator and denominator are ITEM
    counts, and a class with a 1-item gap must not weigh the same as one with
    20. Averaging the per-class ratios did exactly that.
    """
    num = sum(np.mean([b["reachable"] for b in agg[k]["bands"][band]])
              for k in agg if band in agg[k]["bands"])
    den = sum(float(np.mean(agg[k]["oracle"])) for k in agg)
    return num / den if den > 0 else 0.0


def report(agg, n_runs):
    """One block per CELL. Never one block per class across cells."""
    cells = []
    for cell, _c in agg:
        if cell not in cells:
            cells.append(cell)
    for cell in sorted(cells, key=lambda x: str(x)):
        if cell is not None:
            print("  CELL %s" % "/".join(str(x) for x in cell))
        for (cl, c) in sorted((k for k in agg if k[0] == cell),
                              key=lambda k: k[1]):
            _report_one(agg[(cl, c)], c)


def _report_one(a, c):
    """One class within one cell. Split from `report` so the cell loop
    stays readable; the run count is taken from this key alone."""
    orc = float(np.mean(a["oracle"]))
    n_runs = len(a["oracle"])
    print("  CLASS %d -- emits K=%.0f, %.0f true in test, %d run(s)"
          % (c, np.mean(a["K"]), np.mean(a["n_pos"]), n_runs))
    print("    unbounded ORACLE gap: %.2f items" % orc)
    print("    %-14s %10s %10s %13s"
          % ("delta", "contested", "reachable", "shuffled ctrl"))
    for nm in a["order"]:
        print("    %-14s %10.1f %10.2f %13.2f"
              % (nm,
                 np.mean([b["contested"] for b in a["bands"][nm]]),
                 np.mean([b["reachable"] for b in a["bands"][nm]]),
                 np.mean([b["reachable"] for b in a["shuf"][nm]])))
    best = max(np.mean([b["reachable"] for b in a["bands"][nm]])
               for nm in a["order"])
    print("    -> at the widest delta, %.2f of the %.2f oracle items are "
          "reachable (%.0f%%)"
          % (best, orc, 100.0 * best / orc if orc > 0 else 0.0))
    print("")


def self_test(n_seeds=5, verbose=False):
    """The two-sided gate: errors AT the cut vs errors buried far below it."""
    rng = np.random.default_rng(0)
    print("SELF-TEST -- can the statistic tell WHERE the errors sit?\n")
    agg, shuf_tot, oracle = {}, {}, {}
    for regime in ("matched", "tailnoise"):
        a = {}
        for seed in range(n_seeds):
            collect(a, probe(make_synthetic(regime, seed), sweep_deltas, rng),
                    SWEEP_NAMES, ("synthetic", regime))
        if verbose:
            print("  REGIME %s" % regime)
            report(a, n_seeds)
        agg[regime] = a
        oracle[regime] = sum(float(np.mean(a[k]["oracle"])) for k in a)
        shuf_tot[regime] = {nm: sum(np.mean([b["reachable"]
                                             for b in a[k]["shuf"][nm]])
                                    for k in a) for nm in SWEEP_NAMES}

    print("  oracle gap: matched %.2f items, tailnoise %.2f items"
          % (oracle["matched"], oracle["tailnoise"]))
    print("  REACHABLE SHARE of that gap, per delta (shuffled reference in ()):")
    print("    %-16s %20s %20s" % ("delta", "matched", "tailnoise"))
    wins, tested = 0, 0
    for nm in SWEEP_NAMES:
        sm = reachable_share(agg["matched"], nm)
        st = reachable_share(agg["tailnoise"], nm)
        # The smallest bands can be 0.00 in BOTH regimes; a tie there is the
        # statistic having no resolution yet, not a failure to discriminate,
        # so it is excluded from the gate rather than counted against it.
        if max(sm, st) > 0:
            tested += 1
            wins += sm > st
        print("    %-16s %12.2f (%5.1f) %12.2f (%5.1f)"
              % (nm, sm, shuf_tot["matched"][nm], st, shuf_tot["tailnoise"][nm]))

    print("")
    print("  the shuffled reference barely moves between regimes -- it depends")
    print("  on n, K and prevalence, not on where the errors are. That is what")
    print("  makes it a reference rather than a second measurement.")
    if tested == 0 or wins * 2 <= tested:
        raise SystemExit(
            "SELF-TEST FAILED: matched beat tailnoise on only %d of %d "
            "resolved deltas. The statistic is not separating errors at the "
            "cut from errors buried far below it, so it is reading the error "
            "RATE and its numbers mean nothing." % (wins, tested))
    print("  -> PASS on %d of %d resolved deltas. Errors at the cut are"
          % (wins, tested))
    print("     reachable, buried ones are not, which is the whole distinction.")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("runs", nargs="*")
    ap.add_argument("--campaign")
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--verbose", action="store_true",
                    help="per-class tables inside --self-test")
    ap.add_argument("--match-contested", action="store_true",
                    help="hold the number of items near the cut FIXED instead "
                         "of the delta -- the only ladder comparable across "
                         "cap levels")
    ap.add_argument("--sweep", action="store_true",
                    help="fractions of the score range instead of the measured "
                         "treated-minus-null displacement")
    ap.add_argument("--limit", type=int, default=16)
    args = ap.parse_args(argv)

    if args.self_test:
        return self_test(verbose=args.verbose)

    runs = list(args.runs)
    if args.campaign:
        runs += [os.path.dirname(p) for p in sorted(glob.glob(
            os.path.join(args.campaign, "**", "final_predictions_raw.csv"),
            recursive=True))]
    if not runs:
        raise SystemExit("no run directories given (or --self-test)")

    print("STRADDLE PROBE -- how much headroom is REACHABLE by a bounded step?")
    print("Everything is in ITEMS. `oracle` is the unbounded gap to a perfect")
    print("allocator; `reachable(d)` is how many FP-out/TP-in swaps a score")
    print("change of size d can perform at the cut.\n")

    rng = np.random.default_rng(0)
    agg, agg_base, n_ok = {}, {}, 0
    pairs = [] if (args.sweep or args.match_contested) else pair_runs(runs)

    if pairs:
        names = MEASURED_NAMES
        print("  DELTA IS MEASURED, from %d treated/null twin pair(s).\n"
              % len(pairs))
        for arm, cell, seed, treated, null in pairs[:args.limit]:
            try:
                data, disp = measured_delta(treated, null)
            except SystemExit as exc:
                print("  skipped %s: %s" % (treated, exc))
                continue
            n_ok += 1
            print("    %-14s %-26s seed %s  |dp| q95 %s"
                  % (arm, "/".join(str(x) for x in cell), seed,
                     " ".join("c%d=%.4f" % (c, v["q"])
                              for c, v in sorted(disp.items()))))
            def ladder(c, _s, _d=disp):
                q = _d[c]["q"]
                return [q, 2 * q, 10 * q]

            # cell + arm for REPORTING: `pair_runs` keys on the cell WITHOUT
            # the arm so a twin can be found, but two treated arms in one
            # campaign must not land in the same row.
            collect(agg, probe(data, ladder, rng), names, cell + (arm,))
            # The NULL at the SAME delta. Without it the probe cannot say how
            # much was reachable from the BASELINE's ranking, which is the
            # reference `headroom.py` quotes -- and the null is a post-hoc
            # clipper at equal compute with the allocator held fixed, so it is
            # the right baseline rather than a convenient one.
            try:
                collect(agg_base, probe(load_real(null, require_features=False),
                                        ladder, rng), names,
                        cell + (arm + "_null",))
            except SystemExit:
                pass
        print("")
    else:
        names = CONTESTED_NAMES if args.match_contested else SWEEP_NAMES
        if not args.sweep and not args.match_contested:
            print("  NO treated/null twin pairs found, so delta cannot be")
            print("  measured and is swept as a fraction of the score range")
            print("  instead. These numbers are NOT calibrated to our step.\n")
        for r in runs[:args.limit]:
            try:
                data = load_real(r, require_features=False)
            except SystemExit as exc:
                print("  skipped %s: %s" % (r, exc))
                continue
            n_ok += 1
            try:
                cell = _meta(r)["cell"]
            except (OSError, ValueError):
                cell = None
            ladder = (matched_deltas_for(data) if args.match_contested
                      else sweep_deltas)
            collect(agg, probe(data, ladder, rng), names, cell)

    if not n_ok:
        raise SystemExit("no probeable runs")
    if agg_base:
        print("  BASELINE -- the `_null` twin's own ranking, at the SAME delta.")
        print("  This is what a post-hoc clipper at equal compute already had")
        print("  within reach BEFORE any constraint was applied.")
        report(agg_base, n_ok)
        print("  TREATED -- the same cells after the constraint.")
    report(agg, n_ok)

    print("  READ THE CONTROL, AND ITS SIGN. `shuffled ctrl` keeps the score")
    print("  DISTRIBUTION and destroys the ORDERING, so it is what reachable")
    print("  would be with no ranking information at the cut. It does NOT go")
    print("  to zero -- it RISES, because a random top-K scatters positives on")
    print("  both sides. reachable << ctrl means the ranking already took the")
    print("  easy swaps; reachable ~= ctrl means this statistic is reading the")
    print("  distribution and means nothing; reachable >> ctrl means positives")
    print("  are parked BELOW the cut, the one case worth training for.")
    print("  A small `reachable` at the MEASURED delta says the constraint as")
    print("  configured cannot collect the oracle gap however it is tuned --")
    print("  a measurement, not a tie, and cheaper than the campaign it saves.")


if __name__ == "__main__":
    main()
