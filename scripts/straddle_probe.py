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

WHERE `delta` COMES FROM. Assuming one is the whole difficulty, so the default
mode MEASURES it: given a treated run and its `_null` twin at the same seed --
same warm-up, same allocator, same RNG, lambda=0 -- the per-item difference in
the capped-class score IS the displacement the constraint delivered. `reachable`
at that measured delta is the ceiling for the constraint AS CONFIGURED, and the
campaign's own effect size can be read against it. `--sweep` falls back to
fractions of the score range when no twin exists, and says so.

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


# ------------------------------------------------- the measured displacement --

def _meta(run_dir):
    with open(os.path.join(run_dir, "config.json"), encoding="utf-8") as fh:
        cfg = json.load(fh)
    return {"arm": cfg.get("arm"),
            "cell": (cfg.get("dataset_mode"), cfg.get("model_name"),
                     cfg.get("constraint_tag")),
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
        by_key[(m["cell"], m["seed"], m["arm"])] = r
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
    a, b = load_real(treated), load_real(null)
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

def collect(agg, rows, names):
    for c, v in rows.items():
        a = agg.setdefault(c, {"oracle": [], "K": [], "n_pos": [],
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
    num = sum(np.mean([b["reachable"] for b in agg[c]["bands"][band]])
              for c in agg if band in agg[c]["bands"])
    den = sum(float(np.mean(agg[c]["oracle"])) for c in agg)
    return num / den if den > 0 else 0.0


def report(agg, n_runs):
    for c in sorted(agg):
        a = agg[c]
        orc = float(np.mean(a["oracle"]))
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
                    SWEEP_NAMES)
        if verbose:
            print("  REGIME %s" % regime)
            report(a, n_seeds)
        agg[regime] = a
        oracle[regime] = sum(float(np.mean(a[c]["oracle"])) for c in a)
        shuf_tot[regime] = {nm: sum(np.mean([b["reachable"]
                                             for b in a[c]["shuf"][nm]])
                                    for c in a) for nm in SWEEP_NAMES}

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
    agg, n_ok = {}, 0
    pairs = [] if args.sweep else pair_runs(runs)

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
            collect(agg, probe(data, lambda c, s, _d=disp:
                               [_d[c]["q"], 2 * _d[c]["q"], 10 * _d[c]["q"]],
                               rng), names)
        print("")
    else:
        names = SWEEP_NAMES
        if not args.sweep:
            print("  NO treated/null twin pairs found, so delta cannot be")
            print("  measured and is swept as a fraction of the score range")
            print("  instead. These numbers are NOT calibrated to our step.\n")
        for r in runs[:args.limit]:
            try:
                data = load_real(r)
            except SystemExit as exc:
                print("  skipped %s: %s" % (r, exc))
                continue
            n_ok += 1
            collect(agg, probe(data, sweep_deltas, rng), names)

    if not n_ok:
        raise SystemExit("no probeable runs")
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
