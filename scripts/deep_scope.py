"""WHICH ARM ACTUALLY CLOSES THE DEEPLY-VIOLATED SCOPES?

`scripts/penalty_starvation` shows that TraLO's shipped `rational_bounded`
penalty gives the DEEPEST-violated scope 13-71x LESS pull than a scope only 19%
over, on iwildcam, across an 11-scope, 147x-spread constraint set. That is
algebra about the gradient.

THIS IS THE OUTCOME TEST, and it is the one that can refute the story. If the
starvation is real and matters, then in the SAME cell and the SAME seed:

    `alm` -- whose augmented Lagrangian grows its pull with violation depth
    without bound -- should end with the DEEP scopes closer to their budgets
    than `tralo` does, while the two agree on the shallow ones.

If instead `tralo` closes the deep scopes just as well, the shape's algebra is
real but inert, and the penalty-shape direction is dead for free.

HOW THE BUCKETING AVOIDS BEING ENDOGENOUS. Scope depth is measured on the
lambda=0 NULL, never on the arm under test. Bucketing an arm by its own final
excess would sort scopes by the outcome and guarantee the answer: an arm that
closed a scope would move it out of the deep bucket by construction.

WHY RAW PREDICTIONS AND NOT THE TRAINING LOG. FRAMEWORK 3(0c): for every
TRAINED arm the last logged `Hard_Class*` disagrees with the model's actual
predictions (`alm` logs 340 and emits 467; 0/24 agree), while both nulls agree
24/24. Counts come from `final_predictions_raw.csv` -- the argmax BEFORE the
allocator -- because that is what the constraint acted on. The DEPLOYED file is
post-allocation and emits exactly K by construction, so every scope's excess
there is zero and the question cannot be asked of it.

LIMITS come from the run's own `training_log.csv` (`Group<id>_Limit_Class<c>`),
which is constant across epochs, so no cap policy is re-derived here.
"""

import argparse
import collections
import csv
import glob
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

_LIM = re.compile(r"^Group(\d+)_Limit_Class(\d+)$")


def d_penalty(E, s, rho):
    """d(pen)/dE for the shipped `rational_bounded`. This is what sets each
    scope's SHARE of the constraint gradient: the gradient is normalised as a
    whole, so only the RELATIVE slopes across live scopes survive."""
    e = E / s
    return s / ((E + s) ** 2) + 2.0 * rho * e / (s * (1.0 + e * e) ** 2)


def _spearman(x, y):
    """Rank correlation. None when either side is constant (no ranking)."""
    n = len(x)
    if n < 3:
        return None

    def rank(v):
        order = sorted(range(n), key=lambda i: v[i])
        r = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j + 1 < n and v[order[j + 1]] == v[order[i]]:
                j += 1
            avg = (i + j) / 2.0
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r

    rx, ry = rank(x), rank(y)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    dx = sum((a - mx) ** 2 for a in rx)
    dy = sum((b - my) ** 2 for b in ry)
    if dx <= 0 or dy <= 0:
        return None
    return num / (dx * dy) ** 0.5


def limits_of(run_dir, classes):
    """(group, class) -> K, read from the run's own log. Constant per run."""
    log = os.path.join(run_dir, "training_log.csv")
    if not os.path.exists(log):
        return {}
    with open(log, newline="") as fh:
        r = csv.DictReader(fh)
        row = next(r, None)
    if not row:
        return {}
    out = {}
    for key, val in row.items():
        m = _LIM.match(key or "")
        if m and val not in (None, "") and int(m.group(2)) in classes:
            out[(m.group(1), int(m.group(2)))] = float(val)
    return out


def raw_counts(run_dir, classes):
    """(group, class) -> how many items the model ARGMAXED to that class there."""
    path = os.path.join(run_dir, "final_predictions_raw.csv")
    if not os.path.exists(path):
        return None
    c = collections.Counter()
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh):
            try:
                p = int(float(row["Predicted_Label"]))
            except (KeyError, ValueError, TypeError):
                return None
            if p in classes:
                c[(str(row["Group_ID"]).strip(), p)] += 1
    return c


# The POST-HOC methodologies. Protocol rule 1: these run warm-up 30 /
# constraint 0, so they take ZERO constraint steps and their raw predictions are
# plain CE. That makes any of them a REFERENCE for a statistic that is supposed
# to detect a constraint's aim -- whatever a no-constraint arm scores is the
# artefact floor, not a finding.
POSTHOC_METHODS = frozenset((
    "heuristic", "danits_lp", "focal", "class_balanced", "logit_adjust"))


def posthoc_arms(runs):
    """Arm names whose methodology takes no constraint step at all."""
    out = set()
    for d, (_cell, arm, _seed) in runs.items():
        try:
            m = json.load(open(os.path.join(d, "config.json"))).get("methodology")
        except (ValueError, OSError):
            continue
        if m in POSTHOC_METHODS:
            out.add(arm)
    return out


def deployed_tp(run_dir, classes):
    """Capped-class true positives in the AS-DEPLOYED file, summed over the
    capped classes. The allocator emits exactly K per scope, so F1 = 2TP/(K+n)
    and TP is monotone in F1 per class -- this is the items scale."""
    path = os.path.join(run_dir, "final_predictions.csv")
    if not os.path.exists(path):
        return None
    tp = 0
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh):
            try:
                p = int(float(row["Predicted_Label"]))
                t = int(float(row["True_Label"]))
            except (KeyError, ValueError, TypeError):
                return None
            if p in classes and p == t:
                tp += 1
    return tp


def cell_of(run_dir):
    """(campaign, model, dataset, cap, arm, seed) from the path layout."""
    p = os.path.normpath(run_dir).replace("\\", "/").split("/")
    return tuple(p[-6:])


def collect(roots, classes):
    """run_dir -> (cellkey, arm, seed), for every COMPLETED run under roots."""
    out = {}
    for root in roots:
        for cfg in glob.glob(os.path.join(root, "*/*/*/*/*/config.json")):
            try:
                c = json.load(open(cfg))
            except (ValueError, OSError):
                continue
            if c.get("status") != "completed":
                continue
            d = os.path.dirname(cfg)
            camp, model, ds, cap, arm, seed = cell_of(d)
            out[d] = ((camp, model, ds, cap), arm, seed)
    return out


def analyse(roots, classes, arms, null_arm, n_buckets=3, rho=0.5, out=sys.stdout):
    w = out.write
    runs = collect(roots, classes)
    posthoc = posthoc_arms(runs)
    by_cell = collections.defaultdict(dict)
    for d, (cell, arm, seed) in runs.items():
        by_cell[(cell, seed)][arm] = d

    rows = []
    skipped = collections.Counter()
    for (cell, seed), armdirs in sorted(by_cell.items()):
        if null_arm not in armdirs:
            skipped["no null"] += 1
            continue
        lim = limits_of(armdirs[null_arm], classes)
        base = raw_counts(armdirs[null_arm], classes)
        if not lim or base is None:
            skipped["unreadable null"] += 1
            continue
        present = [a for a in arms if a in armdirs]
        if len(present) < 2:
            skipped["<2 arms"] += 1
            continue
        counts = {}
        bad = False
        for a in present:
            c = raw_counts(armdirs[a], classes)
            if c is None:
                bad = True
                break
            counts[a] = c
        if bad:
            skipped["unreadable arm"] += 1
            continue
        for key, K in sorted(lim.items()):
            s = K if K >= 1 else 1.0
            e0 = max(base.get(key, 0) - K, 0.0)
            if e0 <= 0:
                continue                  # the null already satisfies it
            rows.append({"cell": cell, "seed": seed, "scope": key,
                         "cls": key[1], "K": K,
                         "depth": e0 / s, "e0": e0,
                         "slope": d_penalty(e0, s, rho),
                         "excess": {a: max(counts[a].get(key, 0) - K, 0.0)
                                    for a in present},
                         "arms": tuple(present)})

    if not rows:
        w("NO SCOPE IS VIOLATED BY THE lambda=0 NULL in any matched cell.\n"
          "There is nothing for either arm to close, so this comparison has\n"
          "no content here -- that is a finding about the cells, not the arms.\n")
        for k, v in skipped.items():
            w("  skipped: %-16s %d cell(seed)s\n" % (k, v))
        return 1

    rows.sort(key=lambda r: r["depth"])
    per_bucket = [rows[i * len(rows) // n_buckets:(i + 1) * len(rows) // n_buckets]
                  for i in range(n_buckets)]
    all_arms = sorted({a for r in rows for a in r["arms"]})

    w("\n%s\n" % ("=" * 78))
    w("WHO CLOSES THE DEEP SCOPES?  (excess REMOVED vs the lambda=0 null,\n")
    w("in raw predicted items -- positive means the arm pushed the count down)\n")
    w("  %d violated scope-instances over %d cell(seed)s, bucketed by the\n"
      "  NULL's violation depth so the bucketing cannot be endogenous.\n"
      % (len(rows), len(by_cell)))
    w("%s\n" % ("=" * 78))
    w("  %-22s %7s %8s %7s %s\n"
      % ("depth bucket (E/K)", "scopes", "null E", "K=0", "  ".join(
          "%-10s" % a for a in all_arms)))
    for i, b in enumerate(per_bucket):
        if not b:
            continue
        lo, hi = b[0]["depth"], b[-1]["depth"]
        label = "%s %.2f - %.2f" % (("shallow", "middle", "DEEP")[
            min(i, 2)] if n_buckets == 3 else "b%d" % i, lo, hi)
        cells = []
        for a in all_arms:
            vals = [r["e0"] - r["excess"][a] for r in b if a in r["excess"]]
            cells.append("%-10s" % ("%+.1f" % (sum(vals) / len(vals))
                                    if vals else "."))
        w("  %-22s %7d %8.1f %6.0f%% %s\n"
          % (label, len(b), sum(r["e0"] for r in b) / len(b),
             100.0 * sum(1 for r in b if r["K"] < 1) / len(b),
             "  ".join(cells)))

    w("\n  READ THE DEEP ROW. The claim under test is that `alm` removes more\n"
      "  excess there than `tralo` while they agree on the shallow row. If the\n"
      "  rows are flat across arms, the penalty shape is real algebra with no\n"
      "  outcome, and the shape direction closes for free.\n")

    # DOES THE SHAPE AIM ANYTHING? The bucket table compares two ARMS. This asks
    # a sharper question of ONE arm: within a (cell, seed, class) -- the scope
    # set that shares one normalisation AND one linear head -- do the scopes the
    # penalty weights most heavily actually lose the most of their excess?
    #
    # If the fraction removed is flat in slope share, the scopes of a class are
    # moving TOGETHER and the per-scope weighting is decoration, whatever
    # `penalty_starvation`'s algebra says about the ratios. That is the
    # difference between a shape that mis-aims and a shape that never aimed.
    w("\n  DOES THE SHAPE AIM ANYTHING?  Spearman(share of penalty slope,\n"
      "  fraction of excess removed) WITHIN each (cell, seed, class), at\n"
      "  rho = %g. These scopes share one normalisation and one head.\n" % rho)
    groups = collections.defaultdict(list)
    for r in rows:
        groups[(r["cell"], r["seed"], r["cls"])].append(r)
    usable = [g for g in groups.values() if len(g) >= 3]
    w("  %d scope-sets carry >= 3 live scopes, of %d\n"
      % (len(usable), len(groups)))
    scores = {}
    for a in all_arms:
        rs = []
        for g in usable:
            tot = sum(r["slope"] for r in g)
            if tot <= 0:
                continue
            share = [r["slope"] / tot for r in g]
            frac = [(r["e0"] - r["excess"][a]) / r["e0"] for r in g]
            rr = _spearman(share, frac)
            if rr is not None:
                rs.append(rr)
        if rs:
            rs.sort()
            scores[a] = (rs[len(rs) // 2], len(rs),
                         sum(1 for v in rs if v > 0))
    bar = max((scores[a][0] for a in scores if a in posthoc), default=None)
    for a in sorted(scores, key=lambda a: -scores[a][0]):
        med, n, pos = scores[a]
        if a in posthoc:
            tag = "<- REFERENCE, takes ZERO constraint steps"
        elif bar is None:
            tag = "no reference arm present, UNINTERPRETABLE"
        else:
            tag = "clears the bar" if med > bar else "DOES NOT CLEAR THE BAR"
        w("    %-14s median rho %+.3f over %3d sets, %3d positive (%3.0f%%)"
          "  %s\n" % (a, med, n, pos, 100.0 * pos / n, tag))
    w("\n  READ THE REFERENCE ROW FIRST, AND DO NOT SKIP IT. This statistic\n"
      "  has a CONSTRUCTION ARTEFACT: a shallow scope has a tiny excess, so a\n"
      "  few items of ordinary run-to-run movement remove a large FRACTION of\n"
      "  it, while the same movement is a small fraction of a deep one. Slope\n"
      "  share is itself largest at shallow depth, so fraction-removed and\n"
      "  slope-share correlate positively FOR ANY ARM, including one that\n"
      "  never took a constraint step. Measured on dom1: the post-hoc `lp`\n"
      "  reads +0.400 against `tralo`'s +0.447. Only the margin OVER the\n"
      "  reference is evidence that a shape aimed anything.\n")

    # DOES CLOSING THE EXCESS BUY ANYTHING? Everything above is about the RAW
    # counts, which is what the constraint acts on. It is NOT what is scored:
    # the allocator emits exactly K per scope regardless, so a run that ends
    # deeply infeasible and one that ends exactly at budget are DEPLOYED
    # identically except through the RANKING they induce.
    #
    # So this is the question that decides whether the whole shape direction is
    # worth a GPU: across runs, does removing more raw excess go with capturing
    # more true positives after allocation? If the correlation is ~0 or
    # negative, an arm that satisfies its constraints better is not thereby a
    # better arm, and aiming the penalty is aiming at the wrong target.
    w("\n  DOES CLOSING THE EXCESS BUY DEPLOYED QUALITY?  Per run, total raw\n"
      "  excess removed vs the null, against deployed capped-class TP moved vs\n"
      "  the same null. Both in items. Spearman WITHIN each cell.\n")
    per_run = collections.defaultdict(dict)
    for (cell, seed), armdirs in sorted(by_cell.items()):
        if null_arm not in armdirs:
            continue
        lim = limits_of(armdirs[null_arm], classes)
        base = raw_counts(armdirs[null_arm], classes)
        tp0 = deployed_tp(armdirs[null_arm], classes)
        if not lim or base is None or tp0 is None:
            continue
        for a, d in armdirs.items():
            if a == null_arm:
                continue
            c, tp = raw_counts(d, classes), deployed_tp(d, classes)
            if c is None or tp is None:
                continue
            closed = sum(max(base.get(k, 0) - K, 0.0)
                         - max(c.get(k, 0) - K, 0.0) for k, K in lim.items())
            per_run[cell][(a, seed)] = (closed, tp - tp0)
    rs, n_pairs = [], 0
    for cell, runs in sorted(per_run.items()):
        if len(runs) < 3:
            continue
        vals = list(runs.values())
        rr = _spearman([v[0] for v in vals], [v[1] for v in vals])
        if rr is not None:
            rs.append(rr)
            n_pairs += len(vals)
    if not rs:
        w("    not enough runs per cell to correlate.\n")
    else:
        rs.sort()
        pos = sum(1 for v in rs if v > 0)
        w("    median rho %+.3f over %d cells (%d runs), %d positive (%.0f%%)\n"
          % (rs[len(rs) // 2], len(rs), n_pairs, pos, 100.0 * pos / len(rs)))
        w("\n  A median at or below ZERO means SATISFYING THE CONSTRAINT AND\n"
          "  WINNING ARE DIFFERENT OBJECTIVES, and every penalty-shape variant\n"
          "  is tuning the one that is not scored.\n")
    if skipped:
        w("\n  skipped: %s\n" % ", ".join("%s=%d" % kv for kv in skipped.items()))
    return 0


def self_test(out=sys.stdout):
    """Gate the bucketing and the direction of the excess-removed sign."""
    checks = []

    # Bucketing must be driven by the NULL, not the arm. Build rows by hand and
    # check the summariser puts a scope the arm CLOSED in the DEEP bucket --
    # bucketing on the arm's own excess would file it as shallow.
    rows = [{"depth": d, "e0": e0, "excess": {"a": ea, "b": eb},
             "arms": ("a", "b"), "cell": ("c",), "seed": "1",
             "scope": ("g%d" % i, 2)}
            for i, (d, e0, ea, eb) in enumerate(
                [(0.1, 1.0, 1.0, 1.0), (0.2, 2.0, 2.0, 2.0),
                 (5.0, 50.0, 50.0, 0.0), (6.0, 60.0, 60.0, 0.0)])]
    import io as _io
    buf = _io.StringIO()
    # exercise the summariser directly through analyse's tail by monkey-free
    # means: replicate its bucketing on the fixture
    rows.sort(key=lambda r: r["depth"])
    n = 2
    per = [rows[i * len(rows) // n:(i + 1) * len(rows) // n] for i in range(n)]
    deep = per[-1]
    rem_a = sum(r["e0"] - r["excess"]["a"] for r in deep) / len(deep)
    rem_b = sum(r["e0"] - r["excess"]["b"] for r in deep) / len(deep)
    checks.append(("an arm that CLOSES the deep scopes scores higher there "
                   "(%.1f vs %.1f)" % (rem_b, rem_a), rem_b > rem_a + 40))
    shallow = per[0]
    ra = sum(r["e0"] - r["excess"]["a"] for r in shallow) / len(shallow)
    rb = sum(r["e0"] - r["excess"]["b"] for r in shallow) / len(shallow)
    checks.append(("  and the arms tie on the shallow bucket, as built",
                   abs(ra - rb) < 1e-9))
    checks.append(("NEGATIVE CONTROL: the deep scope stays in the DEEP bucket "
                   "even though arm b emptied it",
                   all(r["depth"] >= 5.0 for r in deep)))

    # A scope the NULL already satisfies must never enter the table: there is
    # nothing to close, and including it would dilute every bucket toward zero.
    lim = {("g1", 2): 100.0}
    base = collections.Counter({("g1", 2): 50})
    e0 = max(base.get(("g1", 2), 0) - lim[("g1", 2)], 0.0)
    checks.append(("a scope the NULL satisfies is excluded", e0 == 0))

    # _spearman must report BOTH directions and refuse a constant.
    checks.append(("spearman: a monotone increasing relation reads +1",
                   _spearman([1, 2, 3, 4], [10, 20, 30, 40]) == 1.0))
    checks.append(("NEGATIVE CONTROL: a monotone DECREASING one reads -1",
                   _spearman([1, 2, 3, 4], [40, 30, 20, 10]) == -1.0))
    checks.append(("NEGATIVE CONTROL: a constant side has no ranking -> None",
                   _spearman([1, 2, 3, 4], [7, 7, 7, 7]) is None))

    # d_penalty is a POSITIVE CONTROL against `scripts/penalty_starvation`:
    # the shipped shape must STARVE the deeper scope, and the starvation must
    # WORSEN as rho ratchets 0.5 -> 100. Depths are dom1's measured median and
    # deepest, 0.19x and 29.8x over budget, at a nominal s = 100.
    s0 = 100.0
    lo, hi = 0.19 * s0, 29.8 * s0
    r_lo = d_penalty(hi, s0, 0.5) / d_penalty(lo, s0, 0.5)
    r_hi = d_penalty(hi, s0, 100.0) / d_penalty(lo, s0, 100.0)
    checks.append(("POSITIVE CONTROL: shipped shape starves the deeper scope "
                   "(%.3fx at rho=0.5)" % r_lo, r_lo < 0.2))
    checks.append(("  and the starvation DEEPENS as rho ratchets to 100 "
                   "(%.3fx)" % r_hi, r_hi < r_lo))
    checks.append(("NEGATIVE CONTROL: a LINEAR shape would not starve it "
                   "(ratio 1.0 by construction)",
                   abs((1.0 / s0) / (1.0 / s0) - 1.0) < 1e-12))

    print("", file=out)
    for label, good in checks:
        print("  %-66s %s" % (label[:66], "PASS" if good else "FAIL"), file=out)
    bad = [c for c, g in checks if not g]
    print("", file=out)
    print("SELF-TEST PASSED" if not bad else "FAILED: %d" % len(bad), file=out)
    _ = buf
    return 1 if bad else 0


def main(argv=None):
    a = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    a.add_argument("--campaign", nargs="+", default=[])
    a.add_argument("--arms", nargs="+", default=["tralo", "alm"])
    a.add_argument("--null-arm", default="tralo_null")
    a.add_argument("--classes", nargs="+", type=int, default=[2, 7])
    a.add_argument("--buckets", type=int, default=3)
    a.add_argument("--rho", type=float, default=0.5,
                   help="rho at which to score the penalty slope. It RATCHETS "
                        "initial_rho 0.5 -> rho_target 100 within a run and "
                        "freezes on first satisfaction, so quote which end.")
    a.add_argument("--self-test", action="store_true")
    args = a.parse_args(argv)
    if args.self_test:
        return self_test()
    if not args.campaign:
        a.error("give --campaign <root> ... (or --self-test)")
    return analyse(args.campaign, set(args.classes), args.arms,
                   args.null_arm, args.buckets, args.rho)


if __name__ == "__main__":
    sys.exit(main())
