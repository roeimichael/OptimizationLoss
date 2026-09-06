"""DOES THE PENALTY SHAPE STARVE THE WORST-VIOLATED SCOPE, ON iwildcam?

TraLO's shipped penalty is `rational_bounded` -- the manuscript's Eq. 4:

    pen(E) = E/(E+s) + rho * e^2/(1+e^2),   e = E/s,   s = max(K, 1)

Both terms are BOUNDED, so `d(pen)/dE` is NON-MONOTONE in the violation: about
`1/s` at the boundary, peaking a little above it, and decaying to zero for
anything deeply violated. `src/losses/transductive_loss.py` states the
consequence and FRAMEWORK 2(a2) reproduces it to four decimals: a scope violated
by 8x its budget receives ~167x LESS pull than one violated by 58%.

WITH ONE TERM THAT IS HARMLESS. The constraint gradient is clipped (and under
`constraint_grad_mode: normalize`, rescaled to exactly `clip`) as a whole, so a
single scope's shape is a scalar times a fixed direction and divides out.

WITH SEVERAL TERMS IT SETS THEIR RELATIVE WEIGHTS, AND IT SETS THEM BACKWARDS.
The deepest violator is the one the shape starves. iwildcam runs 1 global + 14
local terms with SEVEN zero-K local ceilings, so it is the many-term case by a
wide margin.

WHY THIS IS WORTH MEASURING AGAIN. The starvation was demonstrated on
**dermmnist** (classes 2+4, L30_G20, 2026-08-20), which is REMOVED, leaks 38.7%
of its test set, and whose LOCAL scope was empty -- `lp_fallback_used` was False
with 0 candidates on all 52 runs. So the one dataset where the effect was shown
is the one where the many-term case barely existed. The comment itself notes
single-class runs never showed it because their spread across scopes is ~1.5x
against dermmnist's ~30x. iwildcam's spread has never been measured.

AND IT IS THE CLEAREST STRUCTURAL DIFFERENCE FROM `alm`. An augmented Lagrangian
grows its pull with violation depth without bound; this shape shrinks it. If
iwildcam's scopes are violated heterogeneously, TraLO is systematically
down-weighting exactly the constraints that are furthest from satisfied, and
`alm` is not -- which is a mechanism, not a tuning difference.

`linear` (e) and `squared` (e^2) are already implemented in the same function
and give constant and growing pull respectively. Neither has ever run on
iwildcam.

WHAT THIS SCRIPT MEASURES, from a real run's `training_log.csv` -- no GPU, no
model, no re-run:

  spread     the ratio of the deepest to the median violation depth across the
             live scopes in an epoch. If this is ~1 the shape cannot matter,
             whatever its algebra, and the direction is closed for free.
  starved    under each shape, the pull the DEEPEST violator receives divided
             by the pull the MEDIAN violator receives. Below 1 means the
             deepest scope is being pulled LESS than a milder one.

READ `spread` FIRST. A shape argument on a dataset whose scopes are all violated
by the same factor is arithmetic about nothing.
"""

import argparse
import csv
import glob
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

EPSILON = 1e-8

# Group columns look like `Group130_Soft_Class2` / `Group130_Limit_Class2`.
_GRP = re.compile(r"^Group(\d+)_(Soft|Limit)_Class(\d+)$")


def d_penalty(E, s, rho, shape):
    """d(pen)/dE for each implemented shape. Derivatives of the SAME formulas
    in `src/losses/transductive_loss.py::_penalty`, not a re-derivation of the
    penalty: what sets a term's relative weight is its slope, not its value."""
    e = E / (s + EPSILON)
    if shape == "linear":
        return 1.0 / (s + EPSILON)
    if shape == "squared":
        return 2.0 * e / (s + EPSILON)
    # rational_bounded: d/dE[E/(E+s)] + rho * d/dE[e^2/(1+e^2)]
    return (s / ((E + s + EPSILON) ** 2)
            + 2.0 * rho * e / ((s + EPSILON) * (1.0 + e * e + EPSILON) ** 2))


def scopes_of_row(row, classes):
    """(label, soft, K) for every constraint term live in this epoch.

    GLOBAL and LOCAL both, because the shape sets weights ACROSS them and a
    per-class global term is one of the fifteen.
    """
    out = []
    for c in classes:
        k = row.get("Limit_Class%d" % c)
        s = row.get("Soft_Class%d" % c)
        if k not in (None, "") and s not in (None, ""):
            out.append(("global_c%d" % c, float(s), float(k)))
    seen = {}
    for key, val in row.items():
        m = _GRP.match(key or "")
        if not m or val in (None, ""):
            continue
        gid, kind, cls = m.group(1), m.group(2), int(m.group(3))
        if cls not in classes:
            continue
        seen.setdefault((gid, cls), {})[kind] = float(val)
    for (gid, cls), d in sorted(seen.items()):
        if "Soft" in d and "Limit" in d:
            out.append(("g%s_c%d" % (gid, cls), d["Soft"], d["Limit"]))
    return out


def analyse_row(row, classes, rho):
    """spread and per-shape starvation for ONE epoch, or None if nothing binds."""
    live = []
    for label, soft, K in scopes_of_row(row, classes):
        s = K if K >= 1 else 1.0
        E = max(soft - K, 0.0)
        if E <= 0:
            continue                      # satisfied: contributes no gradient
        live.append((label, E, s, E / s))
    if len(live) < 2:
        return None
    live.sort(key=lambda t: t[3])
    deep = live[-1]
    med = live[len(live) // 2]
    out = {"n_live": len(live), "deepest": deep[0],
           "depth_deep": deep[3], "depth_med": med[3],
           "spread": deep[3] / (med[3] + EPSILON)}
    for shape in ("rational_bounded", "linear", "squared"):
        pd_ = d_penalty(deep[1], deep[2], rho, shape)
        pm = d_penalty(med[1], med[2], rho, shape)
        out[shape] = pd_ / (pm + EPSILON)
    return out


def run(paths, classes, rho, out=sys.stdout):
    w = out.write
    rows = []
    for p in paths:
        with open(p, newline="") as fh:
            for row in csv.DictReader(fh):
                r = analyse_row(row, classes, rho)
                if r:
                    r["run"] = p
                    rows.append(r)
    if not rows:
        w("NO EPOCH HAS TWO VIOLATED SCOPES. The shape sets relative weights\n"
          "between terms; with fewer than two live terms it divides out, so\n"
          "there is nothing here for it to get wrong.\n")
        return 1

    def med(key):
        v = sorted(r[key] for r in rows)
        return v[len(v) // 2]

    w("\n%s\n" % ("=" * 76))
    w("PENALTY SHAPE -- is the deepest violator being starved?\n")
    w("  %d epoch(s) with >= 2 violated scopes, over %d run(s), rho=%.3g\n"
      % (len(rows), len(set(r["run"] for r in rows)), rho))
    w("%s\n" % ("=" * 76))
    w("  median live scopes per epoch   %.0f\n" % med("n_live"))
    w("  median deepest depth  E/s      %.3g\n" % med("depth_deep"))
    w("  median median  depth  E/s      %.3g\n" % med("depth_med"))
    w("  median SPREAD (deep/median)    %.3gx\n" % med("spread"))
    w("\n  pull(deepest) / pull(median), by shape:\n")
    for shape in ("rational_bounded", "linear", "squared"):
        m = med(shape)
        flag = "  <-- STARVED" if m < 1.0 else ""
        w("    %-18s %12.4g%s\n" % (shape, m, flag))

    sp = med("spread")
    rb = med("rational_bounded")
    w("\n")
    if sp < 1.5:
        w("  VERDICT: SPREAD IS %.2gx. The scopes are violated to nearly the\n"
          "           same depth, so the shape has almost no relative weight to\n"
          "           set. This direction is CLOSED without a GPU.\n" % sp)
    elif rb < 1.0:
        w("  VERDICT: LIVE AND BACKWARDS. Scopes differ %.3gx in depth and the\n"
          "           shipped shape pulls the DEEPEST one %.3gx as hard as a\n"
          "           median one -- it is down-weighting the constraints\n"
          "           furthest from satisfied. `linear` (%.3g) and `squared`\n"
          "           (%.3g) do not. Worth an arm.\n"
          % (sp, rb, med("linear"), med("squared")))
    else:
        w("  VERDICT: spread is %.3gx but the shipped shape still pulls the\n"
          "           deepest scope %.3gx as hard as the median, so it is not\n"
          "           starving anything here. The algebra is real; this regime\n"
          "           does not reach the decaying branch.\n" % (sp, rb))
    return 0


def self_test(out=sys.stdout):
    """Gate the derivative against FRAMEWORK 2(a2)'s published figure."""
    checks = []
    rho = 3.93

    # POSITIVE CONTROL against FRAMEWORK 2(a2)'s autograd table, at BOTH of its
    # rho rows. One row could be matched by a wrong formula with a compensating
    # error; two rows an octave apart could not.
    #
    # RHO IS THE WHOLE ANSWER HERE and it is easy to get wrong: `initial_rho` is
    # 0.5 and `rho_target` is 100, so the starvation ratio moves 51x -> 167x
    # ACROSS A SINGLE RUN. Evaluating at rho=1 gives 65x and looks like a failed
    # reproduction of a 167x claim; it is the right formula at the wrong rho.
    s = 100.0

    def ratio_at(r):
        return (d_penalty(0.58 * s, s, r, "rational_bounded")
                / d_penalty(8.0 * s, s, r, "rational_bounded"))

    r393, r100 = ratio_at(3.93), ratio_at(100.0)
    checks.append(("FRAMEWORK 2(a2) rho=3.93 row: 109x (got %.0fx)" % r393,
                   105 < r393 < 113))
    checks.append(("FRAMEWORK 2(a2) rho=100 row: 167x (got %.0fx)" % r100,
                   162 < r100 < 172))

    # The shape must be NON-MONOTONE: pull rises then falls.
    pulls = [d_penalty(f * s, s, rho, "rational_bounded")
             for f in (0.1, 0.58, 2.0, 8.0)]
    checks.append(("  and the pull is non-monotone: rises then decays",
                   pulls[1] > pulls[0] and pulls[1] > pulls[2] > pulls[3]))

    # NEGATIVE CONTROL: linear is constant and squared GROWS, so neither can
    # starve. A probe that reported starvation for these is computing the wrong
    # derivative.
    lin = [d_penalty(f * s, s, rho, "linear") for f in (0.58, 8.0)]
    sq = [d_penalty(f * s, s, rho, "squared") for f in (0.58, 8.0)]
    checks.append(("NEGATIVE CONTROL: `linear` pull is constant in depth",
                   abs(lin[0] - lin[1]) < 1e-12))
    checks.append(("NEGATIVE CONTROL: `squared` pull GROWS with depth",
                   sq[1] > sq[0] * 10))

    # A satisfied scope contributes nothing, so it must never enter the ranking.
    row = {"Limit_Class2": "100", "Soft_Class2": "50",     # satisfied
           "Limit_Class7": "100", "Soft_Class7": "158",    # 58% over
           "Group1_Limit_Class2": "10", "Group1_Soft_Class2": "90"}  # 8x over
    r = analyse_row(row, (2, 7), rho)
    checks.append(("a SATISFIED scope is excluded from the ranking",
                   r is not None and r["n_live"] == 2))
    checks.append(("  and the deepest is correctly identified",
                   r["deepest"] == "g1_c2"))

    # With one live scope the shape divides out and there is nothing to report.
    r1 = analyse_row({"Limit_Class2": "100", "Soft_Class2": "158"}, (2, 7), rho)
    checks.append(("NEGATIVE CONTROL: one live scope reports nothing",
                   r1 is None))

    print("", file=out)
    for label, good in checks:
        print("  %-68s %s" % (label[:68], "PASS" if good else "FAIL"), file=out)
    bad = [c for c, g in checks if not g]
    print("", file=out)
    print("SELF-TEST PASSED" if not bad else "FAILED: %d" % len(bad), file=out)
    return 1 if bad else 0


def main(argv=None):
    a = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    a.add_argument("--glob", nargs="+", default=[],
                   help="run directories holding training_log.csv")
    a.add_argument("--rho", type=float, nargs="+", default=None,
                   help="rho value(s) to evaluate at. Default: the run's own "
                        "initial_rho AND rho_target, because rho ratchets "
                        "between them and the starvation ratio moves with it.")
    a.add_argument("--self-test", action="store_true")
    args = a.parse_args(argv)
    if args.self_test:
        return self_test()
    if not args.glob:
        a.error("give --glob '<runs>/seed_*' (or --self-test)")

    paths, classes, rhos = [], set(), args.rho
    for pat in args.glob:
        for d in sorted(glob.glob(pat)):
            log = os.path.join(d, "training_log.csv")
            cfg = os.path.join(d, "config.json")
            if not os.path.exists(log) or not os.path.exists(cfg):
                continue
            c = json.load(open(cfg))
            classes.update(c["dataset_config"]["constrained_class"])
            hp = c["hyperparams"]
            # RHO RATCHETS DURING THE RUN: `initial_rho` 0.5 -> `rho_target`
            # 100, and the starvation ratio moves 51x -> 167x with it. It is
            # not logged per epoch, so reporting ONE value would be a choice
            # dressed as a measurement. Report both ends and let the reader see
            # the range. The key is `initial_rho`, NOT `rho` -- defaulting to
            # `rho` reads 1.0 and understates the effect by ~3x.
            if rhos is None:
                rhos = sorted({float(hp.get("initial_rho", 0.5)),
                               float(hp.get("rho_target", 100.0))})
            paths.append(log)
    if not paths:
        print("no run with a training_log.csv matched", file=sys.stderr)
        return 1
    rc = 0
    for r in rhos:
        rc |= run(paths, tuple(sorted(classes)), r)
    return rc


if __name__ == "__main__":
    sys.exit(main())
