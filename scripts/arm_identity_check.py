"""Are two arms the SAME METHOD? Test it on the deployed predictions.

WHY. `dual_cone_probe` shows that at any fixed model state `fioretto_alm` and
`fioretto_ldf` build weight vectors that are both proportional to
`relu(S_j - K_j)`, so their constraint-gradient DIRECTIONS are identical
(cos > 0.999999 in 192 of 192 measured states). Under
`constraint_grad_mode: normalize` the magnitude is discarded outright, and the
magnitude is the only thing their dual rules disagree about.

That is a PREDICTION about the deployed output, and this script is the test:
if the two arms are one rule, `|alm - fioretto|` must sit at the same scale as
the pure-RNG floor `|tralo_null - tralo_reseed|`, not at a method scale. If it
sits far above the floor, the algebra is missing a channel and the prediction
is wrong -- which is the outcome that would matter most.

WHAT IS MEASURED. Captured true positives per capped class, from
`final_predictions.csv` -- as DEPLOYED, not the panel's re-derived allocation.
`full_panel` is allocator-blind by construction, so it cannot answer this.

Everything is paired within (cap, seed). Nothing is pooled across cap levels.
"""
import argparse
import glob
import itertools
import os
import statistics as st
import sys

import pandas as pd

from scripts import quarantine


def captured(run_dir, classes):
    """{class: true positives among the emitted predictions}, as deployed."""
    p = os.path.join(run_dir, "final_predictions.csv")
    if not os.path.exists(p):
        return None
    d = pd.read_csv(p)
    y, yh = d["True_Label"].to_numpy(), d["Predicted_Label"].to_numpy()
    return {c: int(((yh == c) & (y == c)).sum()) for c in classes}


def collect(root, classes, dead=()):
    """{(model, cap, seed): {arm: {class: tp}}}

    `dead` is the arm set a PARTIAL quarantine marker disqualifies, and it is
    a PARAMETER rather than a global because the gate runs in `main` while the
    enumeration runs here. The FOR-SCALE block at the bottom of `main` walks
    whatever arms this returns and prints every trained-arm contrast it finds,
    so a dead arm reaching this dict is a disqualified contrast printed beside
    the live ones with nothing to distinguish them.
    """
    out = {}
    for cfg in quarantine.drop_dead_runs(
            glob.glob(os.path.join(root, "*", "*", "*", "*", "seed_*")),
            dead, label="run"):
        parts = cfg.replace("\\", "/").split("/")
        model, cap, arm, seed = parts[-5], parts[-3], parts[-2], parts[-1]
        tp = captured(cfg, classes)
        if tp is not None:
            out.setdefault((model, cap, seed), {})[arm] = tp
    return out


def gaps(cells, a, b, classes):
    """|a - b| per (cell, class), paired within the cell. Absolute items."""
    g = []
    for key, arms in sorted(cells.items()):
        if a in arms and b in arms:
            for c in classes:
                g.append(abs(arms[a][c] - arms[b][c]))
    return g


def line(w, label, g):
    if not g:
        w("  %-34s  no paired cells\n" % label)
        return None
    s = sorted(g)
    w("  %-34s n=%3d   median %6.1f   mean %6.1f   max %5d\n"
      % (label, len(s), st.median(s), sum(s) / len(s), s[-1]))
    return st.median(s)


def self_test(w=sys.stdout.write):
    """The comparison must be able to say BOTH 'identical' and 'different'."""
    ok = True

    def check(good, label):
        nonlocal ok
        w("  %-4s %s\n" % ("PASS" if good else "FAIL", label))
        ok = ok and good

    cells = {
        ("M", "L90", "seed_1"): {"x": {2: 100, 7: 50}, "y": {2: 100, 7: 50},
                                 "z": {2: 130, 7: 20}},
        ("M", "L90", "seed_2"): {"x": {2: 110, 7: 55}, "y": {2: 111, 7: 54},
                                 "z": {2: 80, 7: 90}},
    }
    same = gaps(cells, "x", "y", [2, 7])
    diff = gaps(cells, "x", "z", [2, 7])
    check(st.median(same) <= 1.0, "identical arms read as identical (median %.1f)"
          % st.median(same))
    check(st.median(diff) >= 20.0,
          "LIVENESS: genuinely different arms read as different (median %.1f)"
          % st.median(diff))
    check(len(gaps(cells, "x", "absent", [2, 7])) == 0,
          "a missing arm yields no pairs rather than a silent zero")
    w("\nSELF-TEST %s\n" % ("PASSED" if ok else "FAILED"))
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root")
    ap.add_argument("--classes", type=int, nargs="+", default=[2, 7])
    ap.add_argument("--pairs", nargs="+", default=["alm:fioretto"],
                    help="armA:armB, the identity hypotheses to test")
    ap.add_argument("--floor", default="tralo_null:tralo_reseed",
                    help="the RNG-only reference contrast")
    ap.add_argument("--allow-quarantined", action="store_true",
                    help="compare arms in a campaign `scripts.quarantine` "
                         "marked dead")
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()

    if a.self_test:
        return self_test()
    if not a.root:
        ap.error("--root is required (or use --self-test)")

    # 🛑 THE QUARANTINE GATE. Audited 2026-09-04: this tool had NONE, and its
    # `--pairs` DEFAULTS to `alm:fioretto` -- `fioretto` is a dead arm in
    # `dom1`, `dom1b` and `equaldose1`, so the bare invocation tested an
    # identity hypothesis on an arm that ran at a different constraint dose.
    # No fallback import -- if the gate cannot load, the tool must break.
    from scripts.quarantine import gate
    blocked, dead = gate([a.root], a.allow_quarantined, "compare")
    if blocked:
        return 1

    # TWO enforcement shapes, because this tool has two ways to name an arm.
    #   * `--pairs` and `--floor` are NAMED on the command line, so the answer
    #     is a REFUSAL: filtering them would leave the tool printing "no
    #     paired cells" for a hypothesis it was explicitly asked to test, and
    #     an absent line reads as absence of evidence.
    #   * the FOR-SCALE block enumerates whatever arms are on disk, so that
    #     half is FILTERED in `collect` -- otherwise a dead arm's contrast
    #     prints there however live the pair on the command line was.
    # PER CAMPAIGN, never a union: a union would disqualify an arm here
    # because a DIFFERENT campaign marked it.
    here = dead.for_path(a.root) if hasattr(dead, "for_path") else dead
    named = [x for spec in list(a.pairs) + [a.floor] for x in spec.split(":")]
    bad = sorted(set(named) & set(here))
    if bad:
        print("REFUSING: %s is a DEAD arm of campaign `%s` (partial "
              "quarantine) -- it ran at a different constraint dose, so "
              "|arm - arm| against it is not the identity test this tool "
              "claims, and as a --floor it would misprice every ratio. "
              "--allow-quarantined governs the campaign marker, not this. "
              "Name live arms instead."
              % (", ".join(bad), quarantine.campaign_name(a.root)))
        return 1

    cells = collect(a.root, a.classes, here)
    if not cells:
        raise SystemExit("no runs with final_predictions.csv under %s" % a.root)
    w = sys.stdout.write
    w("captured true positives per capped class, AS DEPLOYED, paired within "
      "(model, cap, seed)\n")
    w("root: %s   cells: %d\n\n" % (a.root, len(cells)))

    fa, fb = a.floor.split(":")
    floor = line(w, "FLOOR  |%s - %s|" % (fa, fb), gaps(cells, fa, fb, a.classes))
    w("\n")
    for pr in a.pairs:
        x, y = pr.split(":")
        m = line(w, "TEST   |%s - %s|" % (x, y), gaps(cells, x, y, a.classes))
        if m is not None and floor:
            w("  %-34s  ratio to floor: %.2fx\n" % ("", m / floor if floor else float("nan")))
    w("\n")
    # Every other arm pair, for scale: what does a REAL method difference read?
    arms = sorted({k for v in cells.values() for k in v})
    ref = [p for p in itertools.combinations(arms, 2)
           if set(p) <= {"tralo", "alm", "fioretto", "hounie", "clip"}]
    w("for scale, the other trained-arm contrasts:\n")
    for x, y in ref:
        line(w, "       |%s - %s|" % (x, y), gaps(cells, x, y, a.classes))
    return 0


if __name__ == "__main__":
    sys.exit(main())
