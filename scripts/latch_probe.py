"""DOES TraLO STOP ADAPTING PART-WAY THROUGH ITS OWN CONSTRAINT PHASE?

`src/methodologies/tralo/train.py` gates BOTH of TraLO's adaptive channels on a
single latch:

    is_satisfied  = global_satisfied and local_satisfied      # ALL scopes, AND
    ratchet_gate  = satisfaction_epoch is None                # line 540
    if is_satisfied and satisfaction_epoch is None:
        satisfaction_epoch = epoch + 1                        # never reset
        rho_frozen = True

`satisfaction_epoch` is set once and NEVER cleared, and `ratchet_gate` is not
per-scope. So the FIRST epoch in which every scope happens to be satisfied
simultaneously permanently disables, for the whole rest of the run and for every
scope:

  * the lambda ratchet  (`lambda += lambda_step`), and
  * the rho ramp        (`initial_rho` 0.5 -> `rho_target` 100).

After that epoch the only thing still responding to a violation is `pen'(E)`,
which `scripts/penalty_starvation` measured as bounded and non-monotone. This is
a LATCH, not a controller, and it is the one structural feature none of the
three rival duals has: LDF accumulates `step_size * viol` forever, ALM
accumulates `eta * r` and adds an instantaneous `mu*(r)^+` on top, and Hounie's
`lam` keeps ascending against a learned slack. All three scale their update by
the violation; TraLO's is a constant increment behind an on/off gate.

WHETHER THAT COSTS ANYTHING IS AN EMPIRICAL QUESTION AND THIS ANSWERS IT, from
runs already on disk. Two numbers decide it:

  1. WHAT FRACTION OF RUNS LATCH AT ALL, and at which epoch. On iwildcam 7 of
     14 per-group ceilings are K=0, and satisfaction is a global AND over every
     scope, so the latch may simply never fire -- in which case this whole
     direction is closed for free and must be reported as closed.
  2. HOW MUCH VIOLATION HAPPENS AFTER IT. A latch that fires at epoch 28 of 29
     costs nothing. A latch that fires at epoch 3 and is followed by hundreds of
     violated scope-epochs, with lambda and rho both frozen, is the mechanism.

⛔ THE EXISTING `no_freeze` NULL DOES NOT SETTLE THIS. FRAMEWORK's leave-one-out
ablation reads `no_freeze` at +0.13 pp with a budget ratio of 1.002 -- a clean
null -- but it was measured on `dermmnist + octmnist` at `L30_G30`/`L40_G40`.
That carries two of 2(z44)'s six contaminants, and one of them is fatal here:
octmnist's `synth_group` is `index % 3`, so its LOCAL scope is empty by
construction, and dermmnist ran `lp_fallback_used=False` with 0 LP candidates on
all 52 runs. The latch is a local-scope mechanism and it was ablated where the
local scope barely existed. The caps were also outside every measured task
window. Re-measure on iwildcam or do not quote it.
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

_SOFT = re.compile(r"^Group(\d+)_(Hard|Limit)_Class(\d+)$")


def _truthy(v):
    """The log writes these booleans as True/False/1/0 depending on writer."""
    s = str(v).strip().lower()
    return s in ("true", "1", "1.0", "yes")


def read_log(run_dir, classes):
    """Per epoch: (all-satisfied flag, {scope: excess}) from `training_log.csv`.

    `satisfaction_epoch` is NOT persisted to disk -- `runner.py` puts it in
    `best_metrics` and only a subset of that reaches `config.json`. But the log
    writes `Global_Satisfied` and `Local_Satisfied`, which are the exact two
    booleans the latch ANDs, so the latch epoch is reconstructible rather than
    approximated.
    """
    path = os.path.join(run_dir, "training_log.csv")
    if not os.path.exists(path):
        return None
    out = []
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh):
            if "Global_Satisfied" not in row or "Local_Satisfied" not in row:
                return None
            hard, lim = {}, {}
            for k, v in row.items():
                m = _SOFT.match(k or "")
                if not m or v in (None, ""):
                    continue
                c = int(m.group(3))
                if c not in classes:
                    continue
                (hard if m.group(2) == "Hard" else lim)[(m.group(1), c)] = float(v)
            excess = {key: max(hard.get(key, 0.0) - K, 0.0)
                      for key, K in lim.items()}
            sat = _truthy(row["Global_Satisfied"]) and _truthy(row["Local_Satisfied"])
            out.append((sat, excess, lim))
    return out or None


def latch_epoch(log):
    """First epoch index (0-based) at which every scope was satisfied, or None.

    Mirrors `tralo/train.py`: the latch is set on the FIRST such epoch and is
    never cleared, so everything after it runs with lambda and rho frozen.
    """
    for i, (sat, _e, _l) in enumerate(log):
        if sat:
            return i
    return None


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
            for k in range(i, j + 1):
                r[order[k]] = (i + j) / 2.0
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


def weight_rankings(log, latch, lam0, step):
    """TraLO's per-scope multiplier against the one ALM/LDF would have built.

    TraLO ratchets a CONSTANT while a scope is violated, and the gate closes at
    the latch, so its multiplier is exactly

        lam_c = lam0 + step * (epochs scope c was violated, before the latch)

    -- a FREQUENCY counter. LDF/ALM accumulate `eta * violation`, i.e. the
    cumulative violation MAGNITUDE. Under `constraint_grad_mode: normalize` the
    whole summed gradient is scaled by ONE norm over `model.parameters()`, so
    the overall size of either multiplier divides out and only the RATIOS
    ACROSS SCOPES survive. These two rankings ARE the two search directions.

    Returns (lam_by_scope, mag_by_scope). If they rank scopes the same way, the
    frequency-vs-magnitude distinction is cosmetic and the direction closes.
    """
    end = len(log) if latch is None else latch
    freq, mag = collections.Counter(), collections.Counter()
    for sat, excess, _lim in log[:end]:
        for key, e in excess.items():
            if e > 0:
                freq[key] += 1
                mag[key] += e
    lam = {k: lam0 + step * n for k, n in freq.items()}
    return lam, dict(mag)


def analyse(roots, classes, arms, lam0=0.01, step=0.05, out=sys.stdout):
    w = out.write
    rows = []
    unreadable = 0
    for root in roots:
        for cfg in sorted(glob.glob(os.path.join(root, "*/*/*/*/*/config.json"))):
            try:
                c = json.load(open(cfg))
            except (ValueError, OSError):
                continue
            if c.get("status") != "completed":
                continue
            d = os.path.dirname(cfg)
            arm = os.path.normpath(d).replace("\\", "/").split("/")[-2]
            if arms and arm not in arms:
                continue
            log = read_log(d, classes)
            if log is None:
                unreadable += 1
                continue
            hp = c.get("hyperparams") or {}
            lat = latch_epoch(log)
            lam, mag = weight_rankings(
                log, lat, float(hp.get("lambda_local", lam0)),
                float(hp.get("lambda_step", step)))
            rows.append({"arm": arm, "dir": d, "latch": lat, "log": log,
                         "lam": lam, "mag": mag,
                         "trace": [sum(1 for e in ex.values() if e > 0)
                                   for _s, ex, _l in log]})
    unrecorded = unreadable

    if not rows:
        w("no completed runs recorded `satisfaction_epoch` under these roots.\n")
        if unrecorded:
            w("  %d completed runs carried no such field -- wrong arm family?\n"
              % unrecorded)
        return 1

    w("\n%s\n" % ("=" * 74))
    w("DOES THE LATCH FIRE, AND DOES VIOLATION CONTINUE AFTER IT?\n")
    w("  `satisfaction_epoch` disables the lambda ratchet AND the rho ramp\n")
    w("  permanently, for every scope, the first time all scopes are\n")
    w("  simultaneously satisfied. %d completed runs.\n" % len(rows))
    w("%s\n" % ("=" * 74))
    w("  %-16s %6s %8s %9s %14s %s\n"
      % ("arm", "runs", "latched", "med epoch", "epochs after", "viol scope-ep after"))
    by_arm = collections.defaultdict(list)
    for r in rows:
        by_arm[r["arm"]].append(r)
    for arm in sorted(by_arm):
        rs = by_arm[arm]
        lat = [r for r in rs if r["latch"]]
        if not lat:
            w("  %-16s %6d %8s %9s %14s %s\n"
              % (arm, len(rs), "0", "-", "-", "NEVER LATCHES -- no cost here"))
            continue
        eps = sorted(r["latch"] for r in lat)
        med = eps[len(eps) // 2]
        after, viol = [], []
        for r in lat:
            tail = r["trace"][r["latch"]:]
            after.append(len(tail))
            viol.append(sum(tail))
        w("  %-16s %6d %8d %9d %14.1f %.1f\n"
          % (arm, len(rs), len(lat), med,
             sum(after) / len(after), sum(viol) / len(viol)))

    w("\n  READ THE LAST TWO COLUMNS TOGETHER. `epochs after` is how much of\n"
      "  the constraint phase ran with BOTH adaptive channels frozen;\n"
      "  `viol scope-ep after` is how many (group, class, epoch) violations\n"
      "  happened in that frozen tail. Near zero in either column means the\n"
      "  latch costs nothing and this direction is CLOSED. A late latch is\n"
      "  as good as no latch.\n")

    # THE KILL-SWITCH. Everything above is about WHEN TraLO stops adapting.
    # This is about WHAT it was aiming at while it did. Under `normalize` the
    # summed constraint gradient takes ONE norm over model.parameters(), so the
    # overall size of the multiplier divides out and only the ratios across
    # scopes steer. TraLO's ratios come from a FREQUENCY counter; LDF's and
    # ALM's from cumulative violation MAGNITUDE. If those two rank the scopes
    # the same way, the distinction is cosmetic and `tralo_dualprop` is not
    # worth a GPU.
    w("\n%s\n" % ("=" * 74))
    w("FREQUENCY vs MAGNITUDE: are they even different search directions?\n")
    w("  Spearman(TraLO's lambda_c, the cumulative violation ALM/LDF would\n")
    w("  have integrated) ACROSS the scopes of one run. rho = 1.000 means\n")
    w("  TraLO's constant ratchet already orders scopes exactly as a\n")
    w("  magnitude integrator would, and the direction is CLOSED.\n")
    w("%s\n" % ("=" * 74))
    for arm in sorted(by_arm):
        rs, out_r = by_arm[arm], []
        for r in rs:
            keys = [k for k in r["lam"] if k in r["mag"]]
            if len(keys) >= 3:
                rr = _spearman([r["lam"][k] for k in keys],
                               [r["mag"][k] for k in keys])
                if rr is not None:
                    out_r.append(rr)
        if not out_r:
            w("  %-16s fewer than 3 co-violated scopes per run -- no ranking\n"
              % arm)
            continue
        out_r.sort()
        med = out_r[len(out_r) // 2]
        # SPEARMAN IS NOT ENOUGH, AND ON ITS OWN IT MISLEADS HERE. Under
        # `normalize` the search direction is the RATIO between scope weights,
        # and a rank correlation is invariant to any monotone rescaling -- so
        # two weightings can order scopes almost identically while one is
        # nearly uniform and the other is enormously selective. The dynamic
        # range is what actually steers, so it is reported beside the rho.
        spreads = []
        for r in rs:
            keys = [k for k in r["lam"] if k in r["mag"]]
            if len(keys) < 3:
                continue
            lv = [r["lam"][k] for k in keys]
            mv = [r["mag"][k] for k in keys if r["mag"][k] > 0]
            if lv and mv and min(lv) > 0 and min(mv) > 0:
                spreads.append((max(lv) / min(lv), max(mv) / min(mv)))
        if spreads:
            spreads.sort(key=lambda t: t[0])
            lo = spreads[len(spreads) // 2]
            w("  %-16s median rho %+.3f over %3d runs | RANGE tralo %5.1fx vs "
              "magnitude %7.1fx\n" % (arm, med, len(out_r), lo[0], lo[1]))
        else:
            w("  %-16s median rho %+.3f over %3d runs\n" % (arm, med, len(out_r)))
    w("\n  READ THE RANGE, NOT ONLY THE RHO. Spearman is invariant to any\n"
      "  monotone rescaling, so two weightings can ORDER the scopes almost\n"
      "  identically while one is nearly uniform and the other is sharply\n"
      "  selective. Under `normalize` it is the RATIOS that steer, so a small\n"
      "  tralo range beside a large magnitude range means TraLO is spreading\n"
      "  its fixed-norm step almost evenly over scopes that ALM concentrates.\n"
      "  That is a difference a high rho actively hides.\n")
    if unrecorded:
        w("\n  %d completed runs recorded no `satisfaction_epoch` (non-tralo\n"
          "  families do not have one) and were skipped.\n" % unrecorded)
    return 0


def _log(rows):
    """Fixture: rows of (satisfied, {scope: excess}). Limits are unused here."""
    return [(sat, ex, {k: 0.0 for k in ex}) for sat, ex in rows]


def self_test(out=sys.stdout):
    checks = []

    # ---- the latch ------------------------------------------------------
    lg = _log([(False, {("1", 2): 5.0}), (False, {("1", 2): 3.0}),
               (True, {("1", 2): 0.0}), (False, {("1", 2): 9.0})])
    checks.append(("latch fires on the FIRST all-satisfied epoch",
                   latch_epoch(lg) == 2))
    checks.append(("  and a later violation does NOT clear it, as in train.py",
                   latch_epoch(lg) == 2))
    checks.append(("NEGATIVE CONTROL: a run that never satisfies has no latch",
                   latch_epoch(_log([(False, {("1", 2): 5.0})] * 5)) is None))

    # ---- frequency vs magnitude: the kill-switch ------------------------
    # Scope A is violated in EVERY epoch by 1 item; scope B in ONE epoch by 100.
    # TraLO's constant ratchet must rank A above B; a magnitude integrator must
    # rank B above A. If this fixture does not invert, the tool cannot detect
    # the difference it exists to detect.
    rows = [(False, {("A", 2): 1.0, ("B", 2): 0.0}) for _ in range(9)]
    rows.append((False, {("A", 2): 1.0, ("B", 2): 100.0}))
    lam, mag = weight_rankings(_log(rows), None, 0.01, 0.05)
    checks.append(("frequency ranks the OFTEN-violated scope first (%.2f > %.2f)"
                   % (lam[("A", 2)], lam[("B", 2)]),
                   lam[("A", 2)] > lam[("B", 2)]))
    checks.append(("  and magnitude ranks the DEEPLY-violated one first "
                   "(%.0f > %.0f)" % (mag[("B", 2)], mag[("A", 2)]),
                   mag[("B", 2)] > mag[("A", 2)]))

    # The latch must truncate the accumulation. A scope violated only AFTER the
    # latch earns no multiplier, because the gate is shut -- if this counted it,
    # every run would look adaptive to the end.
    rows = [(False, {("A", 2): 1.0, ("B", 2): 0.0}),
            (True, {("A", 2): 0.0, ("B", 2): 0.0}),
            (False, {("A", 2): 0.0, ("B", 2): 50.0})]
    lam2, mag2 = weight_rankings(_log(rows), latch_epoch(_log(rows)), 0.01, 0.05)
    checks.append(("NEGATIVE CONTROL: violation AFTER the latch earns no "
                   "multiplier", ("B", 2) not in lam2 and ("B", 2) not in mag2))

    checks.append(("spearman: increasing reads +1",
                   _spearman([1, 2, 3, 4], [10, 20, 30, 40]) == 1.0))
    checks.append(("NEGATIVE CONTROL: decreasing reads -1",
                   _spearman([1, 2, 3, 4], [40, 30, 20, 10]) == -1.0))
    checks.append(("NEGATIVE CONTROL: a constant side has no ranking",
                   _spearman([1, 2, 3, 4], [7, 7, 7, 7]) is None))

    # A latch at the last epoch must show an EMPTY frozen tail -- that is the
    # "costs nothing" case the tool has to be able to report.
    trace = [3] * 29
    tail = trace[29:]
    checks.append(("NEGATIVE CONTROL: a latch at the final epoch leaves no "
                   "frozen tail", len(tail) == 0 and sum(tail) == 0))

    # An early latch followed by continued violation is the positive case.
    tail = trace[3:]
    checks.append(("an early latch leaves a long, still-violated tail "
                   "(%d epochs, %d scope-epochs)" % (len(tail), sum(tail)),
                   len(tail) == 26 and sum(tail) == 78))

    # A frozen tail with NO violation is also a null, and must read as one.
    quiet = [3, 3, 3] + [0] * 26
    checks.append(("NEGATIVE CONTROL: a frozen tail with zero violation reads "
                   "zero", sum(quiet[3:]) == 0))

    # violation_trace counts a scope as violated only STRICTLY over budget --
    # a count exactly AT the ceiling is satisfied, and off-by-one here would
    # invent violations in every run.
    hard, lim = {("1", 2): 10.0}, {("1", 2): 10.0}
    n = sum(1 for k, K in lim.items() if hard.get(k, 0.0) > K)
    checks.append(("NEGATIVE CONTROL: a count exactly AT the ceiling is not a "
                   "violation", n == 0))
    hard = {("1", 2): 11.0}
    n = sum(1 for k, K in lim.items() if hard.get(k, 0.0) > K)
    checks.append(("  and one item over IS", n == 1))

    print("", file=out)
    for label, good in checks:
        print("  %-66s %s" % (label[:66], "PASS" if good else "FAIL"), file=out)
    bad = [c for c, g in checks if not g]
    print("", file=out)
    print("SELF-TEST PASSED" if not bad else "FAILED: %d" % len(bad), file=out)
    return 1 if bad else 0


def main(argv=None):
    a = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    a.add_argument("--campaign", nargs="+", default=[])
    a.add_argument("--arms", nargs="+", default=[])
    a.add_argument("--classes", nargs="+", type=int, default=[2, 7])
    a.add_argument("--self-test", action="store_true")
    args = a.parse_args(argv)
    if args.self_test:
        return self_test()
    if not args.campaign:
        a.error("give --campaign <root> ... (or --self-test)")
    return analyse(args.campaign, set(args.classes), set(args.arms))


if __name__ == "__main__":
    sys.exit(main())
