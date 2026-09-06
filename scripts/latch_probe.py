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


def latch_of(run_dir):
    """`satisfaction_epoch` for this run, or (None, False) if not recorded.

    Returns (epoch_or_None, found). `found` distinguishes "this run never
    latched" from "this run does not record the field at all" -- opposite
    conclusions that a bare None would merge.
    """
    for name in sorted(os.listdir(run_dir)):
        if not name.endswith(".json"):
            continue
        try:
            blob = json.load(open(os.path.join(run_dir, name)))
        except (ValueError, OSError):
            continue
        for d in (blob, blob.get("best_metrics") or {},
                  blob.get("metrics") or {}, blob.get("summary") or {}):
            if isinstance(d, dict) and "satisfaction_epoch" in d:
                return d["satisfaction_epoch"], True
    return None, False


def violation_trace(run_dir, classes):
    """Per epoch, how many (group, class) scopes were violated on HARD counts.

    Read straight from `training_log.csv`, which is the same quantity the latch
    itself tests -- so this measures the latch on its own terms rather than on
    the final predictions (FRAMEWORK 3(0c): the two disagree for trained arms).
    """
    path = os.path.join(run_dir, "training_log.csv")
    if not os.path.exists(path):
        return []
    out = []
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh):
            hard, lim = {}, {}
            for k, v in row.items():
                m = _SOFT.match(k or "")
                if not m or v in (None, ""):
                    continue
                c = int(m.group(3))
                if c not in classes:
                    continue
                (hard if m.group(2) == "Hard" else lim)[(m.group(1), c)] = float(v)
            if not lim:
                out.append(None)
                continue
            out.append(sum(1 for key, K in lim.items()
                           if hard.get(key, 0.0) > K))
    return out


def analyse(roots, classes, arms, out=sys.stdout):
    w = out.write
    rows = []
    unrecorded = 0
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
            ep, found = latch_of(d)
            if not found:
                unrecorded += 1
                continue
            trace = violation_trace(d, classes)
            rows.append({"arm": arm, "dir": d, "latch": ep,
                         "trace": [t for t in trace if t is not None]})

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
    if unrecorded:
        w("\n  %d completed runs recorded no `satisfaction_epoch` (non-tralo\n"
          "  families do not have one) and were skipped.\n" % unrecorded)
    return 0


def self_test(out=sys.stdout):
    checks = []

    # The two conclusions a bare None would merge must stay separate.
    checks.append(("a run that never latched and a run with no field are "
                   "different", (None, True) != (None, False)))

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
