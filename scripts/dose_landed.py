"""How many constraint steps actually LANDED -- readable on a RUNNING campaign.

WHY THIS EXISTS SEPARATELY FROM `full_panel`.

`full_panel` already prints a `CONSTRAINT DOSE` block and already refuses to
compare two arms whose landing rates differ. But it is the SCORER: it loads
predictions, pairs seeds, and takes minutes over a finished campaign. So the
question "is the arm I am testing getting its treatment" was only ever asked at
the END, and twice it was answered too late:

    results/uniform1   tralo_uniform landed 1 of 29 steps (3.4%) while `tralo`
                       and `tralo_head` landed 29 of 29 in the same campaign.
                       Caught at 4 of 252 runs -- by hand, not by a tool.
    results/iwc3       716 of 1044 (68.6%), at least one step lost in 36 of 36
                       runs. Read after all 180 runs finished.

This reads `results.constraint_steps_applied` / `_attempted` out of each
`config.json` and nothing else. No predictions, no pairing, no GPU. It runs in
seconds on a campaign that is 1% done, which is the only time the answer is
still worth anything.

WHAT A LOST STEP IS. `finish_constraint_step` returns `applied=False` when the
constraint gradient comes back non-finite; the epoch ran, no update landed, and
the run still writes `status: completed`. Nothing in the predictions records it
except the effect it did not have. Two arms can take 29 and 1 steps and be
reported as the same treatment at the same dose.

THE TWO CAUSES SEEN SO FAR, and they are distinguishable from this output:

    the loss shape   an arm whose count takes a logarithm can produce inf/NaN
                     from a probability the guard failed to keep off 1.0. It
                     hits ONE arm and leaves its siblings at 100%.
                     (FRAMEWORK 2(u); fixed by `clamp_probability`.)
    the loss scale   FP16 + GradScaler SKIPS an optimizer step whose gradient
                     overflows. It hits EVERY trained arm on that host, at
                     roughly 25-31%, and BF16 hosts show 100%.
                     (Fixed for a campaign by `--constraint-fp32`.)

So read the `amp` column beside the percentage: one arm low is the loss, every
arm low is the host.

    python -m scripts.dose_landed <root>
    python -m scripts.dose_landed <root> --tolerance 0.05
    python -m scripts.dose_landed --self-test
"""

import argparse
import collections
import glob
import json
import os
import sys

DOSE_FRACTION_TOLERANCE = 0.05


def read_root(root):
    """(arm -> [applied, attempted, runs_with_counts, runs_without], amps)."""
    per = collections.defaultdict(lambda: [0, 0, 0, 0])
    amps = collections.defaultdict(set)
    for path in glob.glob(os.path.join(root, "**", "config.json"),
                          recursive=True):
        try:
            cfg = json.load(open(path))
        except (ValueError, IOError):
            continue
        arm = os.path.basename(os.path.dirname(os.path.dirname(path)))
        res = cfg.get("results") or {}
        rt = res.get("runtime") or {}
        if rt.get("amp_dtype"):
            amps[arm].add(str(rt["amp_dtype"]))
        cell = per[arm]
        app = res.get("constraint_steps_applied")
        att = res.get("constraint_steps_attempted")
        if app is None or att is None:
            cell[3] += 1
            continue
        cell[0] += int(app)
        cell[1] += int(att)
        cell[2] += 1
    return per, amps


def report(per, amps, tolerance=DOSE_FRACTION_TOLERANCE, out=sys.stdout):
    """Print the table. Returns the number of PROBLEMS found."""
    trained = {a: v for a, v in per.items() if v[1] > 0}
    posthoc = sorted(a for a, v in per.items() if v[1] == 0 and v[2])
    blind = sorted(a for a, v in per.items() if v[1] == 0 and not v[2] and v[3])

    if not trained:
        out.write("no completed run records a constraint-step count yet.\n")
        if blind:
            out.write("  %d arm(s) have runs but no counts: %s\n"
                      % (len(blind), ", ".join(blind)))
        out.write("  This is the normal state at the very start of a "
                  "campaign. Re-run it once a TRAINED arm completes.\n")
        return 0

    problems = 0
    out.write("CONSTRAINT DOSE -- steps that LANDED, against steps attempted\n")
    fracs = {}
    for arm in sorted(trained):
        app, att, n, nb = trained[arm]
        frac = app / float(att)
        fracs[arm] = frac
        amp = "/".join(sorted(amps.get(arm) or ["?"]))
        flag = "" if app == att else "   *** %d STEP(S) LOST" % (att - app)
        out.write("  %-16s %6d / %-6d  %6.1f%%  %2d run(s)  amp=%-9s%s\n"
                  % (arm, app, att, 100.0 * frac, n, amp, flag))
        if app != att:
            problems += 1
    for arm in posthoc:
        out.write("  %-16s %s\n" % (arm, "post-hoc: 0 steps attempted, as it "
                                         "should be"))
    for arm in blind:
        out.write("  %-16s no counts recorded (%d run(s) predate the field)\n"
                  % (arm, per[arm][3]))

    if problems:
        out.write("\n  A lost step is a SILENT dose reduction: the epoch ran, "
                  "the gradient was\n"
                  "  non-finite, no update landed, and the run still reports "
                  "`status: completed`.\n")

    if len(fracs) > 1:
        lo = min(fracs, key=fracs.get)
        hi = max(fracs, key=fracs.get)
        if fracs[hi] - fracs[lo] > tolerance:
            problems += 1
            out.write("\n  *** THESE ARMS DID NOT RUN AT THE SAME DOSE.\n")
            out.write("      `%s` landed %.1f%% and `%s` landed %.1f%%. An "
                      "arm-vs-arm delta across\n"
                      "      that gap is confounded with how much constraint "
                      "phase each one got.\n"
                      % (hi, 100.0 * fracs[hi], lo, 100.0 * fracs[lo]))
            spread = len([a for a in fracs if fracs[a] < fracs[hi] - tolerance])
            if spread == 1:
                out.write("      ONE arm is low and its siblings are not, so "
                          "this is the LOSS SHAPE,\n"
                          "      not the host: see FRAMEWORK 2(u).\n")
            else:
                out.write("      %d arms are low, which points at the HOST "
                          "rather than any one loss.\n"
                          "      Check the amp column: FP16 + GradScaler skips "
                          "an overflowing step.\n" % spread)
            out.write("      Fix and RELAUNCH -- a dropped step cannot be "
                      "recovered from the outputs.\n")
    return problems


def self_test(out=sys.stdout):
    """The gate. A reporter that cannot fail is not a check."""
    ok = True

    per = {"a": [29, 29, 1, 0], "b": [1, 29, 1, 0]}
    n = report(per, {"a": {"bfloat16"}, "b": {"bfloat16"}}, out=open(os.devnull, "w"))
    if n < 2:
        out.write("SELF-TEST FAIL: a 3.4%% arm beside a 100%% one reported "
                  "%d problem(s), expected at least 2\n" % n)
        ok = False

    per = {"a": [29, 29, 1, 0], "b": [29, 29, 1, 0]}
    n = report(per, {}, out=open(os.devnull, "w"))
    if n != 0:
        out.write("SELF-TEST FAIL: two arms both at 100%% reported %d "
                  "problem(s), expected 0\n" % n)
        ok = False

    # Every arm low: the HOST case, which must still be caught even though the
    # arms agree with each other.
    per = {"a": [716, 1044, 36, 0], "b": [720, 1044, 36, 0]}
    n = report(per, {"a": {"float16"}, "b": {"float16"}},
               out=open(os.devnull, "w"))
    if n < 2:
        out.write("SELF-TEST FAIL: two arms both at ~69%% reported %d "
                  "problem(s); agreeing with each other is not the same as "
                  "landing\n" % n)
        ok = False

    out.write("SELF-TEST %s\n" % ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("root", nargs="?", help="campaign root, e.g. results/iwc4")
    ap.add_argument("--tolerance", type=float, default=DOSE_FRACTION_TOLERANCE,
                    help="max landing-rate spread between arms (default 0.05)")
    ap.add_argument("--self-test", action="store_true",
                    help="check the reporter against known-bad inputs")
    args = ap.parse_args()

    if args.self_test:
        return self_test()
    if not args.root:
        ap.error("give a campaign root, or --self-test")
    if not os.path.isdir(args.root):
        print("no such campaign root: %s" % args.root)
        return 2

    per, amps = read_root(args.root)
    problems = report(per, amps, tolerance=args.tolerance)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
