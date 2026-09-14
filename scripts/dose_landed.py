"""Compare planned, attempted and applied constraint updates in campaign logs."""

import argparse
import collections
import glob
import json
import os
import sys

DOSE_FRACTION_TOLERANCE = 0.05


def read_root(root):
    per = collections.defaultdict(lambda: [0, 0, 0, 0, 0, 0])
    amps = collections.defaultdict(set)
    for path in glob.glob(os.path.join(root, "**", "config.json"), recursive=True):
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
            if str(cfg.get("status", "pending")) != "completed":
                cell[3] += 1
            elif (cfg.get("hyperparams") or {}).get("constraint_epochs") == 0:
                cell[5] += 1
            else:
                cell[4] += 1
            continue
        cell[0] += int(app)
        cell[1] += int(att)
        cell[2] += 1
    return (per, amps)


def report(per, amps, tolerance=DOSE_FRACTION_TOLERANCE, out=sys.stdout):

    def slot(v, i):
        return v[i] if len(v) > i else 0

    def state(arm):
        v = per[arm]
        bits = []
        if v[2]:
            bits.append(
                "%d finished with 0 steps attempted, as a lambda=0 twin does" % v[2]
            )
        if slot(v, 5):
            bits.append("%d finished post-hoc, which attempts none" % slot(v, 5))
        if slot(v, 4):
            bits.append(
                "%d finished with NO counts: they predate the field" % slot(v, 4)
            )
        if slot(v, 3):
            bits.append("%d still pending or running" % slot(v, 3))
        return "; ".join(bits) or "no runs"

    trained = {a: v for (a, v) in per.items() if v[1] > 0}
    others = sorted((a for a in per if a not in trained))
    if not trained:
        out.write("no completed run records a constraint-step count yet.\n")
        for arm in others:
            out.write("  %-16s %s\n" % (arm, state(arm)))
        out.write(
            "  This is the normal state at the very start of a campaign. Re-run it once a TRAINED arm completes.\n"
        )
        return 0
    problems = 0
    out.write("CONSTRAINT DOSE -- steps that LANDED, against steps attempted\n")
    fracs = {}
    for arm in sorted(trained):
        (app, att, n) = (trained[arm][0], trained[arm][1], trained[arm][2])
        frac = app / float(att)
        fracs[arm] = frac
        amp = "/".join(sorted(amps.get(arm) or ["?"]))
        flag = "" if app == att else "   *** %d STEP(S) LOST" % (att - app)
        out.write(
            "  %-16s %6d / %-6d  %6.1f%%  %2d run(s)  amp=%-9s%s\n"
            % (arm, app, att, 100.0 * frac, n, amp, flag)
        )
        if app != att:
            problems += 1
    for arm in others:
        out.write("  %-16s %s\n" % (arm, state(arm)))
    if problems:
        out.write(
            "\n  A lost step is a SILENT dose reduction: the epoch ran, the gradient was\n  non-finite, no update landed, and the run still reports `status: completed`.\n"
        )
    if len(fracs) > 1:
        lo = min(fracs, key=fracs.get)
        hi = max(fracs, key=fracs.get)
        if fracs[hi] - fracs[lo] > tolerance:
            problems += 1
            out.write("\n  *** THESE ARMS DID NOT RUN AT THE SAME DOSE.\n")
            out.write(
                "      `%s` landed %.1f%% and `%s` landed %.1f%%. An arm-vs-arm delta across\n      that gap is confounded with how much constraint phase each one got.\n"
                % (hi, 100.0 * fracs[hi], lo, 100.0 * fracs[lo])
            )
            spread = len([a for a in fracs if fracs[a] < fracs[hi] - tolerance])
            if spread == 1:
                out.write(
                    "      ONE arm is low and its siblings are not, so this is the LOSS SHAPE,\n      not the host: see FRAMEWORK 2(u).\n"
                )
            else:
                out.write(
                    "      %d arms are low, which points at the HOST rather than any one loss.\n      Check the amp column: FP16 + GradScaler skips an overflowing step.\n"
                    % spread
                )
            out.write(
                "      Fix and RELAUNCH -- a dropped step cannot be recovered from the outputs.\n"
            )
    cross_arm_attempts(per, out)
    return problems


def self_test(out=sys.stdout):
    ok = True
    per = {"a": [29, 29, 1, 0, 0, 0], "b": [1, 29, 1, 0, 0, 0]}
    n = report(per, {"a": {"bfloat16"}, "b": {"bfloat16"}}, out=open(os.devnull, "w"))
    if n < 2:
        out.write(
            "SELF-TEST FAIL: a 3.4%% arm beside a 100%% one reported %d problem(s), expected at least 2\n"
            % n
        )
        ok = False
    per = {"a": [29, 29, 1, 0, 0, 0], "b": [29, 29, 1, 0, 0, 0]}
    n = report(per, {}, out=open(os.devnull, "w"))
    if n != 0:
        out.write(
            "SELF-TEST FAIL: two arms both at 100%% reported %d problem(s), expected 0\n"
            % n
        )
        ok = False
    per = {"a": [716, 1044, 36, 0, 0, 0], "b": [720, 1044, 36, 0, 0, 0]}
    n = report(per, {"a": {"float16"}, "b": {"float16"}}, out=open(os.devnull, "w"))
    if n < 2:
        out.write(
            "SELF-TEST FAIL: two arms both at ~69%% reported %d problem(s); agreeing with each other is not the same as landing\n"
            % n
        )
        ok = False
    import io as _io

    buf = _io.StringIO()
    report({"tralo_uniform": [0, 0, 0, 36, 0, 0]}, {}, out=buf)
    if "predate" in buf.getvalue() or "still pending or running" not in buf.getvalue():
        out.write(
            "SELF-TEST FAIL: 36 unstarted runs reported as predating the field:\n%s"
            % buf.getvalue()
        )
        ok = False
    buf = _io.StringIO()
    report({"tralo": [0, 0, 0, 0, 36, 0]}, {}, out=buf)
    if "predate" not in buf.getvalue():
        out.write(
            "SELF-TEST FAIL: 36 COMPLETED runs with no counts must be reported as predating the field:\n%s"
            % buf.getvalue()
        )
        ok = False
    buf = _io.StringIO()
    report({"tralo": [29, 29, 1, 0, 0, 0], "clip": [0, 0, 0, 34, 0, 2]}, {}, out=buf)
    if "predate" in buf.getvalue() or "post-hoc" not in buf.getvalue():
        out.write(
            "SELF-TEST FAIL: a finished post-hoc run must not read as predating the field:\n%s"
            % buf.getvalue()
        )
        ok = False
    if buf.getvalue().count("  clip ") > 1:
        out.write(
            "SELF-TEST FAIL: one arm printed on more than one line:\n%s"
            % buf.getvalue()
        )
        ok = False
    out.write("SELF-TEST %s\n" % ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


def cross_arm_attempts(per, out):
    rate = {}
    for arm, v in per.items():
        (attempted, runs) = (v[1], v[2])
        if attempted and runs:
            rate[arm] = attempted / float(runs)
    if len(set((round(r, 3) for r in rate.values()))) <= 1:
        return
    hi = max(rate.values())
    out.write(
        chr(10)
        + "CROSS-ARM ATTEMPTS PER RUN -- the asymmetry the percentages above cannot show"
        + chr(10)
    )
    for arm in sorted(rate, key=lambda a: (-rate[a], a)):
        r = rate[arm]
        tail = ""
        if abs(r - hi) > 1e-09:
            tail = "   <-- %.1f%% fewer steps than the top arm" % (
                100.0 * (hi - r) / hi
            )
        out.write("  %-18s %6.2f attempted/run%s" % (arm, r, tail) + chr(10))
    out.write(
        "  Every arm above can still read 100%, because that figure is applied/attempted"
        + chr(10)
    )
    out.write(
        "  WITHIN an arm. A dominance claim across these arms is NOT at equal dose."
        + chr(10)
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("root", nargs="?", help="campaign root, e.g. results/iwc4")
    ap.add_argument(
        "--tolerance",
        type=float,
        default=DOSE_FRACTION_TOLERANCE,
        help="max landing-rate spread between arms (default 0.05)",
    )
    ap.add_argument(
        "--self-test",
        action="store_true",
        help="check the reporter against known-bad inputs",
    )
    args = ap.parse_args()
    if args.self_test:
        return self_test()
    if not args.root:
        ap.error("give a campaign root, or --self-test")
    if not os.path.isdir(args.root):
        print("no such campaign root: %s" % args.root)
        return 2
    (per, amps) = read_root(args.root)
    problems = report(per, amps, tolerance=args.tolerance)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
