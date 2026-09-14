"""Compare planned, attempted and applied constraint updates in campaign logs."""

import argparse
import collections
import glob
import json
import os
import sys

DOSE_FRACTION_TOLERANCE = 0.05

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from configs.gen_campaign import load_protocol

ARMS = load_protocol()["arms"]


def read_root(root):
    per = collections.defaultdict(lambda: [0, 0, 0, 0, 0, 0])
    amps = collections.defaultdict(set)
    for path in glob.glob(os.path.join(root, "**", "config.json"), recursive=True):
        try:
            with open(path, encoding="utf-8") as fh:
                cfg = json.load(fh)
        except (ValueError, IOError) as exc:
            raise ValueError("unreadable config %s: %s" % (path, exc)) from exc
        arm = cfg.get("arm")
        # Was a hardcoded seven-arm whitelist, so every arm added since --
        # focal_tralo, aug_tralo, aug_clip -- crashed this with "unknown arm"
        # at the score stage, after the campaign had already been paid for.
        if arm not in ARMS:
            raise ValueError("%s: arm %r is not declared in the protocol" % (path, arm))
        res = cfg.get("results") or {}
        rt = res.get("runtime") or {}
        if rt.get("amp_dtype"):
            amps[arm].add(str(rt["amp_dtype"]))
        cell = per[arm]
        app = res.get("constraint_steps_applied")
        att = res.get("constraint_steps_attempted")
        # Also a hardcoded list, which silently made aug_clip a TRAINED arm.
        posthoc = ARMS[arm]["phase"] == "posthoc"
        hp = cfg.get("hyperparams") or {}
        # The epoch BUDGET is campaign-relative and `check_parity` owns it --
        # this used to demand exactly (30, 0) or (1, 29) and raised on any other
        # budget, which would have aborted every short-horizon campaign. All
        # that matters here is that the phase and the dose agree.
        con = hp.get("constraint_epochs")
        if type(con) is not int or con < 0 or (con == 0) != posthoc:
            raise ValueError(
                "%s: arm %r is %s but declares constraint_epochs=%r"
                % (path, arm, "post-hoc" if posthoc else "trained", con))
        if cfg.get("status") != "completed":
            if app is not None or att is not None:
                raise ValueError("%s: counts on a non-completed run" % path)
            cell[3] += 1
            continue
        # Current post-hoc runtime emits no constraint-step counters. Its
        # declared zero-epoch phase is the only missing-count exception.
        if posthoc and app is None and att is None:
            cell[5] += 1
            continue
        expected = 0 if posthoc or arm == "tralo_null" else hp["constraint_epochs"]
        if type(app) is not int or type(att) is not int:
            raise ValueError(
                "%s: completed run requires integer applied/attempted counts" % path
            )
        if att != expected or app != expected:
            raise ValueError(
                "%s: dose applied=%s attempted=%s expected=%s"
                % (path, app, att, expected)
            )
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
            bits.append("%d finished with 0 steps attempted (null or post-hoc)" % v[2])
        if slot(v, 5):
            bits.append("%d finished post-hoc, which attempts none" % slot(v, 5))
        if slot(v, 4):
            bits.append("%d finished with NO counts: dose is unverified" % slot(v, 4))
        if slot(v, 3):
            bits.append("%d still pending or running" % slot(v, 3))
        return "; ".join(bits) or "no runs"

    trained = {a: v for (a, v) in per.items() if v[1] > 0}
    others = sorted((a for a in per if a not in trained))
    problems = sum(slot(v, 4) for v in per.values())
    completed = sum(v[2] + slot(v, 5) for v in per.values())
    if not trained:
        for arm in others:
            out.write("  %-16s %s\n" % (arm, state(arm)))
        missing_trained = any(
            a not in {"tralo_null", "clip", "focal_clip"} for a in per
        )
        if not completed or missing_trained:
            out.write("INCOMPLETE: no completed run establishes its constraint dose.\n")
        return problems + int(not completed or missing_trained)
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
            "\n  Applied and attempted counts differ; the cause is not established by counts alone.\n"
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
    problems += cross_arm_attempts(per, out)
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
    if "unverified" not in buf.getvalue():
        out.write(
            "SELF-TEST FAIL: completed runs with no counts must be unverified:\n%s"
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
        return 0
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
    return 1


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
    try:
        (per, amps) = read_root(args.root)
    except ValueError as exc:
        print("FAIL: " + str(exc))
        return 1
    problems = report(per, amps, tolerance=args.tolerance)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
