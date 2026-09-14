"""Run the selected software validation stage."""

import argparse
import subprocess
import sys

STAGES = [
    ("data", "stage1_data", "the slice, before a single image is loaded"),
    ("budget", "stage2_budget", "the cap arithmetic, before a config is written"),
    ("model", "stage3_model", "the backbone and warm-up cache, before training"),
    ("grid", "stage4_grid", "apples-to-apples, before launch"),
    ("trainlog", "stage5_trainlog", "CE saturation, dose, collapse, divergence"),
    ("results", "stage6_results", "what may be read off the output"),
]
BY_NAME = {n: m for (n, m, _) in STAGES}
BEFORE_LAUNCH = ["data", "budget", "model", "grid"]


def run(markers, extra):
    expr = " or ".join(markers)
    cmd = [
        sys.executable,
        "-m",
        "pytest",
        "tests/gates",
        "-m",
        expr,
        "-q",
        "--no-header",
    ] + list(extra)
    print("+ %s" % " ".join(cmd))
    return subprocess.call(cmd)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument(
        "--stage",
        nargs="*",
        default=None,
        help="one or more of: %s, or `all`" % ", ".join((n for (n, _, _) in STAGES)),
    )
    ap.add_argument(
        "--before-launch",
        action="store_true",
        help="stages 1-4: everything answerable from configs and labels alone. This is THE launch gate",
    )
    ap.add_argument("--list", action="store_true")
    (args, extra) = ap.parse_known_args()
    if args.list:
        print("STAGE     MARKER            WHAT IT GATES")
        for n, m, d in STAGES:
            print("  %-8s %-17s %s" % (n, m, d))
        return 0
    bad = [n for n in args.stage or [] if n not in BY_NAME and n != "all"]
    if bad:
        ap.error("unknown stage(s): %s" % ", ".join(bad))
    if args.before_launch:
        names = BEFORE_LAUNCH
    elif not args.stage or "all" in args.stage:
        names = [n for (n, _, _) in STAGES]
    else:
        names = args.stage
    bad = [n for n in names if n not in BY_NAME]
    if bad:
        ap.error(
            "unknown stage(s): %s. Known: %s" % (", ".join(bad), ", ".join(BY_NAME))
        )
    print("PRE-FLIGHT -- %d stage(s): %s" % (len(names), ", ".join(names)))
    for n in names:
        print("   %-9s %s" % (n, dict(((a, c) for (a, _, c) in STAGES))[n]))
    print("")
    return run([BY_NAME[n] for n in names], extra)


if __name__ == "__main__":
    sys.exit(main())
