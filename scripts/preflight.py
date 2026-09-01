"""THE PRE-FLIGHT GATES -- run the right bucket at the right moment.

`tests/` gates the CODE. `tests/gates/` gates the EXPERIMENT: six buckets, one
per stage of the pipeline, each encoding failure modes this project actually
paid for, at the point where each is still cheap to catch.

    1 data      the slice, before a single image is loaded
    2 budget    the cap arithmetic, before a config is written
    3 model     the backbone and warm-up cache, before training
    4 grid      apples-to-apples across the campaign, before launch
    5 trainlog  what the optimiser did -- CE saturation, dose, collapse
    6 results   what may be read off the output, before any claim

WHY BUCKETS AND NOT ONE SUITE. Each stage answers a question you can only ask
once you are standing at that point, and the cost of a miss falls by orders of
magnitude the earlier you catch it. A dead dataset caught at stage 1 costs
minutes of CPU; the same defect caught at stage 6 has already cost a week of
GPU and produced a plausible, publishable, wrong table. `dermmnist` leaked
38.7% of its test set and was found at stage 6.

    python -m scripts.preflight --stage data       # before downloading a set
    python -m scripts.preflight --stage budget     # before choosing cap tags
    python -m scripts.preflight --stage grid       # before every launch
    python -m scripts.preflight --stage trainlog   # on the FIRST completed run
    python -m scripts.preflight --stage results    # before quoting a number
    python -m scripts.preflight --before-launch    # stages 1-4, the launch gate
    python -m scripts.preflight --stage all

Exit code is pytest's, so this drops straight into CI or a pre-commit hook.
"""
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
BY_NAME = {n: m for n, m, _ in STAGES}
BEFORE_LAUNCH = ["data", "budget", "model", "grid"]


def run(markers, extra):
    """One pytest invocation over the union of the given markers."""
    expr = " or ".join(markers)
    cmd = [sys.executable, "-m", "pytest", "tests/gates", "-m", expr,
           "-q", "--no-header"] + list(extra)
    print("+ %s" % " ".join(cmd))
    return subprocess.call(cmd)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--stage", nargs="*", default=None,
                    help="one or more of: %s, or `all`"
                         % ", ".join(n for n, _, _ in STAGES))
    ap.add_argument("--before-launch", action="store_true",
                    help="stages 1-4: everything answerable from configs and "
                         "labels alone. This is THE launch gate")
    ap.add_argument("--list", action="store_true")
    args, extra = ap.parse_known_args()

    if args.list:
        print("STAGE     MARKER            WHAT IT GATES")
        for n, m, d in STAGES:
            print("  %-8s %-17s %s" % (n, m, d))
        return 0

    if args.before_launch:
        names = BEFORE_LAUNCH
    elif not args.stage or "all" in args.stage:
        names = [n for n, _, _ in STAGES]
    else:
        names = args.stage

    bad = [n for n in names if n not in BY_NAME]
    if bad:
        # NOT a silent skip. A typo'd stage name that runs nothing and exits 0
        # is the same shape as the gate this suite exists to prevent.
        ap.error("unknown stage(s): %s. Known: %s"
                 % (", ".join(bad), ", ".join(BY_NAME)))

    print("PRE-FLIGHT -- %d stage(s): %s" % (len(names), ", ".join(names)))
    for n in names:
        print("   %-9s %s" % (n, dict((a, c) for a, _, c in STAGES)[n]))
    print("")
    return run([BY_NAME[n] for n in names], extra)


if __name__ == "__main__":
    sys.exit(main())
