"""Reset runs that CRASHED so the dispatcher will retry them -- and nothing else.

Written after doing this by hand and getting it wrong: a loop over
`**/error_log.json` also matched two runs that had crashed once, been retried,
and FINISHED. Setting those back to `pending` would have re-run them and
overwritten a good result with a fresh one. The bug is not the loop, it is that
the loop had no idea what "already finished" looks like.

So the rule here is the inverse of the obvious one. A run is eligible only if it
has NO usable result: no `results.accuracy` in its config and no full
`training_log.csv`. A crash log is not evidence of anything on its own -- the
dispatcher retries, so a run can carry a crash log and still be the real result.

Use it after fixing the cause of a crash:

    python -m scripts.reset_crashed <campaign-root>            # dry run
    python -m scripts.reset_crashed <campaign-root> --apply
"""
import argparse
import glob
import json
import os

import pandas as pd

MIN_ROWS = 5  # a header-only or few-row log is a death, not a short run


def _rows(run_dir):
    f = os.path.join(run_dir, "training_log.csv")
    if not os.path.exists(f):
        return 0
    try:
        return len(pd.read_csv(f))
    except Exception:
        return 0


def eligible(cfg, rows):
    """Only a run with nothing worth keeping may be reset.

    Returns (ok, why). `why` is printed either way, so a refusal is as visible
    as an action -- the point of the script is that it says NO out loud.
    """
    if cfg.get("status") == "running":
        return False, "still running"
    has_result = bool((cfg.get("results") or {}).get("accuracy"))
    if has_result:
        return False, "HAS RESULTS (accuracy present) -- resetting would " \
                      "overwrite a finished run"
    if rows >= MIN_ROWS:
        return False, "has %d epochs logged but no results; inspect it by " \
                      "hand rather than discarding" % rows
    return True, "no results, %d epoch(s) logged" % rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--apply", action="store_true",
                    help="write the change; without it this is a dry run")
    args = ap.parse_args()

    reset = refused = 0
    for cfgp in sorted(glob.glob(os.path.join(args.root, "**", "config.json"),
                                 recursive=True)):
        d = os.path.dirname(cfgp)
        crash = glob.glob(os.path.join(d, "error_log*.json"))
        if not crash:
            continue
        cfg = json.load(open(cfgp, encoding="utf-8"))
        ok, why = eligible(cfg, _rows(d))
        rel = os.path.relpath(d, args.root)
        if not ok:
            refused += 1
            print("  SKIP  %-46s %s" % (rel, why))
            continue
        reset += 1
        print("  RESET %-46s %s (failures %s -> 0)"
              % (rel, why, cfg.get("failures", 0)))
        if args.apply:
            cfg["status"] = "pending"
            cfg["failures"] = 0
            json.dump(cfg, open(cfgp, "w", encoding="utf-8"), indent=2)

    print("\n%d reset, %d refused%s"
          % (reset, refused, "" if args.apply else "  (DRY RUN -- pass --apply)"))
    # The crash logs stay on disk under their own names. full_panel globs
    # error_log*.json, so a run that died and was retried still reports as
    # having crashed -- renaming them away is how a dead arm goes back to
    # looking merely unstarted.


if __name__ == "__main__":
    main()
