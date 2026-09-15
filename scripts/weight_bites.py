"""Did the per-item weighting actually do anything? (gate:weight_bites)

WHY THIS GATE EXISTS. Five flags in this project were found to be INERT after
they had already produced campaign results that looked healthy -- `logit_adjust`,
`class_balanced`, `hounie_alpha`, `graph_probe --dump`, and the local cap in one
window. Each one logged, ran, and changed nothing. `constraint_weight` is
exactly that shape of knob: a string in the hyperparameters that silently
reduces to the reference arm if anything in the chain is wrong.

So the weighting arm does not get to be read as a result until this passes.

WHAT IT CHECKS, per campaign:

  1. An arm that declares a non-uniform `constraint_weight` must log a weight
     spread (`constraint_weight_cv`) strictly above --min-cv on a real fraction
     of its constraint epochs. A cv of 0 is an inert arm.
  2. An arm that declares `uniform` -- every reference and null arm -- must log
     a cv of EXACTLY 0.0. Anything else means the control was perturbed and the
     comparison is void.
  3. The two must not agree. If a weighting arm's predictions are byte-identical
     to its uniform twin at the same seed, the weights reached the log but never
     reached the gradient, which is the failure that stale bytecode and dropped
     dtype casts both produce.

Check 3 is the one that catches a knob that is live in the count but dead in the
step, so it is not optional once a twin exists.
"""
import argparse
import collections
import glob
import hashlib
import json
import os
import sys

MIN_CV = 1e-6
MIN_EPOCH_FRACTION = 0.5


def _events(run_dir):
    path = os.path.join(run_dir, "constraint_events.jsonl")
    if not os.path.exists(path):
        return []
    rows = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except ValueError:
                continue
    return [r for r in rows if r.get("phase") == "constraint"]


def _pred_hash(run_dir):
    """Hash of the deployed predictions, or None when the run has not finished."""
    path = os.path.join(run_dir, "final_predictions.csv")
    if not os.path.exists(path):
        return None
    with open(path, "rb") as handle:
        return hashlib.sha256(handle.read()).hexdigest()


def read_root(root):
    per = collections.defaultdict(lambda: {"cvs": [], "modes": set(), "runs": 0})
    twins = collections.defaultdict(dict)
    for cfg_path in glob.glob(os.path.join(root, "**", "config.json"), recursive=True):
        run_dir = os.path.dirname(cfg_path)
        with open(cfg_path, encoding="utf-8") as handle:
            cfg = json.load(handle)
        arm = cfg.get("arm")
        if arm is None:
            continue
        rows = _events(run_dir)
        if not rows:
            continue
        entry = per[arm]
        entry["runs"] += 1
        for r in rows:
            entry["modes"].add(r.get("constraint_weight", "uniform"))
            entry["cvs"].append(float(r.get("constraint_weight_cv") or 0.0))
        hp = cfg.get("hyperparams") or {}
        key = (cfg.get("backbone"), cfg.get("dataset"), cfg.get("constraint_tag"),
               hp.get("seed"))
        twins[key][arm] = _pred_hash(run_dir)
    return per, twins


def report(per, twins, pairs, min_cv=MIN_CV, min_fraction=MIN_EPOCH_FRACTION):
    problems = []
    print("%-24s %6s %14s %10s  %s"
          % ("arm", "runs", "declared", "median cv", "verdict"))
    for arm in sorted(per):
        entry = per[arm]
        modes = sorted(entry["modes"])
        cvs = sorted(entry["cvs"])
        median = cvs[len(cvs) // 2] if cvs else 0.0
        declared = ",".join(modes)
        if len(modes) > 1:
            verdict = "FAIL -- mixed declarations within one arm"
            problems.append((arm, verdict))
        elif modes == ["uniform"]:
            if any(c != 0.0 for c in cvs):
                verdict = "FAIL -- control was perturbed, cv != 0"
                problems.append((arm, verdict))
            else:
                verdict = "ok (reference, exactly unweighted)"
        else:
            live = sum(1 for c in cvs if c > min_cv)
            frac = live / float(len(cvs)) if cvs else 0.0
            if frac < min_fraction:
                verdict = ("FAIL -- INERT on %.0f%% of constraint epochs"
                           % (100.0 * (1.0 - frac)))
                problems.append((arm, verdict))
            else:
                verdict = "ok (weights bite on %.0f%% of epochs)" % (100.0 * frac)
        print("%-24s %6d %14s %10.5f  %s"
              % (arm, entry["runs"], declared, median, verdict))

    if pairs:
        print("")
        print("Twin check -- a weighting arm must not reproduce its uniform twin:")
        for weighted, control in pairs:
            checked = identical = 0
            for key, byarm in twins.items():
                a, b = byarm.get(weighted), byarm.get(control)
                if a is None or b is None:
                    continue
                checked += 1
                identical += int(a == b)
            if checked == 0:
                print("  %s vs %s: no completed pair yet, NOT CHECKED"
                      % (weighted, control))
                continue
            if identical:
                verdict = ("FAIL -- %d/%d pairs byte-identical, the weights "
                           "never reached the gradient" % (identical, checked))
                problems.append((weighted, verdict))
            else:
                verdict = "ok (%d/%d pairs differ)" % (checked - identical, checked)
            print("  %s vs %s: %s" % (weighted, control, verdict))

    print("")
    if problems:
        print("gate:weight_bites FAIL -- %d problem(s)" % len(problems))
    else:
        print("gate:weight_bites PASS")
    return problems


def self_test():
    """Known-bad inputs the gate must reject, and one it must accept."""
    cases = [
        ("an inert weighting arm",
         {"tralo_stab": {"cvs": [0.0] * 8, "modes": {"knn_disagree"}, "runs": 4}},
         {}, [], 1),
        ("a perturbed uniform control",
         {"tralo": {"cvs": [0.0, 0.0, 1e-4], "modes": {"uniform"}, "runs": 4}},
         {}, [], 1),
        ("an arm declaring two modes at once",
         {"tralo_stab": {"cvs": [0.2] * 8, "modes": {"uniform", "knn_disagree"},
                         "runs": 4}},
         {}, [], 1),
        ("weights that never reached the gradient",
         {"tralo_stab": {"cvs": [0.2] * 8, "modes": {"knn_disagree"}, "runs": 2},
          "tralo": {"cvs": [0.0] * 8, "modes": {"uniform"}, "runs": 2}},
         {("mn3", "fmow2", "L80", 1): {"tralo_stab": "aaa", "tralo": "aaa"}},
         [("tralo_stab", "tralo")], 1),
        ("a healthy campaign",
         {"tralo_stab": {"cvs": [0.2] * 8, "modes": {"knn_disagree"}, "runs": 2},
          "tralo": {"cvs": [0.0] * 8, "modes": {"uniform"}, "runs": 2}},
         {("mn3", "fmow2", "L80", 1): {"tralo_stab": "aaa", "tralo": "bbb"}},
         [("tralo_stab", "tralo")], 0),
    ]
    bad = 0
    for name, per, twins, pairs, expected in cases:
        print("--- self-test: %s (expect %s)"
              % (name, "FAIL" if expected else "PASS"))
        problems = report(per, twins, pairs)
        got = 1 if problems else 0
        if got != expected:
            print("  SELF-TEST BROKEN: gate returned %d, expected %d"
                  % (got, expected))
            bad += 1
    print("self-test: %s" % ("FAILED" if bad else "all cases behaved"))
    return 1 if bad else 0


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("root", nargs="?", help="campaign results root")
    ap.add_argument("--pair", action="append", default=[], metavar="WEIGHTED:CONTROL",
                    help="a weighting arm and the uniform twin it must differ from")
    ap.add_argument("--min-cv", type=float, default=MIN_CV,
                    help="a weight spread at or below this counts as inert")
    ap.add_argument("--self-test", action="store_true",
                    help="check the gate against known-bad inputs")
    args = ap.parse_args()
    if args.self_test:
        return self_test()
    if not args.root:
        ap.error("give a campaign root, or --self-test")
    if not os.path.isdir(args.root):
        print("no such campaign root: %s" % args.root)
        return 2
    pairs = []
    for spec in args.pair:
        if ":" not in spec:
            ap.error("--pair wants WEIGHTED:CONTROL, got %r" % spec)
        pairs.append(tuple(spec.split(":", 1)))
    per, twins = read_root(args.root)
    if not per:
        print("no constraint events under %s -- NOTHING CHECKED" % args.root)
        return 2
    return 1 if report(per, twins, pairs, min_cv=args.min_cv) else 0


if __name__ == "__main__":
    sys.exit(main())
