"""How much does the SAME config, run twice, disagree with itself?

This is the noise floor. Every arm-vs-arm delta in this project has to clear
it, and until 2026-08-20 nobody had measured it on the real pipeline.

WHY IT EXISTS. Cross-checking a post-hoc arm across two campaigns turned up
`clip` at seed 1 scoring macro-F1 0.6709 in one and 0.7015 in the other --
same arm, same seed, same config, same data, same GPU. 0.0306 apart, against a
headline TraLO-vs-clip effect of 0.0017. Epoch 1 was bit-identical and the runs
diverged from epoch 2, so it is not seeding or data order; it is
nondeterministic kernels, which `cudnn.deterministic` alone does not cover.

If that floor is real and stays where it is, no campaign this project can
afford will resolve the effects it is trying to measure, and the honest report
is the floor rather than the ranking.

    python -m scripts.variance_probe results/vit_diag/ViTB16/dermmnist/L30_G30/clip/seed_1 --repeats 3

Copies the run's config into N fresh directories, runs each through the SAME
entry point a campaign uses, and reports the spread of every metric. It does
not average anything: the spread IS the result.
"""

import argparse
import json
import os
import shutil
import statistics
import subprocess
import sys
from pathlib import Path

import pandas as pd

RUNNER = "src.experiments.runner"

# The metrics an arm-vs-arm claim is actually made on. Deliberately not every
# column: `Flips Required` and the raw-count family are not metrics in this
# project, and reporting their spread would invite quoting them.
WATCH = ["Accuracy", "F1 (Macro)", "Precision (Macro)", "Recall (Macro)",
         "F1_Class4", "Precision_Class4", "Recall_Class4",
         "ECE", "Brier Score", "Raw Total Excess"]


def one_run(src_cfg, dest, gpu):
    dest.mkdir(parents=True, exist_ok=True)
    cfg = json.loads(Path(src_cfg).read_text(encoding="utf-8"))
    cfg["status"] = "pending"
    out = dest / "config.json"
    out.write_text(json.dumps(cfg, indent=2), encoding="utf-8")

    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    # Before torch loads in the child, for the same reason main.py sets it.
    env.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    # A PRIVATE cache per repeat. Without this the probe measures nothing:
    # repeat 0 trains the warm-up and writes it to the shared cache, and every
    # later repeat LOADS that file instead of training. The spread then comes
    # out 0.0000 on every metric, which looks like perfect determinism and is
    # actually just "reading the same .pt twice gives the same weights".
    # Observed before this fix: warm-up 1180.9s, then 1.9s, then 1.0s.
    env["OPTLOSS_MODEL_CACHE"] = str((dest / "_cache").resolve())
    r = subprocess.run([sys.executable, "-m", RUNNER, str(out)],
                       env=env, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout[-2000:])
        print(r.stderr[-2000:])
        raise SystemExit("run failed in %s" % dest)
    m = dest / "evaluation_metrics.csv"
    if not m.exists():
        raise SystemExit("no evaluation_metrics.csv in %s" % dest)
    df = pd.read_csv(m)
    return dict(zip(df["Metric"], df["Value"]))


def _assert_really_retrained(results):
    """A repeat that skipped its warm-up did not measure anything.

    The private cache dir prevents this; the assertion is here so that if the
    isolation ever breaks, the probe REFUSES to report a floor instead of
    reporting 0.0000 and being believed.
    """
    times = []
    for r in results:
        try:
            times.append(float(r.get("Warmup Time")))
        except (TypeError, ValueError):
            return
    if not times or max(times) <= 0:
        return
    slowest = max(times)
    for i, t in enumerate(times):
        if t < 0.25 * slowest:
            raise SystemExit(
                "repeat %d warmed up in %.1fs against a slowest of %.1fs -- it "
                "LOADED a cached model instead of training one. Every metric "
                "would read as identical and the floor would be a fiction. "
                "Cache isolation is broken." % (i, t, slowest))


def main():
    a = argparse.ArgumentParser(description=__doc__)
    a.add_argument("run_dir", help="a completed run to replicate")
    a.add_argument("--repeats", type=int, default=3)
    a.add_argument("--gpu", default="0")
    a.add_argument("--out", default=None,
                   help="where to put the repeat directories")
    args = a.parse_args()

    src = Path(args.run_dir) / "config.json"
    if not src.exists():
        raise SystemExit("no config.json in %s" % args.run_dir)
    root = Path(args.out or (Path(args.run_dir).parent / "_variance_probe"))
    if root.exists():
        shutil.rmtree(root)

    cfg = json.loads(src.read_text(encoding="utf-8"))
    print("replicating %s  arm=%s seed=%s  x%d"
          % (args.run_dir, cfg.get("arm"),
             cfg.get("hyperparams", {}).get("seed"), args.repeats))
    print("code_version %s\n" % cfg.get("code_version"))

    results = []
    for i in range(args.repeats):
        d = root / ("rep_%d" % i)
        print("  run %d/%d ..." % (i + 1, args.repeats), flush=True)
        results.append(one_run(src, d, args.gpu))

    _assert_really_retrained(results)
    print("warm-up times: %s" % ", ".join(
        str(r.get("Warmup Time")) for r in results))
    print("\n%-22s %10s %10s %12s" % ("metric", "min", "max", "SPREAD"))
    print("-" * 58)
    worst = None
    for k in WATCH:
        vals = []
        for r in results:
            try:
                vals.append(float(r[k]))
            except (KeyError, TypeError, ValueError):
                pass
        if len(vals) < 2:
            continue
        spread = max(vals) - min(vals)
        sd = statistics.stdev(vals) if len(vals) > 2 else float("nan")
        print("%-22s %10.4f %10.4f %12.4f%s"
              % (k, min(vals), max(vals), spread,
                 "" if len(vals) < 3 else "   sd=%.4f" % sd))
        if k == "F1 (Macro)":
            worst = spread

    print()
    if worst is None:
        print("no macro-F1 recorded; cannot state a floor")
        return
    print("NOISE FLOOR on macro-F1 over %d identical runs: %.4f" % (args.repeats, worst))
    print()
    print("Reference points from this project:")
    print("   tralo vs clip, the headline effect          0.0017")
    print("   fioretto vs clip, this campaign seed 1      0.0059")
    print("   mean headroom over the whole corpus         0.0669")
    if worst >= 0.0017:
        print()
        print("*** The floor is at or above the headline effect. An arm-vs-arm")
        print("    delta smaller than this floor is not a measurement, however")
        print("    many seeds are averaged -- averaging shrinks the STANDARD")
        print("    ERROR, and the floor is what each draw is drawn from.")
    else:
        print()
        print("The floor sits below the headline effect; arm-vs-arm deltas")
        print("above it are resolvable at this seed count.")


if __name__ == "__main__":
    main()
