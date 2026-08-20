"""What constraint step size actually moves the count? Short runs, watched.

WHY. The constraint step's delivered magnitude is not what any config says it
is. Under `constraint_grad_mode: normalize` + `constraint_step_rule: sgd` the
step norm is exactly `lr_constraint * constraint_grad_clip`, and the protocol's
default of lr 1e-4 x clip 1.0 = **1e-4**. The Adam step it replaces has norm
~`lr * sqrt(N)` = 1e-4 * sqrt(86M) ~ **0.93** -- about 9,300x larger. So
recovering the constraint DIRECTION at the default clip cuts the MAGNITUDE by
four orders of magnitude, and measured over four epochs the capped-class count
went the wrong way exactly as before (class 4: 122 -> 346).

That is a dosing problem, not a refutation, and `configs/protocol.yml` already
names this knob "the only dose axis the protocol admits". This sweeps it.

WHY SHORT RUNS. A 29-epoch run is ~30 minutes and this project has hit a new
bug in nearly every one. Four epochs is enough to see the DIRECTION of the
count, and direction is what picks the dose. Nothing here is a result: it is a
dose-finding scan, at n=1, on a leaked test set. It says which dose to take to
a real campaign, and nothing else.

WATCH BOTH COLUMNS. The known failure of an over-dose is not that the count
stays high -- it is that the count is crushed while the classifier is
destroyed. `joint_objective` held the cap on 98.8% of epochs and lost 0.067 AP
doing it. A dose that drives the count to the budget while train accuracy
collapses is not a win, so accuracy is printed next to the count and the
summary refuses to call a dose good on the count alone.

    python -m scripts.dose_scan <a completed tralo run dir> --clips 100 1000 10000
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd

RUNNER = "src.experiments.runner"


def one(base_cfg, dest, gpu, epochs, clip, cache, chunk):
    dest.mkdir(parents=True, exist_ok=True)
    cfg = json.loads(json.dumps(base_cfg))
    hp = cfg["hyperparams"]
    hp["constraint_epochs"] = epochs
    hp["constraint_chunk_size"] = chunk
    hp["constraint_grad_mode"] = "normalize"
    hp["constraint_step_rule"] = "sgd"
    hp["constraint_grad_clip"] = float(clip)
    cfg["status"] = "pending"
    out = dest / "config.json"
    out.write_text(json.dumps(cfg, indent=2), encoding="utf-8")

    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    env["OPTLOSS_MODEL_CACHE"] = str(cache.resolve())
    r = subprocess.run([sys.executable, "-m", RUNNER, str(out)],
                       env=env, capture_output=True, text=True)
    log = dest / "training_log.csv"
    if r.returncode != 0 or not log.exists():
        tail = (r.stdout + r.stderr)[-400:].replace("\n", " ")
        return None, tail
    try:
        return pd.read_csv(log), None
    except Exception as exc:                        # empty/partial file
        return None, str(exc)


def main():
    a = argparse.ArgumentParser(description=__doc__)
    a.add_argument("run_dir")
    a.add_argument("--clips", type=float, nargs="+",
                   default=[1.0, 100.0, 1000.0, 10000.0])
    a.add_argument("--epochs", type=int, default=4)
    a.add_argument("--chunk", type=int, default=128,
                   help="constraint_chunk_size. 256 OOMs at ViTB16 under "
                        "constraint_fp32 on a 22GB card; 128 fits.")
    a.add_argument("--gpu", default="0")
    a.add_argument("--out", default="/tmp/dose_scan")
    args = a.parse_args()

    base = json.loads((Path(args.run_dir) / "config.json").read_text(encoding="utf-8"))
    lr = float(base["hyperparams"]["lr_constraint"])
    root = Path(args.out)
    if root.exists():
        shutil.rmtree(root)
    cache = root / "_cache"

    caps = [c for c in range(base.get("num_classes", 7))
            if base.get("global_constraints", [])
            and base["global_constraints"][c] < 1e9]
    print("dose scan: %s  lr_constraint=%g  capped classes=%s  %d epochs each"
          % (args.run_dir, lr, caps, args.epochs))
    print("step norm delivered = lr_constraint x clip\n")

    rows = []
    for clip in args.clips:
        d, err = one(base, root / ("clip_%g" % clip), args.gpu, args.epochs,
                     clip, cache, args.chunk)
        print("#### clip=%-8g  step norm = %.3g ####" % (clip, lr * clip))
        if d is None:
            print("   FAILED: %s" % err)
            continue
        first = last = None
        for _, r in d.iterrows():
            cols = " ".join("c%d=%4d/%d" % (c, int(r["Hard_Class%d" % c]),
                                            int(r["Limit_Class%d" % c]))
                            for c in caps)
            print("   ep%-3d acc=%.4f  %s  satG=%d satL=%d  raw=%.4g"
                  % (r["Epoch"], r["Train_Acc"], cols,
                     r["Global_Satisfied"], r["Local_Satisfied"], r["Grad_Norm"]))
            if first is None:
                first = r
            last = r
        if first is not None and last is not None and len(d) > 1:
            tot0 = sum(max(0, int(first["Hard_Class%d" % c]) - int(first["Limit_Class%d" % c])) for c in caps)
            tot1 = sum(max(0, int(last["Hard_Class%d" % c]) - int(last["Limit_Class%d" % c])) for c in caps)
            rows.append((clip, tot0, tot1, float(first["Train_Acc"]),
                         float(last["Train_Acc"])))
    print()
    if not rows:
        print("no dose produced a readable trajectory")
        return
    print("%-10s %10s %10s %9s %10s %10s"
          % ("clip", "excess0", "excessN", "d(excess)", "acc0", "accN"))
    print("-" * 64)
    for clip, e0, e1, a0, a1 in rows:
        print("%-10g %10d %10d %9+d %10.4f %10.4f" % (clip, e0, e1, e1 - e0, a0, a1))
    print()
    print("A dose is only interesting if excess FALLS and accuracy does NOT.")
    print("Crushing the count while the classifier degrades is the joint arm's")
    print("failure (cap held 98.8% of epochs, AP -0.067), not a win. And this")
    print("is n=1 dose-finding on a leaked test set -- it picks the dose to")
    print("take to a real campaign, it is not itself a result.")


if __name__ == "__main__":
    main()
