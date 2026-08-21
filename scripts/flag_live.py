"""Does this flag DO anything? md5 the predictions across arms before believing it.

Inert flags are this project's most frequent failure mode -- four occurrences
and counting, including three arms whose `focal_clip` was a second `clip` and a
`tralo_uniform` whose treatment was a no-op in `train.py`. Every one of them
passed `audit_config` (the key existed and had a reader), passed `smoke_arms`
(the arm ran), and produced a full campaign of numbers that meant nothing.

Neither existing gate can catch it: one reads configs, the other checks that an
arm does not crash. This runs the arms end to end on the SAME synthetic inputs
with the SAME seed and hashes what the model produces, which is the only thing
that distinguishes a live treatment from a renamed control.

    python -m scripts.flag_live tralo tralo_margin tralo_null

Reads the constraint-phase length up so a slow treatment has room to act -- a
flag that needs 29 epochs to separate looks inert at 2. Exit 1 if any two arms
produce bit-identical predictions.

NOT A RESULT. Synthetic tensors, random labels, a 4-layer net, n=1. It answers
"is this knob connected" and nothing else -- in particular a count that moves
here may still be worthless on real data, and a count that CRUSHES the hard
count here is showing the over-dose failure mode, not a win.
"""
import argparse
import hashlib
import sys
import tempfile

import numpy as np
import torch

from configs.gen_campaign import load_protocol
from scripts.smoke_arms import make_inputs
from src.experiments.runner import TRAIN_FNS


def main():
    a = argparse.ArgumentParser(description=__doc__)
    a.add_argument("arms", nargs="+")
    a.add_argument("--constraint-epochs", type=int, default=6)
    a.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3],
                   help="one seed is not enough. Measured: at seed 1 the arms "
                        "spread 12 to 35 against K=11, at seed 2 nothing is "
                        "predicted the capped class at all and every arm is "
                        "bit-identical, at seed 3 every item is and they are "
                        "all within 1. A single seed can show anything.")
    args = a.parse_args()

    P = load_protocol()
    unknown = [x for x in args.arms if x not in P["arms"]]
    if unknown:
        print("unknown arm(s): %s" % " ".join(unknown))
        return 1

    tmp = tempfile.mkdtemp(prefix="flag_live_")
    per_seed = {}
    for seed in args.seeds:
        rows, K = [], None
        for arm in args.arms:
            torch.manual_seed(seed)
            np.random.seed(seed)
            inp, gcon, _ = make_inputs(P, arm, tmp, seed=seed)
            inp.hyperparams["constraint_epochs"] = args.constraint_epochs
            K = int(gcon[1])
            res = TRAIN_FNS[P["arms"][arm]["methodology"]](inp)
            res.model.eval()
            with torch.no_grad():
                p = torch.softmax(res.model(inp.X_test), dim=1).numpy().astype(np.float64)
            rows.append((arm,
                         hashlib.md5(np.round(p, 8).tobytes()).hexdigest()[:12],
                         float(p[:, 1].sum()), int((p.argmax(1) == 1).sum()),
                         inp.hyperparams.get("soft_count_mode", "-")))
        per_seed[seed] = (rows, K)

    print("%d constraint epochs, capped class 1, seeds %s"
          % (args.constraint_epochs, " ".join(str(x) for x in args.seeds)))
    print()
    print("%-16s %-8s %-14s %6s %10s %9s"
          % ("arm", "count", "md5", "seed", "sum_p(c1)", "hard(c1)"))
    print("-" * 70)
    for seed, (rows, K) in per_seed.items():
        for arm, h, sp, hard, mode in rows:
            print("%-16s %-8s %-14s %6d %10.3f %9d"
                  % (arm, mode, h, seed, sp, hard))
        print("%-16s %-8s %-14s %6s %10s %9s"
              % ("", "", "", "", "budget K =", K))
    print()

    # A cell where the cap is already satisfied takes NO constraint step -- the
    # penalty is relu(count - K) and it is zero. Every arm is then correctly
    # bit-identical, and calling that "inert" is a false alarm. Judge only on
    # seeds where the constraint actually bound.
    binding = [sd for sd, (rows, K) in per_seed.items()
               if any(r[3] > K for r in rows)]
    vacuous = [sd for sd in per_seed if sd not in binding]
    if vacuous:
        print("SKIPPED seed(s) %s: no arm exceeded the budget, so the penalty"
              % " ".join(str(x) for x in vacuous))
        print("is identically zero and every arm is correctly identical there.")
        print()
    if not binding:
        print("VACUOUS -- the cap never bound on any seed, so this run compares")
        print("nothing. Raise --constraint-epochs or pick a tighter cap.")
        return 2

    dupes = []
    dupes = []
    for sd in binding:
        rows = per_seed[sd][0]
        for i in range(len(rows)):
            for j in range(i + 1, len(rows)):
                if rows[i][1] == rows[j][1]:
                    dupes.append("seed %d: %s == %s" % (sd, rows[i][0], rows[j][0]))
    if dupes:
        print("INERT -- bit-identical predictions: %s" % "; ".join(dupes))
        print("Whatever differs between those arms in the config is not")
        print("reaching the model. Do not launch a campaign on it.")
        return 1
    print("Every flag is connected on every binding seed. That is ALL this says.")
    print()
    print("DO NOT READ THE COUNTS AS A RESULT. Random labels, a 4-layer net that")
    print("reaches chance accuracy, one dose. Measured across seeds at K=11 the")
    print("same four arms gave 31/35/32/12, then 0/0/0/0, then 120/120/119/119 --")
    print("a single seed here can show a large separation, none, or saturation.")
    print("Ordering on this harness is noise; only connectedness survives it.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
