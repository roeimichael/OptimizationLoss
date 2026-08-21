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
    a.add_argument("--seed", type=int, default=1)
    args = a.parse_args()

    P = load_protocol()
    unknown = [x for x in args.arms if x not in P["arms"]]
    if unknown:
        print("unknown arm(s): %s" % " ".join(unknown))
        return 1

    tmp = tempfile.mkdtemp(prefix="flag_live_")
    rows, K = [], None
    for arm in args.arms:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        inp, gcon, _ = make_inputs(P, arm, tmp, seed=args.seed)
        inp.hyperparams["constraint_epochs"] = args.constraint_epochs
        K = int(gcon[1])
        res = TRAIN_FNS[P["arms"][arm]["methodology"]](inp)
        res.model.eval()
        with torch.no_grad():
            p = torch.softmax(res.model(inp.X_test), dim=1).numpy().astype(np.float64)
        rows.append((arm, hashlib.md5(np.round(p, 8).tobytes()).hexdigest()[:12],
                     float(p[:, 1].sum()), int((p.argmax(1) == 1).sum()),
                     inp.hyperparams.get("soft_count_mode", "-")))

    print("%d constraint epochs, seed %d, capped class 1, K=%d"
          % (args.constraint_epochs, args.seed, K))
    print()
    print("%-16s %-8s %-14s %10s %10s" % ("arm", "count", "md5", "sum_p(c1)", "hard(c1)"))
    print("-" * 64)
    for arm, h, sp, hard, mode in rows:
        print("%-16s %-8s %-14s %10.3f %10d" % (arm, mode, h, sp, hard))
    print()

    dupes = []
    for i in range(len(rows)):
        for j in range(i + 1, len(rows)):
            if rows[i][1] == rows[j][1]:
                dupes.append("%s == %s" % (rows[i][0], rows[j][0]))
    if dupes:
        print("INERT -- bit-identical predictions: %s" % "; ".join(dupes))
        print("Whatever differs between those arms in the config is not")
        print("reaching the model. Do not launch a campaign on it.")
        return 1
    print("Every arm produced different predictions, so every flag is connected.")
    print("That is ALL this says. It is n=1 on random labels: a hard count")
    print("driven far BELOW K here is the over-dose failure mode (the joint arm")
    print("held the cap 98.8% of epochs and lost 0.067 AP doing it), not a win.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
