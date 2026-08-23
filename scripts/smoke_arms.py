"""Actually RUN every arm, on tiny synthetic tensors, before launching anything.

`audit_config` proves the config keys line up and `check_parity` proves the
campaign is a fair comparison -- but both read configs, and neither executes a
single line of a methodology. Three of the ten arms once shipped with an
undefined name in `train()`: they burned all 29 constraint epochs, raised
`NameError`, were reset to `pending` by the runner, and came back looking like
"still queued". Both audits passed on that campaign.

This calls each registered `train_fn` end to end with a 4-layer model and ~120
synthetic items, in a few seconds, on CPU. It does not check that an arm is
CORRECT -- only that it runs, returns the contract, and respects its caps.

    python -m scripts.smoke_arms            # every arm
    python -m scripts.smoke_arms tralo lp   # named arms only

Exit code 1 if any arm fails, so it can gate a launch.
"""
import argparse
import os
import shutil
import sys
import tempfile
import traceback
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs.gen_campaign import (build_hyperparams, cap_pair,      # noqa: E402
                                  load_protocol)
from src.experiments.runner import TRAIN_FNS                         # noqa: E402
from src.pipeline.contracts import TrainInputs                       # noqa: E402
from src.training.constraints import (compute_global_constraints,    # noqa: E402
                                      compute_local_constraints)
from src.utils.constants import UNLIMITED                            # noqa: E402

N_TEST, N_TRAIN, N_CLASSES, N_GROUPS, SIDE = 120, 96, 4, 3, 8


class TinyNet(nn.Module):
    """Small enough to train in a second, real enough to have conv gradients."""

    def __init__(self, n_classes):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 8, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten())
        self.classifier = nn.Linear(8, n_classes)

    def forward(self, x):
        return self.classifier(self.features(x))


def make_inputs(P, arm, tmp, seed=1):
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    y_test = rng.integers(0, N_CLASSES, size=N_TEST)
    groups = rng.integers(0, N_GROUPS, size=N_TEST)
    # every (group, capped class) cell non-empty, so K=0 is not what we test here
    for g in range(N_GROUPS):
        idx = np.where(groups == g)[0][:2]
        y_test[idx] = np.arange(len(idx)) % N_CLASSES

    import pandas as pd
    df = pd.DataFrame({"label": y_test, "grp": groups})
    local_pct, global_pct = cap_pair("L50_G30")
    gcon = compute_global_constraints(df, "label", global_pct,
                                      constrained_class=[1], num_classes=N_CLASSES)
    lcon = compute_local_constraints(df, "label", local_pct, "grp",
                                     constrained_class=[1], num_classes=N_CLASSES)

    spec = P["arms"][arm]
    hp = build_hyperparams(P, spec, seed)
    hp["warmup_epochs"], hp["constraint_epochs"] = 1, 2
    hp["batch_size"] = 32

    path = Path(tmp) / arm
    path.mkdir(parents=True, exist_ok=True)
    config = {"methodology": spec["methodology"], "model_name": "TinyNet",
              "hyperparams": hp, "dataset_mode": "smoke", "arm": arm,
              "constraint": [local_pct, global_pct],
              "dataset_config": {"num_classes": N_CLASSES, "constrained_class": 1}}
    return TrainInputs(
        model=TinyNet(N_CLASSES),
        X_train=torch.randn(N_TRAIN, 3, SIDE, SIDE),
        y_train=torch.randint(0, N_CLASSES, (N_TRAIN,)),
        X_test=torch.randn(N_TEST, 3, SIDE, SIDE),
        y_test=y_test, group_ids=groups,
        global_con=gcon, local_con=lcon, constrained_classes=[1],
        num_classes=N_CLASSES, config=config, hyperparams=hp,
        device=torch.device("cpu"),
        experiment_path=path, csv_log_path=path / "training_log.csv"), gcon, lcon


def violations(y_pred, groups, gcon, lcon):
    bad = []
    for c in range(N_CLASSES):
        if gcon[c] < UNLIMITED and int((y_pred == c).sum()) > gcon[c]:
            bad.append("global c%d %d>%d" % (c, (y_pred == c).sum(), gcon[c]))
    for g, bounds in lcon.items():
        m = groups == g
        for c in range(N_CLASSES):
            if bounds[c] < UNLIMITED and int((y_pred[m] == c).sum()) > bounds[c]:
                bad.append("local g%s c%d" % (g, c))
    return bad


def main():
    P = load_protocol()
    a = argparse.ArgumentParser()
    a.add_argument("arms", nargs="*", default=None)
    a.add_argument("-v", "--verbose", action="store_true")
    args = a.parse_args()
    arms = args.arms or sorted(P["arms"])

    tmp = tempfile.mkdtemp(prefix="smoke_")
    fails = []
    print("smoke-testing %d arm(s): %d train / %d test items, %d classes, "
          "%d groups\n" % (len(arms), N_TRAIN, N_TEST, N_CLASSES, N_GROUPS))
    for arm in arms:
        meth = P["arms"][arm]["methodology"]
        try:
            inputs, gcon, lcon = make_inputs(P, arm, tmp)
            out = TRAIN_FNS[meth](inputs)
            assert out.model is not None, "TrainOutputs.model is None"
            note = ""
            if out.precomputed_predictions is not None:
                yp = np.asarray(out.precomputed_predictions)
                assert yp.shape == (N_TEST,), "predictions shape %s" % (yp.shape,)
                bad = violations(yp, inputs.group_ids, gcon, lcon)
                if bad:
                    raise AssertionError("emitted predictions violate %s" % bad[:3])
                note = "preds ok, caps satisfied"
            else:
                note = "trained, post-hoc adjustment downstream"
            print("  OK    %-11s -> %-15s %s" % (arm, meth, note))
        except Exception as e:                     # noqa: BLE001 - report them all
            fails.append((arm, meth, e))
            print("  FAIL  %-11s -> %-15s %s: %s"
                  % (arm, meth, type(e).__name__, e))
            if args.verbose:
                traceback.print_exc()
    shutil.rmtree(tmp, ignore_errors=True)

    print()
    if fails:
        print("%d of %d arm(s) CANNOT RUN:" % (len(fails), len(arms)))
        for arm, meth, e in fails:
            print("  %-11s (%s)  %s: %s" % (arm, meth, type(e).__name__, e))
        return 1
    print("All %d arms run end to end and respect their caps." % len(arms))
    return 0


if __name__ == "__main__":
    sys.exit(main())
