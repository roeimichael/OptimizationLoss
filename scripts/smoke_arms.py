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


def matrix(P, arms):
    """Sweep capped-class count x cap tag over the TRAINED arms, caps verified.

    The single-point smoke above proves an arm RUNS. This proves the thing the
    project actually promises: that the caps hold after the post-hoc correction,
    for EVERY capped class, in the coupled multi-class case, and under a cap tag
    where the global bound actually binds (G < L).
    """
    import pandas as pd
    import torch
    import torch.nn.functional as F

    from src.utils.posthoc_adjustment import targeted_correction

    trained = [a for a in arms if P["arms"][a].get("phase") == "trained"]
    if not trained:
        print("no trained arms selected; --matrix has nothing to do")
        return []
    print("\nMATRIX: %d trained arm(s) x {1, 2} capped classes x "
          "{L30_G30, L50_G30}" % len(trained))
    print("  L50_G30 is the only tag here where the GLOBAL cap binds "
          "(G < L; see FRAMEWORK section 1).\n")

    tmp = tempfile.mkdtemp(prefix="matrix_")
    fails = []
    try:
        for capped in ([1], [1, 2]):
            for tag in ("L30_G30", "L50_G30"):
                for arm in trained:
                    label = "%-9s %-6s %-11s" % (tag, str(capped), arm)
                    try:
                        inputs, _, _ = make_inputs(P, arm, tmp)
                        df = pd.DataFrame({"label": inputs.y_test,
                                           "grp": inputs.group_ids})
                        local_pct, global_pct = cap_pair(tag)
                        gcon = compute_global_constraints(
                            df, "label", global_pct, constrained_class=capped,
                            num_classes=N_CLASSES)
                        lcon = compute_local_constraints(
                            df, "label", local_pct, "grp",
                            constrained_class=capped, num_classes=N_CLASSES)
                        inputs.global_con = gcon
                        inputs.local_con = lcon
                        inputs.constrained_classes = capped
                        inputs.config["dataset_config"]["constrained_class"] = capped

                        out = TRAIN_FNS[P["arms"][arm]["methodology"]](inputs)
                        out.model.eval()
                        with torch.no_grad():
                            proba = F.softmax(out.model(inputs.X_test),
                                              dim=1).cpu().numpy()
                        y_pred = targeted_correction(
                            proba, inputs.group_ids, gcon, lcon, capped)[0]
                        y_pred = np.asarray(y_pred)

                        bad = violations_for(y_pred, inputs.group_ids, gcon,
                                             lcon, capped)
                        if bad:
                            fails.append("%s: %s" % (label.strip(), bad[:3]))
                            print("  FAIL  %s %s" % (label, bad[:3]))
                        else:
                            ks = ", ".join("c%d K=%d" % (c, gcon[c]) for c in capped)
                            print("  OK    %s caps hold  (%s)" % (label, ks))
                    except Exception as exc:
                        fails.append("%s: %s: %s" % (label.strip(),
                                                     type(exc).__name__, exc))
                        print("  FAIL  %s %s: %s"
                              % (label, type(exc).__name__, str(exc)[:70]))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return fails


def violations_for(y_pred, groups, gcon, lcon, classes):
    """Cap check over an EXPLICIT class list.

    `violations` above loops `range(N_CLASSES)`, which is right for the smoke
    harness but hides which capped class failed. Naming the classes is what the
    multi-class case needs -- the scorer's `cls[0]` bug measured class 1 and
    silently ignored every class after it.
    """
    bad = []
    for c in classes:
        if gcon[c] < UNLIMITED and int((y_pred == c).sum()) > int(gcon[c]):
            bad.append("global c%d %d>%d"
                       % (c, int((y_pred == c).sum()), int(gcon[c])))
        for g, bounds in lcon.items():
            m = groups == g
            if bounds[c] < UNLIMITED and int((y_pred[m] == c).sum()) > int(bounds[c]):
                bad.append("local g%s c%d %d>%d"
                           % (g, c, int((y_pred[m] == c).sum()), int(bounds[c])))
    return bad


def main():
    P = load_protocol()
    a = argparse.ArgumentParser()
    a.add_argument("arms", nargs="*", default=None)
    a.add_argument("-v", "--verbose", action="store_true")
    a.add_argument("--matrix", action="store_true",
                   help="also sweep {1,2} capped classes x {L30_G30, L50_G30} "
                        "over the trained arms, verifying caps after the "
                        "post-hoc correction. Run this before any multi-class "
                        "campaign.")
    args = a.parse_args()
    arms = args.arms or sorted(P["arms"])

    tmp = tempfile.mkdtemp(prefix="smoke_")
    fails = []
    unchecked = []
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
                note = "runs; CAPS NOT CHECKED HERE (--matrix does)"
                unchecked.append(arm)
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
    checked = len(arms) - len(unchecked)
    print("All %d arm(s) run end to end." % len(arms))
    print("Caps VERIFIED for %d of %d: the arms that emit predictions directly."
          % (checked, len(arms)))
    if unchecked:
        # This line used to read "All N arms run end to end and respect their
        # caps" unconditionally, while only the post-hoc arms were ever cap
        # checked. The trained arms enforce their caps in targeted_correction,
        # downstream of where this harness stops, so it had no basis for the
        # claim -- in the gate CLAUDE.md tells every user to trust before
        # launching.
        print("Caps NOT verified here for %d trained arm(s): %s"
              % (len(unchecked), " ".join(sorted(unchecked))))
        print("  They enforce caps in targeted_correction, downstream of this")
        print("  harness. Run --matrix to check them.")

    if args.matrix:
        mfails = matrix(P, arms)
        if mfails:
            print("\n%d MATRIX COMBINATION(S) FAILED:" % len(mfails))
            for f in mfails:
                print("  " + f)
            return 1
        print("\nEvery matrix combination satisfies every cap, for every "
              "capped class.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
