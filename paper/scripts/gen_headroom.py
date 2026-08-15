"""Compute-matched headroom probe. RUN ON THE SERVER from ~/OptimizationLoss.

The paper's comparisons all sit at warm-up 50, where the CE-saturation gate has
already fired: nothing is learning during the constraint phase, so every method
can only re-threshold a frozen score vector. Optimal re-thresholding is exactly
what the post-hoc clipper does, which is why the clipper wins that regime and
why the budget-equalized control found no constrained-class advantage there.

The open question is the other regime. At short warm-up the representation is
still plastic and the constraint term can shape what gets learned. The corpus
already hints at this -- TraLO's advantage over the clippers is large at warm-up
1 -- but that hint is worthless as evidence, because at warm-up 1 the clipper
trained for ONE epoch while TraLO trained for about twenty-six. That gap
measures compute, not objectives.

So this campaign fixes total optimizer epochs across every arm:

    post-hoc arms   warmup_epochs = B                (no constraint phase)
    trained arms    warmup_epochs = 1, constraint_epochs = B - 1

Identical epoch budget, identical data, identical warm-up cache, only the
objective differs. The metric that decides it is average precision on the
constrained class: allocation-free, so no amount of quota-filling can move it,
which means a clipper cannot manufacture the result the way it manufactured the
tight-cap cc-F1 lead.

TraLO early-stops on five consecutive satisfied epochs, so it will often spend
LESS than B. That is left in deliberately and the realized epoch count is
logged: winning on less compute is a stronger claim than winning on equal
compute, and hiding it would repeat the mistake this campaign exists to fix.

    python paper/scripts/gen_headroom.py [--budget 30] [--lanes 4]
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.abspath("."))
from src.config_generators.generate_configs import compute_base_model_id  # noqa: E402

DS_CFG = {
    "tissuemnist": {"data_dir": "data/tissuemnist/slice_1", "num_classes": 8,
                    "image_size": 224, "target_column": "label",
                    "group_column": "synth_group", "constrained_class": 4},
    "dermmnist": {"data_dir": "data/dermmnist/slice_1", "num_classes": 7,
                  "image_size": 224, "target_column": "label",
                  "group_column": "loc_group", "constrained_class": 4},
    "octmnist": {"data_dir": "data/octmnist/slice_1", "num_classes": 4,
                 "image_size": 224, "target_column": "label",
                 "group_column": "synth_group", "constrained_class": 2},
}
SHARED_HP = {"lr": 1e-4, "dropout": 0.3, "batch_size": 64,
             "pretrained": True, "class_weighted_ce": False,
             "constraint_chunk_size": 256}

# `lr_constraint` is set per invocation, and getting it wrong invalidates the
# whole comparison. The warm-up phase optimizes at `lr` (1e-4); the constraint
# phase optimizes at `lr_constraint`. The project default is 5e-6 -- twenty
# times smaller -- which is correct for the warm-up-50 regime, where the model
# has already converged and the constraint phase should only nudge it.
#
# In the SHORT WARM-UP regime that default silently ruins the experiment. The
# post-hoc arms run `warmup_epochs=B` entirely at 1e-4 (they have no constraint
# phase at all, `constraint_epochs=0`, so `lr_constraint` never applies to
# them), while the trained arms run one epoch at 1e-4 and then B-1 epochs at
# 5e-6. That is not a comparison of objectives, it is a comparison of learning
# rates, and it makes the trained arms look catastrophically worse for reasons
# that have nothing to do with the constraint. A first pass at B=30 produced
# DermMNIST macro-F1 of 0.43 for the trained arms against 0.72 for plain CE
# purely from this.
#
# To compare objectives, `lr_constraint` must equal `lr`. It is NOT part of the
# warm-up cache key (`compute_base_model_id` hashes `lr` but not
# `lr_constraint`), so changing it reuses the cached warm-ups and only the
# constraint phase is recomputed.
DEFAULT_LR_CONSTRAINT = 1e-4

TRALO = {"lambda_global": 0.05, "lambda_local": 0.05, "lambda_step": 0.002,
         "initial_rho": 5.0, "rho_target": 100.0, "alpha_kl": 0.0,
         "penalty_mode": "both", "enable_ce_skip": True,
         "hybrid_mode": "undershoot_hinge", "fior_beta": 0.50,
         "reset_optimizer_at_sat": True}

# Post-hoc arms get the whole budget as warm-up; trained arms split it.
POSTHOC = {"heuristic": {}, "danits_lp": {}}
TRAINED = {
    "tralo": TRALO,
    "fioretto_ldf": {"fioretto_step_size": 0.005},
    "hounie_rcl": {"hounie_eta_lambda": 0.01, "hounie_eta_u": 0.01,
                   "hounie_alpha": 10.0},
}

SEEDS = [1, 2, 3, 4]
BACKBONES = ["MobileNetV3", "RegNetY400MF"]
CAPS = ["L30_G30", "L50_G50"]
DATASETS = ["octmnist", "dermmnist", "tissuemnist"]


def pair(tag):
    p = tag.split("_")
    return [int(p[0][1:]) / 100, int(p[1][1:]) / 100]


def cells(budget, trained_only=False):
    for ds in DATASETS:
        for model in BACKBONES:
            for tag in CAPS:
                if not trained_only:
                    for method, extra in POSTHOC.items():
                        for seed in SEEDS:
                            yield (ds, model, tag, method, seed, extra,
                                   {"warmup_epochs": budget,
                                    "constraint_epochs": 0})
                for method, extra in TRAINED.items():
                    for seed in SEEDS:
                        yield (ds, model, tag, method, seed, extra,
                               {"warmup_epochs": 1,
                                "constraint_epochs": budget - 1})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=int, default=30,
                    help="total optimizer epochs, identical for every arm")
    ap.add_argument("--lanes", type=int, default=4)
    ap.add_argument("--root", default="results/headroom")
    ap.add_argument("--lr-constraint", type=float, default=DEFAULT_LR_CONSTRAINT,
                    help="constraint-phase LR; must equal --lr to compare "
                         "objectives rather than learning rates")
    ap.add_argument("--trained-only", action="store_true",
                    help="skip the post-hoc arms; their runs are unaffected by "
                         "lr_constraint (constraint_epochs=0) so they never "
                         "need regenerating")
    ap.add_argument("--no-ce-skip", action="store_true",
                    help="disable the CE-saturation gate. The gate stops the "
                         "cross-entropy term once train accuracy passes 0.995, "
                         "after which the ONLY objective is the count penalty "
                         "and nothing anchors the representation. Forensics "
                         "measured it firing at constraint epoch 4-5 in "
                         "93-99.7%% of warm-up-50 TraLO runs, leaving L_CE "
                         "identically zero for 85-95%% of the constraint phase, "
                         "and showed that disabling it removes essentially all "
                         "of the resulting AP damage (derm -0.040 -> +0.001, "
                         "tissue -0.006 -> +0.001). It also fires in the "
                         "plastic regime: 39 of 79 corrected runs have zero-CE "
                         "epochs. Since the whole hypothesis is that the "
                         "constraint shapes the boundary WHILE CE is still "
                         "active, leaving the gate on tests the opposite of "
                         "what is claimed.")
    args = ap.parse_args()

    camp = "headroom_b%d_lrc%g%s" % (args.budget, args.lr_constraint,
                                     "_noceskip" if args.no_ce_skip else "")
    todo = list(cells(args.budget, args.trained_only))
    # Interleave by method so the slow arms spread across lanes instead of
    # piling into one and leaving three GPUs idle at the end.
    todo.sort(key=lambda c: (c[4], c[3], c[0], c[1], c[2]))

    counts = [0] * args.lanes
    per_method = {}
    for i, (ds, model, tag, method, seed, extra, ep) in enumerate(todo):
        lane = i % args.lanes
        dc = DS_CFG[ds]
        hp = {**SHARED_HP, **extra, **ep, "seed": seed,
              "lr_constraint": args.lr_constraint}
        if args.no_ce_skip and "enable_ce_skip" in hp:
            hp["enable_ce_skip"] = False
        bmid = compute_base_model_id(model, hp, ds, dc["data_dir"], dc)
        path = ("%s/%s/lane%d/%s/%s/%s/%s/seed_%d"
                % (args.root, camp, lane, model, ds, tag, method, seed))
        cfg = {"methodology": method, "model_name": model,
               "constraint": pair(tag), "constraint_tag": tag,
               "dataset_mode": ds, "dataset_config": dc, "hyperparams": hp,
               "base_model_id": bmid, "sweep_tag": camp, "arm": method,
               "epoch_budget": args.budget,
               "exp_name": "%s_%s_%s_%s_%s_seed%d" % (camp, model, ds, method,
                                                      tag, seed),
               "experiment_path": path, "status": "pending"}
        os.makedirs(path, exist_ok=True)
        json.dump(cfg, open(path + "/config.json", "w"), indent=4)
        counts[lane] += 1
        per_method[method] = per_method.get(method, 0) + 1

    print("wrote %d configs -> %s/%s/lane{0..%d}   (budget = %d epochs/arm)"
          % (sum(counts), args.root, camp, args.lanes - 1, args.budget))
    for k, v in sorted(per_method.items()):
        print("   %-14s %4d" % (k, v))
    print("per-lane: %s" % counts)


if __name__ == "__main__":
    main()
