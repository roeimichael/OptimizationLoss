"""Generate the round-4 campaigns. RUN ON THE SERVER from ~/OptimizationLoss.

Three campaigns, each written into its own root so nothing touches the frozen
grid and so every comparison inside a campaign is within-campaign by
construction:

  r4_ablation  96 runs -- the component ablation with ALL FOUR arms run together.
      The published hinge row was paired across campaigns against the frozen
      grid, where a same-seed rerun moves cc-F1 by 0.027 -- close to the effect
      it reports. Running full / -reset / -hinge / -both side by side, from the
      same warm-up caches, gives the hinge row the within-campaign control the
      reset row already had.

  r4_mnv2     280 runs -- MobileNetV2 at the cap levels it is missing, so the
      fourth backbone stops being a partial column: OctMNIST L10/L40/L60/L90 and
      the mid caps L40/L50/L60 on DermMNIST and TissueMNIST, all seven methods.

  r4_almprobe  12 runs -- the warm-up-1 mechanism probe with ALM added, so the
      dual-escalation figure covers the augmented-Lagrangian rule too.

    python paper/scripts/gen_r4.py [--lanes 4]
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
SHARED_HP = {"lr": 1e-4, "lr_constraint": 5e-6, "dropout": 0.3, "batch_size": 64,
             "warmup_epochs": 50, "constraint_epochs": 300, "pretrained": True,
             "class_weighted_ce": False, "constraint_chunk_size": 256}

TRALO = {"lambda_global": 0.05, "lambda_local": 0.05, "lambda_step": 0.002,
         "initial_rho": 5.0, "rho_target": 100.0, "alpha_kl": 0.0,
         "penalty_mode": "both", "enable_ce_skip": True,
         "hybrid_mode": "undershoot_hinge", "fior_beta": 0.50,
         "reset_optimizer_at_sat": True}
PER_METHOD = {
    "heuristic": {}, "danits_lp": {},
    "fioretto_ldf": {"fioretto_step_size": 0.005},
    "hounie_rcl": {"hounie_eta_lambda": 0.01, "hounie_eta_u": 0.01,
                   "hounie_alpha": 10.0},
    "fioretto_alm": {"alm_eta": 0.005, "alm_mu0": 0.01, "alm_mu_step": 0.01},
    "tralo_bounded": {"lambda_global": 0.05, "lambda_local": 0.05,
                      "lambda_step": 0.002, "initial_rho": 5.0,
                      "rho_target": 100.0, "alpha_kl": 0.0,
                      "penalty_mode": "both", "enable_ce_skip": True},
    "tralo": TRALO,
}

# The four ablation arms are all methodology `tralo`; only the two knobs move.
# They are named in the PATH, because `methodology` cannot distinguish them --
# which is exactly how an earlier ablation campaign shipped with flags that had
# never taken effect and nobody noticed.
ABLATION_ARMS = {
    "full":     dict(TRALO),
    "noreset":  {**TRALO, "reset_optimizer_at_sat": False},
    "nohinge":  {**TRALO, "hybrid_mode": "bounded_only", "fior_beta": 0.0},
    "neither":  {**TRALO, "reset_optimizer_at_sat": False,
                 "hybrid_mode": "bounded_only", "fior_beta": 0.0},
}

SEEDS = [1, 2, 3, 4]
BB3 = ["MobileNetV3", "RegNetY400MF", "ViTB16"]
ALL_METHODS = list(PER_METHOD)


def pair(tag):
    p = tag.split("_")
    return [int(p[0][1:]) / 100, int(p[1][1:]) / 100]


def cells_ablation():
    for arm, hp_extra in ABLATION_ARMS.items():
        for model in BB3:
            for tag in ["L30_G30", "L40_G40"]:
                for seed in SEEDS:
                    yield ("r4_ablation", model, "octmnist", tag, "tralo", seed,
                           hp_extra, arm, {})


def cells_mnv2():
    plan = [("octmnist", ["L10_G10", "L40_G40", "L60_G60", "L90_G90"]),
            ("dermmnist", ["L40_G40", "L50_G50", "L60_G60"]),
            ("tissuemnist", ["L40_G40", "L50_G50", "L60_G60"])]
    for ds, tags in plan:
        for tag in tags:
            for method in ALL_METHODS:
                for seed in SEEDS:
                    yield ("r4_mnv2", "MobileNetV2", ds, tag, method, seed,
                           PER_METHOD[method], method, {})


def cells_almprobe():
    # Warm-up 1 keeps the classifier learning through the constraint phase, so
    # the two objectives visibly compete -- the regime the mechanism figure is
    # about. Same cell as the existing probe, with ALM added.
    for method in ["tralo", "fioretto_ldf", "fioretto_alm"]:
        for seed in SEEDS:
            yield ("r4_almprobe", "RegNetY400MF", "dermmnist", "L30_G30", method,
                   seed, PER_METHOD[method], method, {"warmup_epochs": 1})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lanes", type=int, default=4)
    ap.add_argument("--root", default="results/r4")
    args = ap.parse_args()

    cells = list(cells_ablation()) + list(cells_mnv2()) + list(cells_almprobe())
    # Interleave by method then seed so the slow arms spread over the lanes
    # instead of piling into one.
    cells.sort(key=lambda c: (c[4], c[5], c[1], c[2], c[3]))

    counts, per_campaign = [0] * args.lanes, {}
    for i, (camp, model, ds, tag, method, seed, hp_extra, arm, hp_over) in enumerate(cells):
        lane = i % args.lanes
        dc = DS_CFG[ds]
        hp = {**SHARED_HP, **hp_extra, **hp_over, "seed": seed}
        bmid = compute_base_model_id(model, hp, ds, dc["data_dir"], dc)
        ep = f"{args.root}/{camp}/lane{lane}/{model}/{ds}/{tag}/{arm}/seed_{seed}"
        cfg = {"methodology": method, "model_name": model,
               "constraint": pair(tag), "constraint_tag": tag,
               "dataset_mode": ds, "dataset_config": dc, "hyperparams": hp,
               "base_model_id": bmid, "sweep_tag": camp, "arm": arm,
               "exp_name": f"{camp}_{model}_{ds}_{arm}_{tag}_seed{seed}",
               "experiment_path": ep, "status": "pending"}
        os.makedirs(ep, exist_ok=True)
        json.dump(cfg, open(f"{ep}/config.json", "w"), indent=4)
        counts[lane] += 1
        per_campaign[camp] = per_campaign.get(camp, 0) + 1

    print("wrote %d configs -> %s/{%s}/lane{0..%d}"
          % (sum(counts), args.root, ",".join(per_campaign), args.lanes - 1))
    for k, v in per_campaign.items():
        print("   %-14s %4d" % (k, v))
    print("per-lane: %s" % counts)


if __name__ == "__main__":
    main()
