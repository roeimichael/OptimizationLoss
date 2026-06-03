"""Overnight CIFAR-100 sweep testing TWO operatives that should genuinely
prevent train_acc=1.0 saturation:

  Arm A: pretrained=False on full CIFAR-100 (no ImageNet head start)
         -> train_acc should plateau ~0.65-0.75 in 50 warmup epochs

  Arm B: CIFAR-100N (Wei et al. NeurIPS 2022, ~40% real human-noisy labels)
         -> train_acc mathematically bounded by ~0.60 (can't fit
            contradictory labels)

Each arm: 3 methods x 2 seeds x 2 tightness = 12 cells. 24 total.

constraint_epochs capped at 100 to control wall-time. Predicted result
under refined headroom theory: TraLO should win Hounie in both arms IF
train_acc stays below 0.995 throughout training.
"""
from src.config_generators.generate_configs import (
    compute_base_model_id, save_configs,
)

ARMS = {
    "pretrained_off": {
        "data_dir": "data/cifar100/slice_1",
        "dataset_mode": "cifar100",
        "pretrained": False,
    },
    "noisy_labels": {
        "data_dir": "data/cifar100n/slice_1",
        "dataset_mode": "cifar100n",
        "pretrained": True,  # standard backbone; noise comes from labels
    },
}
TIGHT = ["L30_G30", "L50_G50"]
SEEDS = [1, 2]
METHODS = ["tralo", "fioretto_ldf", "hounie_rcl"]
MODEL = "MobileNetV3"

SHARED_HP = {
    "lr": 1e-4, "lr_constraint": 5e-6, "dropout": 0.3, "batch_size": 64,
    "warmup_epochs": 50, "constraint_epochs": 100,
    "class_weighted_ce": False, "constraint_chunk_size": 256,
    "fioretto_step_size": 0.01,
}
TRALO_HP = {
    "lambda_global": 0.05, "lambda_local": 0.05, "lambda_step": 0.002,
    "initial_rho": 5.0, "rho_target": 100.0, "alpha_kl": 0.0,
    "penalty_mode": "both", "enable_ce_skip": True,
    "hybrid_mode": "undershoot_hinge", "fior_beta": 0.50,
    "reset_optimizer_at_sat": True,
    "disable_freeze_on_satisfy": False,
}


def _pair(tag):
    p = tag.split("_")
    return (int(p[0][1:])/100, int(p[1][1:])/100)


def make_cfg(arm, arm_opts, tight, method, seed):
    ds_config = {
        "num_classes": 100, "image_size": 224, "target_column": "label",
        "group_column": "synth_group", "constrained_class": 0,
        "data_dir": arm_opts["data_dir"],
    }
    hp = {**SHARED_HP, "seed": seed, "pretrained": arm_opts["pretrained"]}
    if method == "tralo":
        hp.update(TRALO_HP)
    pair = _pair(tight)
    bmid = compute_base_model_id(
        MODEL, hp, dataset_mode=arm_opts["dataset_mode"],
        data_dir=arm_opts["data_dir"], dataset_config=ds_config,
    )
    return {
        "methodology": method, "model_name": MODEL,
        "constraint": list(pair), "constraint_tag": tight,
        "dataset_mode": arm_opts["dataset_mode"], "dataset_config": ds_config,
        "hyperparams": hp, "base_model_id": bmid,
        "experiment_path": (
            f"results/pending_runs/cifar100_overnight/{arm}/{MODEL}/{tight}/{method}/seed_{seed}"
        ),
    }


def main():
    cfgs = []
    for arm, arm_opts in ARMS.items():
        for tight in TIGHT:
            for method in METHODS:
                for seed in SEEDS:
                    cfgs.append(make_cfg(arm, arm_opts, tight, method, seed))
    print(f"Generated {len(cfgs)} configs (2 arms x 12 cells)")
    save_configs(cfgs, output_dir="results/pending_runs")


if __name__ == "__main__":
    main()
