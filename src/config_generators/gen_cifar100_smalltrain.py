"""CIFAR-100 small-train probe.

Tests the 'limit training data quantity' headroom hypothesis. Subsampled
train sets prevent the model from memorizing the entire train set even
with full warmup -> CE gradient stays alive -> TraLO has room.

Single arm to start: subset_50 (50 samples/class = 5000 total). If train_acc
still saturates above ~0.82, follow up with subset_20.

12 cells: 3 methods x 2 seeds x 2 tightness. constraint_epochs=100 (capped).
"""
from src.config_generators.generate_configs import (
    compute_base_model_id, save_configs,
)

ARMS = {
    "subset10": {"data_dir": "data/cifar100_subset10/slice_1"},
    "subset5":  {"data_dir": "data/cifar100_subset5/slice_1"},
}
TIGHT = ["L30_G30", "L50_G50"]
SEEDS = [1, 2]
METHODS = ["tralo", "fioretto_ldf", "hounie_rcl"]
MODEL = "MobileNetV3"

SHARED_HP = {
    "lr": 1e-4, "lr_constraint": 5e-6, "dropout": 0.3, "batch_size": 64,
    "warmup_epochs": 50, "constraint_epochs": 100, "pretrained": True,
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
    hp = {**SHARED_HP, "seed": seed}
    if method == "tralo":
        hp.update(TRALO_HP)
    pair = _pair(tight)
    bmid = compute_base_model_id(
        MODEL, hp, dataset_mode="cifar100",
        data_dir=arm_opts["data_dir"], dataset_config=ds_config,
    )
    sweep_root = "results/pending_runs/cifar100_smalltrain"
    return {
        "methodology": method, "model_name": MODEL,
        "constraint": list(pair), "constraint_tag": tight,
        "dataset_mode": "cifar100", "dataset_config": ds_config,
        "hyperparams": hp, "base_model_id": bmid,
        "experiment_path": (
            f"{sweep_root}/{arm}/{MODEL}/{tight}/{method}/seed_{seed}"
        ),
    }


def main():
    cfgs = []
    for arm, arm_opts in ARMS.items():
        for tight in TIGHT:
            for method in METHODS:
                for seed in SEEDS:
                    cfgs.append(make_cfg(arm, arm_opts, tight, method, seed))
    print(f"Generated {len(cfgs)} configs")
    save_configs(cfgs, output_dir="results/pending_runs")


if __name__ == "__main__":
    main()
