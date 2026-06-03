"""Probes for 3 candidate new datasets: RetinaMNIST, BloodMNIST, CIFAR-100.

Tests TraLO universal claim on datasets OUTSIDE our active 3. Same backbone
(MobileNetV3), same HP as the headline grid. 3 methods, 2 seeds, 2 tightness.

Per dataset:
  3 methods x 2 seeds x 2 tight = 12 cells
Total: 36 cells across the 3 candidate datasets.

We pick a constrained class with low support (per dataset class distribution
inspected at prep time).
"""
from src.config_generators.generate_configs import (
    compute_base_model_id, save_configs,
)

TIGHT = ["L30_G30", "L50_G50"]
SEEDS = [1, 2]
METHODS = ["tralo", "fioretto_ldf", "hounie_rcl"]
MODEL = "MobileNetV3"

# Per-dataset config. constrained_class chosen for meaningful binding.
DATASETS = {
    "retinamnist":  {"num_classes": 5,   "group_column": "synth_group",
                      "constrained_class": 4,  # 5.0% support, smallest
                      "data_dir": "data/retinamnist/slice_1"},
    "bloodmnist":   {"num_classes": 8,   "group_column": "synth_group",
                      "constrained_class": 4,  # 7.1% support, smallest
                      "data_dir": "data/bloodmnist/slice_1"},
    "cifar100":     {"num_classes": 100, "group_column": "synth_group",
                      "constrained_class": 0,  # balanced 1% each; pick any rare
                      "data_dir": "data/cifar100/slice_1"},
}

SHARED_HP = {
    "lr": 1e-4, "lr_constraint": 5e-6, "dropout": 0.3, "batch_size": 64,
    "warmup_epochs": 50, "constraint_epochs": 300, "pretrained": True,
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


def _tight_pair(tag):
    p = tag.split("_")
    return (int(p[0][1:])/100, int(p[1][1:])/100)


def make_cfg(dataset, ds_base, tight, method, seed):
    ds_config = {
        **ds_base, "image_size": 224, "target_column": "label",
    }
    hp = {**SHARED_HP, "seed": seed}
    if method == "tralo":
        hp.update(TRALO_HP)
    pair = _tight_pair(tight)
    bmid = compute_base_model_id(
        MODEL, hp, dataset_mode=dataset, data_dir=ds_base["data_dir"],
        dataset_config=ds_config,
    )
    sweep_root = "results/pending_runs/new_dataset_probes"
    return {
        "methodology": method, "model_name": MODEL,
        "constraint": list(pair), "constraint_tag": tight,
        "dataset_mode": dataset, "dataset_config": ds_config,
        "hyperparams": hp, "base_model_id": bmid,
        "experiment_path": (
            f"{sweep_root}/{dataset}/{MODEL}/{tight}/{method}/seed_{seed}"
        ),
    }


def main():
    cfgs = []
    for dataset, ds_base in DATASETS.items():
        for tight in TIGHT:
            for method in METHODS:
                for seed in SEEDS:
                    cfgs.append(make_cfg(dataset, ds_base, tight, method, seed))
    print(f"Generated {len(cfgs)} configs")
    save_configs(cfgs, output_dir="results/pending_runs")


if __name__ == "__main__":
    main()
