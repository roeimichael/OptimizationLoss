"""Scale octmnist MobileNetV2 L50 cls 2 warmup=1 push-pull cell to 8 seeds.

Adds seeds 4-8 to push the n=3 paired-t into n=8 territory.
"""
from src.config_generators.generate_configs import (
    compute_base_model_id, save_configs,
)

SEEDS = [4, 5, 6, 7, 8]
METHODS = ["tralo", "fioretto_ldf", "hounie_rcl", "danits_lp", "heuristic"]
BACKBONE = "MobileNetV2"
TIGHT = "L50_G50"

SHARED_HP = {
    "lr": 1e-4, "lr_constraint": 5e-6, "dropout": 0.3, "batch_size": 64,
    "warmup_epochs": 1, "constraint_epochs": 100,
    "pretrained": True,
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


def main():
    cfgs = []
    ds_cfg = {
        "num_classes": 4, "image_size": 224, "target_column": "label",
        "group_column": "synth_group", "constrained_class": 2,
        "data_dir": "data/octmnist/slice_1",
    }
    for seed in SEEDS:
        for method in METHODS:
            hp = {**SHARED_HP, "seed": seed}
            if method == "tralo":
                hp.update(TRALO_HP)
            lo, hi = 0.5, 0.5
            bmid = compute_base_model_id(
                BACKBONE, hp, dataset_mode="octmnist",
                data_dir=ds_cfg["data_dir"], dataset_config=ds_cfg,
            )
            cfgs.append({
                "methodology": method, "model_name": BACKBONE,
                "constraint": [lo, hi], "constraint_tag": TIGHT,
                "dataset_mode": "octmnist", "dataset_config": ds_cfg,
                "hyperparams": hp, "base_model_id": bmid,
                "experiment_path": (
                    f"results/pending_runs/pushpull_octmnist_w1/{BACKBONE}/"
                    f"octmnist/{TIGHT}/{method}/seed_{seed}"
                ),
            })
    print(f"Generated {len(cfgs)} configs (seeds 4-8 x 5 methods)")
    save_configs(cfgs, output_dir="results/pending_runs")


if __name__ == "__main__":
    main()
