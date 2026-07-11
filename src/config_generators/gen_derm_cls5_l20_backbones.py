"""Derm cls 5 L20 × 4 backbones — mirror of gen_aider_cls3_l20_backbones."""
from src.config_generators.generate_configs import (
    compute_base_model_id, save_configs,
)

BACKBONES = ["MobileNetV3", "MobileNetV2", "RegNetY400MF", "ShuffleNetV2"]
SEEDS = [1, 2, 3, 4, 5]
METHODS = ["tralo", "fioretto_ldf", "hounie_rcl", "danits_lp", "heuristic"]
TIGHT = "L20_G20"

SHARED_HP = {
    "lr": 1e-4, "lr_constraint": 5e-6, "dropout": 0.3, "batch_size": 64,
    "warmup_epochs": 50, "constraint_epochs": 300,
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


def _pair(tag):
    p = tag.split("_")
    return (int(p[0][1:]) / 100, int(p[1][1:]) / 100)


def make_cfg(backbone, seed, method):
    ds_config = {
        "num_classes": 7, "image_size": 224, "target_column": "label",
        "group_column": "loc_group", "constrained_class": 5,
        "data_dir": "data/dermmnist/slice_1",
    }
    hp = {**SHARED_HP, "seed": seed}
    if method == "tralo":
        hp.update(TRALO_HP)
    pair = _pair(TIGHT)
    bmid = compute_base_model_id(
        backbone, hp, dataset_mode="dermmnist", data_dir="data/dermmnist/slice_1",
        dataset_config=ds_config,
    )
    return {
        "methodology": method, "model_name": backbone,
        "constraint": list(pair), "constraint_tag": TIGHT,
        "dataset_mode": "dermmnist", "dataset_config": ds_config,
        "hyperparams": hp, "base_model_id": bmid,
        "experiment_path": (
            f"results/pending_runs/derm_cls5_l20_backbones/"
            f"{backbone}/{method}/seed_{seed}"
        ),
    }


def main():
    cfgs = [make_cfg(b, s, m) for b in BACKBONES for s in SEEDS for m in METHODS]
    print(f"Generated {len(cfgs)} configs "
          f"({len(BACKBONES)} backbones x {len(SEEDS)} seeds x {len(METHODS)} methods)")
    save_configs(cfgs, output_dir="results/pending_runs")


if __name__ == "__main__":
    main()
