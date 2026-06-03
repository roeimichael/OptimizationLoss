"""OctMNIST expansion: confirm the smoke probe win with full method panel +
more seeds + asymmetric tightness coverage.

  5 methods (tralo, fioretto_ldf, hounie_rcl, danits_lp, heuristic)
  4 seeds (1..4)
  3 tightness levels (L30 sym, L50 sym, L30_G50 asym)
= 60 cells

Constrained class: c2 (drusen) — same low-binding regime as smoke.
Backbone: MobileNetV3 (smoke winner).
"""
from src.config_generators.generate_configs import (
    compute_base_model_id, save_configs,
)

DATASET = "octmnist"
DATA_DIR = "data/octmnist/slice_1"
NUM_CLASSES = 4
CONSTRAINED_CLASS = 2
GROUP_COLUMN = "synth_group"

TIGHT = ["L30_G30", "L50_G50", "L30_G50"]
SEEDS = [1, 2, 3, 4]
METHODS = ["tralo", "fioretto_ldf", "hounie_rcl", "danits_lp", "heuristic"]
MODEL = "MobileNetV3"

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


def _pair(tag):
    p = tag.split("_")
    return (int(p[0][1:])/100, int(p[1][1:])/100)


def make_cfg(tight, method, seed):
    ds_config = {
        "num_classes": NUM_CLASSES, "image_size": 224, "target_column": "label",
        "group_column": GROUP_COLUMN, "constrained_class": CONSTRAINED_CLASS,
        "data_dir": DATA_DIR,
    }
    hp = {**SHARED_HP, "seed": seed}
    if method == "tralo":
        hp.update(TRALO_HP)
    pair = _pair(tight)
    bmid = compute_base_model_id(
        MODEL, hp, dataset_mode=DATASET, data_dir=DATA_DIR,
        dataset_config=ds_config,
    )
    return {
        "methodology": method, "model_name": MODEL,
        "constraint": list(pair), "constraint_tag": tight,
        "dataset_mode": DATASET, "dataset_config": ds_config,
        "hyperparams": hp, "base_model_id": bmid,
        "experiment_path": (
            f"results/pending_runs/octmnist_expansion/{MODEL}/{tight}/{method}/seed_{seed}"
        ),
    }


def main():
    cfgs = []
    for tight in TIGHT:
        for method in METHODS:
            for seed in SEEDS:
                cfgs.append(make_cfg(tight, method, seed))
    print(f"Generated {len(cfgs)} configs")
    save_configs(cfgs, output_dir="results/pending_runs")


if __name__ == "__main__":
    main()
