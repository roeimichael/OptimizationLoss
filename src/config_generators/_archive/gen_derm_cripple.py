"""DermMNIST cripple-training experiment with medical-style corruptions.

Tests the headroom hypothesis across multiple corruption modalities and
tightness levels. The AIDER cripple experiment (gen_aider_cripple.py)
showed that heavy Gaussian noise flipped the in-training-vs-post-hoc
ranking. This experiment extends to (a) clinically-realistic corruption
types and (b) multiple constraint tightness levels.

Corruption types (both train AND test corrupted):
  noise: Gaussian sensor noise (sigma=0.15)
  blur:  Diagonal motion blur (kernel 11px) - hand-held dermoscopy
  jpeg:  JPEG quality=15 - telemedicine transmission artifact
  color: HSV jitter (hue=0.15, sat=0.4, bri=0.25) - scanner/lighting variation

Grid:
  4 corruption x 3 tightness (L20, L30, L50) x 4 methods x 2 seeds = 96 cells.

Output: results/pending_runs/derm_cripple/
"""
from pathlib import Path

from src.config_generators.generate_configs import (
    compute_base_model_id, save_configs,
)

SWEEP_ROOT = "results/pending_runs/derm_cripple"

CORRUPTIONS = {
    "noise": "data/dermmnist_noise/slice_1",
    "blur":  "data/dermmnist_blur/slice_1",
    "jpeg":  "data/dermmnist_jpeg/slice_1",
    "color": "data/dermmnist_color/slice_1",
}

TIGHTNESS = ["L20_G20", "L30_G30", "L50_G50"]
SEEDS = [1, 2]
MODEL = "MobileNetV3"
CLS = 4  # MEL, matches DermMNIST headline
METHODS = ["tralo", "fioretto_ldf", "danits_lp", "heuristic"]

DATASET_BASE = {
    "num_classes": 7, "image_size": 224, "target_column": "label",
    "group_column": "loc_group", "constrained_class": CLS,
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
    parts = tag.split("_")
    return (int(parts[0][1:]) / 100, int(parts[1][1:]) / 100)


def make_cfg(corr_name, data_dir, method, tight, seed):
    ds_config = {**DATASET_BASE, "data_dir": data_dir}
    hp = {**SHARED_HP, "seed": seed}
    if method == "tralo":
        hp.update(TRALO_HP)
    pair = _tight_pair(tight)
    bmid = compute_base_model_id(
        MODEL, hp, dataset_mode="dermmnist",
        data_dir=data_dir, dataset_config=ds_config,
    )
    return {
        "methodology": method,
        "model_name": MODEL,
        "constraint": list(pair),
        "constraint_tag": tight,
        "dataset_mode": "dermmnist",
        "dataset_config": ds_config,
        "hyperparams": hp,
        "base_model_id": bmid,
        "exp_name": f"derm_{corr_name}_{tight}_{method}_seed{seed}",
        "experiment_path": str(
            Path(SWEEP_ROOT) / corr_name / tight / method / f"seed_{seed}"),
    }


def build():
    cfgs = []
    for corr_name, data_dir in CORRUPTIONS.items():
        for tight in TIGHTNESS:
            for method in METHODS:
                for seed in SEEDS:
                    cfgs.append(make_cfg(corr_name, data_dir, method, tight, seed))
    save_configs(cfgs, output_dir=SWEEP_ROOT)
    print(f"\nGenerated {len(cfgs)} derm-cripple configs -> {SWEEP_ROOT}")


if __name__ == "__main__":
    build()
