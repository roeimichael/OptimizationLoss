"""Tissue low-warmup validation sweep (AAAI headroom test).

Question: does TraLO's F1 advantage GROW at low warmup on TissueMNIST across
backbones (vs the existing warmup=50 grid)? Replicates the derm phase-transition
design on the *winning* dataset, across several backbones.

  tissuemnist x cls 4 x L50_G50
  backbones in {MobileNetV3, MobileNetV2, RegNetY400MF, ResNet18}
  warmup_epochs in {1,2,3,4,5}
  6 methods, 4 seeds
  = 4 x 5 x 6 x 4 = 480 runs

HP mirror gen_phase_transition.py exactly (constraint_epochs=100, same lr/rho/...)
so results are directly comparable to the existing tablef_shortwarm tissue w1 runs
and the warmup=50 headline grid.
"""
from src.config_generators.generate_configs import compute_base_model_id, save_configs

BACKBONES = ["MobileNetV3", "MobileNetV2", "RegNetY400MF", "ResNet18"]
WARMUPS = [1, 2, 3, 4, 5]
SEEDS = [1, 2, 3, 4]
METHODS = ["tralo", "tralo_bounded", "fioretto_ldf",
           "hounie_rcl", "danits_lp", "heuristic"]
ROOT = "results/pending_runs/tissue_lowwarm_validation"

TISSUE_DS = {
    "num_classes": 8, "image_size": 224, "target_column": "label",
    "group_column": "synth_group", "constrained_class": 4,
    "data_dir": "data/tissuemnist/slice_1",
}


def _shared_hp(warmup, seed):
    return {
        "lr": 1e-4, "lr_constraint": 5e-6, "dropout": 0.3, "batch_size": 64,
        "warmup_epochs": warmup, "constraint_epochs": 100,
        "pretrained": True,
        "class_weighted_ce": False, "constraint_chunk_size": 256,
        "fioretto_step_size": 0.01,
        "seed": seed,
    }


def _tralo_hp(bounded=False):
    return {
        "lambda_global": 0.05, "lambda_local": 0.05, "lambda_step": 0.002,
        "initial_rho": 5.0, "rho_target": 100.0, "alpha_kl": 0.0,
        "penalty_mode": "both", "enable_ce_skip": True,
        "hybrid_mode": "off" if bounded else "undershoot_hinge",
        "fior_beta": 0.50,
        "reset_optimizer_at_sat": True,
        "disable_freeze_on_satisfy": False,
    }


def _cfg(backbone, warmup, seed, method):
    hp = _shared_hp(warmup, seed)
    if method == "tralo":
        hp.update(_tralo_hp(bounded=False))
    elif method == "tralo_bounded":
        hp.update(_tralo_hp(bounded=True))
    bmid = compute_base_model_id(
        backbone, hp, dataset_mode="tissuemnist",
        data_dir=TISSUE_DS["data_dir"], dataset_config=TISSUE_DS,
    )
    return {
        "methodology": method, "model_name": backbone,
        "constraint": [0.5, 0.5], "constraint_tag": "L50_G50",
        "dataset_mode": "tissuemnist", "dataset_config": dict(TISSUE_DS),
        "hyperparams": hp, "base_model_id": bmid,
        "experiment_path": (
            f"{ROOT}/{backbone}/tissuemnist/L50_G50/"
            f"w{warmup}/{method}/seed_{seed}"
        ),
    }


def build_configs(backbones=BACKBONES, warmups=WARMUPS, seeds=SEEDS,
                  methods=METHODS):
    cfgs = []
    for b in backbones:
        for w in warmups:
            for s in seeds:
                for m in methods:
                    cfgs.append(_cfg(b, w, s, m))
    return cfgs


def main():
    cfgs = build_configs()
    print(f"Generated {len(cfgs)} configs: "
          f"{len(BACKBONES)} backbones x {len(WARMUPS)} warmups x "
          f"{len(METHODS)} methods x {len(SEEDS)} seeds")
    save_configs(cfgs, output_dir="results/pending_runs")


if __name__ == "__main__":
    main()
