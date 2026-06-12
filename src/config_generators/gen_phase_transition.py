"""Phase-transition sweep on the §5.1 headline cell.

dermmnist × MobileNetV3 × cls 4 (MEL) × L50_G50
warmup_epochs ∈ {1, 3, 10, 25}  (50 already in headline data)
6 methods (matches paper Table 1)
4 seeds (matches paper Table 1)
= 96 cells

Plus Gap 1: tissue MobileNetV2 cls 4 L50 warmup=1 × 5 methods × 3 seeds = 15 cells
(derm + aider warmup=1 L50 already exist).
"""
from src.config_generators.generate_configs import compute_base_model_id, save_configs

PHASE_WARMUPS = [1, 3, 10, 25]
PHASE_SEEDS = [1, 2, 3, 4]
PHASE_METHODS = ["tralo", "tralo_bounded", "fioretto_ldf",
                 "hounie_rcl", "danits_lp", "heuristic"]

TABLEF_SEEDS = [1, 2, 3]
TABLEF_METHODS = ["tralo", "fioretto_ldf", "hounie_rcl",
                  "danits_lp", "heuristic"]


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


def _phase_cfg(warmup, seed, method):
    backbone = "MobileNetV3"
    ds_cfg = {
        "num_classes": 7, "image_size": 224, "target_column": "label",
        "group_column": "loc_group", "constrained_class": 4,
        "data_dir": "data/dermmnist/slice_1",
    }
    hp = _shared_hp(warmup, seed)
    if method == "tralo":
        hp.update(_tralo_hp(bounded=False))
    elif method == "tralo_bounded":
        hp.update(_tralo_hp(bounded=True))
    bmid = compute_base_model_id(
        backbone, hp, dataset_mode="dermmnist",
        data_dir=ds_cfg["data_dir"], dataset_config=ds_cfg,
    )
    method_canon = method if method != "tralo_bounded" else "tralo"
    return {
        "methodology": method, "model_name": backbone,
        "constraint": [0.5, 0.5], "constraint_tag": "L50_G50",
        "dataset_mode": "dermmnist", "dataset_config": ds_cfg,
        "hyperparams": hp, "base_model_id": bmid,
        "experiment_path": (
            f"results/pending_runs/phase_transition/{backbone}/"
            f"dermmnist/L50_G50/w{warmup}/{method}/seed_{seed}"
        ),
    }


def _tablef_cfg(seed, method):
    backbone = "MobileNetV2"
    ds_cfg = {
        "num_classes": 8, "image_size": 224, "target_column": "label",
        "group_column": "synth_group", "constrained_class": 4,
        "data_dir": "data/tissuemnist/slice_1",
    }
    hp = _shared_hp(1, seed)
    if method == "tralo":
        hp.update(_tralo_hp())
    bmid = compute_base_model_id(
        backbone, hp, dataset_mode="tissuemnist",
        data_dir=ds_cfg["data_dir"], dataset_config=ds_cfg,
    )
    return {
        "methodology": method, "model_name": backbone,
        "constraint": [0.5, 0.5], "constraint_tag": "L50_G50",
        "dataset_mode": "tissuemnist", "dataset_config": ds_cfg,
        "hyperparams": hp, "base_model_id": bmid,
        "experiment_path": (
            f"results/pending_runs/tablef_shortwarm/{backbone}/"
            f"tissuemnist/L50_G50/{method}/seed_{seed}"
        ),
    }


def main():
    cfgs = []
    for w in PHASE_WARMUPS:
        for s in PHASE_SEEDS:
            for m in PHASE_METHODS:
                cfgs.append(_phase_cfg(w, s, m))
    for s in TABLEF_SEEDS:
        for m in TABLEF_METHODS:
            cfgs.append(_tablef_cfg(s, m))
    print(f"Generated {len(cfgs)} configs "
          f"({len(PHASE_WARMUPS)*len(PHASE_SEEDS)*len(PHASE_METHODS)} phase + "
          f"{len(TABLEF_SEEDS)*len(TABLEF_METHODS)} tablef)")
    save_configs(cfgs, output_dir="results/pending_runs")


if __name__ == "__main__":
    main()
