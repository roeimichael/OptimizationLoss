"""Two demo cells with cache-miss forced, to capture warmup train_acc + CE curves.

Same recipe as the cells in the audit (dermmnist MobileNetV2 cls 4), but with
a tweaked base_model_id so the model_cache misses and the warmup gets logged
to training_log.csv from epoch 0 onwards.
"""
from src.config_generators.generate_configs import compute_base_model_id, save_configs


def _ds_cfg():
    return {
        "num_classes": 7, "image_size": 224, "target_column": "label",
        "group_column": "loc_group", "constrained_class": 4,
        "data_dir": "data/dermmnist/slice_1",
    }


def _make(warmup_epochs, tag):
    ds_cfg = _ds_cfg()
    hp = {
        "lr": 1e-4, "lr_constraint": 5e-6, "dropout": 0.3, "batch_size": 64,
        "warmup_epochs": warmup_epochs, "constraint_epochs": 100,
        "pretrained": True,
        "class_weighted_ce": False, "constraint_chunk_size": 256,
        "fioretto_step_size": 0.01,
        "lambda_global": 0.05, "lambda_local": 0.05, "lambda_step": 0.002,
        "initial_rho": 5.0, "rho_target": 100.0, "alpha_kl": 0.0,
        "penalty_mode": "both", "enable_ce_skip": True,
        "hybrid_mode": "undershoot_hinge", "fior_beta": 0.50,
        "reset_optimizer_at_sat": True,
        "disable_freeze_on_satisfy": False,
        "seed": 99,  # unique seed to force cache miss
    }
    # Hash will differ because seed is included
    bmid = compute_base_model_id(
        "MobileNetV2", hp, dataset_mode="dermmnist",
        data_dir=ds_cfg["data_dir"], dataset_config=ds_cfg,
    )
    return {
        "methodology": "tralo", "model_name": "MobileNetV2",
        "constraint": [0.5, 0.5], "constraint_tag": "L50_G50",
        "dataset_mode": "dermmnist", "dataset_config": ds_cfg,
        "hyperparams": hp, "base_model_id": bmid,
        "experiment_path": (
            f"results/pending_runs/dynamics_demo/MobileNetV2/{tag}/tralo/seed_99"
        ),
    }


def main():
    cfgs = [_make(1, "warmup1"), _make(50, "warmup50")]
    print(f"Generated {len(cfgs)} dynamics-demo configs")
    save_configs(cfgs, output_dir="results/pending_runs")


if __name__ == "__main__":
    main()
