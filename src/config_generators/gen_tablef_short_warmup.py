"""Table F short_warmup row.

Replaces the broken `no_warmup` row (warmup=0 → catastrophic F1 collapse)
with `short_warmup` (warmup=1 → push-pull regime, meaningful comparison).

Matches g5_component_ablation recipe exactly: MobileNetV3 × L30_G30 × 3 seeds.
"""
from src.config_generators.generate_configs import compute_base_model_id, save_configs

SEEDS = [1, 2, 3]


def _shared_hp(seed):
    return {
        "lr": 1e-4, "lr_constraint": 5e-6, "dropout": 0.3, "batch_size": 64,
        "warmup_epochs": 1, "constraint_epochs": 100,
        "pretrained": True,
        "class_weighted_ce": False, "constraint_chunk_size": 256,
        "fioretto_step_size": 0.01,
        "lambda_global": 0.05, "lambda_local": 0.05, "lambda_step": 0.002,
        "initial_rho": 5.0, "rho_target": 100.0, "alpha_kl": 0.0,
        "penalty_mode": "both", "enable_ce_skip": True,
        "hybrid_mode": "undershoot_hinge", "fior_beta": 0.50,
        "reset_optimizer_at_sat": True,
        "disable_freeze_on_satisfy": False,
        "seed": seed,
    }


CELLS = [
    ("dermmnist", 7, "loc_group", 4, "data/dermmnist/slice_1"),
    ("tissuemnist", 8, "synth_group", 4, "data/tissuemnist/slice_1"),
    ("aider", 4, "synth_group", 0, "data/aider/slice_1"),
]


def main():
    cfgs = []
    for ds_mode, n_cls, grp_col, cls, data_dir in CELLS:
        ds_cfg = {
            "num_classes": n_cls, "image_size": 224,
            "target_column": "label", "group_column": grp_col,
            "constrained_class": cls, "data_dir": data_dir,
        }
        for seed in SEEDS:
            hp = _shared_hp(seed)
            bmid = compute_base_model_id(
                "MobileNetV3", hp, dataset_mode=ds_mode,
                data_dir=data_dir, dataset_config=ds_cfg,
            )
            cfgs.append({
                "methodology": "tralo", "model_name": "MobileNetV3",
                "constraint": [0.3, 0.3], "constraint_tag": "L30_G30",
                "dataset_mode": ds_mode, "dataset_config": ds_cfg,
                "hyperparams": hp, "base_model_id": bmid,
                "experiment_path": (
                    f"results/pending_runs/g5_short_warmup/"
                    f"{ds_mode}/L30_G30/short_warmup/seed_{seed}"
                ),
            })
    print(f"Generated {len(cfgs)} short_warmup configs (3 ds x 3 seeds, MobileNetV3 L30)")
    save_configs(cfgs, output_dir="results/pending_runs")


if __name__ == "__main__":
    main()
