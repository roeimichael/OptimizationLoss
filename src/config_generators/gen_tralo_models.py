"""TraLO across 3 models × 3 cls × 3 tightness × 3 seeds (54 runs new).

MobileNetV3 already done in overnight_2026_05_14. This adds ResNet18 + EfficientNetB0.
Same fix1 settings (CE-skip True, no KL, no beta). Pure breadth validation.
"""
from pathlib import Path
from src.config_generators.generate_configs import (
    compute_base_model_id, constraint_tag, save_configs,
)

DATA_DIR = "data/tissuemnist/slice_1"
SWEEP_ROOT = "results/pending_runs/tralo_models"
TIGHTNESS = [(0.3, 0.3), (0.5, 0.5), (0.7, 0.7)]
SEEDS = [1, 2, 3]
ALL_CLASSES = [(4,), (3, 4), (1, 4, 7)]
MODELS = ["ResNet18", "EfficientNetB0"]

SHARED_HP = {
    "lr": 1e-4, "lr_constraint": 5e-6, "dropout": 0.3, "batch_size": 64,
    "warmup_epochs": 50, "constraint_epochs": 300,
    "use_sum_loss": True, "kl_temperature": 1.0, "pretrained": True,
    "class_weighted_ce": False, "constraint_chunk_size": 256,
    "lambda_global": 0.05, "lambda_local": 0.05, "lambda_step": 0.002,
    "initial_rho": 5.0, "rho_target": 100.0, "alpha_kl": 0.0,
    "penalty_mode": "both", "enable_ce_skip": True,
}


def main():
    cfgs = []
    for model in MODELS:
        for classes in ALL_CLASSES:
            cls_tag = "_".join(str(c) for c in classes)
            constrained_class = list(classes) if len(classes) > 1 else classes[0]
            for pair in TIGHTNESS:
                tag = constraint_tag(pair)
                for seed in SEEDS:
                    hp = dict(SHARED_HP)
                    hp["seed"] = seed
                    ds = {
                        "target_column": "label", "group_column": "synth_group",
                        "num_classes": 8, "image_size": 224, "data_dir": DATA_DIR,
                        "constrained_class": constrained_class,
                    }
                    cfgs.append({
                        "methodology": "tralo",
                        "model_name": model,
                        "constraint": list(pair),
                        "constraint_tag": tag,
                        "dataset_mode": "tissuemnist",
                        "dataset_config": ds,
                        "hyperparams": hp,
                        "base_model_id": compute_base_model_id(
                            model, hp, dataset_mode="tissuemnist",
                            data_dir=DATA_DIR, dataset_config=ds),
                        "exp_name": f"tralo_{model}_cls{cls_tag}_{tag}_seed{seed}",
                        "status": "pending",
                        "experiment_path": str(
                            Path(SWEEP_ROOT) / model / f"cls_{cls_tag}" / tag / f"seed_{seed}"),
                    })
    save_configs(cfgs, output_dir=SWEEP_ROOT)
    print(f"Total: {len(cfgs)} configs ({len(MODELS)} models)")


if __name__ == "__main__":
    main()
