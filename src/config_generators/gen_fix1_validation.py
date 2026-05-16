"""Fix-1 (CE-skip restored) validation sweep.

3 methods x 3 tightness x 3 seeds on hardest cell (1,4,7), 300 epochs.
Goal: confirm fix1 generalizes (not seed=1 fluke) AND beats Fioretto/Hounie
on accuracy + satisfaction.
"""
from pathlib import Path
from src.config_generators.generate_configs import (
    compute_base_model_id, constraint_tag, save_configs,
)

DATA_DIR = "data/tissuemnist/slice_1"
SWEEP_ROOT = "results/pending_runs/fix1_validation"
MODEL = "MobileNetV3"
METHODS = ["tralo", "fioretto_ldf", "hounie_rcl"]
TIGHTNESS_PAIRS = [(0.3, 0.3), (0.5, 0.5), (0.7, 0.7)]
CLASSES = (1, 4, 7)
SEEDS = [1, 2, 3]

SHARED_HP = {
    "lr": 1e-4, "lr_constraint": 5e-6, "dropout": 0.3, "batch_size": 64,
    "warmup_epochs": 50, "constraint_epochs": 300,
    "use_sum_loss": True, "kl_temperature": 1.0, "pretrained": True,
    "class_weighted_ce": False, "constraint_chunk_size": 256,
}

PER_METHOD_HP = {
    "tralo": {"lambda_global": 0.05, "lambda_local": 0.05, "lambda_step": 0.002,
              "initial_rho": 5.0, "rho_target": 100.0, "alpha_kl": 0.0,
              "penalty_mode": "both", "enable_ce_skip": True},
    "fioretto_ldf": {"fioretto_step_size": 0.005},
    "hounie_rcl": {"hounie_eta_lambda": 0.01, "hounie_eta_u": 0.01,
                   "hounie_alpha": 10.0},
}


def main():
    cls_tag = "_".join(str(c) for c in CLASSES)
    constrained_class = list(CLASSES)
    cfgs = []
    for seed in SEEDS:
        for pair in TIGHTNESS_PAIRS:
            tag = constraint_tag(pair)
            for method in METHODS:
                hp = dict(SHARED_HP)
                hp.update(PER_METHOD_HP[method])
                hp["seed"] = seed
                ds = {
                    "target_column": "label", "group_column": "synth_group",
                    "num_classes": 8, "image_size": 224, "data_dir": DATA_DIR,
                    "constrained_class": constrained_class,
                }
                cfgs.append({
                    "methodology": method,
                    "model_name": MODEL,
                    "constraint": list(pair),
                    "constraint_tag": tag,
                    "dataset_mode": "tissuemnist",
                    "dataset_config": ds,
                    "hyperparams": hp,
                    "base_model_id": compute_base_model_id(
                        MODEL, hp, dataset_mode="tissuemnist",
                        data_dir=DATA_DIR, dataset_config=ds),
                    "exp_name": f"fix1val_cls{cls_tag}_{tag}_{method}_seed{seed}",
                    "status": "pending",
                    "experiment_path": str(
                        Path(SWEEP_ROOT) / f"seed_{seed}" / tag / method),
                })

    save_configs(cfgs, output_dir=SWEEP_ROOT)
    print(f"Total: {len(cfgs)} configs ({len(METHODS)} methods x "
          f"{len(TIGHTNESS_PAIRS)} tightness x {len(SEEDS)} seeds)")


if __name__ == "__main__":
    main()
