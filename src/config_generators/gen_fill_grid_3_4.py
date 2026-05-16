"""Fill missing benchmark cells: Fioretto + Hounie on (3,4) × 3 tightness × 3 seeds.

Completes the (3,4) cell coverage so we have head-to-head data vs TraLO
across another constraint-class scenario. 18 runs, ~6.6h single GPU.
"""
from pathlib import Path
from src.config_generators.generate_configs import (
    compute_base_model_id, constraint_tag, save_configs,
)

DATA_DIR = "data/tissuemnist/slice_1"
SWEEP_ROOT = "results/pending_runs/fill_grid_3_4"
MODEL = "MobileNetV3"
TIGHTNESS = [(0.3, 0.3), (0.5, 0.5), (0.7, 0.7)]
SEEDS = [1, 2, 3]
CLASSES = (3, 4)

SHARED_HP = {
    "lr": 1e-4, "lr_constraint": 5e-6, "dropout": 0.3, "batch_size": 64,
    "warmup_epochs": 50, "constraint_epochs": 300,
    "use_sum_loss": True, "kl_temperature": 1.0, "pretrained": True,
    "class_weighted_ce": False, "constraint_chunk_size": 256,
}

PER_METHOD = {
    "fioretto_ldf": {"fioretto_step_size": 0.005},
    "hounie_rcl": {"hounie_eta_lambda": 0.01, "hounie_eta_u": 0.01,
                   "hounie_alpha": 10.0},
}


def main():
    cfgs = []
    cls_tag = "_".join(str(c) for c in CLASSES)
    for method in ("fioretto_ldf", "hounie_rcl"):
        for pair in TIGHTNESS:
            for seed in SEEDS:
                hp = dict(SHARED_HP)
                hp.update(PER_METHOD[method])
                hp["seed"] = seed
                ds = {
                    "target_column": "label", "group_column": "synth_group",
                    "num_classes": 8, "image_size": 224, "data_dir": DATA_DIR,
                    "constrained_class": list(CLASSES),
                }
                cfgs.append({
                    "methodology": method,
                    "model_name": MODEL,
                    "constraint": list(pair),
                    "constraint_tag": constraint_tag(pair),
                    "dataset_mode": "tissuemnist",
                    "dataset_config": ds,
                    "hyperparams": hp,
                    "base_model_id": compute_base_model_id(
                        MODEL, hp, dataset_mode="tissuemnist",
                        data_dir=DATA_DIR, dataset_config=ds),
                    "exp_name": f"fill34_cls{cls_tag}_{constraint_tag(pair)}_{method}_seed{seed}",
                    "status": "pending",
                    "experiment_path": str(
                        Path(SWEEP_ROOT) / method / constraint_tag(pair) / f"seed_{seed}"),
                })
    save_configs(cfgs, output_dir=SWEEP_ROOT)
    print(f"Total: {len(cfgs)} configs")


if __name__ == "__main__":
    main()
