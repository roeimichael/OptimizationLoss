"""DermMNIST + small-CNN full-pipeline smoke (the actual headroom test).

After gen_derm_smallcnn_smoke confirmed all 3 backbones land warmup train_acc
in [0.71, 0.92] (non-saturating), this runs the full method panel with the
real percentile constraint L30_G30 (loc_group local + global both active).

Hypothesis: TraLO's paired d_F1 vs baselines GROWS compared to the saturated
MobileNetV3 paper baseline (where TraLO d_F1 ~+0.005 on derm tight cells).
If yes -> headroom is the operative mechanism. If no -> hypothesis rejected.

Single seed for triage; if the signal is positive, expand to 3 seeds.
"""
from src.config_generators.generate_configs import (
    compute_base_model_id, save_configs,
)

DATASET = "dermmnist"
DATA_DIR = "data/dermmnist/slice_1"
NUM_CLASSES = 7
CONSTRAINED_CLASS = 4  # MEL
GROUP_COLUMN = "loc_group"

BACKBONES = ["TinyCNN", "SmallCNN", "MediumCNN"]
TIGHT = "L30_G30"
SEED = 1
METHODS = ["tralo", "fioretto_ldf", "hounie_rcl", "danits_lp", "heuristic"]

# warmup=30 keeps MediumCNN below saturation; TinyCNN/SmallCNN sit ~0.71/0.76
SHARED_HP = {
    "lr": 1e-4, "lr_constraint": 5e-6, "dropout": 0.3, "batch_size": 64,
    "warmup_epochs": 30, "constraint_epochs": 100,
    "pretrained": False,
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


def make_cfg(model, method):
    ds_config = {
        "num_classes": NUM_CLASSES, "image_size": 224, "target_column": "label",
        "group_column": GROUP_COLUMN, "constrained_class": CONSTRAINED_CLASS,
        "data_dir": DATA_DIR,
    }
    hp = {**SHARED_HP, "seed": SEED}
    if method == "tralo":
        hp.update(TRALO_HP)
    pair = _pair(TIGHT)
    bmid = compute_base_model_id(
        model, hp, dataset_mode=DATASET, data_dir=DATA_DIR,
        dataset_config=ds_config,
    )
    return {
        "methodology": method, "model_name": model,
        "constraint": list(pair), "constraint_tag": TIGHT,
        "dataset_mode": DATASET, "dataset_config": ds_config,
        "hyperparams": hp, "base_model_id": bmid,
        "experiment_path": (
            f"results/pending_runs/derm_smallcnn_full/{model}/{method}/seed_{SEED}"
        ),
    }


def main():
    cfgs = [make_cfg(m, meth) for m in BACKBONES for meth in METHODS]
    print(f"Generated {len(cfgs)} configs "
          f"({len(BACKBONES)} backbones x {len(METHODS)} methods)")
    save_configs(cfgs, output_dir="results/pending_runs")


if __name__ == "__main__":
    main()
