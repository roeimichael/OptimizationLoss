"""DermMNIST + small CNN smoke probe.

Warmup-only run (heuristic methodology, constraint_epochs=0) on 3 small
backbones to find one whose end-of-warmup train_acc lands in the headroom
window [0.50, 0.99]. The post-hoc step will fire but won't move much given
how few epochs of CE training; we only care about the train_acc trajectory
in training_log.csv.

Decision rules:
  end_acc >= 0.99 -> saturated, useless for headroom test
  end_acc <  0.50 -> too weak, will generalize badly
  else           -> proceed to full pipeline sweep on that backbone
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
TIGHT = "L30_G30"  # unused (constraint_epochs=0) but required field
SEED = 1
METHOD = "heuristic"  # CE warmup + post-hoc only

HP = {
    "lr": 1e-4, "lr_constraint": 5e-6, "dropout": 0.3, "batch_size": 64,
    "warmup_epochs": 30, "constraint_epochs": 0,
    "pretrained": False,  # small CNNs have no pretrained weights
    "class_weighted_ce": False, "constraint_chunk_size": 256,
    "seed": SEED,
}


def _pair(tag):
    p = tag.split("_")
    return (int(p[0][1:]) / 100, int(p[1][1:]) / 100)


def make_cfg(model):
    ds_config = {
        "num_classes": NUM_CLASSES, "image_size": 224, "target_column": "label",
        "group_column": GROUP_COLUMN, "constrained_class": CONSTRAINED_CLASS,
        "data_dir": DATA_DIR,
    }
    pair = _pair(TIGHT)
    bmid = compute_base_model_id(
        model, HP, dataset_mode=DATASET, data_dir=DATA_DIR,
        dataset_config=ds_config,
    )
    return {
        "methodology": METHOD, "model_name": model,
        "constraint": list(pair), "constraint_tag": TIGHT,
        "dataset_mode": DATASET, "dataset_config": ds_config,
        "hyperparams": HP, "base_model_id": bmid,
        "experiment_path": (
            f"results/pending_runs/derm_smallcnn_smoke/{model}/seed_{SEED}"
        ),
    }


def main():
    cfgs = [make_cfg(m) for m in BACKBONES]
    print(f"Generated {len(cfgs)} configs (3 backbones x warmup-only 30 epochs)")
    save_configs(cfgs, output_dir="results/pending_runs")


if __name__ == "__main__":
    main()
