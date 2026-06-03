"""DivideMix / ELR replication: vanilla CE on ResNet-18 + CIFAR-100 with
80% symmetric label noise. Published: train_acc plateau ~0.20-0.30 for 50+
epochs. Sanity test that our pipeline can reproduce that curve.

Uses methodology='heuristic' which runs a CE-only warmup then a post-hoc
adjustment — the warmup IS the CE training loop we want to inspect. We log
train_acc per warmup epoch in training_log.csv (heuristic writes warmup rows).

2 seeds, 2 tightness (irrelevant — heuristic doesn't constraint-train, but
runner needs the field). Total 2 cells. ~1h on a Turing GPU.
"""
from src.config_generators.generate_configs import (
    compute_base_model_id, save_configs,
)

DATASET = "cifar100_symnoise80"
DATA_DIR = "data/cifar100_symnoise80/slice_1"
NUM_CLASSES = 100
CONSTRAINED_CLASS = 0  # irrelevant for heuristic-only run
GROUP_COLUMN = "synth_group"

MODEL = "ResNet18"
SEEDS = [1, 2]
TIGHT = "L30_G30"  # ignored
METHOD = "heuristic"  # CE warmup + post-hoc; the warmup curve is the signal

SHARED_HP = {
    "lr": 1e-4, "lr_constraint": 5e-6, "dropout": 0.3, "batch_size": 64,
    "warmup_epochs": 100, "constraint_epochs": 0, "pretrained": True,
    "class_weighted_ce": False, "constraint_chunk_size": 256,
}


def _pair(tag):
    p = tag.split("_")
    return (int(p[0][1:]) / 100, int(p[1][1:]) / 100)


def make_cfg(seed):
    ds_config = {
        "num_classes": NUM_CLASSES, "image_size": 224, "target_column": "label",
        "group_column": GROUP_COLUMN, "constrained_class": CONSTRAINED_CLASS,
        "data_dir": DATA_DIR,
    }
    hp = {**SHARED_HP, "seed": seed}
    pair = _pair(TIGHT)
    bmid = compute_base_model_id(
        MODEL, hp, dataset_mode=DATASET, data_dir=DATA_DIR,
        dataset_config=ds_config,
    )
    return {
        "methodology": METHOD, "model_name": MODEL,
        "constraint": list(pair), "constraint_tag": TIGHT,
        "dataset_mode": DATASET, "dataset_config": ds_config,
        "hyperparams": hp, "base_model_id": bmid,
        "experiment_path": (
            f"results/pending_runs/symnoise_replication/{MODEL}/seed_{seed}"
        ),
    }


def main():
    cfgs = [make_cfg(s) for s in SEEDS]
    print(f"Generated {len(cfgs)} configs (CIFAR-100 80% symnoise / "
          f"{MODEL} / {METHOD}, warmup_epochs=100)")
    save_configs(cfgs, output_dir="results/pending_runs")


if __name__ == "__main__":
    main()
