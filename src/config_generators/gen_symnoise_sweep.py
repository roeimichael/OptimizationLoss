"""Sweep symmetric-label-noise conditions to find a regime where warmup train_acc
genuinely plateaus below ~0.95.

Three probes, designed to be dispatched to separate GPUs in parallel:

  Probe A (GPU1): CIFAR-100 90% symnoise / ResNet18         -> noise-rate ceiling
  Probe B (GPU2): CIFAR-100 80% symnoise / MobileNetV3      -> backbone control
  Probe C (GPU3): CIFAR-100 60% symnoise / ResNet18         -> finds noise cliff

Each writes to results/pending_runs/symnoise_sweep/{tag}/ so they can be picked
up by independent dispatchers via EXPERIMENT_DIR.

All probes use methodology=heuristic with warmup_epochs=100, constraint_epochs=0.
The signal is the per-epoch train_acc in training_log.csv.
"""
from src.config_generators.generate_configs import (
    compute_base_model_id, save_configs,
)

PROBES = [
    {"tag": "p90_resnet18",  "noise": 90, "model": "ResNet18",     "data": "cifar100_symnoise90"},
    {"tag": "p80_mobilenet", "noise": 80, "model": "MobileNetV3",  "data": "cifar100_symnoise80"},
    {"tag": "p60_resnet18",  "noise": 60, "model": "ResNet18",     "data": "cifar100_symnoise60"},
]
SEEDS = [1, 2]

SHARED_HP = {
    "lr": 1e-4, "lr_constraint": 5e-6, "dropout": 0.3, "batch_size": 64,
    "warmup_epochs": 100, "constraint_epochs": 0, "pretrained": True,
    "class_weighted_ce": False, "constraint_chunk_size": 256,
}


def make_cfg(probe, seed):
    ds_config = {
        "num_classes": 100, "image_size": 224, "target_column": "label",
        "group_column": "synth_group", "constrained_class": 0,
        "data_dir": f"data/{probe['data']}/slice_1",
    }
    hp = {**SHARED_HP, "seed": seed}
    bmid = compute_base_model_id(
        probe["model"], hp, dataset_mode=probe["data"],
        data_dir=ds_config["data_dir"], dataset_config=ds_config,
    )
    return {
        "methodology": "heuristic", "model_name": probe["model"],
        "constraint": [0.3, 0.3], "constraint_tag": "L30_G30",
        "dataset_mode": probe["data"], "dataset_config": ds_config,
        "hyperparams": hp, "base_model_id": bmid,
        "experiment_path": (
            f"results/pending_runs/symnoise_sweep/{probe['tag']}/{probe['model']}/seed_{seed}"
        ),
    }


def main():
    cfgs = []
    for probe in PROBES:
        for seed in SEEDS:
            cfgs.append(make_cfg(probe, seed))
    print(f"Generated {len(cfgs)} configs across {len(PROBES)} probes")
    save_configs(cfgs, output_dir="results/pending_runs")


if __name__ == "__main__":
    main()
