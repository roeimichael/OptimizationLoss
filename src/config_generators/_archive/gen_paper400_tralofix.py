"""Apply the v3 Adam-reset breakthrough to paper400 cells.

Re-runs TraLO-equivalent on the cells where vanilla TraLO tied/lost to
Fioretto, using the bidirectional penalty + Adam-reset variant proved
out in hybrid_v3:
   methodology: tralo
   hybrid_mode: undershoot_hinge
   fior_beta:   0.5
   reset_optimizer_at_sat: True

Anchored against the existing paper400 TraLO/Fior/Hounie results
(same cells, same seeds, same backbone). Aggregator will side-by-side.

Cells: 2 datasets x 3 tightness x 4 seeds = 24 configs.
"""
from pathlib import Path

from src.config_generators.generate_configs import (
    compute_base_model_id, save_configs,
)

SWEEP_ROOT = "results/pending_runs/paper400_tralofix"

DATASETS = {
    "tissuemnist": {"data_dir": "data/tissuemnist/slice_1", "num_classes": 8,
                    "image_size": 224, "target_column": "label",
                    "group_column": "synth_group"},
    "eurosat": {"data_dir": "data/eurosat/slice_1", "num_classes": 10,
                "image_size": 224, "target_column": "label",
                "group_column": "synth_group"},
}
TIGHTNESS = ["L20_G20", "L30_G30", "L50_G50"]
SEEDS = [1, 2, 3, 4]
MODEL = "MobileNetV3"
CLS = 4

SHARED_HP = {
    "lr": 1e-4, "lr_constraint": 5e-6, "dropout": 0.3, "batch_size": 64,
    "warmup_epochs": 50, "constraint_epochs": 300, "pretrained": True,
    "class_weighted_ce": False, "constraint_chunk_size": 256,
}

TRALOFIX_HP = {
    "lambda_global": 0.05, "lambda_local": 0.05, "lambda_step": 0.002,
    "initial_rho": 5.0, "rho_target": 100.0, "alpha_kl": 0.0,
    "penalty_mode": "both", "enable_ce_skip": True,
    "hybrid_mode": "undershoot_hinge",
    "fior_beta": 0.50,
    "reset_optimizer_at_sat": True,
}


def _tight_pair(tag):
    parts = tag.split("_")
    return (int(parts[0][1:]) / 100, int(parts[1][1:]) / 100)


def make_cfg(dataset, tight_tag, seed):
    ds_meta = DATASETS[dataset]
    hp = {**SHARED_HP, **TRALOFIX_HP, "seed": seed}
    ds_config = {**ds_meta, "constrained_class": CLS}
    pair = _tight_pair(tight_tag)
    bmid = compute_base_model_id(
        MODEL, hp, dataset_mode=dataset,
        data_dir=ds_meta["data_dir"], dataset_config=ds_config,
    )
    return {
        "methodology": "tralo",
        "model_name": MODEL,
        "constraint": list(pair),
        "constraint_tag": tight_tag,
        "dataset_mode": dataset,
        "dataset_config": ds_config,
        "hyperparams": hp,
        "base_model_id": bmid,
        "exp_name": f"tralofix_{dataset}_{MODEL}_cls{CLS}_{tight_tag}_seed{seed}",
        "experiment_path": str(
            Path(SWEEP_ROOT) / dataset / tight_tag / f"seed_{seed}"),
    }


def build():
    cfgs = []
    for dataset in DATASETS:
        for tight in TIGHTNESS:
            for seed in SEEDS:
                cfgs.append(make_cfg(dataset, tight, seed))
    save_configs(cfgs, output_dir=SWEEP_ROOT)
    print(f"\nGenerated {len(cfgs)} paper400_tralofix configs -> {SWEEP_ROOT}")


if __name__ == "__main__":
    build()
