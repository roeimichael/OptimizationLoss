"""Hybrid v2 sweep: symmetric quadratic + undershoot hinge.

Tests whether a BIDIRECTIONAL penalty (pushes up below K, down above K)
beats TraLO's asymmetric bounded penalty that allows the model to drift
below K with no opposing force.

Cells:
  - baseline_tralo (anchored same code)
  - hybrid symquad        L = lam_T * ((soft - K)/K)^2
  - hybrid undershoot beta=0.2/0.5/1.0   L = bounded + lam_T * beta * relu(K - soft)/K

Target: tissuemnist L20_G20 + L30_G30, MobileNetV3, 2 seeds.
"""
from pathlib import Path

from src.config_generators.generate_configs import (
    compute_base_model_id, save_configs,
)

SWEEP_ROOT = "results/pending_runs/hybrid_v2"
DATASET = "tissuemnist"
DS_META = {
    "data_dir": "data/tissuemnist/slice_1", "num_classes": 8,
    "image_size": 224, "target_column": "label",
    "group_column": "synth_group",
}
MODEL = "MobileNetV3"
CLS = 4
SEEDS = [1, 2]
TIGHTNESS = ["L20_G20", "L30_G30"]

SHARED_HP = {
    "lr": 1e-4, "lr_constraint": 5e-6, "dropout": 0.3, "batch_size": 64,
    "warmup_epochs": 50, "constraint_epochs": 150, "pretrained": True,
    "class_weighted_ce": False, "constraint_chunk_size": 256,
}

TRALO_KNOBS = {
    "lambda_global": 0.05, "lambda_local": 0.05, "lambda_step": 0.002,
    "initial_rho": 5.0, "rho_target": 100.0, "alpha_kl": 0.0,
    "penalty_mode": "both", "enable_ce_skip": True,
}

CELLS = [
    # Anchor
    ("baseline_tralo", "tralo", TRALO_KNOBS),
    # Pure symmetric quadratic (no bounded). Test if parking AT K (rather
    # than below it) changes anything.
    ("symquad", "tralo_fioretto",
     {**TRALO_KNOBS, "hybrid_mode": "symquad", "fior_beta": 0.0}),
    # Undershoot hinge: bounded above + linear push-up below. Sweep beta.
    ("undershoot_b020", "tralo_fioretto",
     {**TRALO_KNOBS, "hybrid_mode": "undershoot_hinge", "fior_beta": 0.20}),
    ("undershoot_b050", "tralo_fioretto",
     {**TRALO_KNOBS, "hybrid_mode": "undershoot_hinge", "fior_beta": 0.50}),
    ("undershoot_b100", "tralo_fioretto",
     {**TRALO_KNOBS, "hybrid_mode": "undershoot_hinge", "fior_beta": 1.00}),
]


def _tight_pair(tag):
    parts = tag.split("_")
    return (int(parts[0][1:]) / 100, int(parts[1][1:]) / 100)


def make_cfg(cell_name, method, method_hp, tight_tag, seed):
    hp = {**SHARED_HP, **method_hp, "seed": seed}
    ds_config = {**DS_META, "constrained_class": CLS}
    pair = _tight_pair(tight_tag)
    bmid = compute_base_model_id(
        MODEL, hp, dataset_mode=DATASET,
        data_dir=DS_META["data_dir"], dataset_config=ds_config,
    )
    return {
        "methodology": method,
        "model_name": MODEL,
        "constraint": list(pair),
        "constraint_tag": tight_tag,
        "dataset_mode": DATASET,
        "dataset_config": ds_config,
        "hyperparams": hp,
        "base_model_id": bmid,
        "exp_name": f"{cell_name}_{tight_tag}_seed{seed}",
        "experiment_path": str(
            Path(SWEEP_ROOT) / tight_tag / cell_name / f"seed_{seed}"),
    }


def build():
    cfgs = []
    for tight in TIGHTNESS:
        for cell_name, method, method_hp in CELLS:
            for seed in SEEDS:
                cfgs.append(make_cfg(cell_name, method, method_hp, tight, seed))
    save_configs(cfgs, output_dir=SWEEP_ROOT)
    print(f"\nGenerated {len(cfgs)} hybrid_v2 configs -> {SWEEP_ROOT}")


if __name__ == "__main__":
    build()
