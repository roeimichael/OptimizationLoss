"""Hybrid v3: test Adam-state hangover fixes on symquad + undershoot_hinge.

v2 revealed both bidirectional designs drift further below K post-satisfaction
because Adam accumulated "decrease soft_4" momentum during descent. The
hinge/symquad gradient is too small to overcome that residual momentum.

This sweep applies two fixes at first satisfaction:
  reset_adam:  rebuild optimizer with fresh m/v buffers (Adam continues)
  sgd_post_sat: switch to plain SGD (no momentum at all)

Anchored against baseline_tralo (no fix needed, gradient drops to 0).

Cells: 5 x 2 tightness x 2 seeds = 20 runs.
"""
from pathlib import Path

from src.config_generators.generate_configs import (
    compute_base_model_id, save_configs,
)

SWEEP_ROOT = "results/pending_runs/hybrid_v3"
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
    ("baseline_tralo", "tralo", TRALO_KNOBS),
    # Symmetric quadratic + post-sat fixes
    ("symquad_resetAdam", "tralo_fioretto",
     {**TRALO_KNOBS, "hybrid_mode": "symquad", "fior_beta": 0.0,
      "reset_optimizer_at_sat": True}),
    ("symquad_sgd", "tralo_fioretto",
     {**TRALO_KNOBS, "hybrid_mode": "symquad", "fior_beta": 0.0,
      "post_sat_optimizer": "sgd"}),
    # Undershoot hinge + post-sat fixes (use b=0.5, mid of v2 range)
    ("undershoot_b050_resetAdam", "tralo_fioretto",
     {**TRALO_KNOBS, "hybrid_mode": "undershoot_hinge", "fior_beta": 0.50,
      "reset_optimizer_at_sat": True}),
    ("undershoot_b050_sgd", "tralo_fioretto",
     {**TRALO_KNOBS, "hybrid_mode": "undershoot_hinge", "fior_beta": 0.50,
      "post_sat_optimizer": "sgd"}),
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
    print(f"\nGenerated {len(cfgs)} hybrid_v3 configs -> {SWEEP_ROOT}")


if __name__ == "__main__":
    build()
