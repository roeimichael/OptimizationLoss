"""Leave-one-out component ablation for TraLO.

Reference (full TraLO): bounded + undershoot_hinge + reset_optimizer_at_sat
                        + freeze_on_satisfy + CE_saturation_skip + rho_schedule
                        + warmup + posthoc_clamp

Variants disable ONE component each:
  -hinge       : hybrid_mode=bounded_only  (only bounded penalty above K)
  -reset       : reset_optimizer_at_sat=False
  -freeze      : disable_freeze_on_satisfy=True
  -ce_skip     : enable_ce_skip=False
  -rho_sched   : rho_target = initial_rho (constant rho through training)
  -warmup      : warmup_epochs=0 (cold start)
  (-posthoc not ablated here; pipeline-level skip is non-trivial)

Cells: 2 datasets x 1 tightness (L30, where wins are cleanest)
       x 6 variants x 2 seeds = 24 runs
Output: results/pending_runs/component_ablation/
"""
from pathlib import Path

from src.config_generators.generate_configs import (
    compute_base_model_id, save_configs,
)

SWEEP_ROOT = "results/pending_runs/component_ablation"
DATASETS = {
    "tissuemnist": {"data_dir": "data/tissuemnist/slice_1", "num_classes": 8,
                    "image_size": 224, "target_column": "label",
                    "group_column": "synth_group"},
    "eurosat": {"data_dir": "data/eurosat/slice_1", "num_classes": 10,
                "image_size": 224, "target_column": "label",
                "group_column": "synth_group"},
}
TIGHTNESS = ["L30_G30"]
SEEDS = [1, 2]
MODEL = "MobileNetV3"
CLS = 4

SHARED_HP = {
    "lr": 1e-4, "lr_constraint": 5e-6, "dropout": 0.3, "batch_size": 64,
    "warmup_epochs": 50, "constraint_epochs": 300,
    "use_sum_loss": True, "kl_temperature": 1.0, "pretrained": True,
    "class_weighted_ce": False, "constraint_chunk_size": 256,
}

# Full TraLO ("reference") hyperparams. Each variant overrides exactly
# one knob.
FULL_TRALO = {
    "lambda_global": 0.05, "lambda_local": 0.05, "lambda_step": 0.002,
    "initial_rho": 5.0, "rho_target": 100.0, "alpha_kl": 0.0,
    "penalty_mode": "both", "enable_ce_skip": True,
    "hybrid_mode": "undershoot_hinge", "fior_beta": 0.50,
    "reset_optimizer_at_sat": True,
    "disable_freeze_on_satisfy": False,
}

VARIANTS = {
    "full":         {},                                          # reference
    "no_hinge":     {"hybrid_mode": "bounded_only"},
    "no_reset":     {"reset_optimizer_at_sat": False},
    "no_freeze":    {"disable_freeze_on_satisfy": True},
    "no_ce_skip":   {"enable_ce_skip": False},
    "no_rho_sched": {"rho_target": 5.0},                         # = initial_rho
    "no_warmup":    {"warmup_epochs": 0},
}


def _tight_pair(tag):
    parts = tag.split("_")
    return (int(parts[0][1:]) / 100, int(parts[1][1:]) / 100)


def make_cfg(variant_name, overrides, dataset, tight_tag, seed):
    ds_meta = DATASETS[dataset]
    hp = {**SHARED_HP, **FULL_TRALO, **overrides, "seed": seed}
    ds_config = {**ds_meta, "constrained_class": CLS}
    pair = _tight_pair(tight_tag)
    bmid = compute_base_model_id(
        MODEL, hp, dataset_mode=dataset,
        data_dir=ds_meta["data_dir"], dataset_config=ds_config,
    )
    return {
        "methodology": "tralo_fioretto",
        "model_name": MODEL,
        "constraint": list(pair),
        "constraint_tag": tight_tag,
        "dataset_mode": dataset,
        "dataset_config": ds_config,
        "hyperparams": hp,
        "base_model_id": bmid,
        "exp_name": f"ablate_{variant_name}_{dataset}_{tight_tag}_seed{seed}",
        "experiment_path": str(
            Path(SWEEP_ROOT) / dataset / tight_tag / variant_name / f"seed_{seed}"),
    }


def build():
    cfgs = []
    for dataset in DATASETS:
        for tight in TIGHTNESS:
            for variant_name, overrides in VARIANTS.items():
                for seed in SEEDS:
                    cfgs.append(make_cfg(variant_name, overrides, dataset, tight, seed))
    save_configs(cfgs, output_dir=SWEEP_ROOT)
    print(f"\nGenerated {len(cfgs)} component-ablation configs -> {SWEEP_ROOT}")


if __name__ == "__main__":
    build()
