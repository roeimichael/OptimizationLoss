"""Diagnostic: 4 fix variants on the known-oscillator seed.

Tests three hypotheses about why TraLO drifts back to violation after
first touch (at E~70-130 in our seed_3 runs):

  H1: /n_chunks divisor in the constraint loss silently scales the
      constraint gradient by 1/n_chunks. With n_test=2400, chunk_size=256,
      n_chunks=10 -> constraint pull is 10x too weak.
  H2: Single Adam optimizer accumulates 150 CE-batch momentum steps per
      epoch, then the single constraint step is dominated by stale CE
      momentum pointing toward over-prediction.
  H3: Unclipped CE gradients can spike and pump Adam's m/v buffers,
      compounding H2.

Variants:
  baseline   : current code (control)
  chunk_fix  : drop /n_chunks from constraint loss only
  optim_fix  : separate CE and constraint optimizers + CE clip_grad_norm_
  all_fixes  : chunk_fix + optim_fix combined

All four use the pre-PI-controller winning headline HP (alpha_kl=0,
penalty_mode=both, lambda_step=0.002, initial_rho=5, rho_target=100,
100 constraint epochs). One seed (3) which is the oscillating seed.
Each run ~7.5 min on MobileNetV3 -> ~30 min total on GPU 1.

Usage: python -m src.config_generators.gen_diag_convergence
"""
from pathlib import Path
from src.config_generators.generate_configs import (
    compute_base_model_id, constraint_tag, save_configs,
)

SWEEP_ROOT = "results/pending_runs/diag_convergence"
SEED = 3
MODEL = "MobileNetV3"
PAIR = (0.5, 0.5)
DATA_DIR = "data/tissuemnist/slice_1"

DS = {
    "target_column": "label", "group_column": "synth_group",
    "num_classes": 8, "image_size": 224, "data_dir": DATA_DIR,
    "constrained_class": 4,
}

BASE_HP = {
    "lr": 1e-4, "lr_constraint": 5e-6, "dropout": 0.3, "batch_size": 64,
    "warmup_epochs": 50, "constraint_epochs": 100,
    "use_sum_loss": True, "kl_temperature": 1.0, "pretrained": True,
    "class_weighted_ce": False, "constraint_chunk_size": 256,
    "lambda_global": 0.05, "lambda_local": 0.05, "lambda_step": 0.002,
    "initial_rho": 5.0, "rho_target": 100.0,
    "penalty_mode": "both",
    "alpha_kl": 0.0,
}

VARIANTS = {
    "baseline":  dict(),
    "chunk_fix": dict(fix_chunk_scaling=True),
    "optim_fix": dict(separate_optimizers=True, ce_grad_clip=True),
    "all_fixes": dict(fix_chunk_scaling=True, separate_optimizers=True, ce_grad_clip=True),
}


def main():
    cfgs = []
    for tag, overrides in VARIANTS.items():
        hp = dict(BASE_HP)
        hp["seed"] = SEED
        hp.update(overrides)
        cfg = {
            "methodology": "tralo",
            "model_name": MODEL,
            "constraint": list(PAIR),
            "constraint_tag": constraint_tag(PAIR),
            "dataset_mode": "tissuemnist",
            "dataset_config": dict(DS),
            "hyperparams": hp,
            "base_model_id": compute_base_model_id(
                MODEL, hp, dataset_mode="tissuemnist",
                data_dir=DATA_DIR, dataset_config=DS),
            "exp_name": f"diag_{tag}_seed{SEED}",
            "status": "pending",
            "experiment_path": str(Path(SWEEP_ROOT) / tag),
        }
        cfgs.append(cfg)
    save_configs(cfgs, output_dir=SWEEP_ROOT)
    print(f"Created {len(cfgs)} configs: {list(VARIANTS)}")


if __name__ == "__main__":
    main()
