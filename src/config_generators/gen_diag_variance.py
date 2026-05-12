"""Variance stress test: same config run 5x independently.

Goal: measure CUDA non-determinism across runs of identical
(seed=3, MobileNetV3, TissueMNIST L50_G50) baseline TraLO. The earlier
diag_convergence baseline run satisfied early at E56 with max amplitude
36 and triggered early-stop at E90. The convergence-compare run (same
seed) showed first_sat=72 with max drift 116 and final excess 98.

Same code, same seed -> wildly different convergence trajectories. This
sweep runs the identical config 5 times in separate experiment paths to
measure the run-to-run variance band:
  - If 5/5 converge with max_amp < 50, method is robust to noise
  - If 2/5 diverge to 100+, we have a real stability problem

Each run also drops stable_count_threshold from 5 to 3 (early-stop fires
on 3 consecutive satisfied epochs, not 5) since the user observed that
the model visibly settles fast and 3 is enough signal.

5 runs at ~7.5 min on MobileNetV3 -> ~37 min on GPU 1.

Usage: python -m src.config_generators.gen_diag_variance
"""
from pathlib import Path
from src.config_generators.generate_configs import (
    compute_base_model_id, constraint_tag, save_configs,
)

SWEEP_ROOT = "results/pending_runs/diag_variance"
SEED = 3
N_RUNS = 5
MODEL = "MobileNetV3"
PAIR = (0.5, 0.5)
DATA_DIR = "data/tissuemnist/slice_1"

DS = {
    "target_column": "label", "group_column": "synth_group",
    "num_classes": 8, "image_size": 224, "data_dir": DATA_DIR,
    "constrained_class": 4,
}

HP = {
    "lr": 1e-4, "lr_constraint": 5e-6, "dropout": 0.3, "batch_size": 64,
    "warmup_epochs": 50, "constraint_epochs": 100,
    "use_sum_loss": True, "kl_temperature": 1.0, "pretrained": True,
    "class_weighted_ce": False, "constraint_chunk_size": 256,
    "lambda_global": 0.05, "lambda_local": 0.05, "lambda_step": 0.002,
    "initial_rho": 5.0, "rho_target": 100.0,
    "penalty_mode": "both",
    "alpha_kl": 0.0,
    "stable_count_threshold": 3,
    "seed": SEED,
}


def main():
    cfgs = []
    for run_idx in range(1, N_RUNS + 1):
        hp = dict(HP)
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
            "exp_name": f"diag_var_run{run_idx}_seed{SEED}",
            "status": "pending",
            "experiment_path": str(Path(SWEEP_ROOT) / f"run_{run_idx}"),
        }
        cfgs.append(cfg)
    save_configs(cfgs, output_dir=SWEEP_ROOT)
    print(f"Created {len(cfgs)} runs of identical config (seed={SEED}, "
          f"stable_count_threshold=3)")


if __name__ == "__main__":
    main()
