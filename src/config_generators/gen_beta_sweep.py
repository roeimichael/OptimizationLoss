"""Beta sweep on the worst-divergence cell.

Cell: (1, 4, 7) L30_G30, MobileNetV3, TissueMNIST slice_1, seed=1.
This cell went from end_ex=90 at 100 epochs to end_ex=411 at 300 epochs
(C4 hard count 184 -> 239) in the prior sweep. Diagnostic: bounded
gradient saturates at E/K=3.7, leaving CE momentum to win.

Fix candidate: add +beta*E/K (linear_sat_tail) — unbounded gradient
beta/K regardless of how deep the violation. From analytic crossover:

  beta = 1.07 matches the bounded gradient at E/K=3.7

We bracket the optimum: {0, 0.5, 1.0, 2.0}. The 0-baseline reproduces
the existing 300-epoch result for sanity.

4 runs x ~22 min on MobileNetV3 (warmup cached) = ~88 min on GPU 0.
"""
from pathlib import Path
from src.config_generators.generate_configs import (
    compute_base_model_id, constraint_tag, save_configs,
)

DATA_DIR = "data/tissuemnist/slice_1"
SWEEP_ROOT = "results/pending_runs/beta_sweep"
SEED = 1
MODEL = "MobileNetV3"
PAIR = (0.3, 0.3)
CLASSES = (1, 4, 7)
BETAS = [0.0, 0.5, 1.0, 2.0]

DS = {
    "target_column": "label", "group_column": "synth_group",
    "num_classes": 8, "image_size": 224, "data_dir": DATA_DIR,
    "constrained_class": list(CLASSES),
}

BASE_HP = {
    "lr": 1e-4, "lr_constraint": 5e-6, "dropout": 0.3, "batch_size": 64,
    "warmup_epochs": 50, "constraint_epochs": 300,
    "use_sum_loss": True, "kl_temperature": 1.0, "pretrained": True,
    "class_weighted_ce": False, "constraint_chunk_size": 256,
    "lambda_global": 0.05, "lambda_local": 0.05, "lambda_step": 0.002,
    "initial_rho": 5.0, "rho_target": 100.0, "alpha_kl": 0.0,
    "penalty_mode": "both",
    "seed": SEED,
}


def main():
    cfgs = []
    for beta in BETAS:
        hp = dict(BASE_HP)
        hp["linear_sat_tail"] = beta
        tag = f"beta_{beta:.1f}".replace(".", "_")
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
            "exp_name": f"beta_sweep_{tag}_seed{SEED}",
            "status": "pending",
            "experiment_path": str(Path(SWEEP_ROOT) / tag),
        }
        cfgs.append(cfg)
    save_configs(cfgs, output_dir=SWEEP_ROOT)
    print(f"Created {len(cfgs)} configs: betas={BETAS}")


if __name__ == "__main__":
    main()
