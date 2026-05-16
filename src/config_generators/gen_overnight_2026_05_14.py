"""Overnight sweep 2026-05-14: TraLO convergence + benchmark comparison.

Plan:
- TraLO: 3 cls × 3 tightness × 3 seeds = 27 runs (~2h with CE-skip early-stop)
- Fioretto + Hounie: 1 cls (1,4,7 hardest) × 3 tightness × 3 seeds × 2 methods
  = 18 runs (~7h, full 300 epochs)
Total ≈ 45 runs, ~9h on single GPU.

Goals:
1. Confirm TraLO converges across (4), (3,4), (1,4,7) cells × 3 tightness × 3 seeds.
2. Compare F1m + acc to Fioretto + Hounie on hardest (1,4,7) cell with N=3 seeds.
3. Identify cells where TraLO struggles to inform future fixes.
"""
from pathlib import Path
from src.config_generators.generate_configs import (
    compute_base_model_id, constraint_tag, save_configs,
)

DATA_DIR = "data/tissuemnist/slice_1"
SWEEP_ROOT = "results/pending_runs/overnight_2026_05_14"
MODEL = "MobileNetV3"
TIGHTNESS = [(0.3, 0.3), (0.5, 0.5), (0.7, 0.7)]
SEEDS = [1, 2, 3]
ALL_CLASSES = [(4,), (3, 4), (1, 4, 7)]
HARD_CLASSES = (1, 4, 7)

SHARED_HP = {
    "lr": 1e-4, "lr_constraint": 5e-6, "dropout": 0.3, "batch_size": 64,
    "warmup_epochs": 50, "constraint_epochs": 300,
    "use_sum_loss": True, "kl_temperature": 1.0, "pretrained": True,
    "class_weighted_ce": False, "constraint_chunk_size": 256,
}

PER_METHOD = {
    "tralo": {"lambda_global": 0.05, "lambda_local": 0.05, "lambda_step": 0.002,
              "initial_rho": 5.0, "rho_target": 100.0, "alpha_kl": 0.0,
              "penalty_mode": "both", "enable_ce_skip": True},
    "fioretto_ldf": {"fioretto_step_size": 0.005},
    "hounie_rcl": {"hounie_eta_lambda": 0.01, "hounie_eta_u": 0.01,
                   "hounie_alpha": 10.0},
}


def make_cfg(method, classes, pair, seed):
    hp = dict(SHARED_HP)
    hp.update(PER_METHOD[method])
    hp["seed"] = seed
    cls_tag = "_".join(str(c) for c in classes)
    tag = constraint_tag(pair)
    constrained_class = list(classes) if len(classes) > 1 else classes[0]
    ds = {
        "target_column": "label", "group_column": "synth_group",
        "num_classes": 8, "image_size": 224, "data_dir": DATA_DIR,
        "constrained_class": constrained_class,
    }
    return {
        "methodology": method,
        "model_name": MODEL,
        "constraint": list(pair),
        "constraint_tag": tag,
        "dataset_mode": "tissuemnist",
        "dataset_config": ds,
        "hyperparams": hp,
        "base_model_id": compute_base_model_id(
            MODEL, hp, dataset_mode="tissuemnist",
            data_dir=DATA_DIR, dataset_config=ds),
        "exp_name": f"overnight_cls{cls_tag}_{tag}_{method}_seed{seed}",
        "status": "pending",
        "experiment_path": str(
            Path(SWEEP_ROOT) / method / f"cls_{cls_tag}" / tag / f"seed_{seed}"),
    }


def main():
    cfgs = []
    # TraLO across full grid
    for classes in ALL_CLASSES:
        for pair in TIGHTNESS:
            for seed in SEEDS:
                cfgs.append(make_cfg("tralo", classes, pair, seed))
    # Fioretto + Hounie only on hardest class config
    for method in ("fioretto_ldf", "hounie_rcl"):
        for pair in TIGHTNESS:
            for seed in SEEDS:
                cfgs.append(make_cfg(method, HARD_CLASSES, pair, seed))

    save_configs(cfgs, output_dir=SWEEP_ROOT)
    n_tralo = sum(1 for c in cfgs if c["methodology"] == "tralo")
    n_bench = len(cfgs) - n_tralo
    print(f"Total: {len(cfgs)} configs ({n_tralo} TraLO + {n_bench} benchmarks)")


if __name__ == "__main__":
    main()
