"""Convergence validation sweep — does TraLO converge across the design space?

The recent diag runs showed TraLO satisfies cleanly on the easy (L50_G50,
single class 4, MobileNetV3, seed_3) cell. Open question: does that
generalise, or were we celebrating the easiest cell?

5 methods x 3 tightnesses x 3 class scenarios x 1 seed = 45 configs.

  Methods   : tralo, hounie_rcl, fioretto_ldf, heuristic, danits_lp
  Tightness : L30_G30 (tight) / L50_G50 (medium) / L70_G70 (loose)
  Classes   : (4,)     -- single, well-studied
              (3, 4)   -- pair
              (1, 4, 7)-- multi-class (3 classes)

Constraint limits K_c are computed per-class from the train data by the
runner (constraint_tag pair gives the L%/G% multipliers). For multi-class
configs each constrained class gets its own K_c.

Backbone: MobileNetV3 (fast, matches headline). TissueMNIST slice_1.
Single seed (1) to keep wall time manageable (~3.5h on 1 GPU). If a cell
looks promising we can re-run with 3 seeds for variance.

Usage:
    python -m src.config_generators.gen_convergence_validation
"""
from pathlib import Path
from src.config_generators.generate_configs import (
    compute_base_model_id, constraint_tag, save_configs,
)

DATA_DIR = "data/tissuemnist/slice_1"
SWEEP_ROOT = "results/pending_runs/convergence_validation"
SEED = 1
MODEL = "MobileNetV3"
METHODS = ["tralo", "fioretto_ldf", "hounie_rcl", "heuristic", "danits_lp"]
TIGHTNESS_PAIRS = [(0.3, 0.3), (0.5, 0.5), (0.7, 0.7)]
CLASS_SCENARIOS = [
    (4,),         # single (existing headline)
    (3, 4),       # pair
    (1, 4, 7),    # 3-class
]

SHARED_HP = {
    "lr": 1e-4, "lr_constraint": 5e-6, "dropout": 0.3, "batch_size": 64,
    "warmup_epochs": 50, "constraint_epochs": 100,
    "use_sum_loss": True, "kl_temperature": 1.0, "pretrained": True,
    "class_weighted_ce": False, "constraint_chunk_size": 256,
}

PER_METHOD_HP = {
    "tralo": {"lambda_global": 0.05, "lambda_local": 0.05, "lambda_step": 0.002,
              "initial_rho": 5.0, "rho_target": 100.0, "alpha_kl": 0.0,
              "penalty_mode": "both"},
    "fioretto_ldf": {"fioretto_step_size": 0.005},
    "hounie_rcl": {"hounie_eta_lambda": 0.01, "hounie_eta_u": 0.01,
                   "hounie_alpha": 10.0},
    "heuristic": {},
    "danits_lp": {},
}


def main():
    cfgs = []
    for classes in CLASS_SCENARIOS:
        cls_tag = "_".join(str(c) for c in classes)
        constrained_class = list(classes) if len(classes) > 1 else classes[0]
        for pair in TIGHTNESS_PAIRS:
            tag = constraint_tag(pair)
            for method in METHODS:
                hp = dict(SHARED_HP)
                hp.update(PER_METHOD_HP[method])
                hp["seed"] = SEED
                ds = {
                    "target_column": "label", "group_column": "synth_group",
                    "num_classes": 8, "image_size": 224, "data_dir": DATA_DIR,
                    "constrained_class": constrained_class,
                }
                cfgs.append({
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
                    "exp_name": f"conv_val_cls{cls_tag}_{tag}_{method}_seed{SEED}",
                    "status": "pending",
                    "experiment_path": str(Path(SWEEP_ROOT) / f"cls_{cls_tag}" / tag / method),
                })

    save_configs(cfgs, output_dir=SWEEP_ROOT)
    print(f"Total: {len(cfgs)}")
    print(f"  classes x tightness x methods x seeds "
          f"= {len(CLASS_SCENARIOS)} x {len(TIGHTNESS_PAIRS)} x {len(METHODS)} x 1 "
          f"= {len(cfgs)}")


if __name__ == "__main__":
    main()
