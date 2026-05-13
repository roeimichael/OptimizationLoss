"""300-epoch version of the convergence validation sweep.

Same 45 configs as gen_convergence_validation.py (5 methods x 3 tight x
3 class-sets x 1 seed) but with constraint_epochs=300 to test whether
TraLO keeps descending past E150 or plateaus / drifts upward.

The 100-epoch run showed TraLO still on a descending sumex trajectory
at E150 for tight/medium cells. This sweep extends the horizon to see
where it lands.

Wall-time estimate (GPU 0, MobileNetV3, warmup cached):
  27 trainable methods x ~22.5 min = ~10 h
  18 heuristic/danits x <1 min = ~10 min
  Total: ~10 hours
"""
from pathlib import Path
from src.config_generators.generate_configs import (
    compute_base_model_id, constraint_tag, save_configs,
)

DATA_DIR = "data/tissuemnist/slice_1"
SWEEP_ROOT = "results/pending_runs/convergence_validation_300"
SEED = 1
MODEL = "MobileNetV3"
METHODS = ["tralo", "fioretto_ldf", "hounie_rcl", "heuristic", "danits_lp"]
TIGHTNESS_PAIRS = [(0.3, 0.3), (0.5, 0.5), (0.7, 0.7)]
CLASS_SCENARIOS = [(4,), (3, 4), (1, 4, 7)]

SHARED_HP = {
    "lr": 1e-4, "lr_constraint": 5e-6, "dropout": 0.3, "batch_size": 64,
    "warmup_epochs": 50, "constraint_epochs": 300,
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
                    "exp_name": f"conv_val_300_cls{cls_tag}_{tag}_{method}_seed{SEED}",
                    "status": "pending",
                    "experiment_path": str(Path(SWEEP_ROOT) / f"cls_{cls_tag}" / tag / method),
                })

    save_configs(cfgs, output_dir=SWEEP_ROOT)
    print(f"Total: {len(cfgs)} configs at constraint_epochs=300")


if __name__ == "__main__":
    main()
