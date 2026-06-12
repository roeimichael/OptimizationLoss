"""Fill the OctMNIST gap for the other two paper backbones (MobileNetV2, RegNetY400MF)
so the final multi-backbone tables/figs have no missing OctMNIST data points.
EXACT same recipe + ds metadata as the octmnist MobileNetV3 runs (scripts/gen_octmnist_probe.py)
so cells are directly comparable: 5 tightness x 6 methods x 4 seeds = 120 cells per backbone.
Per (model, seed) dir -> each GPU lane trains its own warmup once, no cache race.
Run (one lane per model+seed, two GPUs, chained):
  EXPERIMENT_DIR=results/pending_runs/octmnist_MobileNetV2_s1 python main.py <<< 2
  ...
"""
from src.config_generators.generate_configs import compute_base_model_id, save_configs

DATASET = "octmnist"
DATA_DIR = "data/octmnist/slice_1"
NUM_CLASSES = 4
CONSTRAINED_CLASS = 2  # drusen
GROUP_COLUMN = "synth_group"
MODELS = ["MobileNetV2", "RegNetY400MF"]
TIGHT = ["L20_G20", "L30_G30", "L50_G50", "L70_G70", "L80_G80"]
SEEDS = [1, 2, 3, 4]

SHARED_HP = {
    "lr": 1e-4, "lr_constraint": 5e-6, "dropout": 0.3, "batch_size": 64,
    "warmup_epochs": 50, "constraint_epochs": 300, "pretrained": True,
    "class_weighted_ce": False, "constraint_chunk_size": 256,
}
PER_METHOD = {
    "heuristic": {}, "danits_lp": {},
    "fioretto_ldf": {"fioretto_step_size": 0.005},
    "hounie_rcl": {"hounie_eta_lambda": 0.01, "hounie_eta_u": 0.01, "hounie_alpha": 10.0},
    "tralo_bounded": {"lambda_global": 0.05, "lambda_local": 0.05, "lambda_step": 0.002,
                      "initial_rho": 5.0, "rho_target": 100.0, "alpha_kl": 0.0,
                      "penalty_mode": "both", "enable_ce_skip": True},
    "tralo": {"lambda_global": 0.05, "lambda_local": 0.05, "lambda_step": 0.002,
              "initial_rho": 5.0, "rho_target": 100.0, "alpha_kl": 0.0,
              "penalty_mode": "both", "enable_ce_skip": True,
              "hybrid_mode": "undershoot_hinge", "fior_beta": 0.50,
              "reset_optimizer_at_sat": True},
}
METHODS = list(PER_METHOD.keys())


def _pair(tag):
    p = tag.split("_")
    return (int(p[0][1:]) / 100, int(p[1][1:]) / 100)


def make_cfg(model, tight, method, seed, root):
    ds_config = {"num_classes": NUM_CLASSES, "image_size": 224, "target_column": "label",
                 "group_column": GROUP_COLUMN, "constrained_class": CONSTRAINED_CLASS,
                 "data_dir": DATA_DIR}
    hp = {**SHARED_HP, **PER_METHOD[method], "seed": seed}
    bmid = compute_base_model_id(model, hp, dataset_mode=DATASET, data_dir=DATA_DIR,
                                 dataset_config=ds_config)
    return {"methodology": method, "model_name": model, "constraint": list(_pair(tight)),
            "constraint_tag": tight, "dataset_mode": DATASET, "dataset_config": ds_config,
            "hyperparams": hp, "base_model_id": bmid,
            "exp_name": f"octbb_{model}_{method}_{tight}_seed{seed}",
            "experiment_path": f"{root}/{model}/{tight}/{method}/seed_{seed}"}


def main():
    total = 0
    for model in MODELS:
        for seed in SEEDS:
            root = f"results/pending_runs/octmnist_{model}_s{seed}"
            cfgs = [make_cfg(model, t, m, seed, root) for t in TIGHT for m in METHODS]
            save_configs(cfgs, output_dir=root)
            total += len(cfgs)
            print(f"{model} seed{seed}: {len(cfgs)} -> {root}")
    print(f"TOTAL {total} configs ({len(MODELS)} backbones x {len(SEEDS)} seeds x "
          f"{len(TIGHT)} tight x {len(METHODS)} methods)")


if __name__ == "__main__":
    main()
