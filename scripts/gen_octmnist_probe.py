"""OctMNIST probe-smoke: is it (a) in the headroom band, (b) a regime where TraLO separates?
Constrained class 2 (drusen), group synth_group. Headline recipe (warmup50, undershoot_hinge).
6 methods x {L30,L50}. Seed 1 -> octmnist_s1 (GPU2), seed 2 -> octmnist_s2 (GPU3): separate
dirs so each GPU trains its own warmup once (no cache race). Run:
  CUDA_VISIBLE_DEVICES=2 EXPERIMENT_DIR=results/pending_runs/octmnist_s1 python main.py <<< all
  CUDA_VISIBLE_DEVICES=3 EXPERIMENT_DIR=results/pending_runs/octmnist_s2 python main.py <<< all
"""
from src.config_generators.generate_configs import compute_base_model_id, save_configs

DATASET = "octmnist"
DATA_DIR = "data/octmnist/slice_1"
NUM_CLASSES = 4
CONSTRAINED_CLASS = 2  # drusen
GROUP_COLUMN = "synth_group"
MODEL = "MobileNetV3"
TIGHT = ["L20_G20", "L30_G30", "L50_G50", "L70_G70", "L80_G80"]
SEEDS = [1, 2, 3, 4]

SHARED_HP = {
    "lr": 1e-4, "lr_constraint": 5e-6, "dropout": 0.3, "batch_size": 64,
    "warmup_epochs": 50, "constraint_epochs": 300, "pretrained": True,
    "class_weighted_ce": False, "constraint_chunk_size": 256,
}
# Ordered post-hoc FIRST: the warmup-band + over-prediction gate is readable after the
# (fast) heuristic run, before the long trained methods churn -> killable if the gate fails.
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


def make_cfg(tight, method, seed, root):
    ds_config = {"num_classes": NUM_CLASSES, "image_size": 224, "target_column": "label",
                 "group_column": GROUP_COLUMN, "constrained_class": CONSTRAINED_CLASS,
                 "data_dir": DATA_DIR}
    hp = {**SHARED_HP, **PER_METHOD[method], "seed": seed}
    bmid = compute_base_model_id(MODEL, hp, dataset_mode=DATASET, data_dir=DATA_DIR,
                                 dataset_config=ds_config)
    return {"methodology": method, "model_name": MODEL, "constraint": list(_pair(tight)),
            "constraint_tag": tight, "dataset_mode": DATASET, "dataset_config": ds_config,
            "hyperparams": hp, "base_model_id": bmid,
            "exp_name": f"oct_{method}_{tight}_seed{seed}",
            "experiment_path": f"{root}/{MODEL}/{tight}/{method}/seed_{seed}"}


def main():
    for seed in SEEDS:
        root = f"results/pending_runs/octmnist_s{seed}"
        cfgs = [make_cfg(t, m, seed, root) for t in TIGHT for m in METHODS]
        save_configs(cfgs, output_dir=root)
        print(f"seed {seed}: {len(cfgs)} configs -> {root}")


if __name__ == "__main__":
    main()
