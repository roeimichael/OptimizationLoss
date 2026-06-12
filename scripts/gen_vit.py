"""ViT-B/16 (SOTA transformer) smoke. Same recipe as the paper backbones (warmup50,
undershoot_hinge TraLO) for comparability. docs/REJECTED.md: ViT saturated tissue/derm
before -> probe-first. Two lanes:
  GPU2: vit_octmnist_s1  -- full dose-response (6 methods x L30/L50/L70, seed1); the
        one dataset that may stay un-saturated under ViT (drusen is hard).
  GPU3: vit_satcheck     -- tissue+derm saturation check (heuristic/fioretto/tralo x L50
        x seed1): do the methods flatline (saturated) or separate?
Run:
  CUDA_VISIBLE_DEVICES unset, feed GPU index. e.g.
  EXPERIMENT_DIR=results/pending_runs/vit_octmnist_s1 python main.py <<< 2
  EXPERIMENT_DIR=results/pending_runs/vit_satcheck    python main.py <<< 3
"""
from src.config_generators.generate_configs import compute_base_model_id, save_configs

MODEL = "ViTB16"
DSMETA = {
    "tissuemnist": {"data_dir": "data/tissuemnist/slice_1", "num_classes": 8,
                    "image_size": 224, "target_column": "label",
                    "group_column": "synth_group", "constrained_class": 4},
    "dermmnist":   {"data_dir": "data/dermmnist/slice_1", "num_classes": 7,
                    "image_size": 224, "target_column": "label",
                    "group_column": "loc_group", "constrained_class": 4},
    "octmnist":    {"data_dir": "data/octmnist/slice_1", "num_classes": 4,
                    "image_size": 224, "target_column": "label",
                    "group_column": "synth_group", "constrained_class": 2},
}

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
ALL6 = list(PER_METHOD.keys())
PROBE3 = ["heuristic", "fioretto_ldf", "tralo"]

# (dataset, methods, tights, seeds, root)
SPEC = [
    ("octmnist", ALL6, ["L30_G30", "L50_G50", "L70_G70"], [1], "results/pending_runs/vit_octmnist_s1"),
    ("tissuemnist", PROBE3, ["L50_G50"], [1], "results/pending_runs/vit_satcheck"),
    ("dermmnist",   PROBE3, ["L50_G50"], [1], "results/pending_runs/vit_satcheck"),
]


def _pair(tag):
    p = tag.split("_")
    return (int(p[0][1:]) / 100, int(p[1][1:]) / 100)


def make_cfg(ds, method, tight, seed, root):
    ds_config = dict(DSMETA[ds])
    hp = {**SHARED_HP, **PER_METHOD[method], "seed": seed}
    bmid = compute_base_model_id(MODEL, hp, dataset_mode=ds, data_dir=ds_config["data_dir"],
                                 dataset_config=ds_config)
    return {"methodology": method, "model_name": MODEL, "constraint": list(_pair(tight)),
            "constraint_tag": tight, "dataset_mode": ds, "dataset_config": ds_config,
            "hyperparams": hp, "base_model_id": bmid,
            "exp_name": f"vit_{ds}_{method}_{tight}_seed{seed}",
            "experiment_path": f"{root}/{MODEL}/{ds}/{tight}/{method}/seed_{seed}"}


def main():
    by_root = {}
    for ds, methods, tights, seeds, root in SPEC:
        cfgs = [make_cfg(ds, m, t, s, root) for m in methods for t in tights for s in seeds]
        by_root.setdefault(root, []).extend(cfgs)
    for root, cfgs in by_root.items():
        save_configs(cfgs, output_dir=root)
        print(f"{root}: {len(cfgs)} configs")


if __name__ == "__main__":
    main()
