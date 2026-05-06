"""Overnight sweep: compare 5 methodologies across constraint axes.

Axes covered:
  - constraint_pairs: (0.3,0.3), (0.5,0.5), (0.7,0.7) — tightness
  - constrained classes: (4,), (3,4), (1,4,7) — count + identity
  - asymmetric: (0.3,0.5) and (0.5,0.3) — local vs global tightness
  - lambda settings (tralo only): step, alpha_kl, initial_rho

All configs share the same base_model_id (warmup cache reused). Output:
  results/pending_runs/overnight_sweep/<axis>/<scenario>/<methodology>/

Usage:
    python -m src.config_generators.gen_overnight_sweep
"""
from pathlib import Path
from src.config_generators.generate_configs import (
    compute_base_model_id, constraint_tag, save_configs,
)

MODEL = "MobileNetV3"
DATA_DIR = "data/tissuemnist/slice_1"
SEED = 1
SWEEP_ROOT = "results/pending_runs/overnight_sweep"
METHODS = ["tralo", "fioretto_ldf", "hounie_rcl", "heuristic", "danits_lp"]

SHARED_HP = {
    "lr": 1e-4,
    "lr_constraint": 5e-6,
    "dropout": 0.3,
    "batch_size": 64,
    "warmup_epochs": 50,
    "constraint_epochs": 100,
    "use_sum_loss": True,
    "kl_temperature": 1.0,
    "pretrained": True,
    "class_weighted_ce": False,
    "constraint_chunk_size": 256,
    "seed": SEED,
}

PER_METHOD_HP = {
    "tralo": {"lambda_global": 0.01, "lambda_local": 0.01, "lambda_step": 0.002,
              "initial_rho": 5.0, "rho_target": 100.0, "alpha_kl": 0.0},
    "fioretto_ldf": {"fioretto_step_size": 0.005},
    "hounie_rcl": {"hounie_eta_lambda": 0.01, "hounie_eta_u": 0.01, "hounie_alpha": 10.0},
    "heuristic": {},
    "danits_lp": {},
}

DS_BASE = {
    "target_column": "label",
    "group_column": "synth_group",
    "num_classes": 8,
    "image_size": 224,
    "data_dir": DATA_DIR,
}


def build(methodology, pair, classes, scenario_name, axis, hp_override=None):
    hp = dict(SHARED_HP)
    hp.update(PER_METHOD_HP[methodology])
    if hp_override:
        hp.update(hp_override)
    ds = dict(DS_BASE)
    ds["constrained_class"] = list(classes) if len(classes) > 1 else classes[0]
    ctag = constraint_tag(pair)
    path = Path(SWEEP_ROOT) / axis / scenario_name / methodology
    return {
        "methodology": methodology,
        "model_name": MODEL,
        "constraint": list(pair),
        "constraint_tag": ctag,
        "dataset_mode": "tissuemnist",
        "dataset_config": ds,
        "hyperparams": hp,
        "base_model_id": compute_base_model_id(
            MODEL, hp, dataset_mode="tissuemnist",
            data_dir=DATA_DIR, dataset_config=ds),
        "exp_name": f"sweep_{axis}_{scenario_name}_{methodology}",
        "status": "pending",
        "experiment_path": str(path),
    }


def main():
    cfgs = []

    # Axis 1: tightness — vary pair, fix class=4
    for pair in [(0.3, 0.3), (0.7, 0.7)]:  # skip (0.5,0.5), already in smoke_5way
        for m in METHODS:
            cfgs.append(build(m, pair, (4,),
                              f"L{int(pair[0]*100):02d}_G{int(pair[1]*100):02d}_class4",
                              "tightness"))

    # Axis 2: constrained classes — fix pair=(0.5,0.5)
    for classes in [(3, 4), (1, 4, 7)]:
        cls_tag = "_".join(str(c) for c in classes)
        for m in METHODS:
            cfgs.append(build(m, (0.5, 0.5), classes,
                              f"L50_G50_classes_{cls_tag}", "multiclass"))

    # Axis 3: asymmetric — local tighter / global tighter
    for pair, name in [((0.3, 0.5), "local_tighter"), ((0.5, 0.3), "global_tighter")]:
        for m in METHODS:
            cfgs.append(build(m, pair, (4,), f"{name}_class4", "asymmetric"))

    # Axis 4: tralo lambda sweep — fix pair=(0.5,0.5), class=4
    lambda_sweep = [
        ("step_005", {"lambda_step": 0.005}),
        ("step_01", {"lambda_step": 0.01}),
        ("step_05", {"lambda_step": 0.05}),
        ("kl_01", {"alpha_kl": 0.1}),
        ("kl_05", {"alpha_kl": 0.5}),
        ("rho_init_01", {"initial_rho": 0.1}),
        ("rho_init_50", {"initial_rho": 50.0}),
        ("init_lam_05", {"lambda_global": 0.05, "lambda_local": 0.05}),
    ]
    for name, hp_override in lambda_sweep:
        cfgs.append(build("tralo", (0.5, 0.5), (4,),
                          f"lambda_{name}", "tralo_hp", hp_override=hp_override))

    save_configs(cfgs, output_dir=SWEEP_ROOT)
    print(f"Generated {len(cfgs)} configs across "
          f"{len(set(c['experiment_path'].split('/')[2] for c in cfgs))} axes")


if __name__ == "__main__":
    main()
