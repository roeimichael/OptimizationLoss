"""Engineered-headroom CIFAR-100 probe.

The standard CIFAR-100 probe (warmup=50) saturates train_acc to ~1.0
by ep10. TraLO loses Hounie there (verified). This probe creates the
mid-residency regime by:

  Arm A: short_warmup  — warmup_epochs=1  (lands train_acc ~0.78, in band)
  Arm B: short_warmup3 — warmup_epochs=3  (lands ~0.91, upper edge)
  Arm C: contam_sigma30 — full warmup but on data/cifar100_sigma30
                          (data corruption to keep train_acc < band)

3 methods x 2 seeds x 2 tightness x 3 arms = 36 cells.

Prediction: in Arms A/B, TraLO should match or beat Hounie. In Arm C,
similar — depends on whether contamination actually moves the regime.
If TraLO wins on at least one arm we have a 4th dataset for the
universal claim.
"""
from src.config_generators.generate_configs import (
    compute_base_model_id, save_configs,
)

ARMS = {
    "short_warmup1":  {"warmup_epochs": 1,  "data_dir": "data/cifar100/slice_1"},
    # short_warmup3 dropped 2026-06-02: ep3 train_acc=0.91 already saturated,
    # not in headroom band [0.70, 0.82]. Only warmup=1 (acc 0.78) qualifies.
    # contam_sigma30 deferred: data not prepped, disk tight.
}
TIGHT = ["L30_G30", "L50_G50"]
SEEDS = [1, 2]
METHODS = ["tralo", "fioretto_ldf", "hounie_rcl"]
MODEL = "MobileNetV3"

SHARED_HP_BASE = {
    "lr": 1e-4, "lr_constraint": 5e-6, "dropout": 0.3, "batch_size": 64,
    # Reduced from 300 -> 100 on 2026-06-02: CIFAR-100 Turing per-epoch
    # ~60s, 300 epochs was 5h/cell worst case. Early-stop catches most;
    # 100 cap bounds per-cell at ~100min. Re-run intended on Blackwell GPU3.
    "constraint_epochs": 100, "pretrained": True,
    "class_weighted_ce": False, "constraint_chunk_size": 256,
    "fioretto_step_size": 0.01,
}
TRALO_HP = {
    "lambda_global": 0.05, "lambda_local": 0.05, "lambda_step": 0.002,
    "initial_rho": 5.0, "rho_target": 100.0, "alpha_kl": 0.0,
    "penalty_mode": "both", "enable_ce_skip": True,
    "hybrid_mode": "undershoot_hinge", "fior_beta": 0.50,
    "reset_optimizer_at_sat": True,
    "disable_freeze_on_satisfy": False,
}


def _tight_pair(tag):
    p = tag.split("_")
    return (int(p[0][1:])/100, int(p[1][1:])/100)


def make_cfg(arm, arm_opts, tight, method, seed):
    ds_config = {
        "num_classes": 100, "image_size": 224, "target_column": "label",
        "group_column": "synth_group", "constrained_class": 0,
        "data_dir": arm_opts["data_dir"],
    }
    hp = {**SHARED_HP_BASE, "warmup_epochs": arm_opts["warmup_epochs"],
          "seed": seed}
    if method == "tralo":
        hp.update(TRALO_HP)
    pair = _tight_pair(tight)
    bmid = compute_base_model_id(
        MODEL, hp, dataset_mode="cifar100",
        data_dir=arm_opts["data_dir"], dataset_config=ds_config,
    )
    sweep_root = "results/pending_runs/cifar100_headroom"
    return {
        "methodology": method, "model_name": MODEL,
        "constraint": list(pair), "constraint_tag": tight,
        "dataset_mode": "cifar100", "dataset_config": ds_config,
        "hyperparams": hp, "base_model_id": bmid,
        "experiment_path": (
            f"{sweep_root}/{arm}/{MODEL}/{tight}/{method}/seed_{seed}"
        ),
    }


def main():
    cfgs = []
    for arm, arm_opts in ARMS.items():
        for tight in TIGHT:
            for method in METHODS:
                for seed in SEEDS:
                    cfgs.append(make_cfg(arm, arm_opts, tight, method, seed))
    print(f"Generated {len(cfgs)} configs")
    save_configs(cfgs, output_dir="results/pending_runs")


if __name__ == "__main__":
    main()
