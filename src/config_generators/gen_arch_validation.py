"""Architecture validation: reproduce key Turing cells on Blackwell.

Goal: verify TraLO_fix > vanilla + Fioretto + Hounie holds on Blackwell
(dsisco02) too, not just Turing (dsisco01). Cross-GPU determinism is
imperfect (different cudnn kernels, atomicAdd ordering, BF16 paths),
so we re-run the strongest-signal cells and check the direction holds.

Cells: 2 datasets x 2 tightness x 4 methods x 2 seeds = 32 configs.

Setup: existing Turing-built warmup caches for these (model, dataset, seed)
combos MUST be moved aside before launching so dsisco02 rebuilds warmup
on Blackwell from scratch (full Blackwell pipeline, not just constraint).

Output: results/pending_runs/arch_validation
"""
from pathlib import Path

from src.config_generators.generate_configs import (
    compute_base_model_id, save_configs,
)

SWEEP_ROOT = "results/pending_runs/arch_validation"

DATASETS = {
    "tissuemnist": {"data_dir": "data/tissuemnist/slice_1", "num_classes": 8,
                    "image_size": 224, "target_column": "label",
                    "group_column": "synth_group"},
    "eurosat": {"data_dir": "data/eurosat/slice_1", "num_classes": 10,
                "image_size": 224, "target_column": "label",
                "group_column": "synth_group"},
}
TIGHTNESS = ["L30_G30", "L50_G50"]
SEEDS = [1, 2]
MODEL = "MobileNetV3"
CLS = 4

SHARED_HP = {
    "lr": 1e-4, "lr_constraint": 5e-6, "dropout": 0.3, "batch_size": 64,
    "warmup_epochs": 50, "constraint_epochs": 300, "pretrained": True,
    "class_weighted_ce": False, "constraint_chunk_size": 256,
}

PER_METHOD = {
    "tralo": {  # the breakthrough (was tralo_fioretto)
        "lambda_global": 0.05, "lambda_local": 0.05, "lambda_step": 0.002,
        "initial_rho": 5.0, "rho_target": 100.0, "alpha_kl": 0.0,
        "penalty_mode": "both", "enable_ce_skip": True,
        "hybrid_mode": "undershoot_hinge", "fior_beta": 0.50,
        "reset_optimizer_at_sat": True,
    },
    "tralo_bounded": {  # vanilla bounded-only baseline
        "lambda_global": 0.05, "lambda_local": 0.05, "lambda_step": 0.002,
        "initial_rho": 5.0, "rho_target": 100.0, "alpha_kl": 0.0,
        "penalty_mode": "both", "enable_ce_skip": True,
    },
    "fioretto_ldf": {"fioretto_step_size": 0.005},
    "hounie_rcl": {"hounie_eta_lambda": 0.01, "hounie_eta_u": 0.01,
                   "hounie_alpha": 10.0},
}


def _tight_pair(tag):
    parts = tag.split("_")
    return (int(parts[0][1:]) / 100, int(parts[1][1:]) / 100)


def make_cfg(method, dataset, tight_tag, seed):
    ds_meta = DATASETS[dataset]
    hp = {**SHARED_HP, **PER_METHOD[method], "seed": seed}
    ds_config = {**ds_meta, "constrained_class": CLS}
    pair = _tight_pair(tight_tag)
    bmid = compute_base_model_id(
        MODEL, hp, dataset_mode=dataset,
        data_dir=ds_meta["data_dir"], dataset_config=ds_config,
    )
    return {
        "methodology": method,
        "model_name": MODEL,
        "constraint": list(pair),
        "constraint_tag": tight_tag,
        "dataset_mode": dataset,
        "dataset_config": ds_config,
        "hyperparams": hp,
        "base_model_id": bmid,
        "exp_name": f"arch_{method}_{dataset}_{tight_tag}_seed{seed}",
        "experiment_path": str(
            Path(SWEEP_ROOT) / dataset / tight_tag / method / f"seed_{seed}"),
    }


def get_cache_ids_to_invalidate():
    """Return the base_model_id hashes whose warmup caches must be moved
    aside before launching, so Blackwell rebuilds from scratch."""
    ids = set()
    sample_method_hp = PER_METHOD["tralo"]  # warmup HP doesn't depend on method
    for ds, meta in DATASETS.items():
        for seed in SEEDS:
            hp = {**SHARED_HP, **sample_method_hp, "seed": seed}
            ds_config = {**meta, "constrained_class": CLS}
            bmid = compute_base_model_id(
                MODEL, hp, dataset_mode=ds,
                data_dir=meta["data_dir"], dataset_config=ds_config,
            )
            ids.add(bmid)
    return sorted(ids)


def build():
    cfgs = []
    for dataset in DATASETS:
        for tight in TIGHTNESS:
            for method in PER_METHOD:
                for seed in SEEDS:
                    cfgs.append(make_cfg(method, dataset, tight, seed))
    save_configs(cfgs, output_dir=SWEEP_ROOT)
    print(f"\nGenerated {len(cfgs)} arch_validation configs -> {SWEEP_ROOT}")
    print("\nWarmup cache IDs to invalidate (move aside) before Blackwell run:")
    for bmid in get_cache_ids_to_invalidate():
        print(f"  model_cache/{bmid}.pt")


if __name__ == "__main__":
    build()
