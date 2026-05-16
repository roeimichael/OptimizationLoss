"""Master config gen for paper-table rerun (2026-05-15+).

Reads `/tmp/paper_rerun_plan.json` (produced by paper_inventory.py) and emits
a clean config tree under `results/pending_runs/paper_rerun/`. Tier flag picks
which subset to launch.

Usage:
    python -m src.config_generators.gen_paper_rerun --tier 1   # TissueMNIST MobV3 only
    python -m src.config_generators.gen_paper_rerun --tier 2   # +R18/EffB0 TissueMNIST
    python -m src.config_generators.gen_paper_rerun --tier all # full inventory
"""
import argparse
import json
from pathlib import Path
from src.config_generators.generate_configs import (
    compute_base_model_id, constraint_tag as ctag, save_configs,
)

SWEEP_ROOT = "results/pending_runs/paper_rerun"

DATASETS = {
    "tissuemnist": {"data_dir": "data/tissuemnist/slice_1", "num_classes": 8,
                    "image_size": 224, "target_column": "label",
                    "group_column": "synth_group"},
    "dermmnist": {"data_dir": "data/dermmnist/slice_1", "num_classes": 7,
                  "image_size": 224, "target_column": "label",
                  "group_column": "sex"},
    "eurosat": {"data_dir": "data/eurosat/slice_1", "num_classes": 10,
                "image_size": 224, "target_column": "label",
                "group_column": "synth_group"},
    "so2sat": {"data_dir": "data/so2sat", "num_classes": 17,
               "image_size": 224, "target_column": "label",
               "group_column": "city_id"},
}

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


def _tight_pair(tight_tag: str):
    """L30_G30 -> (0.3, 0.3)."""
    parts = tight_tag.split("_")
    return (int(parts[0][1:]) / 100, int(parts[1][1:]) / 100)


def make_cfg(method, dataset, model, cls, tight, seed):
    hp = dict(SHARED_HP)
    hp.update(PER_METHOD[method])
    hp["seed"] = seed
    ds_meta = DATASETS[dataset]
    cls = tuple(cls)
    constrained_class = list(cls) if len(cls) > 1 else cls[0]
    ds_config = {
        **ds_meta,
        "constrained_class": constrained_class,
    }
    pair = _tight_pair(tight)
    cls_tag = "_".join(str(c) for c in cls)
    return {
        "methodology": method,
        "model_name": model,
        "constraint": list(pair),
        "constraint_tag": tight,
        "dataset_mode": dataset,
        "dataset_config": ds_config,
        "hyperparams": hp,
        "base_model_id": compute_base_model_id(
            model, hp, dataset_mode=dataset,
            data_dir=ds_meta["data_dir"], dataset_config=ds_config),
        "exp_name": f"paper_rerun_{method}_{dataset}_{model}_cls{cls_tag}_{tight}_seed{seed}",
        "status": "pending",
        "experiment_path": str(
            Path(SWEEP_ROOT) / dataset / model / f"cls_{cls_tag}" / tight /
            method / f"seed_{seed}"),
    }


def filter_tier(cells, tier):
    """Tier 1: TissueMNIST MobileNetV3 only.
       Tier 2: +TissueMNIST other backbones.
       Tier 3: +other datasets MobileNetV3.
       Tier all: everything."""
    if tier == "all":
        return cells
    if tier == "1":
        return [c for c in cells if c[0] == "tissuemnist" and c[1] == "MobileNetV3"]
    if tier == "2":
        return [c for c in cells if c[0] == "tissuemnist"]
    if tier == "3":
        return [c for c in cells if c[1] == "MobileNetV3"]
    raise ValueError(f"unknown tier {tier!r}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", default="/tmp/paper_rerun_plan.json",
                    help="JSON plan from paper_inventory.py")
    ap.add_argument("--tier", default="1",
                    choices=["1", "2", "3", "all"],
                    help="Subset to materialize")
    args = ap.parse_args()

    plan = json.load(open(args.plan))
    cfgs = []
    counts = {}
    for method, cells in plan.items():
        sel = filter_tier(cells, args.tier)
        counts[method] = len(sel)
        for c in sel:
            dataset, model, cls, tight, seed = c
            # cls is a tuple-as-list when loaded from JSON
            if isinstance(cls, (list, tuple)):
                cls = tuple(cls)
            else:
                cls = (cls,)
            cfgs.append(make_cfg(method, dataset, model, cls, tight, int(seed)))
    save_configs(cfgs, output_dir=SWEEP_ROOT)
    print(f"Tier {args.tier}: {len(cfgs)} configs ({counts})")


if __name__ == "__main__":
    main()
