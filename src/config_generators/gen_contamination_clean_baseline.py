"""Clean (sigma=0) baseline matching the contamination grid exactly.

Same HPs/methods/seeds/tightness as gen_contamination_grid.py but using the
original (uncontaminated) data dirs. Lets us plot sigma=0 as a fair anchor
on the contamination-grid graphs.

Grid: 3 datasets x 4 tightness x 5 methods x 2 seeds = 120 cells.
Output: results/pending_runs/contamination_clean/
"""
from pathlib import Path

from src.config_generators.generate_configs import (
    compute_base_model_id, save_configs,
)
from src.config_generators.gen_contamination_grid import (
    DATASETS, TIGHTNESS, SEEDS, METHODS, MODEL, SHARED_HP, TRALO_HP, _tight_pair,
)

SWEEP_ROOT = "results/pending_runs/contamination_clean"

CLEAN_DATA_DIR = {
    "tissuemnist": "data/tissuemnist/slice_1",
    "dermmnist":   "data/dermmnist/slice_1",
    "aider":       "data/aider/slice_1",
}


def make_cfg(dataset, ds_base, tight, method, seed):
    data_dir = CLEAN_DATA_DIR[dataset]
    ds_config = {**ds_base, "data_dir": data_dir}
    hp = {**SHARED_HP, "seed": seed}
    if method == "tralo":
        hp.update(TRALO_HP)
    pair = _tight_pair(tight)
    bmid = compute_base_model_id(
        MODEL, hp, dataset_mode=dataset,
        data_dir=data_dir, dataset_config=ds_config,
    )
    return {
        "methodology": method,
        "model_name": MODEL,
        "constraint": list(pair),
        "constraint_tag": tight,
        "dataset_mode": dataset,
        "dataset_config": ds_config,
        "hyperparams": hp,
        "base_model_id": bmid,
        "exp_name": f"clean_{dataset}_{tight}_{method}_seed{seed}",
        "experiment_path": str(
            Path(SWEEP_ROOT) / dataset / tight / method / f"seed_{seed}"),
    }


def build():
    cfgs = []
    for dataset, ds_base in DATASETS.items():
        for tight in TIGHTNESS:
            for method in METHODS:
                for seed in SEEDS:
                    cfgs.append(make_cfg(dataset, ds_base, tight, method, seed))
    save_configs(cfgs, output_dir=SWEEP_ROOT)
    print(f"\nGenerated {len(cfgs)} clean-baseline configs -> {SWEEP_ROOT}")


if __name__ == "__main__":
    build()
