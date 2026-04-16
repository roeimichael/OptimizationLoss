"""Multi-class constraint experiments: designed to break LP=heuristic degeneracy.

With a single constrained class + identity cost, LP and heuristic solve the
same trivial 1D sorting problem. Multiple constrained classes create competing
demands for sample reassignment — the LP can find global optima that the
sequential greedy heuristic cannot.

TissueMNIST test distribution (2400 samples, 8 classes):
  CDI=770 CDS=113 CST=85 EPI=223 GE=171 PTC=112 STR=569 TUB=357

Scenarios (all use tight constraint tier L30_G50 to force heavy redistribution):

  A: dual_GE_CST       — [4,2]     two small classes (171+85)
  B: dual_GE_STR       — [4,6]     small+large (171+569) — massive reallocation on STR
  C: triple_GE_CST_PTC — [4,2,5]   three rare classes (171+85+112)
  D: quad_rare          — [4,2,5,1] four rare classes (171+85+112+113)

Each: 3 methods (our_approach, heuristic, danits_lp) × 3 seeds (1,2,3) = 9 runs.
Total: 4 × 9 = 36 runs. our_approach uses diagnostic_level=2.
"""

from __future__ import annotations
from pathlib import Path

from src.config_generators.generate_configs import (
    compute_base_model_id, constraint_tag, save_configs,
)

MODEL = "MobileNetV3"
DATA_DIR = "data/tissuemnist/slice_1"
METHODS = ["our_approach", "heuristic", "danits_lp"]
SEEDS = [1, 2, 3]
# Tight constraints: local=30% of group count, global=50% of class count.
CONSTRAINT_PAIR = (0.3, 0.5)

SCENARIOS = {
    "dual_GE_CST": {
        "constrained_class": [4, 2],
        "desc": "GE(171)+CST(85) — two small competing classes",
    },
    "dual_GE_STR": {
        "constrained_class": [4, 6],
        "desc": "GE(171)+STR(569) — small+large, massive STR reallocation",
    },
    "triple_GE_CST_PTC": {
        "constrained_class": [4, 2, 5],
        "desc": "GE(171)+CST(85)+PTC(112) — three rare classes",
    },
    "quad_rare": {
        "constrained_class": [4, 2, 5, 1],
        "desc": "GE+CST+PTC+CDS — four rare classes (20% of test)",
    },
}

BASELINE_HP = {
    "lr": 0.0001,
    "lr_constraint": 5e-06,
    "dropout": 0.3,
    "batch_size": 64,
    "warmup_epochs": 50,
    "constraint_epochs": 300,
    "lambda_global": 0.01,
    "lambda_local": 0.01,
    "lambda_step": 0.002,
    "use_sum_loss": True,
    "initial_rho": 5.0,
    "rho_target": 100.0,
    "alpha_kl": 0.1,
    "kl_temperature": 1.0,
    "pretrained": True,
    "class_weighted_ce": False,
    "constraint_chunk_size": 64,
    "lambda_mode": "ratchet",
}


def _build(methodology, scenario_name, seed):
    hp = dict(BASELINE_HP)
    hp["seed"] = seed
    if methodology == "our_approach":
        hp["diagnostic_level"] = 2
    else:
        hp.pop("lambda_mode", None)

    sc = SCENARIOS[scenario_name]
    ctag = constraint_tag(CONSTRAINT_PAIR)
    variant = f"s{seed}"
    path = (Path("results/pending_runs") / scenario_name / ctag
            / MODEL / methodology / variant)
    ds = {
        "target_column": "label",
        "group_column": "synth_group",
        "num_classes": 8,
        "image_size": 224,
        "data_dir": DATA_DIR,
        "constrained_class": sc["constrained_class"],
    }
    return {
        "methodology": methodology,
        "model_name": MODEL,
        "constraint": list(CONSTRAINT_PAIR),
        "constraint_tag": ctag,
        "dataset_mode": "tissuemnist",
        "dataset_config": ds,
        "hyperparams": hp,
        "base_model_id": compute_base_model_id(
            MODEL, hp, dataset_mode="tissuemnist", data_dir=DATA_DIR),
        "exp_name": f"mc_{scenario_name}_{ctag}_{methodology}_s{seed}",
        "status": "pending",
        "experiment_path": str(path),
    }


def main():
    cfgs = []
    for sc_name in SCENARIOS:
        for meth in METHODS:
            for seed in SEEDS:
                cfgs.append(_build(meth, sc_name, seed))

    n_oa = sum(1 for c in cfgs if c["methodology"] == "our_approach")
    n_base = len(cfgs) - n_oa
    print("=" * 70)
    print("MULTI-CLASS CONSTRAINT SWEEP")
    print("=" * 70)
    print(f"Constraint tier: L{int(CONSTRAINT_PAIR[0]*100):02d}_G{int(CONSTRAINT_PAIR[1]*100):02d}")
    print(f"Scenarios: {len(SCENARIOS)}")
    for name, sc in SCENARIOS.items():
        print(f"  {name:25s} classes={sc['constrained_class']}  ({sc['desc']})")
    print(f"Methods: {METHODS}")
    print(f"Seeds: {SEEDS}")
    print(f"our_approach: {n_oa}  baselines: {n_base}  total: {len(cfgs)}")

    hashes = sorted({c["base_model_id"] for c in cfgs})
    print(f"Warmup hashes: {len(hashes)}")
    for h in hashes:
        print(f"  {h}")

    save_configs(cfgs, output_dir="results/pending_runs")


if __name__ == "__main__":
    main()
