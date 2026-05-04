"""Generate configs for ablation study (ConvNeXtTiny) and EfficientNetB0 LR sweep.

Ablation: ConvNeXtTiny on L50_G50, both datasets, both scenarios, 5 slices each.
  6 ablation variants × 2 datasets × 2 scenarios × 5 slices = 120 experiments

LR sweep: EfficientNetB0 on L50_G50, both datasets, both scenarios, 5 slices each.
  3 LR values × 2 datasets × 2 scenarios × 5 slices = 60 experiments

Total: 180 new experiments.
Output: results/pending_runs/{dataset}/{scenario}/L50_G50/{model}/{variant}/slice_N/

Usage:
    python -m analysis.generate_ablation_configs [--dry-run]
"""

import argparse
import copy
import hashlib
import json
from pathlib import Path

from src.config_generators.generate_configs import (
    HYPERPARAMS, compute_base_model_id, constraint_tag)

# ── Shared settings ──

CONSTRAINT_PAIR = (0.5, 0.5)  # L50_G50 — best balanced win rate
CTAG = constraint_tag(CONSTRAINT_PAIR)
NUM_SLICES = 5

DATASETS = {
    'dermmnist': {
        'dataset_mode': 'dermmnist',
        'num_classes': 7,
        'target_column': 'label',
        'group_column': 'loc_group',
        'image_size': 224,
        'scenarios': {
            'single_MEL': {'constrained_class': 4},
            'multi_MEL_BCC': {'constrained_class': [4, 1]},
        },
    },
    'tissuemnist': {
        'dataset_mode': 'tissuemnist',
        'num_classes': 8,
        'target_column': 'label',
        'group_column': 'synth_group',
        'image_size': 224,
        'scenarios': {
            'single_GE': {'constrained_class': 4},
            'multi_GE_PTC': {'constrained_class': [4, 5]},
        },
    },
}

# ── Ablation variants (ConvNeXtTiny) ──
# Each variant overrides specific hyperparams from the baseline.

ABLATION_VARIANTS = {
    'no_kl': {
        'description': 'Remove KL regularization (alpha_kl=0)',
        'overrides': {'alpha_kl': 0.0},
    },
    'fixed_rho': {
        'description': 'Fixed rho (no annealing schedule)',
        'overrides': {'initial_rho': 5.0, 'rho_target': 5.0},
    },
    'no_lambda_ratchet': {
        'description': 'Fixed lambdas (no lambda_step increment)',
        'overrides': {'lambda_step': 0.0},
    },
    'high_lambda': {
        'description': '10x initial lambda weights',
        'overrides': {'lambda_global': 0.1, 'lambda_local': 0.1},
    },
    'low_rho': {
        'description': 'Weaker quadratic penalty (rho 1->20)',
        'overrides': {'initial_rho': 1.0, 'rho_target': 20.0},
    },
    'high_rho': {
        'description': 'Stronger quadratic penalty (rho 10->200)',
        'overrides': {'initial_rho': 10.0, 'rho_target': 200.0},
    },
}

# ── EfficientNetB0 LR sweep ──

LR_SWEEP_VALUES = {
    'lr_1e6': {
        'description': 'Constraint LR = 1e-6 (5x lower than default)',
        'overrides': {'lr_constraint': 1e-6},
    },
    'lr_2e5': {
        'description': 'Constraint LR = 2e-5 (4x higher than default)',
        'overrides': {'lr_constraint': 2e-5},
    },
    'lr_5e5': {
        'description': 'Constraint LR = 5e-5 (10x higher than default)',
        'overrides': {'lr_constraint': 5e-5},
    },
}


def build_config(model_name, variant_name, overrides, description,
                 ds_name, ds_info, scenario_name, scenario, slice_idx):
    """Build a single experiment config."""
    hp = copy.deepcopy(HYPERPARAMS)
    hp.update(overrides)

    data_dir = f"data/{ds_name}/slice_{slice_idx}"
    ds_config = {
        'data_dir': data_dir,
        'target_column': ds_info['target_column'],
        'group_column': ds_info['group_column'],
        'num_classes': ds_info['num_classes'],
        'image_size': ds_info['image_size'],
        'constrained_class': scenario['constrained_class'],
    }

    exp_name = (f"{scenario_name}_{CTAG}_{model_name}_{variant_name}_slice{slice_idx}")
    exp_path = Path('results/pending_runs') / ds_name / scenario_name / CTAG / model_name / variant_name / f"slice_{slice_idx}"

    return {
        'methodology': 'our_approach',
        'model_name': model_name,
        'constraint': list(CONSTRAINT_PAIR),
        'constraint_tag': CTAG,
        'dataset_mode': ds_name,
        'dataset_config': ds_config,
        'hyperparams': hp,
        'base_model_id': compute_base_model_id(
            model_name, hp, dataset_mode=ds_name, data_dir=data_dir),
        'exp_name': exp_name,
        'status': 'pending',
        'experiment_path': str(exp_path),
        'ablation_variant': variant_name,
        'ablation_description': description,
    }


def generate_all_configs():
    configs = []

    # ConvNeXtTiny ablation
    for variant_name, variant in ABLATION_VARIANTS.items():
        for ds_name, ds_info in DATASETS.items():
            for sc_name, sc in ds_info['scenarios'].items():
                for s in range(1, NUM_SLICES + 1):
                    cfg = build_config(
                        'ConvNeXtTiny', variant_name,
                        variant['overrides'], variant['description'],
                        ds_name, ds_info, sc_name, sc, s)
                    configs.append(cfg)

    # EfficientNetB0 LR sweep
    for variant_name, variant in LR_SWEEP_VALUES.items():
        for ds_name, ds_info in DATASETS.items():
            for sc_name, sc in ds_info['scenarios'].items():
                for s in range(1, NUM_SLICES + 1):
                    cfg = build_config(
                        'EfficientNetB0', variant_name,
                        variant['overrides'], variant['description'],
                        ds_name, ds_info, sc_name, sc, s)
                    configs.append(cfg)

    return configs


def save_configs(configs, dry_run=False):
    created, skipped = 0, 0
    for config in configs:
        path = Path(config['experiment_path'])
        existing = path / 'config.json'
        if existing.exists():
            try:
                with open(existing) as f:
                    ex = json.load(f)
                if ex.get('status') == 'completed':
                    skipped += 1
                    continue
            except (json.JSONDecodeError, KeyError):
                pass

        if dry_run:
            created += 1
            continue

        path.mkdir(parents=True, exist_ok=True)
        with open(path / 'config.json', 'w') as f:
            json.dump(config, f, indent=2)
        created += 1

    return created, skipped


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dry-run', action='store_true',
                        help='Count configs without writing')
    args = parser.parse_args()

    configs = generate_all_configs()

    # Summary
    ablation_configs = [c for c in configs if c['model_name'] == 'ConvNeXtTiny']
    lr_configs = [c for c in configs if c['model_name'] == 'EfficientNetB0']

    print("=== Ablation Study: ConvNeXtTiny ===")
    print(f"  Constraint: {CTAG} ({CONSTRAINT_PAIR})")
    print(f"  Variants: {list(ABLATION_VARIANTS.keys())}")
    for v, info in ABLATION_VARIANTS.items():
        n = sum(1 for c in ablation_configs if c['ablation_variant'] == v)
        print(f"    {v}: {info['description']} ({n} experiments)")
    print(f"  Total: {len(ablation_configs)}")

    print()
    print("=== LR Sweep: EfficientNetB0 ===")
    print(f"  Constraint: {CTAG} ({CONSTRAINT_PAIR})")
    print(f"  LR values: {list(LR_SWEEP_VALUES.keys())}")
    for v, info in LR_SWEEP_VALUES.items():
        n = sum(1 for c in lr_configs if c['ablation_variant'] == v)
        print(f"    {v}: {info['description']} ({n} experiments)")
    print(f"  Total: {len(lr_configs)}")

    print(f"\n  Grand total: {len(configs)} experiments")

    if args.dry_run:
        print("\n  [DRY RUN — no configs written]")
        created, skipped = save_configs(configs, dry_run=True)
    else:
        created, skipped = save_configs(configs)
        print(f"\n  Created: {created}, Skipped (completed): {skipped}")


if __name__ == '__main__':
    main()
