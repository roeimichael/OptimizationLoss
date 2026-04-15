# Configuration generator for DermMNIST experiments.
# Supports round1 (all 4 scenarios, symmetric constraints) and
# round2 (multi-class only, asymmetric constraints).
# Output: results/pending_runs/{scenario}/{constraint_tag}/{model}/{methodology}/{slice}/

import hashlib
import json
from pathlib import Path

DATASET_CONFIG_TEMPLATE = {
    'target_column': 'label',
    'group_column': 'loc_group',
    'num_classes': 7,
    'image_size': 224,
}

# ── Round definitions ──

ROUND1_SCENARIOS = {
    'single_MEL': {'constrained_class': 4},
    'single_NV': {'constrained_class': 5},
    'multi_MEL_BKL': {'constrained_class': [4, 2]},
    'multi_MEL_BCC_VASC': {'constrained_class': [4, 1, 6]},
}

ROUND1_CONSTRAINT_PAIRS = [
    # Equal: tight, medium, loose
    (0.3, 0.3),
    (0.5, 0.5),
    (0.8, 0.8),
    # Global tighter than local
    (0.5, 0.3),
    (0.8, 0.5),
    (0.8, 0.3),
]

ROUND1_METHODOLOGIES = ['our_approach', 'heuristic']

ROUND2_SCENARIOS = {
    'multi_MEL_BKL': {'constrained_class': [4, 2]},
    'multi_MEL_BCC_VASC': {'constrained_class': [4, 1, 6]},
}

ROUND2_CONSTRAINT_PAIRS = [
    (0.3, 0.6), (0.3, 0.8),
    (0.4, 0.6), (0.5, 0.7),
    (0.6, 0.3), (0.6, 0.4),
    (0.7, 0.5), (0.8, 0.3),
]

ROUND2_METHODOLOGIES = ['our_approach', 'heuristic', 'po_lp']

# Defaults (round2 for backward compat)
SCENARIOS = ROUND2_SCENARIOS
CONSTRAINT_PAIRS = ROUND2_CONSTRAINT_PAIRS
METHODOLOGIES = ROUND2_METHODOLOGIES

MODEL_NAMES = ['MobileNetV3', 'EfficientNetB0', 'ConvNeXtTiny']

NUM_SLICES = 5

HYPERPARAMS = {
    'lr': 0.0001,
    'lr_constraint': 5e-6,
    'dropout': 0.3,
    'batch_size': 64,
    'warmup_epochs': 50,
    'constraint_epochs': 300,
    'lambda_global': 0.01,
    'lambda_local': 0.01,
    'lambda_step': 0.002,
    'use_sum_loss': True,
    'initial_rho': 5.0,
    'rho_target': 100.0,
    'alpha_kl': 0.5,
    'kl_temperature': 1.0,
    'pretrained': True,
    'class_weighted_ce': False,
    'constraint_chunk_size': 64,
}


def constraint_tag(pair):
    local_pct, global_pct = pair
    return f"L{int(local_pct * 100):02d}_G{int(global_pct * 100):02d}"


def model_tag(model_name):
    tags = {'MobileNetV3': 'mv3', 'EfficientNetB0': 'effb0', 'ConvNeXtTiny': 'cnxt'}
    return tags.get(model_name, model_name.lower())


def compute_base_model_id(model_name, hyperparams, dataset_mode='dermmnist',
                          data_dir='data/dermmnist'):
    key = {
        'model_name': model_name,
        'lr': hyperparams['lr'],
        'dropout': hyperparams['dropout'],
        'batch_size': hyperparams['batch_size'],
        'warmup_epochs': hyperparams['warmup_epochs'],
        'pretrained': hyperparams.get('pretrained', False),
        'class_weighted_ce': hyperparams.get('class_weighted_ce', False),
        'dataset_mode': dataset_mode,
        'data_dir': data_dir,
    }
    if 'seed' in hyperparams:
        key['seed'] = hyperparams['seed']
    h = hashlib.md5(json.dumps(key, sort_keys=True).encode()).hexdigest()[:12]
    return f"{model_name}_{dataset_mode}_{h}"


def _build_dataset_config(scenario_name, slice_idx, scenarios=None):
    sc_map = scenarios or SCENARIOS
    sc = sc_map[scenario_name]
    cfg = DATASET_CONFIG_TEMPLATE.copy()
    cfg['data_dir'] = f"data/dermmnist/slice_{slice_idx}"
    cfg['constrained_class'] = sc['constrained_class']
    return cfg


def _build_config(methodology, exp_name, scenario_name, constraint_pair,
                  model_name, slice_idx, hyperparams=None, scenarios=None):
    hp = hyperparams or HYPERPARAMS.copy()
    ds_config = _build_dataset_config(scenario_name, slice_idx, scenarios=scenarios)
    ctag = constraint_tag(constraint_pair)
    return {
        'methodology': methodology,
        'model_name': model_name,
        'constraint': list(constraint_pair),
        'constraint_tag': ctag,
        'dataset_mode': 'dermmnist',
        'dataset_config': ds_config,
        'hyperparams': hp.copy(),
        'base_model_id': compute_base_model_id(model_name, hp, data_dir=ds_config['data_dir']),
        'exp_name': exp_name,
        'status': 'pending',
    }


def _get_round_defaults(round_name):
    if round_name == 'round1':
        return ROUND1_SCENARIOS, ROUND1_CONSTRAINT_PAIRS, ROUND1_METHODOLOGIES
    return ROUND2_SCENARIOS, ROUND2_CONSTRAINT_PAIRS, ROUND2_METHODOLOGIES


def generate_configs(scenarios=None, constraint_pairs=None, model_names=None,
                     methodologies=None, num_slices=None, round_name=None):
    if round_name:
        r_scenarios, r_constraints, r_methods = _get_round_defaults(round_name)
    else:
        r_scenarios, r_constraints, r_methods = SCENARIOS, CONSTRAINT_PAIRS, METHODOLOGIES

    scenario_list = scenarios if scenarios is not None else list(r_scenarios.keys())
    if constraint_pairs is None:
        constraint_pairs = r_constraints
    if model_names is None:
        model_names = MODEL_NAMES
    if methodologies is None:
        methodologies = r_methods
    if num_slices is None:
        num_slices = NUM_SLICES

    configs = []
    for sc in scenario_list:
        for pair in constraint_pairs:
            ctag = constraint_tag(pair)
            for mn in model_names:
                for meth in methodologies:
                    for s in range(1, num_slices + 1):
                        exp_name = f"{sc}_{ctag}_{mn}_{meth}_slice{s}"
                        config = _build_config(meth, exp_name, sc, pair, mn, s,
                                               scenarios=r_scenarios)
                        path = Path('results/pending_runs') / sc / ctag / mn / meth / f"slice_{s}"
                        config['experiment_path'] = str(path)
                        configs.append(config)
    return configs


def save_configs(configs, output_dir='results/pending_runs'):
    from src.utils.filesystem_manager import save_config_to_path
    created, skipped = 0, 0
    for config in configs:
        path = Path(config['experiment_path'])
        existing_config = path / 'config.json'
        if existing_config.exists():
            try:
                with open(existing_config) as f:
                    existing = json.load(f)
                if existing.get('status') == 'completed':
                    skipped += 1
                    continue
            except (json.JSONDecodeError, KeyError):
                pass
        path.mkdir(parents=True, exist_ok=True)
        save_config_to_path(config, str(path))
        created += 1
    print(f"Created {created} configs in '{output_dir}' "
          f"(skipped {skipped} already completed)")


def reset_all_to_pending(results_dir='results'):
    from src.utils.filesystem_manager import get_all_experiment_configs, save_config_to_path
    experiments = get_all_experiment_configs(results_dir)
    count = 0
    for path, config in experiments:
        if config.get('status') != 'pending':
            config['status'] = 'pending'
            save_config_to_path(config, path)
            count += 1
    print(f"Reset {count} experiments to pending")


def main():
    n_scenarios = len(SCENARIOS)
    n_constraints = len(CONSTRAINT_PAIRS)
    n_models = len(MODEL_NAMES)
    n_methods = len(METHODOLOGIES)
    n_slices = NUM_SLICES
    total = n_scenarios * n_constraints * n_models * n_methods * n_slices

    print(f"=== Experiment Config Generator (Round 2) ===")
    print(f"Scenarios: {list(SCENARIOS.keys())}")
    print(f"Constraints: {n_constraints} -> {[constraint_tag(p) for p in CONSTRAINT_PAIRS]}")
    print(f"Models: {MODEL_NAMES}")
    print(f"Methodologies: {METHODOLOGIES}")
    print(f"Slices: {n_slices}")
    print(f"Total: {n_scenarios} x {n_constraints} x {n_models} x {n_methods} x {n_slices} = {total}")
    print()
    print("1. Generate all configs")
    print("9. Reset all to pending")
    print("0. Exit")
    choice = input("\nChoice: ").strip()
    if choice == '1':
        save_configs(generate_configs())
    elif choice == '9':
        reset_all_to_pending()


if __name__ == "__main__":
    main()
