"""Configuration generator for experiment grid."""

import hashlib
import json
from pathlib import Path

# FTTransformer with multiple constraint pairs for comparison
MODELS = ['FTTransformer']
CONSTRAINTS = [
    (0.9, 0.8),
    (0.9, 0.5),
    (0.8, 0.7),
    (0.8, 0.2),
    (0.7, 0.5),
    (0.6, 0.5),
    (0.5, 0.3),
    (0.4, 0.2),
]

# Dataset modes (multiclass support exists but not active - no multiclass data)
DATASET_MODES = ['binary']

HYPERPARAMS = {
    # Basic training params
    'lr': 0.001,
    'lr_constraint': 0.00001,         # 1e-5: constraint phase learning rate
    'dropout': 0.3,
    'batch_size': 64,
    'hidden_dims': [128, 64],

    # Warmup: runs until accuracy saturates (no fixed epoch count)
    'warmup_epochs': 50,              # Minimum warmup (used for cache key + fallback)
    'max_warmup_epochs': 500,         # Safety cap
    'warmup_saturation_threshold': 0.001,  # Min accuracy improvement to continue
    'warmup_saturation_patience': 5,       # Checks without improvement before stopping

    # Constraint phase
    'constraint_epochs': 350,         # Max constraint training epochs after warmup

    # Lambda scheduling (ratchet up until satisfied, then freeze)
    'lambda_global': 0.005,
    'lambda_local': 0.005,
    'lambda_step': 0.001,

    # Constraint loss (ALM)
    'use_sum_loss': True,
    'initial_rho': 0.5,

    # KL-Divergence Regularization
    'alpha_kl': 1.0,
}


def compute_base_model_id(model_name, hyperparams, dataset_mode='binary'):
    """Generate unique ID for model caching based on architecture params."""
    key = {
        'model_name': model_name,
        'lr': hyperparams['lr'],
        'dropout': hyperparams['dropout'],
        'batch_size': hyperparams['batch_size'],
        'hidden_dims': tuple(hyperparams['hidden_dims']),
        'warmup_epochs': hyperparams['warmup_epochs'],
        'dataset_mode': dataset_mode  # Different cache for binary vs multiclass
    }
    h = hashlib.md5(json.dumps(key, sort_keys=True).encode()).hexdigest()[:12]
    return f"{model_name}_{dataset_mode}_{h}"


def generate_configs(methodology='our_approach', dataset_mode='binary'):
    """Generate experiment configs for given methodology and dataset mode."""
    configs = []
    for model in MODELS:
        for constraint in CONSTRAINTS:
            hyperparams = HYPERPARAMS.copy()

            config = {
                'methodology': methodology,
                'model_name': model,
                'constraint': constraint,
                'dataset_mode': dataset_mode,
                'hyperparam_regime': 'standard',
                'variation_name': 'default',
                'hyperparams': hyperparams,
                'base_model_id': compute_base_model_id(model, HYPERPARAMS, dataset_mode),
                'status': 'pending'
            }
            configs.append(config)
    return configs


def save_configs(configs, output_dir='results'):
    """Create directory structure and save configs."""
    from src.utils.filesystem_manager import save_config_to_path

    for config in configs:
        constraint = config['constraint']
        methodology = config.get('methodology', 'our_approach')
        dataset_mode = config.get('dataset_mode', 'binary')
        hyperparam_regime = config.get('hyperparam_regime', 'standard')
        variation_name = config.get('variation_name', 'default')

        # Path structure: results/dataset_mode/methodology/model/constraint/regime/variation
        path = Path(output_dir) / dataset_mode / methodology / config['model_name'] / \
               f"constraint_{constraint[0]}_{constraint[1]}" / hyperparam_regime / variation_name
        path.mkdir(parents=True, exist_ok=True)
        config['experiment_path'] = str(path)
        save_config_to_path(config, str(path))

    print(f"Created {len(configs)} experiment configurations in '{output_dir}'")


def reset_all_to_pending(results_dir='results'):
    """Reset all experiment statuses to pending."""
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
    print("=== Binary Classification (2 classes, constrain class 1) ===")
    print("1. Generate binary our_approach configs")
    print("2. Generate binary heuristic configs")
    print("3. Generate binary both configs")
    print("")
    print("=== Multiclass Classification (5 classes, constrain class 4) ===")
    print("4. Generate multiclass our_approach configs")
    print("5. Generate multiclass heuristic configs")
    print("6. Generate multiclass both configs")
    print("")
    print("=== All Experiments ===")
    print("7. Generate ALL configs (binary + multiclass, both methodologies)")
    print("")
    print("=== Utilities ===")
    print("9. Reset all to pending")
    print("0. Exit")

    choice = input("\nChoice: ").strip()

    if choice == '1':
        configs = generate_configs('our_approach', 'binary')
        save_configs(configs)
    elif choice == '2':
        configs = generate_configs('heuristic', 'binary')
        save_configs(configs)
    elif choice == '3':
        configs_opt = generate_configs('our_approach', 'binary')
        configs_heur = generate_configs('heuristic', 'binary')
        save_configs(configs_opt)
        save_configs(configs_heur)
    elif choice == '4':
        configs = generate_configs('our_approach', 'multiclass')
        save_configs(configs)
    elif choice == '5':
        configs = generate_configs('heuristic', 'multiclass')
        save_configs(configs)
    elif choice == '6':
        configs_opt = generate_configs('our_approach', 'multiclass')
        configs_heur = generate_configs('heuristic', 'multiclass')
        save_configs(configs_opt)
        save_configs(configs_heur)
    elif choice == '7':
        # Generate all configs
        for mode in DATASET_MODES:
            for methodology in ['our_approach', 'heuristic']:
                configs = generate_configs(methodology, mode)
                save_configs(configs)
                print(f"  -> {mode}/{methodology}")
    elif choice == '9':
        reset_all_to_pending()


if __name__ == "__main__":
    main()
