"""Configuration generator for experiment grid."""

import hashlib
import json
from pathlib import Path

MODELS = ['BasicNN', 'TabularResNet', 'FTTransformer']
CONSTRAINTS = [(0.9, 0.8), (0.8, 0.2), (0.5, 0.3)]

HYPERPARAMS = {
    'lr': 0.0001,
    'dropout': 0.3,
    'batch_size': 64,
    'hidden_dims': [128, 64],
    'epochs': 1000,
    'lambda_global': 0.1,
    'lambda_local': 0.1,
    'warmup_epochs': 50,
    'constraint_threshold': 0.02,
    'lambda_step': 0.005
}


def compute_base_model_id(model_name, hyperparams):
    """Generate unique ID for model caching based on architecture params."""
    key = {
        'model_name': model_name,
        'lr': hyperparams['lr'],
        'dropout': hyperparams['dropout'],
        'batch_size': hyperparams['batch_size'],
        'hidden_dims': tuple(hyperparams['hidden_dims']),
        'warmup_epochs': hyperparams['warmup_epochs']
    }
    h = hashlib.md5(json.dumps(key, sort_keys=True).encode()).hexdigest()[:12]
    return f"{model_name}_{h}"


def generate_configs():
    """Generate all experiment configurations."""
    configs = []
    for model in MODELS:
        for constraint in CONSTRAINTS:
            config = {
                'methodology': 'our_approach',
                'model_name': model,
                'constraint': constraint,
                'hyperparam_regime': 'standard',
                'variation_name': 'default',
                'hyperparams': HYPERPARAMS.copy(),
                'base_model_id': compute_base_model_id(model, HYPERPARAMS),
                'status': 'pending'
            }
            configs.append(config)
    return configs


def save_configs(configs, output_dir='results'):
    """Create directory structure and save configs."""
    from src.utils.filesystem_manager import save_config_to_path

    for config in configs:
        constraint = config['constraint']
        path = Path(output_dir) / 'our_approach' / config['model_name'] / \
               f"constraint_{constraint[0]}_{constraint[1]}" / 'standard' / 'default'
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
    print("1. Generate new configs")
    print("2. Reset all to pending")
    print("3. Exit")

    choice = input("\nChoice: ").strip()

    if choice == '1':
        configs = generate_configs()
        save_configs(configs)
    elif choice == '2':
        reset_all_to_pending()


if __name__ == "__main__":
    main()
