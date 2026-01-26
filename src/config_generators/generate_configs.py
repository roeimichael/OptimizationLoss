"""Configuration generator for constraint-based optimization experiments."""

import hashlib
import json
from typing import Dict, Any, List, Tuple

METHODOLOGIES = ['our_approach']

MODELS = ['BasicNN', 'TabularResNet', 'FTTransformer']
CONSTRAINTS = [
    (0.9, 0.8),
    (0.8, 0.2),
    (0.5, 0.3),
]

BASE_HYPERPARAMS = {
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

HYPERPARAM_REGIMES = {
    'standard': {
        'name': 'standard',
        'variations': [
            {'variation_name': 'default', 'params': BASE_HYPERPARAMS.copy()}
        ]
    },
}

def compute_base_model_id(model_name: str, hyperparams: Dict[str, Any]) -> str:
    model_key_params = {
        'model_name': model_name,
        'lr': hyperparams['lr'],
        'dropout': hyperparams['dropout'],
        'batch_size': hyperparams['batch_size'],
        'hidden_dims': tuple(hyperparams['hidden_dims']),
        'warmup_epochs': hyperparams['warmup_epochs']
    }
    config_str = json.dumps(model_key_params, sort_keys=True)
    config_hash = hashlib.md5(config_str.encode()).hexdigest()[:12]
    return f"{model_name}_{config_hash}"


def create_config(methodology: str, model_name: str, constraint: Tuple[float, float], hyperparam_regime: str,
                  variation_name: str, hyperparam_params: Dict[str, Any]) -> Dict[str, Any]:
    base_model_id = compute_base_model_id(model_name, hyperparam_params)
    config = {
        'methodology': methodology,
        'model_name': model_name,
        'constraint': constraint,
        'hyperparam_regime': hyperparam_regime,
        'variation_name': variation_name,
        'hyperparams': hyperparam_params,
        'base_model_id': base_model_id,
        'experiment_path': None,
        'status': 'pending'
    }
    return config


def generate_all_configs() -> List[Dict[str, Any]]:
    all_configs = []
    config_id = 0
    print("Generating experiment configurations...")
    print(f"Methodologies: {len(METHODOLOGIES)}")
    print(f"Models: {len(MODELS)}")
    print(f"Constraints: {len(CONSTRAINTS)}")
    print(f"Hyperparameter Regimes: {len(HYPERPARAM_REGIMES)}")
    print()
    for methodology in METHODOLOGIES:
        for model_name in MODELS:
            for constraint in CONSTRAINTS:
                for regime_name, regime_config in HYPERPARAM_REGIMES.items():
                    for variation in regime_config['variations']:
                        config = create_config(
                            methodology,
                            model_name,
                            constraint,
                            regime_name,
                            variation['variation_name'],
                            variation['params']
                        )
                        all_configs.append(config)
                        config_id += 1
    print(f"Total configurations generated: {len(all_configs)}")
    return all_configs


def save_configs_and_create_structure(configs: List[Dict[str, Any]], output_dir: str = 'results') -> int:
    from src.utils.filesystem_manager import ensure_experiment_path, save_config_to_path
    print(f"\nCreating experiment directory structure in '{output_dir}'...")
    saved_count = 0
    for i, config in enumerate(configs):
        experiment_path = ensure_experiment_path(config)
        config['experiment_path'] = experiment_path
        save_config_to_path(config, experiment_path)
        saved_count += 1
        if (i + 1) % 100 == 0:
            print(f"  Created {i + 1}/{len(configs)} experiment folders...")
    print(f"Successfully created {saved_count} experiment configurations!")
    return saved_count


def reset_all_status_to_pending(results_dir: str = 'results') -> int:
    from src.utils.filesystem_manager import get_all_experiment_configs, save_config_to_path
    print("RESET ALL EXPERIMENT STATUSES")
    print(f"\nScanning directory: {results_dir}")
    all_experiments = get_all_experiment_configs(results_dir)
    reset_count = 0
    for experiment_path, config in all_experiments:
        if config.get('status') != 'pending':
            config['status'] = 'pending'
            save_config_to_path(config, experiment_path)
            reset_count += 1
    print(f"\nTotal experiments found: {len(all_experiments)}")
    print(f"Experiments reset to pending: {reset_count}")
    print(f"Already pending: {len(all_experiments) - reset_count}")
    return reset_count


def main() -> None:
    print("EXPERIMENT CONFIGURATION MANAGER")
    print()
    print("Select an option:")
    print("  1. Generate new experiment configurations")
    print("  2. Reset all experiment statuses to pending")
    print("  3. Exit")
    print()

    while True:
        choice = input("Enter your choice (1-3): ").strip()

        if choice == '1':
            print()
            print("GENERATING CONFIGURATIONS")
            print()
            all_configs = generate_all_configs()
            saved_count = save_configs_and_create_structure(all_configs)
            print("\nCONFIGURATION GENERATION COMPLETE")
            print()
            print("Next step: Run experiments using: python main.py")
            print()
            break

        elif choice == '2':
            print()
            reset_all_status_to_pending()
            print()
            break

        elif choice == '3':
            print("\nExiting...")
            break

        else:
            print("Invalid choice. Please enter 1, 2, or 3.")


if __name__ == "__main__":
    main()
