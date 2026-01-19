"""Configuration generator for student dropout prediction experiments.

This module generates experiment configurations for systematic evaluation of
constraint-based optimization across different models, constraints, and hyperparameters.

CURRENT MODE: LAMBDA STRATEGY COMPARISON (72 configurations)
===============================================================================
Configuration breakdown:
  - 3 tabular models: BasicNN, TabularResNet, FTTransformer
  - 3 constraint pairs: [Soft,Soft], [Soft,Hard], [Hard,Hard]
  - 2 learning rates: 0.0001 (low), 0.00005 (very low)
  - 4 lambda strategies: linear, transfer, balanced, combined
  - Total: 3 × 3 × 2 × 4 = 72 experiments

Lambda Strategies:
  - linear: Baseline - increase lambda linearly when constraint not satisfied
  - transfer: Transfer lambda step from satisfied to unsatisfied constraint
  - balanced: Initialize lambdas based on initial loss ratio, then linear
  - combined: Balanced initialization + Transfer adjustment (best of both)

To restore previous experiments:
  - Modify CONSTRAINTS, LR_VALUES, and LAMBDA_STRATEGIES as needed
===============================================================================
"""

import hashlib
import json
from typing import Dict, Any, List, Tuple

METHODOLOGIES = ['our_approach']

# ============================================================================
# FOCUSED EXPERIMENTAL CONFIGURATION (36 total experiments)
# ============================================================================

# LAMBDA STRATEGY EXPERIMENT: 3 tabular-specific models
MODELS = ['BasicNN', 'TabularResNet', 'FTTransformer']

# LAMBDA STRATEGY EXPERIMENT: 3 focused constraint pairs
CONSTRAINTS = [
    (0.9, 0.8),  # [Soft, Soft] - Both permissive
    (0.8, 0.2),  # [Soft, Hard] - Global permissive, Local restrictive
    (0.5, 0.3),  # [Hard, Hard] - Both restrictive
]

# LAMBDA STRATEGY EXPERIMENT: 4 lambda adjustment strategies
LAMBDA_STRATEGIES = ['linear', 'transfer', 'balanced', 'combined']

BASE_HYPERPARAMS = {
    'lr': 0.0001,
    'dropout': 0.3,
    'batch_size': 64,
    'hidden_dims': [128, 64],
    'epochs': 1000,
    'lambda_global': 0.1,
    'lambda_local': 0.1,
    'lambda_strategy': 'linear',  # Default strategy
    'warmup_epochs': 50,
    'constraint_threshold': 0.02,
    'lambda_step': 0.005
}

# LAMBDA STRATEGY EXPERIMENT: Learning rate and lambda strategy variations
# Vary both learning rate and lambda adjustment strategy
HYPERPARAM_REGIMES = {
    'lr_lambda_test': {
        'name': 'lr_lambda_test',
        'variations': [
            {
                'variation_name': f'lr_{lr}_lambda_{strategy}',
                'params': {**BASE_HYPERPARAMS, 'lr': lr, 'lambda_strategy': strategy}
            }
            for lr in [0.0001, 0.00005]  # Low and very low learning rates
            for strategy in LAMBDA_STRATEGIES  # All 3 lambda strategies
        ]
    },
}


# FULL EXPERIMENT: All hyperparameter regimes (uncomment to restore)
# HYPERPARAM_REGIMES = {
#     'standard': {
#         'name': 'standard',
#         'variations': [
#             {'variation_name': 'default', 'params': BASE_HYPERPARAMS.copy()}
#         ]
#     },
#     'lr_test': {
#         'name': 'lr_test',
#         'variations': [
#             {'variation_name': f'lr_{lr}', 'params': {**BASE_HYPERPARAMS, 'lr': lr}}
#             for lr in [0.0001, 0.0005, 0.001, 0.005, 0.01]
#         ]
#     },
#     'dropout_test': {
#         'name': 'dropout_test',
#         'variations': [
#             {'variation_name': f'dropout_{dropout}', 'params': {**BASE_HYPERPARAMS, 'dropout': dropout}}
#             for dropout in [0.1, 0.2, 0.3, 0.4, 0.5]
#         ]
#     },
#     'batch_test': {
#         'name': 'batch_test',
#         'variations': [
#             {'variation_name': f'batch_{batch}', 'params': {**BASE_HYPERPARAMS, 'batch_size': batch}}
#             for batch in [32, 64, 128, 256, 512]
#         ]
#     }
# }

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
    print("=" * 80)
    print("RESET ALL EXPERIMENT STATUSES")
    print("=" * 80)
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
    print("\n" + "=" * 80)
    print("RESET COMPLETE")
    print("=" * 80)
    return reset_count


def main() -> None:
    print("=" * 80)
    print("EXPERIMENT CONFIGURATION MANAGER")
    print("=" * 80)
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
            print("=" * 80)
            print("GENERATING CONFIGURATIONS")
            print("=" * 80)
            print()
            all_configs = generate_all_configs()
            saved_count = save_configs_and_create_structure(all_configs)
            print("\n" + "=" * 80)
            print("CONFIGURATION GENERATION COMPLETE")
            print("=" * 80)
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
