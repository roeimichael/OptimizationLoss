"""Configuration generator for static lambda experiments - Version 2 (Fine-Tuned).

This module generates experiment configurations for the static lambda methodology,
where lambda values remain constant throughout training (no adaptive increase).

STATIC LAMBDA METHODOLOGY V2 (48 configurations)
===============================================================================
Configuration breakdown:
  - 3 tabular models: BasicNN, TabularResNet, FTTransformer
  - 4 constraint pairs: [Soft,Soft], [Hard,Soft], [Soft,Hard], [Hard,Hard]
  - 4 lambda values: 0.02, 0.03, 0.05, 0.07 (fine-tuned sweet spot)
  - Total: 3 × 4 × 4 = 48 experiments

Rationale for lambda value selection:
  Based on initial experiments (v1), we found:
  - λ ≤ 0.01: Best predictions (86% Graduate) but low convergence (33%)
  - λ = 0.1: Moderate convergence (56%) but biased (93% Graduate)
  - λ ≥ 1.0: High convergence (100%) but severe bias (95%+ Graduate)

  V2 focuses on the sweet spot (0.02-0.07) to find optimal balance between:
  - Constraint satisfaction (convergence rate)
  - Prediction quality (avoid overfitting to constraints)

Key differences from adaptive methodology:
  - No warmup phase (warmup_epochs = 0)
  - Fixed epochs: 300 (not 10,000)
  - Static lambda values (no lambda_step)
  - Experiments may fail if constraints not met
===============================================================================
"""

import hashlib
import json
from typing import Dict, Any, List, Tuple

METHODOLOGIES = ['static_lambda']

# ============================================================================
# MODEL AND CONSTRAINT CONFIGURATION (same as adaptive methodology)
# ============================================================================

# 3 tabular-specific models
MODELS = ['BasicNN', 'TabularResNet', 'FTTransformer']

# 4 constraint pairs covering the space of constraint tightness
# Each pair is (local_percentage, global_percentage)
# - Soft = high percentage (more capacity, ~0.8-0.9)
# - Hard = low percentage (less capacity, ~0.2-0.3)
CONSTRAINTS = [
    (0.9, 0.8),  # [Soft, Soft] - Both permissive
    (0.3, 0.8),  # [Hard, Soft] - Local restrictive, Global permissive
    (0.8, 0.3),  # [Soft, Hard] - Local permissive, Global restrictive
    (0.3, 0.3),  # [Hard, Hard] - Both restrictive
]

# ============================================================================
# STATIC LAMBDA HYPERPARAMETERS
# ============================================================================

BASE_HYPERPARAMS = {
    'lr': 0.001,  # Fixed learning rate for fair comparison
    'dropout': 0.3,
    'batch_size': 64,
    'hidden_dims': [128, 64],
    'epochs': 300,  # Fixed at 300 epochs (not 10,000)
    'warmup_epochs': 0,  # No warmup phase!
    'constraint_threshold': 1e-6,  # For checking constraint satisfaction
    # Note: No lambda_step (lambdas don't change)
}

# Lambda value combinations to test - V2 Fine-Tuned Sweet Spot
# Focus on range 0.02-0.07 to balance convergence and prediction quality
STATIC_LAMBDA_REGIMES = {
    'lambda_fine_tune': {
        'name': 'lambda_fine_tune',
        'variations': [
            # Fine-grained search in the sweet spot (0.02-0.07)
            # Goal: Find lambda that gives 70-90% convergence with 75-85% Graduate predictions
            {
                'variation_name': 'lambda_0.2',
                'params': {
                    **BASE_HYPERPARAMS,
                    'lambda_global': 0.2,
                    'lambda_local': 0.2
                },
                'description': 'Just above v1 low (0.01) - expect better convergence, still good predictions'
            },
            {
                'variation_name': 'lambda_0.3',
                'params': {
                    **BASE_HYPERPARAMS,
                    'lambda_global': 0.3,
                    'lambda_local': 0.3
                },
                'description': 'Mid-low range - likely sweet spot for balanced performance'
            },
            {
                'variation_name': 'lambda_0.5',
                'params': {
                    **BASE_HYPERPARAMS,
                    'lambda_global': 0.5,
                    'lambda_local': 0.5
                },
                'description': 'Mid-range - testing higher convergence while maintaining quality'
            },
            {
                'variation_name': 'lambda_0.7',
                'params': {
                    **BASE_HYPERPARAMS,
                    'lambda_global': 0.7,
                    'lambda_local': 0.7
                },
                'description': 'Just below v1 medium (0.1) - upper bound before predictions degrade'
            },
        ]
    }
}


def compute_base_model_id(model_name: str, hyperparams: Dict[str, Any]) -> str:
    model_key_params = {
        'model_name': model_name,
        'lr': hyperparams['lr'],
        'dropout': hyperparams['dropout'],
        'batch_size': hyperparams['batch_size'],
        'hidden_dims': tuple(hyperparams['hidden_dims']),
        'methodology': 'static_lambda'
    }
    config_str = json.dumps(model_key_params, sort_keys=True)
    config_hash = hashlib.md5(config_str.encode()).hexdigest()[:12]
    return f"{model_name}_static_{config_hash}"


def create_config(
    methodology: str,
    model_name: str,
    constraint: Tuple[float, float],
    hyperparam_regime: str,
    variation_name: str,
    hyperparam_params: Dict[str, Any]
) -> Dict[str, Any]:
    """Create a single experiment configuration.

    Args:
        methodology: Experiment methodology ('static_lambda')
        model_name: Model architecture name
        constraint: Tuple of (local_percentage, global_percentage)
        hyperparam_regime: Name of hyperparameter regime
        variation_name: Name of specific variation
        hyperparam_params: Hyperparameter values

    Returns:
        Configuration dictionary
    """
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
    """Generate all static lambda experiment configurations.

    Returns:
        List of configuration dictionaries
    """
    all_configs = []
    config_id = 0

    print("Generating static lambda experiment configurations...")
    print(f"Methodologies: {len(METHODOLOGIES)}")
    print(f"Models: {len(MODELS)}")
    print(f"Constraints: {len(CONSTRAINTS)}")
    print(f"Lambda Regimes: {len(STATIC_LAMBDA_REGIMES)}")
    print()

    for methodology in METHODOLOGIES:
        for model_name in MODELS:
            for constraint in CONSTRAINTS:
                for regime_name, regime_config in STATIC_LAMBDA_REGIMES.items():
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


def save_configs_and_create_structure(
    configs: List[Dict[str, Any]],
    output_dir: str = 'results'
) -> int:
    from src.utils.filesystem_manager import ensure_experiment_path, save_config_to_path

    print(f"\nCreating experiment directory structure in '{output_dir}'...")
    saved_count = 0

    for i, config in enumerate(configs):
        experiment_path = ensure_experiment_path(config)
        config['experiment_path'] = experiment_path
        save_config_to_path(config, experiment_path)
        saved_count += 1

        if (i + 1) % 20 == 0:
            print(f"  Created {i + 1}/{len(configs)} experiment folders...")

    print(f"Successfully created {saved_count} experiment configurations!")
    return saved_count


def reset_all_status_to_pending(results_dir: str = 'results/static_lambda') -> int:
    from src.utils.filesystem_manager import get_all_experiment_configs, save_config_to_path

    print("=" * 80)
    print("RESET ALL STATIC LAMBDA EXPERIMENT STATUSES")
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
    print("STATIC LAMBDA V2 EXPERIMENT CONFIGURATION MANAGER (Fine-Tuned)")
    print("=" * 80)
    print()
    print("Select an option:")
    print("  1. Generate new static lambda V2 configurations (48 experiments)")
    print("  2. Reset all experiment statuses to pending")
    print("  3. Exit")
    print()

    while True:
        choice = input("Enter your choice (1-3): ").strip()

        if choice == '1':
            print()
            print("=" * 80)
            print("GENERATING STATIC LAMBDA V2 CONFIGURATIONS")
            print("=" * 80)
            print()
            all_configs = generate_all_configs()
            saved_count = save_configs_and_create_structure(all_configs)
            print("\n" + "=" * 80)
            print("CONFIGURATION GENERATION COMPLETE")
            print("=" * 80)
            print()
            print("Next steps:")
            print("1. Review the static_lambda_v2_plan_summary.txt file")
            print("2. Run experiments using: python run_static_lambda_experiment.py <config_path>")
            print("3. Or batch run with: python main.py (set ACTIVE_METHODOLOGIES=['static_lambda'])")
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
