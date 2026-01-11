"""Configuration generator for static lambda experiments.

This module generates experiment configurations for the static lambda methodology,
where lambda values remain constant throughout training (no adaptive increase).

STATIC LAMBDA METHODOLOGY (84 configurations)
===============================================================================
Configuration breakdown:
  - 3 tabular models: BasicNN, TabularResNet, FTTransformer
  - 4 constraint pairs: [Soft,Soft], [Hard,Soft], [Soft,Hard], [Hard,Hard]
  - 7 lambda value combinations: very_low, low, medium, high, very_high,
    global_heavy, local_heavy
  - Total: 3 × 4 × 7 = 84 experiments

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

# Lambda value combinations to test
# These test different strengths of constraint enforcement
STATIC_LAMBDA_REGIMES = {
    'lambda_search': {
        'name': 'lambda_search',
        'variations': [
            # Symmetric lambda combinations (global = local)
            {
                'variation_name': 'very_low',
                'params': {
                    **BASE_HYPERPARAMS,
                    'lambda_global': 0.001,
                    'lambda_local': 0.001
                },
                'description': 'Very weak constraint enforcement'
            },
            {
                'variation_name': 'low',
                'params': {
                    **BASE_HYPERPARAMS,
                    'lambda_global': 0.01,
                    'lambda_local': 0.01
                },
                'description': 'Weak constraint enforcement (adaptive baseline)'
            },
            {
                'variation_name': 'medium',
                'params': {
                    **BASE_HYPERPARAMS,
                    'lambda_global': 0.1,
                    'lambda_local': 0.1
                },
                'description': 'Medium constraint enforcement'
            },
            {
                'variation_name': 'high',
                'params': {
                    **BASE_HYPERPARAMS,
                    'lambda_global': 1.0,
                    'lambda_local': 1.0
                },
                'description': 'Strong constraint enforcement'
            },
            {
                'variation_name': 'very_high',
                'params': {
                    **BASE_HYPERPARAMS,
                    'lambda_global': 10.0,
                    'lambda_local': 10.0
                },
                'description': 'Very strong constraint enforcement'
            },
            # Asymmetric lambda combinations (global ≠ local)
            {
                'variation_name': 'global_heavy',
                'params': {
                    **BASE_HYPERPARAMS,
                    'lambda_global': 1.0,
                    'lambda_local': 0.01
                },
                'description': 'Prioritize global constraints over local'
            },
            {
                'variation_name': 'local_heavy',
                'params': {
                    **BASE_HYPERPARAMS,
                    'lambda_global': 0.01,
                    'lambda_local': 1.0
                },
                'description': 'Prioritize local constraints over global'
            },
        ]
    }
}


def compute_base_model_id(model_name: str, hyperparams: Dict[str, Any]) -> str:
    """Compute unique identifier for base model configuration.

    Args:
        model_name: Name of the model architecture
        hyperparams: Hyperparameter dictionary

    Returns:
        Unique hash-based identifier string
    """
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
    """Save configurations and create directory structure.

    Args:
        configs: List of configuration dictionaries
        output_dir: Root output directory

    Returns:
        Number of configurations saved
    """
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


def generate_summary_report(
    configs: List[Dict[str, Any]],
    output_file: str = 'static_lambda_plan_summary.txt'
) -> None:
    """Generate summary report of experiment plan.

    Args:
        configs: List of configuration dictionaries
        output_file: Output file path
    """
    with open(output_file, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("STATIC LAMBDA EXPERIMENT PLAN SUMMARY\n")
        f.write("=" * 80 + "\n\n")

        f.write(f"Total Experiments: {len(configs)}\n\n")

        f.write("Methodology: Static Lambda\n")
        f.write("  - No warmup phase (constraints from epoch 0)\n")
        f.write("  - Fixed epochs: 300\n")
        f.write("  - Constant lambda values (no adaptive increase)\n")
        f.write("  - Experiments fail if constraints not met\n\n")

        f.write("By Model:\n")
        for model in MODELS:
            count = sum(1 for c in configs if c['model_name'] == model)
            f.write(f"  {model}: {count}\n")
        f.write("\n")

        f.write("By Constraint:\n")
        for constraint in CONSTRAINTS:
            count = sum(1 for c in configs if c['constraint'] == constraint)
            f.write(f"  {constraint}: {count}\n")
        f.write("\n")

        f.write("By Lambda Configuration:\n")
        for regime_name, regime_config in STATIC_LAMBDA_REGIMES.items():
            for variation in regime_config['variations']:
                name = variation['variation_name']
                params = variation['params']
                count = sum(1 for c in configs if c['variation_name'] == name)
                f.write(f"  {name}: {count} experiments "
                       f"(λ_global={params['lambda_global']}, "
                       f"λ_local={params['lambda_local']})\n")
        f.write("\n")

        unique_base_models = len(set(c['base_model_id'] for c in configs))
        f.write(f"Unique Base Models: {unique_base_models}\n")
        f.write("\n")

        f.write("=" * 80 + "\n")

    print(f"\nSummary report saved to: {output_file}")


def reset_all_status_to_pending(results_dir: str = 'results/static_lambda') -> int:
    """Reset all experiment statuses to pending.

    Args:
        results_dir: Directory containing experiments

    Returns:
        Number of experiments reset
    """
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
    """Main entry point for configuration management."""
    print("=" * 80)
    print("STATIC LAMBDA EXPERIMENT CONFIGURATION MANAGER")
    print("=" * 80)
    print()
    print("Select an option:")
    print("  1. Generate new static lambda configurations")
    print("  2. Reset all experiment statuses to pending")
    print("  3. Exit")
    print()

    while True:
        choice = input("Enter your choice (1-3): ").strip()

        if choice == '1':
            print()
            print("=" * 80)
            print("GENERATING STATIC LAMBDA CONFIGURATIONS")
            print("=" * 80)
            print()
            all_configs = generate_all_configs()
            saved_count = save_configs_and_create_structure(all_configs)
            generate_summary_report(all_configs)
            print("\n" + "=" * 80)
            print("CONFIGURATION GENERATION COMPLETE")
            print("=" * 80)
            print()
            print("Next steps:")
            print("1. Review the static_lambda_plan_summary.txt file")
            print("2. Run experiments using: python run_static_lambda_experiment.py <config_path>")
            print("3. Or batch run with: find results/static_lambda -name 'config.json'")
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
