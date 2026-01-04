"""
Experiment Configuration Generator
Generates all experiment configurations across 4 dimensions:
1. Methodologies
2. Models
3. Constraints
4. Hyperparameter Regimes
"""
import json
import hashlib
from pathlib import Path
from itertools import product


# ============================================================================
# DIMENSION DEFINITIONS
# ============================================================================

METHODOLOGIES = [
    'our_approach',
    # 'benchmark',          # Placeholder for future
    # 'train_then_optimize', # Placeholder for future
    # 'hybrid'               # Placeholder for future
]

MODELS = [
    'BasicNN',
    'ResNet56',
    'DenseNet121',
    'InceptionV3',
    'VGG19'
]

CONSTRAINTS = [
    (0.9, 0.8),
    (0.9, 0.5),
    (0.8, 0.7),
    (0.8, 0.2),
    (0.7, 0.5),
    (0.6, 0.5),
    (0.5, 0.3),
    (0.4, 0.2)
]

# Base hyperparameters
BASE_HYPERPARAMS = {
    'lr': 0.001,
    'dropout': 0.3,
    'batch_size': 64,
    'hidden_dims': [128, 64],
    'epochs': 10000,
    'lambda_global': 0.01,
    'lambda_local': 0.01,
    'max_lambda_global': 0.5,
    'max_lambda_local': 0.5,
    'gradient_clip': 1.0,
    'warmup_epochs': 250,
    'constraint_threshold': 1e-6,
    'lambda_step': 0.01
}

# Hyperparameter regime definitions
HYPERPARAM_REGIMES = {
    'standard': {
        'name': 'standard',
        'variations': [BASE_HYPERPARAMS.copy()]
    },
    'lr_test': {
        'name': 'lr_test',
        'variations': [
            {**BASE_HYPERPARAMS, 'lr': lr}
            for lr in [0.0001, 0.0005, 0.001, 0.005, 0.01]
        ]
    },
    'dropout_test': {
        'name': 'dropout_test',
        'variations': [
            {**BASE_HYPERPARAMS, 'dropout': dropout}
            for dropout in [0.1, 0.2, 0.3, 0.4, 0.5]
        ]
    },
    'batch_test': {
        'name': 'batch_test',
        'variations': [
            {**BASE_HYPERPARAMS, 'batch_size': batch}
            for batch in [32, 64, 128, 256, 512]
        ]
    }
}


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def compute_base_model_id(model_name, hyperparams):
    """
    Compute unique ID for model based on architecture and hyperparameters
    This ID is used to identify models that can share pre-trained weights
    (excludes constraints which don't affect the base model training)

    Args:
        model_name: Name of the model architecture
        hyperparams: Dictionary of hyperparameters

    Returns:
        str: Unique hash-based identifier
    """
    # Extract only the params that affect model architecture and training
    # Exclude constraint-related params
    model_key_params = {
        'model_name': model_name,
        'lr': hyperparams['lr'],
        'dropout': hyperparams['dropout'],
        'batch_size': hyperparams['batch_size'],
        'hidden_dims': tuple(hyperparams['hidden_dims']),
        'warmup_epochs': hyperparams['warmup_epochs']
    }

    # Create deterministic hash
    config_str = json.dumps(model_key_params, sort_keys=True)
    config_hash = hashlib.md5(config_str.encode()).hexdigest()[:12]

    return f"{model_name}_{config_hash}"


def create_config(methodology, model_name, constraint, hyperparam_regime, hyperparam_variation):
    """
    Create a single experiment configuration

    Returns:
        dict: Complete experiment configuration
    """
    base_model_id = compute_base_model_id(model_name, hyperparam_variation)

    config = {
        'methodology': methodology,
        'model_name': model_name,
        'constraint': constraint,
        'hyperparam_regime': hyperparam_regime,
        'hyperparams': hyperparam_variation,
        'base_model_id': base_model_id,
        'experiment_path': None,  # Will be set by filesystem_manager
        'status': 'pending'
    }

    return config


def generate_all_configs():
    """
    Generate all experiment configurations across all dimensions

    Returns:
        list: List of all configuration dictionaries
    """
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
                            variation
                        )

                        all_configs.append(config)
                        config_id += 1

    print(f"Total configurations generated: {len(all_configs)}")
    return all_configs


def save_configs_and_create_structure(configs, output_dir='results'):
    """
    Save all configs to their respective experiment folders
    Creates the complete directory structure

    Args:
        configs: List of configuration dictionaries
        output_dir: Base output directory for results

    Returns:
        int: Number of configs saved
    """
    from utils.filesystem_manager import ensure_experiment_path, save_config_to_path

    print(f"\nCreating experiment directory structure in '{output_dir}'...")

    saved_count = 0
    for i, config in enumerate(configs):
        # Create experiment path
        experiment_path = ensure_experiment_path(config)

        # Update config with experiment path
        config['experiment_path'] = experiment_path

        # Save config to experiment folder
        save_config_to_path(config, experiment_path)

        saved_count += 1

        if (i + 1) % 100 == 0:
            print(f"  Created {i + 1}/{len(configs)} experiment folders...")

    print(f"Successfully created {saved_count} experiment configurations!")
    return saved_count


def generate_summary_report(configs, output_file='experiment_plan_summary.txt'):
    """
    Generate a human-readable summary of the experiment plan

    Args:
        configs: List of configuration dictionaries
        output_file: Output file for summary
    """
    with open(output_file, 'w') as f:
        f.write("="*80 + "\n")
        f.write("EXPERIMENT PLAN SUMMARY\n")
        f.write("="*80 + "\n\n")

        # Overall stats
        f.write(f"Total Experiments: {len(configs)}\n\n")

        # By methodology
        f.write("By Methodology:\n")
        for methodology in METHODOLOGIES:
            count = sum(1 for c in configs if c['methodology'] == methodology)
            f.write(f"  {methodology}: {count}\n")
        f.write("\n")

        # By model
        f.write("By Model:\n")
        for model in MODELS:
            count = sum(1 for c in configs if c['model_name'] == model)
            f.write(f"  {model}: {count}\n")
        f.write("\n")

        # By constraint
        f.write("By Constraint:\n")
        for constraint in CONSTRAINTS:
            count = sum(1 for c in configs if c['constraint'] == constraint)
            f.write(f"  {constraint}: {count}\n")
        f.write("\n")

        # By regime
        f.write("By Hyperparameter Regime:\n")
        for regime_name in HYPERPARAM_REGIMES.keys():
            count = sum(1 for c in configs if c['hyperparam_regime'] == regime_name)
            f.write(f"  {regime_name}: {count}\n")
        f.write("\n")

        # Unique base models
        unique_base_models = len(set(c['base_model_id'] for c in configs))
        f.write(f"Unique Base Models (for pre-training): {unique_base_models}\n")
        f.write("\n")

        f.write("="*80 + "\n")

    print(f"\nSummary report saved to: {output_file}")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """
    Main execution: Generate all configs and create directory structure
    """
    print("="*80)
    print("EXPERIMENT CONFIGURATION GENERATOR")
    print("="*80)
    print()

    # Generate all configurations
    all_configs = generate_all_configs()

    # Save configs and create directory structure
    saved_count = save_configs_and_create_structure(all_configs)

    # Generate summary report
    generate_summary_report(all_configs)

    print("\n" + "="*80)
    print("CONFIGURATION GENERATION COMPLETE")
    print("="*80)
    print()
    print("Next steps:")
    print("1. Review the experiment_plan_summary.txt file")
    print("2. Run experiments using: python run_all_experiments.py")
    print()


if __name__ == "__main__":
    main()
