"""Configuration generator for experiment grid."""

import hashlib
import json
from pathlib import Path

# FTTransformer with multiple constraint pairs for comparison
MODELS = ['FTTransformer']
CONSTRAINTS = [
    (0.5, 0.3),  # Tight: 30% of class 1 allowed
    (0.9, 0.8),  # Loose: 80% of class 1 allowed
]

# Ablation study variations - each disables one component
ABLATION_VARIATIONS = {
    'full': {},  # Full pipeline (baseline for ablation comparison)
    'no_temp_scaling': {
        # Disable temperature scaling: all temps = 1.0
        'warmup_temp': 1.0,
        'drop_start_temp': 1.0,
        'drop_end_temp': 1.0,
        'conv_temp': 1.0,
    },
    'no_alm': {
        # Disable Augmented Lagrangian Method: rho = 0 (only linear penalty)
        'initial_rho': 0.0,
    },
    'no_margin': {
        # Disable constraint margin: aim for exact limit
        'constraint_margin': 0.0,
    },
}

HYPERPARAMS = {
    # Basic training params
    'lr': 0.001,
    'lr_constraint': 0.00001,  # Very low for precise adjustment in drop phase
    'dropout': 0.3,
    'batch_size': 64,
    'hidden_dims': [128, 64],

    # Epoch configuration (500 total: 50 warmup + 300 drop + 150 convergence)
    # Extended for very gradual, stable convergence
    'epochs': 500,
    'warmup_epochs': 50,
    'drop_epochs': 300,
    'conv_epochs': 150,

    # Lambda scheduling - VERY GENTLE for stable convergence over 500 epochs
    # Constraint loss ~5 at start, so lambda=0.005 gives effective weight 0.025 (vs CE ~0.3)
    # This is ~12x weaker than CE initially, building up very gradually
    'lambda_global': 0.005,
    'lambda_local': 0.005,
    'lambda_step': 0.001,  # Very slow increase (takes ~100 epochs to reach 0.1)
    'constraint_threshold': 0.02,

    # Constraint loss improvements (Augmented Lagrangian Method + Gumbel-Softmax)
    'constraint_margin': 0.05,  # Aim for 95% of constraint (5% safety margin)
    'use_sum_loss': True,       # Sum constraint losses instead of averaging
    'constraint_update_frequency': 10,  # Apply constraint gradient every N batches (~7x per epoch)
    'initial_rho': 0.5,         # Lower initial quadratic penalty for gentler start
    'gumbel_temp': 0.5,         # Gumbel-Softmax temperature (lower = more like hard argmax)

    # Temperature scaling
    'use_temperature_scaling': True,
    'learnable_temperature': False,
    'warmup_temp': 1.0,
    'drop_start_temp': 1.5,
    'drop_end_temp': 0.5,
    'conv_temp': 0.5,

    # Learning rate recovery (if constraints not satisfied by convergence phase)
    'recovery_lr_multiplier': 2.0,
    'recovery_interval': 25,

    # Saturation detection (infeasible constraints)
    'saturation_window': 20,
    'saturation_threshold': 1e-4,

    # Refinement after satisfaction
    'refinement_epochs': 20,  # Continue for 20 epochs after satisfaction for stability
    'maintenance_lambda_factor': 0.5,  # Keep lambdas at 50% after satisfaction (gentler reduction)
    'violation_boost_factor': 1.2,  # Gentle boost if violation occurs (was 2.0)
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


def generate_configs(methodology='our_approach', ablation=None):
    """Generate all experiment configurations.

    Args:
        methodology: Either 'our_approach' or 'heuristic'
        ablation: Optional ablation variation name (e.g., 'no_temp_scaling', 'no_alm', 'no_margin')
                  If None, generates standard configs. If specified, applies ablation overrides.
    """
    configs = []
    for model in MODELS:
        for constraint in CONSTRAINTS:
            # Start with base hyperparams
            hyperparams = HYPERPARAMS.copy()

            # Apply ablation overrides if specified
            variation_name = 'default'
            hyperparam_regime = 'standard'
            if ablation and ablation in ABLATION_VARIATIONS:
                overrides = ABLATION_VARIATIONS[ablation]
                hyperparams.update(overrides)
                variation_name = ablation
                hyperparam_regime = 'ablation'

            config = {
                'methodology': methodology,
                'model_name': model,
                'constraint': constraint,
                'hyperparam_regime': hyperparam_regime,
                'variation_name': variation_name,
                'hyperparams': hyperparams,
                'base_model_id': compute_base_model_id(model, HYPERPARAMS),  # Use base for cache sharing
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
        hyperparam_regime = config.get('hyperparam_regime', 'standard')
        variation_name = config.get('variation_name', 'default')

        # Path structure: results/methodology/model/constraint/regime/variation
        path = Path(output_dir) / methodology / config['model_name'] / \
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
    print("=== Standard Configs ===")
    print("1. Generate our_approach configs (standard)")
    print("2. Generate heuristic configs")
    print("3. Generate both standard configs")
    print("")
    print("=== Ablation Studies ===")
    print("4. Generate ablation: no_temp_scaling (all temps = 1.0)")
    print("5. Generate ablation: no_alm (rho = 0)")
    print("6. Generate ablation: no_margin (margin = 0)")
    print("7. Generate ablation: full (baseline for comparison)")
    print("8. Generate ALL ablation configs")
    print("")
    print("=== Utilities ===")
    print("9. Reset all to pending")
    print("0. Exit")

    choice = input("\nChoice: ").strip()

    if choice == '1':
        configs = generate_configs('our_approach')
        save_configs(configs)
    elif choice == '2':
        configs = generate_configs('heuristic')
        save_configs(configs)
    elif choice == '3':
        configs_opt = generate_configs('our_approach')
        configs_heur = generate_configs('heuristic')
        save_configs(configs_opt)
        save_configs(configs_heur)
    elif choice == '4':
        configs = generate_configs('our_approach', ablation='no_temp_scaling')
        save_configs(configs)
    elif choice == '5':
        configs = generate_configs('our_approach', ablation='no_alm')
        save_configs(configs)
    elif choice == '6':
        configs = generate_configs('our_approach', ablation='no_margin')
        save_configs(configs)
    elif choice == '7':
        configs = generate_configs('our_approach', ablation='full')
        save_configs(configs)
    elif choice == '8':
        # Generate all ablation configs
        for ablation_name in ABLATION_VARIATIONS.keys():
            configs = generate_configs('our_approach', ablation=ablation_name)
            save_configs(configs)
            print(f"  -> {ablation_name}")
    elif choice == '9':
        reset_all_to_pending()


if __name__ == "__main__":
    main()
