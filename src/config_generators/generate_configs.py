"""Configuration generator for experiment grid."""

import hashlib
import json
from pathlib import Path

# ── Dataset configurations (replaces config/experiment_config.py) ────────────

DATASET_CONFIGS = {
    # 'binary': {
    #     'train_path': 'data/adult/train_dataset_cleaned.csv',
    #     'test_path': 'data/adult/test_dataset_cleaned.csv',
    #     'target_column': 'income',
    #     'group_column': 'race',
    #     'num_classes': 2,
    #     'constrained_class': 1,
    # },
    'dermmnist': {
        'data_dir': 'data/dermmnist',
        'target_column': 'label',
        'group_column': 'sex',  # 0=male, 1=female (from DermaMNIST-C metadata)
        'num_classes': 7,
        'constrained_class': 4,  # Melanoma
        'image_size': 224,
    },
}

# ── Model + constraint grid ──────────────────────────────────────────────────

# MODELS_TABULAR = ['FTTransformer']
MODELS_IMAGERY = ['ResNet18', 'ResNet50']

CONSTRAINTS = [
    (0.5, 0.3),
]

ALPHA_KL_VALUES = [0.0, 0.1, 1.0]

# ── Hyperparameters ──────────────────────────────────────────────────────────

# HYPERPARAMS_TABULAR = {
#     'lr': 0.001,
#     'lr_constraint': 0.00001,
#     'dropout': 0.3,
#     'batch_size': 64,
#     'hidden_dims': [128, 64],
#     'warmup_epochs': 5,
#     'constraint_epochs': 350,
#     'lambda_global': 0.005,
#     'lambda_local': 0.005,
#     'lambda_step': 0.001,
#     'use_sum_loss': True,
#     'initial_rho': 0.5,
#     'alpha_kl': 1.0,
# }

HYPERPARAMS_IMAGERY = {
    'lr': 0.0001,
    'lr_constraint': 0.00001,
    'dropout': 0.3,
    'batch_size': 64,
    'warmup_epochs': 5,
    'constraint_epochs': 100,
    'lambda_global': 0.005,
    'lambda_local': 0.005,
    'lambda_step': 0.001,
    'use_sum_loss': True,
    'initial_rho': 0.5,
    'alpha_kl': 1.0,
    'pretrained': False,
}


# ── Helpers ──────────────────────────────────────────────────────────────────

def compute_base_model_id(model_name, hyperparams, dataset_mode='binary'):
    """Generate unique ID for model caching based on architecture params."""
    key = {
        'model_name': model_name,
        'lr': hyperparams['lr'],
        'dropout': hyperparams['dropout'],
        'batch_size': hyperparams['batch_size'],
        'warmup_epochs': hyperparams['warmup_epochs'],
        'dataset_mode': dataset_mode,
    }
    # Only include hidden_dims for tabular models
    if 'hidden_dims' in hyperparams:
        key['hidden_dims'] = tuple(hyperparams['hidden_dims'])
    h = hashlib.md5(json.dumps(key, sort_keys=True).encode()).hexdigest()[:12]
    return f"{model_name}_{dataset_mode}_{h}"


def generate_configs(methodology='our_approach', dataset_mode='dermmnist'):
    """Generate experiment configs for given methodology and dataset mode.

    Creates one config per (model, constraint, alpha_kl) combination.
    All alpha_kl values share the same warmup cache.
    """
    if dataset_mode == 'dermmnist':
        models = MODELS_IMAGERY
        base_hyperparams = HYPERPARAMS_IMAGERY
    else:
        raise ValueError(f"Tabular configs temporarily disabled. Use dataset_mode='dermmnist'.")
        # models = MODELS_TABULAR
        # base_hyperparams = HYPERPARAMS_TABULAR

    ds_config = DATASET_CONFIGS[dataset_mode]

    configs = []
    for model in models:
        for constraint in CONSTRAINTS:
            for alpha_kl in ALPHA_KL_VALUES:
                hyperparams = base_hyperparams.copy()
                hyperparams['alpha_kl'] = alpha_kl

                config = {
                    'methodology': methodology,
                    'model_name': model,
                    'constraint': constraint,
                    'dataset_mode': dataset_mode,
                    'dataset_config': ds_config,
                    'hyperparam_regime': 'standard',
                    'variation_name': f'alpha_kl_{alpha_kl}',
                    'hyperparams': hyperparams,
                    'base_model_id': compute_base_model_id(
                        model, base_hyperparams, dataset_mode),
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
        dataset_mode = config.get('dataset_mode', 'dermmnist')
        hyperparam_regime = config.get('hyperparam_regime', 'standard')
        variation_name = config.get('variation_name', 'default')

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
    print("=== DermMNIST (7 classes, imagery) ===")
    print("1. Generate dermmnist our_approach configs")
    print("2. Generate dermmnist heuristic configs")
    print("3. Generate dermmnist both")
    print("")
    # print("=== Adult Binary (tabular, legacy) ===")
    # print("4. Generate binary our_approach configs")
    # print("5. Generate binary both")
    # print("")
    print("=== Utilities ===")
    print("9. Reset all to pending")
    print("0. Exit")

    choice = input("\nChoice: ").strip()

    if choice == '1':
        save_configs(generate_configs('our_approach', 'dermmnist'))
    elif choice == '2':
        save_configs(generate_configs('heuristic', 'dermmnist'))
    elif choice == '3':
        save_configs(generate_configs('our_approach', 'dermmnist'))
        save_configs(generate_configs('heuristic', 'dermmnist'))
    # elif choice == '4':
    #     save_configs(generate_configs('our_approach', 'binary'))
    # elif choice == '5':
    #     save_configs(generate_configs('our_approach', 'binary'))
    #     save_configs(generate_configs('heuristic', 'binary'))
    elif choice == '9':
        reset_all_to_pending()


if __name__ == "__main__":
    main()
