import hashlib
import json
from pathlib import Path
from typing import Dict, Any, List, Tuple

METHODOLOGIES = ['our_approach']

MODELS = ['BasicNN', 'ResNet56', 'DenseNet121', 'InceptionV3', 'VGG19']

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

BASE_HYPERPARAMS = {
    'lr': 0.001,
    'dropout': 0.3,
    'batch_size': 64,
    'hidden_dims': [128, 64],
    'epochs': 10000,
    'lambda_global': 0.01,
    'lambda_local': 0.01,
    'warmup_epochs': 250,
    'constraint_threshold': 1e-6,
    'lambda_step': 0.01
}

HYPERPARAM_REGIMES = {
    'standard': {
        'name': 'standard',
        'variations': [
            {'variation_name': 'default', 'params': BASE_HYPERPARAMS.copy()}
        ]
    },
    'lr_test': {
        'name': 'lr_test',
        'variations': [
            {'variation_name': f'lr_{lr}', 'params': {**BASE_HYPERPARAMS, 'lr': lr}}
            for lr in [0.0001, 0.0005, 0.001, 0.005, 0.01]
        ]
    },
    'dropout_test': {
        'name': 'dropout_test',
        'variations': [
            {'variation_name': f'dropout_{dropout}', 'params': {**BASE_HYPERPARAMS, 'dropout': dropout}}
            for dropout in [0.1, 0.2, 0.3, 0.4, 0.5]
        ]
    },
    'batch_test': {
        'name': 'batch_test',
        'variations': [
            {'variation_name': f'batch_{batch}', 'params': {**BASE_HYPERPARAMS, 'batch_size': batch}}
            for batch in [32, 64, 128, 256, 512]
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
        'warmup_epochs': hyperparams['warmup_epochs']
    }
    config_str = json.dumps(model_key_params, sort_keys=True)
    config_hash = hashlib.md5(config_str.encode()).hexdigest()[:12]
    return f"{model_name}_{config_hash}"

def create_config(methodology: str, model_name: str, constraint: Tuple[float, float], hyperparam_regime: str, variation_name: str, hyperparam_params: Dict[str, Any]) -> Dict[str, Any]:
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
    from utils.filesystem_manager import ensure_experiment_path, save_config_to_path
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

def generate_summary_report(configs: List[Dict[str, Any]], output_file: str = 'experiment_plan_summary.txt') -> None:
    with open(output_file, 'w') as f:
        f.write("="*80 + "\n")
        f.write("EXPERIMENT PLAN SUMMARY\n")
        f.write("="*80 + "\n\n")
        f.write(f"Total Experiments: {len(configs)}\n\n")
        f.write("By Methodology:\n")
        for methodology in METHODOLOGIES:
            count = sum(1 for c in configs if c['methodology'] == methodology)
            f.write(f"  {methodology}: {count}\n")
        f.write("\n")
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
        f.write("By Hyperparameter Regime:\n")
        for regime_name in HYPERPARAM_REGIMES.keys():
            count = sum(1 for c in configs if c['hyperparam_regime'] == regime_name)
            f.write(f"  {regime_name}: {count}\n")
        f.write("\n")
        unique_base_models = len(set(c['base_model_id'] for c in configs))
        f.write(f"Unique Base Models (for pre-training): {unique_base_models}\n")
        f.write("\n")
        f.write("="*80 + "\n")
    print(f"\nSummary report saved to: {output_file}")

def reset_all_status_to_pending(results_dir: str = 'results') -> int:
    from utils.filesystem_manager import get_all_experiment_configs, save_config_to_path
    print("="*80)
    print("RESET ALL EXPERIMENT STATUSES")
    print("="*80)
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
    print("\n" + "="*80)
    print("RESET COMPLETE")
    print("="*80)
    return reset_count

def main() -> None:
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == '--reset':
        reset_all_status_to_pending()
        return
    print("="*80)
    print("EXPERIMENT CONFIGURATION GENERATOR")
    print("="*80)
    print()
    all_configs = generate_all_configs()
    saved_count = save_configs_and_create_structure(all_configs)
    generate_summary_report(all_configs)
    print("\n" + "="*80)
    print("CONFIGURATION GENERATION COMPLETE")
    print("="*80)
    print()
    print("Next steps:")
    print("1. Review the experiment_plan_summary.txt file")
    print("2. Run experiments using: python main.py")
    print("3. To reset all statuses: python utils/generate_configs.py --reset")
    print()

if __name__ == "__main__":
    main()
