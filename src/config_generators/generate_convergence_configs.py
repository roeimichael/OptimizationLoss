import hashlib
import json
from pathlib import Path
from typing import List, Tuple


def compute_base_model_id(model_name: str, hyperparams: dict) -> str:
    relevant_params = {
        'model_name': model_name,
        'lr': hyperparams['lr'],
        'dropout': hyperparams['dropout'],
        'batch_size': hyperparams['batch_size'],
        'hidden_dims': tuple(hyperparams['hidden_dims']),
        'warmup_epochs': hyperparams['warmup_epochs']
    }
    param_str = json.dumps(relevant_params, sort_keys=True)
    config_hash = hashlib.md5(param_str.encode()).hexdigest()[:12]
    return f"{model_name}_{config_hash}"  # MATCH original format


def create_experiment_config(model_name: str, constraint_pair: List[float]) -> dict:
    """Create experiment configuration."""
    hyperparams = {
        'lr': 0.001,
        'batch_size': 64,
        'warmup_epochs': 50,
        'epochs': 1000,
        'hidden_dims': [128, 64],
        'dropout': 0.3,
        'lambda_global': 0.1,
        'lambda_local': 0.1,
        'lambda_step': 0.005,
        'constraint_threshold': 0.02,
    }

    base_model_id = compute_base_model_id(model_name, hyperparams)

    config = {
        'model_name': model_name,
        'base_model_id': base_model_id,
        'constraint': constraint_pair,
        'hyperparams': hyperparams,
        'status': 'pending'
    }

    return config


def main():
    """Generate experiment configurations for testing."""
    model_name = 'TabularResNet'
    constraint_pairs = [
        [0.5, 0.3],
        [0.8, 0.2],
        [0.9, 0.8]
    ]

    base_results_dir = Path('../../results/test_experiments')
    base_results_dir.mkdir(parents=True, exist_ok=True)
    experiment_count = 0

    for constraint_pair in constraint_pairs:
        constraint_str = f"constraint_{constraint_pair[0]}_{constraint_pair[1]}"
        exp_dir = base_results_dir / model_name / constraint_str
        exp_dir.mkdir(parents=True, exist_ok=True)
        config = create_experiment_config(model_name=model_name, constraint_pair=constraint_pair)
        config_path = exp_dir / 'config.json'
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        experiment_count += 1

    print(f"\nSuccessfully generated {experiment_count} experiment configurations")


if __name__ == '__main__':
    main()
