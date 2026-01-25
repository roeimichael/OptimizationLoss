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


def generate_convergence_combinations() -> List[Tuple[int, int]]:
    combinations = [
        (1, 1),
        (5, 2),  # 40% satisfaction
        (5, 5),  # 100% satisfaction
        (10, 5),  # 50% satisfaction
        (10, 7),  # 70% satisfaction
        (20, 12),  # 60% satisfaction
        (20, 14),  # 70% satisfaction
        (20, 15),  # 75% satisfaction (recommended)
        (30, 20),  # 67% satisfaction
        (30, 24),  # 80% satisfaction
        (30, 27),  # 90% satisfaction
    ]
    return combinations


def create_experiment_config(
        model_name: str,
        constraint_pair: List[float],
        convergence_window: int,
        convergence_required: int
) -> dict:
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
        'lambda_strategy': 'linear',
        'constraint_threshold': 0.02,
        'convergence_window': convergence_window,
        'convergence_required': convergence_required
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
    model_name = 'TabularResNet'
    constraint_pairs = [
        [0.5, 0.3],
        [0.8, 0.2],
        [0.9, 0.8]
    ]

    convergence_combos = generate_convergence_combinations()
    base_results_dir = Path('../../results/longer_saturation')
    base_results_dir.mkdir(parents=True, exist_ok=True)
    experiment_count = 0
    for constraint_pair in constraint_pairs:
        constraint_str = f"constraint_{constraint_pair[0]}_{constraint_pair[1]}"

        for window, required in convergence_combos:
            exp_dir = base_results_dir / model_name / constraint_str / 'convergence_test' / f'conv_{window}_{required}'
            exp_dir.mkdir(parents=True, exist_ok=True)
            config = create_experiment_config(
                model_name=model_name,
                constraint_pair=constraint_pair,
                convergence_window=window,
                convergence_required=required
            )
            config_path = exp_dir / 'config.json'
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=2)

            experiment_count += 1

    print(f"\n✓ Successfully generated {experiment_count} experiment configurations")


if __name__ == '__main__':
    main()
