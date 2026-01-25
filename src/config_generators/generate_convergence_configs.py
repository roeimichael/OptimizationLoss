"""
Generate experiment configurations for testing convergence parameters.

Creates 60 experiments:
- 1 model: TabularResNet
- 3 constraint pairs: [0.5, 0.3], [0.8, 0.2], [0.9, 0.8]
- 20 convergence parameter combinations
- Learning rate: 0.001
- Strategy: linear
- Max epochs: 1000 (matches original experiments)

Hyperparameters match EXACTLY with original experiments except for
convergence_window and convergence_required parameters.

Results saved to: results/longer_saturation/
"""

import hashlib
import json
from pathlib import Path
from typing import List, Tuple


def compute_base_model_id(model_name: str, hyperparams: dict) -> str:
    """
    Compute unique ID for base model (warmup-only parameters).
    MUST match EXACTLY with generate_configs.py logic!
    """
    relevant_params = {
        'model_name': model_name,  # MATCH original (was 'model')
        'lr': hyperparams['lr'],
        'dropout': hyperparams['dropout'],  # Order matters for consistency
        'batch_size': hyperparams['batch_size'],
        'hidden_dims': tuple(hyperparams['hidden_dims']),  # MATCH original (was list)
        'warmup_epochs': hyperparams['warmup_epochs']
    }
    param_str = json.dumps(relevant_params, sort_keys=True)
    config_hash = hashlib.md5(param_str.encode()).hexdigest()[:12]
    return f"{model_name}_{config_hash}"  # MATCH original format


def generate_convergence_combinations() -> List[Tuple[int, int]]:
    """
    Generate 20 convergence parameter combinations (window_size, required_satisfied).

    Option B: Specific test cases covering different satisfaction rates and window sizes.
    """
    combinations = [
        # Immediate (1/1 = 100%) - baseline behavior
        (1, 1),

        # Small windows (5 epochs)
        (5, 2),   # 40% satisfaction
        (5, 5),   # 100% satisfaction

        # Medium windows (10 epochs)
        (10, 5),  # 50% satisfaction
        (10, 7),  # 70% satisfaction

        (20, 12), # 60% satisfaction
        (20, 14), # 70% satisfaction
        (20, 15), # 75% satisfaction (recommended)


        # Very large windows (30 epochs)
        (30, 20), # 67% satisfaction
        (30, 24), # 80% satisfaction
        (30, 27), # 90% satisfaction
    ]

    return combinations


def create_experiment_config(
    model_name: str,
    constraint_pair: List[float],
    convergence_window: int,
    convergence_required: int
) -> dict:
    """Create experiment configuration with convergence parameters."""

    # Fixed hyperparameters for all experiments
    # IMPORTANT: Match EXACTLY with original experiments (generate_configs.py)
    # to ensure fair comparison. Only convergence parameters should differ.
    hyperparams = {
        'lr': 0.001,
        'batch_size': 64,
        'warmup_epochs': 50,  # MATCH original (was 300)
        'epochs': 1000,  # MATCH original (was 2000)
        'hidden_dims': [128, 64],
        'dropout': 0.3,
        'lambda_global': 0.1,
        'lambda_local': 0.1,
        'lambda_step': 0.005,  # MATCH original (was 0.1)
        'lambda_strategy': 'linear',
        'constraint_threshold': 0.02,  # MATCH original (was 0.01)
        # NEW: Convergence parameters (ONLY difference from original)
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
    """Generate all convergence test configurations."""

    # Fixed parameters
    model_name = 'TabularResNet'
    # Use the same constraint pairs as original experiments
    # [local%, global%] where:
    #   - local% applies per-course constraints
    #   - global% applies to total test set
    # Results in these constraints:
    #   [0.5, 0.3]: 43 dropouts, 24 enrolled allowed
    #   [0.8, 0.2]: 28 dropouts, 16 enrolled allowed
    #   [0.9, 0.8]: 114 dropouts, 63 enrolled allowed
    constraint_pairs = [
        [0.5, 0.3],
        [0.8, 0.2],
        [0.9, 0.8]
    ]

    # Get convergence combinations
    convergence_combos = generate_convergence_combinations()

    print(f"Generating {len(constraint_pairs)} × {len(convergence_combos)} = {len(constraint_pairs) * len(convergence_combos)} experiments")
    print(f"Model: {model_name}")
    print(f"Constraints: {constraint_pairs}")
    print(f"Convergence combinations: {len(convergence_combos)}")
    print(f"Max epochs: 1000 (matches original experiments)")
    print(f"Warmup epochs: 50 (matches original experiments)")
    print(f"Lambda step: 0.005 (matches original experiments)")
    print(f"Constraint threshold: 0.02 (matches original experiments)")
    print()

    # Create base results directory
    base_results_dir = Path('../../results/longer_saturation')
    base_results_dir.mkdir(parents=True, exist_ok=True)

    experiment_count = 0

    for constraint_pair in constraint_pairs:
        constraint_str = f"constraint_{constraint_pair[0]}_{constraint_pair[1]}"

        for window, required in convergence_combos:
            # Create experiment directory
            # results/longer_saturation/TabularResNet/constraint_50_30/convergence_test/conv_20_15/
            exp_dir = base_results_dir / model_name / constraint_str / 'convergence_test' / f'conv_{window}_{required}'
            exp_dir.mkdir(parents=True, exist_ok=True)

            # Generate config
            config = create_experiment_config(
                model_name=model_name,
                constraint_pair=constraint_pair,
                convergence_window=window,
                convergence_required=required
            )

            # Save config
            config_path = exp_dir / 'config.json'
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=2)

            experiment_count += 1

            # Print progress every 10 experiments
            if experiment_count % 10 == 0:
                print(f"Generated {experiment_count}/{len(constraint_pairs) * len(convergence_combos)} configs...")

    print(f"\n✓ Successfully generated {experiment_count} experiment configurations")
    print(f"  Location: {base_results_dir}")
    print(f"  Structure: {model_name}/constraint_X_Y/convergence_test/conv_W_R/")
    print(f"\nTo run all experiments:")
    print(f"  python run_all_convergence_experiments.py")


if __name__ == '__main__':
    main()
