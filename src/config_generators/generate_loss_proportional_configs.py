#!/usr/bin/env python3
"""
Configuration generator for loss-proportional adaptive lambda experiments.

This script generates experiment configurations for testing the loss-proportional
adaptive lambda methodology where lambda increases proportionally to constraint loss.
"""

import json
from pathlib import Path
from typing import Dict, List, Any

# Base configuration
BASE_CONFIG = {
    'methodology': 'loss_proportional',
    'data_path': 'data/processed/students_data.csv',
    'seed': 42,
    'test_size': 0.3,
    'num_classes': 3,
    'learning_rate': 0.001,
    'warmup_epochs': 250,
    'epochs': 1000,
    'constraint_threshold': 1e-6,
    'early_stop_patience': 10,
    'max_lambda': 1.0,
}

# Models to test
MODELS = {
    'BasicNN': {
        'model_name': 'BasicNN',
        'model_kwargs': {
            'hidden_dims': [128, 64, 32]
        }
    },
    'TabularResNet': {
        'model_name': 'TabularResNet',
        'model_kwargs': {
            'd_hidden': 128,
            'n_blocks': 3
        }
    },
    'FTTransformer': {
        'model_name': 'FTTransformer',
        'model_kwargs': {
            'd_token': 64,
            'n_blocks': 2,
            'attention_n_heads': 4,
            'attention_dropout': 0.2,
            'ffn_d_hidden': 128,
            'ffn_dropout': 0.1,
            'residual_dropout': 0.0
        }
    }
}

# Constraint configurations to test
CONSTRAINTS = {
    'hard_hard': {
        'name': 'constraint_0.3_0.3',
        'global_constraints': [0.3, 0.3],
        'local_constraints': {
            '0': [0.3, 0.3],
            '1': [0.3, 0.3],
            '2': [0.3, 0.3]
        }
    },
    'hard_soft': {
        'name': 'constraint_0.3_0.8',
        'global_constraints': [0.3, 0.8],
        'local_constraints': {
            '0': [0.3, 0.8],
            '1': [0.3, 0.8],
            '2': [0.3, 0.8]
        }
    },
    'soft_hard': {
        'name': 'constraint_0.8_0.3',
        'global_constraints': [0.8, 0.3],
        'local_constraints': {
            '0': [0.8, 0.3],
            '1': [0.8, 0.3],
            '2': [0.8, 0.3]
        }
    },
    'soft_soft': {
        'name': 'constraint_0.9_0.8',
        'global_constraints': [0.9, 0.8],
        'local_constraints': {
            '0': [0.9, 0.8],
            '1': [0.9, 0.8],
            '2': [0.9, 0.8]
        }
    }
}

# Lambda learning rate variations to test
LAMBDA_LR_REGIMES = {
    'lambda_lr_sweep': {
        'name': 'lambda_lr_sweep',
        'description': 'Sweep different lambda learning rates to find optimal growth speed',
        'variations': [
            {
                'variation_name': 'alpha_0.005',
                'params': {
                    'lambda_learning_rate': 0.005,
                    'initial_lambda': 0.001
                }
            },
            {
                'variation_name': 'alpha_0.01',
                'params': {
                    'lambda_learning_rate': 0.01,
                    'initial_lambda': 0.001
                }
            },
            {
                'variation_name': 'alpha_0.02',
                'params': {
                    'lambda_learning_rate': 0.02,
                    'initial_lambda': 0.001
                }
            },
            {
                'variation_name': 'alpha_0.05',
                'params': {
                    'lambda_learning_rate': 0.05,
                    'initial_lambda': 0.001
                }
            }
        ]
    }
}


def generate_loss_proportional_configs(output_dir: str = 'results/loss_proportional') -> int:
    """
    Generate all experiment configurations for loss-proportional adaptive lambda.

    Args:
        output_dir: Root directory for experiment results

    Returns:
        Number of configurations generated
    """
    output_path = Path(output_dir)
    config_count = 0

    print(f"Generating loss-proportional adaptive lambda configurations...")
    print(f"Output directory: {output_path}")
    print()

    # Iterate through all combinations
    for model_key, model_config in MODELS.items():
        for constraint_key, constraint_config in CONSTRAINTS.items():
            for regime_key, regime in LAMBDA_LR_REGIMES.items():
                for variation in regime['variations']:
                    # Create experiment directory
                    exp_dir = (
                        output_path /
                        model_config['model_name'] /
                        constraint_config['name'] /
                        regime['name'] /
                        variation['variation_name']
                    )
                    exp_dir.mkdir(parents=True, exist_ok=True)

                    # Build configuration
                    config = BASE_CONFIG.copy()
                    config.update(model_config)
                    config.update({
                        'global_constraints': constraint_config['global_constraints'],
                        'local_constraints': constraint_config['local_constraints']
                    })
                    config.update(variation['params'])

                    # Add metadata
                    config['experiment_id'] = f"{model_key}_{constraint_key}_{variation['variation_name']}"
                    config['regime'] = regime['name']
                    config['regime_description'] = regime['description']

                    # Save configuration
                    config_file = exp_dir / 'config.json'
                    with open(config_file, 'w') as f:
                        json.dump(config, f, indent=2)

                    config_count += 1

    print(f"✓ Generated {config_count} experiment configurations")
    print()
    print(f"Breakdown:")
    print(f"  Models: {len(MODELS)}")
    print(f"  Constraint types: {len(CONSTRAINTS)}")
    print(f"  Lambda LR variations: {sum(len(r['variations']) for r in LAMBDA_LR_REGIMES.values())}")
    print(f"  Total: {config_count}")
    print()

    return config_count


if __name__ == '__main__':
    generate_loss_proportional_configs()
