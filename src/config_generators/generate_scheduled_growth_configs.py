#!/usr/bin/env python3

import json
from pathlib import Path

BASE_CONFIG = {
    'methodology': 'scheduled_growth',
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

MODELS = {
    'BasicNN': {
        'model_name': 'BasicNN',
        'model_kwargs': {'hidden_dims': [128, 64, 32]}
    },
    'TabularResNet': {
        'model_name': 'TabularResNet',
        'model_kwargs': {'d_hidden': 128, 'n_blocks': 3}
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

CONSTRAINTS = {
    'hard_hard': {
        'name': 'constraint_0.3_0.3',
        'global_constraints': [0.3, 0.3],
        'local_constraints': {'0': [0.3, 0.3], '1': [0.3, 0.3], '2': [0.3, 0.3]}
    },
    'hard_soft': {
        'name': 'constraint_0.3_0.8',
        'global_constraints': [0.3, 0.8],
        'local_constraints': {'0': [0.3, 0.8], '1': [0.3, 0.8], '2': [0.3, 0.8]}
    },
    'soft_hard': {
        'name': 'constraint_0.8_0.3',
        'global_constraints': [0.8, 0.3],
        'local_constraints': {'0': [0.8, 0.3], '1': [0.8, 0.3], '2': [0.8, 0.3]}
    },
    'soft_soft': {
        'name': 'constraint_0.9_0.8',
        'global_constraints': [0.9, 0.8],
        'local_constraints': {'0': [0.9, 0.8], '1': [0.9, 0.8], '2': [0.9, 0.8]}
    }
}

GROWTH_REGIMES = {
    'growth_factor_sweep': {
        'name': 'growth_factor_sweep',
        'variations': [
            {'variation_name': 'factor_1.05_freq_10', 'params': {'growth_factor': 1.05, 'check_frequency': 10, 'initial_lambda': 0.001}},
            {'variation_name': 'factor_1.1_freq_10', 'params': {'growth_factor': 1.1, 'check_frequency': 10, 'initial_lambda': 0.001}},
            {'variation_name': 'factor_1.1_freq_20', 'params': {'growth_factor': 1.1, 'check_frequency': 20, 'initial_lambda': 0.001}},
            {'variation_name': 'factor_1.2_freq_10', 'params': {'growth_factor': 1.2, 'check_frequency': 10, 'initial_lambda': 0.001}}
        ]
    }
}


def generate_scheduled_growth_configs(output_dir: str = 'results/scheduled_growth') -> int:
    output_path = Path(output_dir)
    config_count = 0

    for model_key, model_config in MODELS.items():
        for constraint_key, constraint_config in CONSTRAINTS.items():
            for regime_key, regime in GROWTH_REGIMES.items():
                for variation in regime['variations']:
                    exp_dir = (
                        output_path /
                        model_config['model_name'] /
                        constraint_config['name'] /
                        regime['name'] /
                        variation['variation_name']
                    )
                    exp_dir.mkdir(parents=True, exist_ok=True)

                    config = BASE_CONFIG.copy()
                    config.update(model_config)
                    config.update({
                        'global_constraints': constraint_config['global_constraints'],
                        'local_constraints': constraint_config['local_constraints']
                    })
                    config.update(variation['params'])
                    config['experiment_id'] = f"{model_key}_{constraint_key}_{variation['variation_name']}"
                    config['regime'] = regime['name']

                    config_file = exp_dir / 'config.json'
                    with open(config_file, 'w') as f:
                        json.dump(config, f, indent=2)

                    config_count += 1

    print(f"Generated {config_count} scheduled_growth configs")
    return config_count


if __name__ == '__main__':
    generate_scheduled_growth_configs()
