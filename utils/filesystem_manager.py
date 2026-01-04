"""
Filesystem Manager for Experiment Organization
Ensures proper directory structure and path validation
"""
import os
from pathlib import Path
import json


def ensure_experiment_path(config):
    """
    Validates and creates experiment directory structure based on config

    Expected config structure:
    {
        'methodology': 'our_approach',
        'model_name': 'ResNet56',
        'constraint': (0.5, 0.3),
        'hyperparams': {...},
        'base_model_id': 'unique_hash'
    }

    Returns:
        str: Valid path to experiment folder
    """
    # Extract configuration details
    methodology = config.get('methodology', 'our_approach')
    model_name = config.get('model_name', 'BasicNN')
    constraint = config.get('constraint', (0.5, 0.3))
    base_model_id = config.get('base_model_id', 'default')

    # Build path: results/methodology/model/constraint/base_model_id/
    constraint_str = f"constraint_{constraint[0]}_{constraint[1]}"

    experiment_path = Path('results') / methodology / model_name / constraint_str / base_model_id

    # Create directory structure if it doesn't exist
    experiment_path.mkdir(parents=True, exist_ok=True)

    return str(experiment_path)


def save_config_to_path(config, experiment_path):
    """
    Save configuration file to experiment folder

    Args:
        config: Dictionary containing experiment configuration
        experiment_path: Path to experiment folder

    Returns:
        str: Path to saved config file
    """
    config_path = Path(experiment_path) / 'config.json'

    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4)

    return str(config_path)


def load_config_from_path(experiment_path):
    """
    Load configuration from experiment folder

    Args:
        experiment_path: Path to experiment folder

    Returns:
        dict: Configuration dictionary
    """
    config_path = Path(experiment_path) / 'config.json'

    if not config_path.exists():
        raise FileNotFoundError(f"No config.json found in {experiment_path}")

    with open(config_path, 'r') as f:
        config = json.load(f)

    return config


def get_all_experiment_configs(results_dir='results'):
    """
    Recursively find all experiment configs in results directory

    Args:
        results_dir: Root results directory

    Returns:
        list: List of (experiment_path, config) tuples
    """
    experiments = []
    results_path = Path(results_dir)

    if not results_path.exists():
        return experiments

    # Find all config.json files
    for config_file in results_path.rglob('config.json'):
        experiment_path = config_file.parent
        try:
            config = load_config_from_path(experiment_path)
            experiments.append((str(experiment_path), config))
        except Exception as e:
            print(f"Warning: Failed to load config from {experiment_path}: {e}")

    return experiments


def mark_experiment_complete(experiment_path):
    """
    Mark experiment as completed by creating a .complete marker file

    Args:
        experiment_path: Path to experiment folder
    """
    complete_marker = Path(experiment_path) / '.complete'
    complete_marker.touch()


def is_experiment_complete(experiment_path):
    """
    Check if experiment has been completed

    Args:
        experiment_path: Path to experiment folder

    Returns:
        bool: True if experiment is complete
    """
    complete_marker = Path(experiment_path) / '.complete'
    return complete_marker.exists()
