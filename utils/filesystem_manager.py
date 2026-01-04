import json
from pathlib import Path
from typing import Dict, Any, List, Tuple


def ensure_experiment_path(config: Dict[str, Any]) -> str:
    methodology = config.get('methodology', 'our_approach')
    model_name = config.get('model_name', 'BasicNN')
    constraint = config.get('constraint', (0.5, 0.3))
    hyperparam_regime = config.get('hyperparam_regime', 'standard')
    variation_name = config.get('variation_name', 'default')
    constraint_str = f"constraint_{constraint[0]}_{constraint[1]}"
    experiment_path = Path('results') / methodology / model_name / constraint_str / hyperparam_regime / variation_name
    experiment_path.mkdir(parents=True, exist_ok=True)
    return str(experiment_path)


def save_config_to_path(config: Dict[str, Any], experiment_path: str) -> str:
    config_path = Path(experiment_path) / 'config.json'
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4)
    return str(config_path)


def load_config_from_path(experiment_path: str) -> Dict[str, Any]:
    config_path = Path(experiment_path) / 'config.json'
    if not config_path.exists():
        raise FileNotFoundError(f"No config.json found in {experiment_path}")
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config


def get_all_experiment_configs(results_dir: str = 'results') -> List[Tuple[str, Dict[str, Any]]]:
    experiments = []
    results_path = Path(results_dir)
    if not results_path.exists():
        return experiments
    for config_file in results_path.rglob('config.json'):
        experiment_path = config_file.parent
        try:
            config = load_config_from_path(experiment_path)
            experiments.append((str(experiment_path), config))
        except Exception as e:
            print(f"Warning: Failed to load config from {experiment_path}: {e}")
    return experiments


def mark_experiment_complete(experiment_path: str) -> None:
    complete_marker = Path(experiment_path) / '.complete'
    complete_marker.touch()


def is_experiment_complete(experiment_path: str) -> bool:
    complete_marker = Path(experiment_path) / '.complete'
    return complete_marker.exists()
