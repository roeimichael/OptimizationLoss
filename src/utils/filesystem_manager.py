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

def update_experiment_status(experiment_path: str, status: str) -> None:
    config = load_config_from_path(experiment_path)
    config['status'] = status
    save_config_to_path(config, experiment_path)

def mark_experiment_complete(experiment_path: str) -> None:
    update_experiment_status(experiment_path, 'completed')

def is_experiment_complete(experiment_path: str) -> bool:
    try:
        config = load_config_from_path(experiment_path)
        return config.get('status', 'pending') == 'completed'
    except:
        return False

def get_experiments_by_status(results_dir: str = 'results') -> Dict[str, List[Tuple[str, Dict[str, Any]]]]:
    all_experiments = get_all_experiment_configs(results_dir)
    by_status = {'pending': [], 'completed': [], 'running': []}
    for exp_path, config in all_experiments:
        status = config.get('status', 'pending')
        if status == 'running':
            by_status['pending'].append((exp_path, config))
        elif status in by_status:
            by_status[status].append((exp_path, config))
        else:
            by_status['pending'].append((exp_path, config))
    return by_status

def print_status_summary(results_dir: str = 'results') -> None:
    by_status = get_experiments_by_status(results_dir)
    total = sum(len(exps) for exps in by_status.values())
    print("\n" + "="*80)
    print("EXPERIMENT STATUS SUMMARY")
    print("="*80)
    print(f"Total experiments: {total}")
    print(f"  Completed: {len(by_status['completed'])}")
    print(f"  Pending: {len(by_status['pending'])} (includes interrupted runs)")
    print("="*80 + "\n")
