import subprocess
import sys
from pathlib import Path
from typing import List, Tuple, Dict, Any
from collections import defaultdict

from utils.filesystem_manager import get_experiments_by_status, print_status_summary

def main() -> None:
    print("="*80)
    print("BATCH EXPERIMENT RUNNER")
    print("="*80)
    print_status_summary('results')
    by_status = get_experiments_by_status('results')
    pending_experiments = by_status['pending']
    if not pending_experiments:
        print("No pending experiments found!")
        print("All experiments are completed or there are no configurations.")
        print("Run 'python utils/generate_configs.py' to create new configurations.")
        return
    print(f"Running {len(pending_experiments)} pending experiments...")
    print()
    by_model = defaultdict(list)
    for exp_path, config in pending_experiments:
        model_name = config.get('model_name', 'Unknown')
        by_model[model_name].append((exp_path, config))
    print("Breakdown by model:")
    for model_name, exps in sorted(by_model.items()):
        print(f"  {model_name}: {len(exps)} pending")
    print()
    completed = 0
    failed = 0
    for i, (experiment_path, config) in enumerate(pending_experiments, 1):
        config_path = Path(experiment_path) / 'config.json'
        model_name = config.get('model_name', 'Unknown')
        constraint = config.get('constraint', 'Unknown')
        regime = config.get('hyperparam_regime', 'Unknown')
        variation = config.get('variation_name', 'Unknown')
        print(f"\n{'='*80}")
        print(f"[{i}/{len(pending_experiments)}] {model_name} | Constraint: {constraint} | {regime}/{variation}")
        print(f"Path: {experiment_path}")
        print(f"{'='*80}")
        try:
            result: subprocess.CompletedProcess = subprocess.run(
                [sys.executable, 'run_experiment.py', str(config_path)],
                capture_output=False,
                text=True
            )
            if result.returncode == 0:
                completed += 1
                print(f"[SUCCESS] Experiment {i}/{len(pending_experiments)} completed")
            else:
                failed += 1
                print(f"[FAILED] Experiment {i}/{len(pending_experiments)} failed (exit code: {result.returncode})")
        except KeyboardInterrupt:
            print("\n\n" + "="*80)
            print("INTERRUPTED BY USER")
            print("="*80)
            print(f"Completed: {completed}")
            print(f"Failed: {failed}")
            print(f"Remaining: {len(pending_experiments) - i}")
            print("="*80)
            print("\nYou can resume by running 'python main.py' again.")
            print("The script will automatically continue from pending experiments.")
            break
        except Exception as e:
            failed += 1
            print(f"[ERROR] Failed to run experiment: {e}")
    else:
        print("\n" + "="*80)
        print("BATCH RUN COMPLETE")
        print("="*80)
        print(f"Successfully completed: {completed}")
        print(f"Failed: {failed}")
        print("="*80)
        print_status_summary('results')

if __name__ == "__main__":
    main()
