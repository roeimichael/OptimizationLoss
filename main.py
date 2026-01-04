import subprocess
import sys
from pathlib import Path

from src.utils.filesystem_manager import get_experiments_by_status, print_status_summary

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
    print(f"Running {len(pending_experiments)} pending experiments...\n")
    completed = 0
    failed = 0
    for i, (experiment_path, config) in enumerate(pending_experiments, 1):
        config_path = Path(experiment_path) / 'config.json'
        if not config_path.exists():
            print(f"[ERROR] Config file not found: {config_path}")
            failed += 1
            continue
        if not Path(experiment_path).exists():
            print(f"[ERROR] Experiment folder not found: {experiment_path}")
            failed += 1
            continue
        print(f"\n{'='*80}")
        print(f"[{i}/{len(pending_experiments)}] Running experiment")
        print(f"Path: {experiment_path}")
        print(f"{'='*80}")
        try:
            result = subprocess.run(
                [sys.executable, 'run_experiment.py', str(config_path)],
                capture_output=False,
                text=True
            )
            if result.returncode == 0:
                completed += 1
            else:
                failed += 1
        except KeyboardInterrupt:
            print("\n\n" + "="*80)
            print("INTERRUPTED BY USER")
            print("="*80)
            print(f"Completed: {completed}")
            print(f"Failed: {failed}")
            print(f"Remaining: {len(pending_experiments) - i}")
            print("="*80)
            print("\nResume by running 'python main.py' again.")
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
