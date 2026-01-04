import argparse
import subprocess
import sys
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional

from utils.filesystem_manager import get_all_experiment_configs, is_experiment_complete

def run_all_experiments(resume: bool = True, max_experiments: Optional[int] = None) -> None:
    print("="*80)
    print("BATCH EXPERIMENT RUNNER")
    print("="*80)
    print()
    print("Scanning for experiments...")
    experiments: List[Tuple[str, Dict[str, Any]]] = get_all_experiment_configs('results')
    if not experiments:
        print("No experiments found!")
        print("Run 'python utils/generate_configs.py' first to create experiment configurations.")
        return
    print(f"Found {len(experiments)} total experiments")
    if resume:
        pending_experiments = [
            (path, config) for path, config in experiments
            if not is_experiment_complete(path)
        ]
        print(f"Pending experiments: {len(pending_experiments)}")
    else:
        pending_experiments = experiments
        print("Running all experiments (including completed ones)")
    if max_experiments is not None:
        pending_experiments = pending_experiments[:max_experiments]
        print(f"Limiting to first {max_experiments} experiments")
    print()
    completed = 0
    failed = 0
    skipped = 0
    for i, (experiment_path, config) in enumerate(pending_experiments, 1):
        config_path = Path(experiment_path) / 'config.json'
        print(f"\n{'='*80}")
        print(f"[{i}/{len(pending_experiments)}] Processing: {experiment_path}")
        print(f"{'='*80}")
        try:
            result: subprocess.CompletedProcess = subprocess.run(
                [sys.executable, 'run_experiment.py', str(config_path)],
                capture_output=False,
                text=True
            )
            if result.returncode == 0:
                completed += 1
            else:
                print(f"[WARNING] Experiment returned non-zero exit code: {result.returncode}")
                failed += 1
        except KeyboardInterrupt:
            print("\n\nInterrupted by user!")
            break
        except Exception as e:
            print(f"[ERROR] Failed to run experiment: {e}")
            failed += 1
    print("\n" + "="*80)
    print("BATCH RUN SUMMARY")
    print("="*80)
    print(f"Total experiments found: {len(experiments)}")
    print(f"Pending experiments: {len(pending_experiments)}")
    print(f"Successfully completed: {completed}")
    print(f"Failed: {failed}")
    print(f"Skipped: {skipped}")
    print("="*80)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run all experiment configurations')
    parser.add_argument('--no-resume', action='store_true',
                      help='Re-run all experiments (including completed ones)')
    parser.add_argument('--max', type=int, default=None,
                      help='Maximum number of experiments to run')
    args = parser.parse_args()
    run_all_experiments(resume=not args.no_resume, max_experiments=args.max)
