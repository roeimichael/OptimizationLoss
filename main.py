"""
Batch Experiment Runner
Iterates through all experiment configurations and executes them using run_experiment.py
"""
import subprocess
import sys
from pathlib import Path

from utils.filesystem_manager import get_all_experiment_configs, is_experiment_complete


def run_all_experiments(resume=True, max_experiments=None):
    """
    Run all pending experiments

    Args:
        resume: If True, skip already completed experiments
        max_experiments: Maximum number of experiments to run (None = all)
    """
    print("="*80)
    print("BATCH EXPERIMENT RUNNER")
    print("="*80)
    print()

    # Get all experiment configurations
    print("Scanning for experiments...")
    experiments = get_all_experiment_configs('results')

    if not experiments:
        print("No experiments found!")
        print("Run 'python utils/generate_configs.py' first to create experiment configurations.")
        return

    print(f"Found {len(experiments)} total experiments")

    # Filter completed experiments if resuming
    if resume:
        pending_experiments = [
            (path, config) for path, config in experiments
            if not is_experiment_complete(path)
        ]
        print(f"Pending experiments: {len(pending_experiments)}")
    else:
        pending_experiments = experiments
        print("Running all experiments (including completed ones)")

    # Limit if max_experiments specified
    if max_experiments is not None:
        pending_experiments = pending_experiments[:max_experiments]
        print(f"Limiting to first {max_experiments} experiments")

    print()

    # Run experiments
    completed = 0
    failed = 0
    skipped = 0

    for i, (experiment_path, config) in enumerate(pending_experiments, 1):
        config_path = Path(experiment_path) / 'config.json'

        print(f"\n{'='*80}")
        print(f"[{i}/{len(pending_experiments)}] Processing: {experiment_path}")
        print(f"{'='*80}")

        try:
            # Call run_experiment.py with the config path
            result = subprocess.run(
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

    # Summary
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
    import argparse

    parser = argparse.ArgumentParser(description='Run all experiment configurations')
    parser.add_argument('--no-resume', action='store_true',
                      help='Re-run all experiments (including completed ones)')
    parser.add_argument('--max', type=int, default=None,
                      help='Maximum number of experiments to run')

    args = parser.parse_args()

    run_all_experiments(resume=not args.no_resume, max_experiments=args.max)

