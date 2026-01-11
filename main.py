"""Main experiment runner with methodology filtering.

This script runs pending experiments from the results directory, filtering by
methodology type and using the appropriate experiment runner for each.

Supported methodologies:
- 'our_approach': Adaptive lambda methodology (uses run_experiment.py)
- 'static_lambda': Static lambda methodology (uses run_static_lambda_experiment.py)
"""

import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Any

from src.utils.filesystem_manager import get_experiments_by_status, print_status_summary


# ============================================================================
# METHODOLOGY CONFIGURATION
# ============================================================================

# Map each methodology to its corresponding experiment runner script
METHODOLOGY_RUNNERS: Dict[str, str] = {
    'our_approach': 'run_experiment.py',
    'static_lambda': 'run_static_lambda_experiment.py',
}

# Specify which methodologies to run
# Change this list to control which experiments are executed
ACTIVE_METHODOLOGIES: List[str] = [
    'static_lambda',  # Only run static lambda experiments
    # 'our_approach',  # Uncomment to also run adaptive lambda experiments
]


# ============================================================================
# EXPERIMENT FILTERING AND EXECUTION
# ============================================================================

def filter_experiments_by_methodology(
    experiments: List[Tuple[str, Dict[str, Any]]],
    methodologies: List[str]
) -> List[Tuple[str, Dict[str, Any]]]:
    """Filter experiments by methodology type.

    Args:
        experiments: List of (experiment_path, config) tuples
        methodologies: List of methodology names to include

    Returns:
        Filtered list of experiments
    """
    filtered = []
    for experiment_path, config in experiments:
        methodology = config.get('methodology', 'our_approach')
        if methodology in methodologies:
            filtered.append((experiment_path, config))
    return filtered


def main() -> None:
    """Main experiment runner with methodology filtering."""
    print("=" * 80)
    print("EXPERIMENT RUNNER")
    print("=" * 80)
    print(f"Active Methodologies: {', '.join(ACTIVE_METHODOLOGIES)}")
    print("=" * 80)
    print()

    # Get status summary and pending experiments
    print_status_summary('results')
    by_status = get_experiments_by_status('results')
    all_pending = by_status['pending']

    if not all_pending:
        print("✓ All experiments are completed or there are no configurations.")
        return

    # Filter by active methodologies
    pending_experiments = filter_experiments_by_methodology(
        all_pending,
        ACTIVE_METHODOLOGIES
    )

    if not pending_experiments:
        print(f"✓ No pending experiments found for methodologies: {ACTIVE_METHODOLOGIES}")
        print(f"  Total pending experiments (all methodologies): {len(all_pending)}")
        return

    print(f"\nFound {len(pending_experiments)} pending experiments "
          f"(filtered from {len(all_pending)} total)")
    print(f"Running experiments for: {', '.join(ACTIVE_METHODOLOGIES)}\n")

    completed = 0
    failed = 0
    skipped = 0

    for i, (experiment_path, config) in enumerate(pending_experiments, 1):
        config_path = Path(experiment_path) / 'config.json'
        methodology = config.get('methodology', 'our_approach')

        # Validate paths
        if not config_path.exists():
            print(f"[ERROR] Config file not found: {config_path}")
            failed += 1
            continue

        if not Path(experiment_path).exists():
            print(f"[ERROR] Experiment folder not found: {experiment_path}")
            failed += 1
            continue

        # Get appropriate runner script for this methodology
        runner_script = METHODOLOGY_RUNNERS.get(methodology)
        if not runner_script:
            print(f"[ERROR] Unknown methodology: {methodology}")
            skipped += 1
            continue

        # Run experiment
        print(f"\n{'=' * 80}")
        print(f"[{i}/{len(pending_experiments)}] Running experiment")
        print(f"Methodology: {methodology}")
        print(f"Runner: {runner_script}")
        print(f"Path: {experiment_path}")
        print(f"{'=' * 80}")

        try:
            result = subprocess.run(
                [sys.executable, runner_script, str(config_path)],
                capture_output=False,
                text=True
            )

            if result.returncode == 0:
                completed += 1
            else:
                failed += 1

        except KeyboardInterrupt:
            print("\n\n" + "=" * 80)
            print("INTERRUPTED BY USER")
            print("=" * 80)
            print(f"Completed: {completed}")
            print(f"Failed: {failed}")
            print(f"Remaining: {len(pending_experiments) - i}")
            print("=" * 80)
            print("\nResume by running 'python main.py' again.")
            break

        except Exception as e:
            failed += 1
            print(f"[ERROR] Failed to run experiment: {e}")

    else:
        # Loop completed without interruption
        print("\n" + "=" * 80)
        print("EXPERIMENT RUN COMPLETE")
        print("=" * 80)
        print(f"✓ Successfully completed: {completed}")
        print(f"✗ Failed: {failed}")
        if skipped > 0:
            print(f"○ Skipped: {skipped}")
        print("=" * 80)
        print()
        print_status_summary('results')


if __name__ == "__main__":
    main()
