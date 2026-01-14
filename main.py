import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Any

from src.utils.filesystem_manager import get_experiments_by_status, print_status_summary

METHODOLOGY_RUNNERS: Dict[str, str] = {
    'our_approach': 'src/experiments/run_experiment.py',
}

ACTIVE_METHODOLOGIES: List[str] = ['our_approach']


def filter_experiments_by_methodology(
    experiments: List[Tuple[str, Dict[str, Any]]],
    methodologies: List[str]
) -> List[Tuple[str, Dict[str, Any]]]:
    filtered = []
    for experiment_path, config in experiments:
        methodology = config.get('methodology', 'our_approach')
        if methodology in methodologies:
            filtered.append((experiment_path, config))
    return filtered


def main() -> None:
    print(f"Active Methodologies: {', '.join(ACTIVE_METHODOLOGIES)}\n")
    print_status_summary('results')
    by_status = get_experiments_by_status('results')
    all_pending = by_status['pending']

    if not all_pending:
        print("All experiments completed or no configurations found")
        return

    pending_experiments = filter_experiments_by_methodology(all_pending, ACTIVE_METHODOLOGIES)

    if not pending_experiments:
        print(f"No pending experiments for: {ACTIVE_METHODOLOGIES}")
        print(f"Total pending (all methodologies): {len(all_pending)}")
        return

    print(f"Found {len(pending_experiments)} pending experiments\n")

    completed = 0
    failed = 0
    skipped = 0

    for i, (experiment_path, config) in enumerate(pending_experiments, 1):
        config_path = Path(experiment_path) / 'config.json'
        methodology = config.get('methodology', 'our_approach')

        if not config_path.exists():
            print(f"[ERROR] Config file not found: {config_path}")
            failed += 1
            continue

        if not Path(experiment_path).exists():
            print(f"[ERROR] Experiment folder not found: {experiment_path}")
            failed += 1
            continue

        runner_script = METHODOLOGY_RUNNERS.get(methodology)
        if not runner_script:
            print(f"[ERROR] Unknown methodology: {methodology}")
            skipped += 1
            continue

        print(f"\n[{i}/{len(pending_experiments)}] {methodology}: {experiment_path}")

        try:
            result = subprocess.run(
                [sys.executable, runner_script, str(config_path)],
                capture_output=True,
                text=True
            )

            if result.returncode == 0:
                completed += 1
            else:
                failed += 1
                if result.stderr:
                    print(f"[ERROR] {result.stderr}")

        except KeyboardInterrupt:
            print(f"\nInterrupted: Completed={completed} Failed={failed} Remaining={len(pending_experiments) - i}")
            print("Resume by running 'python main.py' again")
            break

        except Exception as e:
            failed += 1
            print(f"ERROR: {e}")

    else:
        print(f"\nRun complete: Completed={completed} Failed={failed} Skipped={skipped}")
        print_status_summary('results')


if __name__ == "__main__":
    main()
