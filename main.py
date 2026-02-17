"""Main experiment orchestrator - runs all pending experiments."""

import subprocess
import sys
from pathlib import Path

from src.utils.filesystem_manager import get_experiments_by_status, print_status_summary

# Route to correct runner based on methodology (use module path for proper imports)
OPTIMIZATION_MODULE = 'src.experiments.run_experiment'
HEURISTIC_MODULE = 'src.experiments.run_heuristic'


def main():
    print_status_summary('results')
    pending = get_experiments_by_status('results')['pending']
    if not pending:
        print("No pending experiments found")
        return
    print(f"Running {len(pending)} pending experiments\n")
    completed, failed = 0, 0

    for i, (exp_path, config) in enumerate(pending, 1):
        config_path = Path(exp_path) / 'config.json'

        # Select runner based on methodology
        methodology = config.get('methodology', 'our_approach')
        if methodology == 'heuristic':
            runner_module = HEURISTIC_MODULE
        else:
            runner_module = OPTIMIZATION_MODULE

        print(f"\n[{i}/{len(pending)}] {exp_path} ({methodology})")
        try:
            result = subprocess.run(
                [sys.executable, '-m', runner_module, str(config_path)],
                capture_output=True,
                text=True
            )
            if result.returncode == 0:
                completed += 1
            else:
                failed += 1
                if result.stderr:
                    print(f"[ERROR] {result.stderr[:500]}")

        except KeyboardInterrupt:
            print(f"\nInterrupted. Completed={completed}, Failed={failed}, Remaining={len(pending) - i}")
            break

    print(f"\nDone. Completed={completed}, Failed={failed}")
    print_status_summary('results')


if __name__ == "__main__":
    main()
