"""Main experiment orchestrator: runs all pending experiments via subprocess."""

import logging
import subprocess
import sys
from pathlib import Path

from src.utils.filesystem_manager import get_experiments_by_status, print_status_summary

OPTIMIZATION_MODULE = 'src.experiments.run_experiment'
HEURISTIC_MODULE = 'src.experiments.run_heuristic'

log = logging.getLogger(__name__)


def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(name)s %(levelname)s %(message)s')

    print_status_summary('results')
    pending = get_experiments_by_status('results')['pending']
    if not pending:
        log.info("No pending experiments")
        return

    log.info("Running %d pending experiments", len(pending))
    completed, failed = 0, 0

    for i, (exp_path, config) in enumerate(pending, 1):
        config_path = Path(exp_path) / 'config.json'
        methodology = config.get('methodology', 'our_approach')
        runner = HEURISTIC_MODULE if methodology == 'heuristic' else OPTIMIZATION_MODULE

        log.info("[%d/%d] %s (%s)", i, len(pending), exp_path, methodology)
        try:
            result = subprocess.run(
                [sys.executable, '-m', runner, str(config_path)],
                capture_output=True, text=True)
            if result.returncode == 0:
                completed += 1
            else:
                failed += 1
                if result.stderr:
                    log.error("Error: %s", result.stderr[:500])
        except KeyboardInterrupt:
            log.warning("Interrupted. completed=%d failed=%d remaining=%d",
                        completed, failed, len(pending) - i)
            break

    log.info("Done. completed=%d failed=%d", completed, failed)
    print_status_summary('results')


if __name__ == "__main__":
    main()
