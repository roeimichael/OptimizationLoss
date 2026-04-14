# Regenerate all analysis: metrics and aggregated CSV.
# Usage: python -m src.evaluation.generate_all [results_dir]

import logging
import sys
from pathlib import Path

from src.evaluation.experiment_comparison import (
    generate_comparison_charts, load_predictions
)
from src.training.metrics import compute_metrics
from src.training.logging import save_evaluation_metrics

log = logging.getLogger(__name__)


def recompute_all_metrics(results_dir='results'):
    results_path = Path(results_dir)
    count = 0
    for config_file in sorted(results_path.rglob('config.json')):
        exp_dir = config_file.parent
        pred_file = exp_dir / 'final_predictions.csv'
        if not pred_file.exists():
            continue
        y_true, y_pred, y_proba, _ = load_predictions(exp_dir)
        metrics = compute_metrics(y_true, y_pred, y_proba)
        save_evaluation_metrics(exp_dir / 'evaluation_metrics.csv', metrics)
        count += 1
    log.info("Recomputed metrics for %d experiments", count)


def main():
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(name)s %(levelname)s %(message)s')
    results_dir = sys.argv[1] if len(sys.argv) > 1 else 'results'

    log.info("Step 1/2: Recompute evaluation metrics")
    recompute_all_metrics(results_dir)

    log.info("Step 2/2: Collect all experiments to CSV")
    df = generate_comparison_charts(results_dir)
    if df is not None and len(df) > 0:
        log.info("DONE: %d experiments collected to %s/all_metrics.csv", len(df), results_dir)


if __name__ == '__main__':
    main()
