"""Regenerate all analysis: metrics, training curves, comparison charts.

Usage:
    python -m src.evaluation.generate_all [results_dir]
"""

import logging
import sys
from pathlib import Path

from src.evaluation.training_curves import generate_training_curves
from src.evaluation.experiment_comparison import (
    generate_comparison_charts, collect_all_experiments, load_predictions
)
from src.training.metrics import compute_metrics
from src.training.logging import save_evaluation_metrics

log = logging.getLogger(__name__)


def recompute_all_metrics(results_dir='results'):
    """Recompute evaluation_metrics.csv for all completed experiments with ECE/uncertainty."""
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

    log.info("=" * 60)
    log.info("Analysis Pipeline — DermaMNIST-C Constraint Optimization")
    log.info("=" * 60)

    log.info("Step 1/3: Recompute evaluation metrics (with ECE/calibration)")
    recompute_all_metrics(results_dir)

    log.info("Step 2/3: Generate training curves (per-experiment + overlays)")
    generate_training_curves(results_dir)

    log.info("Step 3/3: Generate comparison charts")
    df = generate_comparison_charts(results_dir)

    # Summary
    figures_dir = Path(results_dir) / 'figures'
    if figures_dir.exists():
        all_figs = list(figures_dir.rglob('*.png'))
        log.info("=" * 60)
        log.info("DONE: %d figures generated in %s/", len(all_figs), figures_dir)
        for d in sorted(set(f.parent for f in all_figs)):
            count = len(list(d.glob('*.png')))
            log.info("  %s/ (%d figures)", d.relative_to(Path(results_dir)), count)
        log.info("  Metrics CSV: %s/all_metrics.csv", results_dir)
        log.info("=" * 60)


if __name__ == '__main__':
    main()
