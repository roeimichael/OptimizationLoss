"""Regenerate all analysis: metrics, training curves, comparison charts."""

import logging
import sys
from pathlib import Path

from src.analysis.training_curves import generate_training_curves
from src.analysis.experiment_comparison import generate_comparison_charts, load_predictions
from src.training.metrics import compute_metrics
from src.training.logging import save_evaluation_metrics

log = logging.getLogger(__name__)


def recompute_all_metrics(results_dir='results'):
    """Recompute evaluation_metrics.csv for all experiments with ECE/uncertainty."""
    results_path = Path(results_dir)
    count = 0
    for method in ['our_approach', 'heuristic']:
        method_dir = results_path / 'binary' / method / 'FTTransformer'
        if not method_dir.exists():
            continue
        for exp_path in sorted(method_dir.glob('constraint_*/standard/default')):
            pred_file = exp_path / 'final_predictions.csv'
            if not pred_file.exists():
                continue
            y_true, y_pred, y_proba, _ = load_predictions(exp_path)
            metrics = compute_metrics(y_true, y_pred, y_proba)
            save_evaluation_metrics(exp_path / 'evaluation_metrics.csv', metrics)
            count += 1
    log.info("Recomputed metrics for %d experiments", count)


def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(name)s %(levelname)s %(message)s')
    results_dir = sys.argv[1] if len(sys.argv) > 1 else 'results'

    log.info("Step 1: Recompute evaluation metrics")
    recompute_all_metrics(results_dir)

    log.info("Step 2: Generate training curves")
    generate_training_curves(results_dir)

    log.info("Step 3: Generate comparison charts")
    generate_comparison_charts(results_dir)

    figures_dir = Path(results_dir) / 'figures'
    if figures_dir.exists():
        all_figs = list(figures_dir.rglob('*.png'))
        log.info("Total figures generated: %d", len(all_figs))


if __name__ == '__main__':
    main()
