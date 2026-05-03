# Single experiment runner: train model with constraint optimization and evaluate.
# Handles warmup, constraint phase, post-hoc adjustment, and metric export.

import argparse
import logging
import time
from pathlib import Path
from typing import Dict, Any, Optional

import numpy as np
import torch

from src.pipeline.data import load_data
from src.utils.error_handler import logger, log_exception
from src.utils.posthoc_adjustment import targeted_correction
from src.training.trainer import ConstraintTrainer
from src.pipeline.warmup import run_warmup
from src.training.metrics import get_predictions_with_probabilities, compute_metrics
from src.training.logging import save_final_predictions, save_evaluation_metrics
from src.utils.filesystem_manager import load_config_from_path, update_experiment_status
from src.pipeline.setup import seed_all
from src.pipeline.io import save_results_to_config
from src.utils.constants import UNLIMITED

log = logging.getLogger(__name__)


@logger()
def run_experiment(config_path: str) -> Optional[Dict[str, Any]]:
    experiment_path = Path(config_path).parent
    config = load_config_from_path(experiment_path)
    if config.get('status', 'pending') == 'completed':
        log.info("Skipping completed: %s", experiment_path)
        return None
    update_experiment_status(experiment_path, 'running')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    seed = config.get('hyperparams', {}).get('seed', None)
    seed_all(seed)
    log.info("Running %s on %s (model=%s)", config_path, device, config['model_name'])
    if torch.cuda.is_available():
        log.info("GPU: %s | CUDA: %s | BF16: %s",
                 torch.cuda.get_device_name(0), torch.version.cuda,
                 torch.cuda.is_bf16_supported())
    data = load_data(config)
    X_train_tensor = data.X_train
    y_train_tensor = data.y_train
    X_test_tensor = data.X_test.to(device)
    groups_test = data.groups_test
    global_con = data.global_con
    local_con = data.local_con
    num_classes = data.num_classes
    constrained_classes = data.constrained_classes
    trainer = ConstraintTrainer(config, str(experiment_path), device, num_classes=num_classes)
    warmup_start = time.time()
    trainer.model, trainer.from_cache = run_warmup(
        config, num_classes, X_train_tensor, y_train_tensor, device,
        csv_log_path=str(trainer.csv_log_path),
    )
    actual_warmup = config['hyperparams']['warmup_epochs']
    warmup_time = time.time() - warmup_start
    log.info("TIMING warmup=%.2fs (%d epochs, cached=%s)", warmup_time, actual_warmup, trainer.from_cache)
    constraint_start = time.time()
    model = trainer.train_constraints(
        X_train=X_train_tensor, y_train=y_train_tensor,
        X_test=X_test_tensor, groups_test=groups_test,
        global_con=global_con, local_con=local_con,
        actual_warmup_epochs=actual_warmup)
    constraint_train_time = time.time() - constraint_start
    training_time = warmup_time + constraint_train_time
    y_true = data.y_test
    group_ids = groups_test
    needs_adjustment = any(global_con[c] < UNLIMITED for c in constrained_classes)
    posthoc_start = time.time()

    def _eval_and_correct(the_model, label='final'):
        the_model.eval()
        y_pred, y_proba = get_predictions_with_probabilities(the_model, X_test_tensor)
        adj = 0
        posthoc_meta = {}
        if needs_adjustment:
            y_pred, adj, posthoc_meta = targeted_correction(
                y_proba, group_ids, global_con, local_con, constrained_classes)
        metrics = compute_metrics(y_true, y_pred, y_proba)
        log.info("[%s] acc=%.4f f1=%.4f adjusted=%d", label, metrics['accuracy'], metrics['f1_macro'], adj)
        return y_pred, y_proba, metrics, adj, posthoc_meta

    # Evaluate final-epoch model. (Bracket-best/F1-on-test selection removed
    # to close audit S2 -- test-set leakage via checkpoint pick.)
    best_pred, best_proba, best_metrics, best_adj, best_meta = _eval_and_correct(model, 'final')
    best_source = 'final'

    for c in range(num_classes):
        pred_count = (best_pred == c).sum()
        limit = int(global_con[c]) if global_con[c] < UNLIMITED else 'INF'
        status = 'OK' if (isinstance(limit, str) or pred_count <= limit) else f'VIOLATED by {pred_count - limit}'
        log.info("Class %d: pred=%d limit=%s %s", c, pred_count, limit, status)
    save_final_predictions(experiment_path / 'final_predictions.csv', y_true, best_pred, best_proba, group_ids)
    # Also save the pre-post-hoc raw argmax of the same probabilities. Same
    # y_true, same y_proba, same group_ids -- only the Predicted_Label column
    # differs. This lets downstream analysis inspect the constraint-trained
    # model's direct output without the targeted_correction flips, so we can
    # test whether the post-hoc step is driving all methods toward the same
    # feasible-vertex saturation. Zero training overhead (reuses best_proba).
    raw_best_pred = best_proba.argmax(axis=1)
    save_final_predictions(experiment_path / 'final_predictions_raw.csv',
                           y_true, raw_best_pred, best_proba, group_ids)
    # Track 1: constraint-specific metrics
    from src.training.metrics import compute_flips, compute_raw_constraint_satisfaction
    flips = compute_flips(raw_best_pred, best_pred)
    raw_sat = compute_raw_constraint_satisfaction(
        raw_best_pred, global_con, local_con, group_ids, constrained_classes)
    best_metrics['flips_required'] = flips
    best_metrics.update(raw_sat)
    best_metrics['satisfaction_epoch'] = getattr(trainer, 'satisfaction_epoch', None)
    best_metrics['soft_hard_gap'] = getattr(trainer, 'final_soft_hard_gap', {})
    log.info("[Track1] flips=%d raw_satisfied=%s excess=%d sat_epoch=%s",
             flips, raw_sat['raw_all_satisfied'], raw_sat['raw_total_excess'],
             best_metrics['satisfaction_epoch'] or 'N/A')
    posthoc_time = time.time() - posthoc_start
    best_metrics['warmup_time'] = float(warmup_time)
    best_metrics['constraint_train_time'] = float(constraint_train_time)
    best_metrics['posthoc_time'] = float(posthoc_time)
    save_evaluation_metrics(experiment_path / 'evaluation_metrics.csv', best_metrics)
    save_results_to_config(config, experiment_path, {
        'accuracy': float(best_metrics['accuracy']),
        'precision_macro': float(best_metrics['precision_macro']),
        'recall_macro': float(best_metrics['recall_macro']),
        'f1_macro': float(best_metrics['f1_macro']),
        'training_time': float(training_time),
        'warmup_time': float(warmup_time),
        'constraint_train_time': float(constraint_train_time),
        'posthoc_time': float(posthoc_time),
        'used_cached_model': trainer.from_cache,
        'samples_adjusted': int(best_adj),
        'lp_fallback_used': best_meta.get('lp_fallback_used', False),
        'lp_fallback_candidates': best_meta.get('lp_fallback_candidates', 0),
    })
    log.info("Done: accuracy=%.4f source=%s time=%.2fs path=%s",
             best_metrics['accuracy'], best_source, training_time, experiment_path)
    return config['results']


def main() -> None:
    parser = argparse.ArgumentParser(description='Run single experiment')
    parser.add_argument('config_path', type=str, help='Path to config.json')
    args = parser.parse_args()
    experiment_path = Path(args.config_path).parent
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(name)s %(levelname)s %(message)s')
    try:
        run_experiment(args.config_path)
    except Exception as e:
        log_exception(e, context=f"Experiment: {experiment_path}")
        update_experiment_status(str(experiment_path), 'pending')
        exit(1)


if __name__ == "__main__":
    main()
