# Single experiment runner: train model with constraint optimization and evaluate.
# Handles warmup, constraint phase, post-hoc adjustment, and metric export.

import argparse
import logging
import time
from pathlib import Path
from typing import Dict, Any, Optional

import numpy as np
import torch

from src.utils.data_loader import load_experiment_data
from src.utils.error_handler import logger, log_exception
from src.utils.posthoc_adjustment import (
    apply_posthoc_adjustment, compute_constraint_delta, enforce_local_constraints,
    adjust_predictions_to_constraint
)
from src.training.trainer import ConstraintTrainer
from src.training.metrics import get_predictions_with_probabilities, compute_metrics
from src.training.logging import save_final_predictions, save_evaluation_metrics
from src.utils.filesystem_manager import load_config_from_path, save_config_to_path, update_experiment_status

log = logging.getLogger(__name__)


def _to_numpy(arr):
    return arr.values if hasattr(arr, 'values') else arr


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
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        log.info("Set random seed: %d", seed)
    log.info("Running %s on %s (model=%s)", config_path, device, config['model_name'])
    data = load_experiment_data(config)
    X_train, X_test, y_train, y_test, groups_test, global_con, local_con, num_classes = data
    ds_config = config.get('dataset_config', {})
    constrained_class = ds_config.get('constrained_class', num_classes - 1)
    if isinstance(constrained_class, (list, tuple)):
        constrained_classes = list(constrained_class)
    else:
        constrained_classes = [constrained_class]
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.LongTensor(_to_numpy(y_train))
    X_test_tensor = torch.FloatTensor(X_test).to(device)
    input_dim = None
    trainer = ConstraintTrainer(config, str(experiment_path), device, num_classes=num_classes)
    trainer.setup_model(input_dim=input_dim, base_model_id=config['base_model_id'])
    start_time = time.time()
    actual_warmup = trainer.train_warmup(X_train_tensor, y_train_tensor, config['base_model_id'])
    model = trainer.train_constraints(
        X_train=X_train_tensor, y_train=y_train_tensor,
        X_test=X_test_tensor, groups_test=groups_test,
        global_con=global_con, local_con=local_con,
        actual_warmup_epochs=actual_warmup)
    training_time = time.time() - start_time
    model.eval()
    y_pred, y_proba = get_predictions_with_probabilities(model, X_test_tensor)
    y_true = _to_numpy(y_test)
    group_ids = _to_numpy(groups_test)
    adjustment_info = {'adjustment_type': 'none', 'samples_adjusted': 0}
    needs_adjustment = any(global_con[c] < 1e9 for c in constrained_classes)
    if needs_adjustment:
        for cc in constrained_classes:
            constraint_limit = int(global_con[cc]) if global_con[cc] < 1e9 else None
            if constraint_limit is None:
                continue
            delta = compute_constraint_delta(y_pred, constraint_limit, cc)
            if delta != 0:
                log.info("Post-hoc: class %d count=%d limit=%d delta=%d",
                         cc, (y_pred == cc).sum(), constraint_limit, delta)
                y_pred, info = adjust_predictions_to_constraint(y_pred, y_proba, constraint_limit, cc)
                adjustment_info['samples_adjusted'] += info.get('samples_adjusted', 0)
                adjustment_info['adjustment_type'] = info.get('adjustment_type', 'none')
    if local_con and needs_adjustment:
        for cc in constrained_classes:
            y_pred, _ = enforce_local_constraints(y_pred, y_proba, group_ids, local_con, cc)
        for cc in constrained_classes:
            constraint_limit = int(global_con[cc]) if global_con[cc] < 1e9 else None
            if constraint_limit is None:
                continue
            delta = compute_constraint_delta(y_pred, constraint_limit, cc)
            if delta > 0:
                log.info("Post-hoc: re-fixing global class %d after local adjustment (delta=%d)", cc, delta)
                y_pred, info = adjust_predictions_to_constraint(y_pred, y_proba, constraint_limit, cc)
                adjustment_info['samples_adjusted'] += info.get('samples_adjusted', 0)
    for c in range(num_classes):
        pred_count = (y_pred == c).sum()
        limit = int(global_con[c]) if global_con[c] < 1e9 else 'INF'
        status = 'OK' if (isinstance(limit, str) or pred_count <= limit) else f'VIOLATED by {pred_count - limit}'
        log.info("Class %d: pred=%d limit=%s %s", c, pred_count, limit, status)
    save_final_predictions(experiment_path / 'final_predictions.csv', y_true, y_pred, y_proba, group_ids)
    metrics = compute_metrics(y_true, y_pred, y_proba)
    save_evaluation_metrics(experiment_path / 'evaluation_metrics.csv', metrics)
    config['results'] = {
        'accuracy': float(metrics['accuracy']),
        'precision_macro': float(metrics['precision_macro']),
        'recall_macro': float(metrics['recall_macro']),
        'f1_macro': float(metrics['f1_macro']),
        'training_time': float(training_time),
        'used_cached_model': trainer.from_cache,
        'post_hoc_adjustment': adjustment_info.get('adjustment_type', 'none'),
        'samples_adjusted': adjustment_info.get('samples_adjusted', 0)
    }
    config['status'] = 'completed'
    save_config_to_path(config, experiment_path)
    log.info("Done: accuracy=%.4f time=%.2fs path=%s", metrics['accuracy'], training_time, experiment_path)
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
