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
from src.methodologies.our_approach.train import train as train_our_approach
from src.methodologies.fioretto_ldf.train import train as train_fioretto_ldf
from src.pipeline.contracts import TrainInputs
from src.pipeline.warmup import run_warmup
from src.pipeline.eval import evaluate_with_posthoc, write_evaluation_outputs
from src.training.logging import save_evaluation_metrics
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
    csv_log_path = experiment_path / 'training_log.csv'
    warmup_start = time.time()
    model, from_cache = run_warmup(
        config, num_classes, X_train_tensor, y_train_tensor, device,
        csv_log_path=str(csv_log_path),
    )
    warmup_time = time.time() - warmup_start
    log.info("TIMING warmup=%.2fs (%d epochs, cached=%s)",
             warmup_time, config['hyperparams']['warmup_epochs'], from_cache)
    constraint_start = time.time()
    train_inputs = TrainInputs(
        model=model,
        X_train=X_train_tensor, y_train=y_train_tensor,
        X_test=X_test_tensor, y_test=data.y_test,
        group_ids=groups_test,
        global_con=global_con, local_con=local_con,
        constrained_classes=constrained_classes,
        num_classes=num_classes,
        config=config, hyperparams=config['hyperparams'],
        device=device,
        experiment_path=experiment_path,
        csv_log_path=csv_log_path,
    )
    methodology = config.get('methodology', 'our_approach')
    if methodology == 'fioretto_ldf':
        train_outputs = train_fioretto_ldf(train_inputs)
    elif methodology == 'our_approach':
        train_outputs = train_our_approach(train_inputs)
    else:
        raise ValueError(f"Unknown methodology for run_experiment: {methodology!r}")
    model = train_outputs.model
    constraint_train_time = time.time() - constraint_start
    training_time = warmup_time + constraint_train_time
    y_true = data.y_test
    group_ids = groups_test
    posthoc_start = time.time()

    result = evaluate_with_posthoc(
        model, X_test_tensor, y_true, group_ids,
        global_con, local_con, constrained_classes, num_classes,
        label='final',
    )
    best_pred = result['y_pred']
    best_proba = result['y_proba']
    best_metrics = result['metrics']
    best_adj = result['adj']
    best_meta = result['posthoc_meta']
    best_source = 'final'

    write_evaluation_outputs(experiment_path, y_true, group_ids, result, num_classes, global_con)
    best_metrics['satisfaction_epoch'] = train_outputs.summary.get('satisfaction_epoch')
    best_metrics['soft_hard_gap'] = train_outputs.summary.get('soft_hard_gap', {})
    if 'checkpoint_source' in train_outputs.summary:
        best_metrics['checkpoint_source'] = train_outputs.summary['checkpoint_source']
    if 'results_comparison' in train_outputs.summary:
        config['results_comparison'] = train_outputs.summary['results_comparison']
    log.info("sat_epoch=%s", best_metrics['satisfaction_epoch'] or 'N/A')
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
        'used_cached_model': from_cache,
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
