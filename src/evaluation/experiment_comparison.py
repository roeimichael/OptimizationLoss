# Cross-experiment data collection and comparison.
# Collects results from all completed experiments into a single DataFrame.
# Handles multi-class constraints and variable number of classes.

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.training.metrics import compute_metrics

log = logging.getLogger(__name__)

DERMMNIST_CLASS_NAMES = {
    0: 'AKIEC', 1: 'BCC', 2: 'BKL', 3: 'DF', 4: 'MEL', 5: 'NV', 6: 'VASC',
}


def load_predictions(experiment_path):
    df = pd.read_csv(Path(experiment_path) / 'final_predictions.csv')
    y_true = df['True_Label'].values
    y_pred = df['Predicted_Label'].values
    prob_cols = [c for c in df.columns if c.startswith('Prob_Class_')]
    y_proba = df[prob_cols].values if prob_cols else None
    group_ids = df['Group_ID'].values if 'Group_ID' in df.columns else None
    return y_true, y_pred, y_proba, group_ids


def collect_all_experiments(results_dir='results'):
    results_path = Path(results_dir)
    records = []

    for config_path in sorted(results_path.rglob('config.json')):
        exp_dir = config_path.parent
        pred_path = exp_dir / 'final_predictions.csv'
        if not pred_path.exists():
            continue

        try:
            with open(config_path) as f:
                cfg = json.load(f)
        except (json.JSONDecodeError, ValueError):
            continue
        if cfg.get('status') != 'completed':
            continue

        method = cfg.get('methodology', 'unknown')
        hp = cfg.get('hyperparams', {})
        res = cfg.get('results', {})
        ds = cfg.get('dataset_config', {})
        num_classes = ds.get('num_classes', 7)

        y_true, y_pred, y_proba, group_ids = load_predictions(exp_dir)
        metrics = compute_metrics(y_true, y_pred, y_proba)

        constrained_class = ds.get('constrained_class', 4)
        if isinstance(constrained_class, int):
            ccs = [constrained_class]
        elif isinstance(constrained_class, list):
            ccs = constrained_class
        else:
            ccs = []

        total_constrained_pred = 0
        total_constrained_tp = 0
        for cc in ccs:
            total_constrained_pred += int((y_pred == cc).sum())
            total_constrained_tp += int(((y_pred == cc) & (y_true == cc)).sum())
        total_constrained_fp = total_constrained_pred - total_constrained_tp

        record = {
            'method': method,
            'model_name': cfg.get('model_name', 'unknown'),
            'scenario': str(exp_dir.relative_to(results_path)).split('\\')[0].split('/')[0],
            'constraint': str(cfg.get('constraint', [])),
            'constraint_tag': cfg.get('constraint_tag', ''),
            'name': cfg.get('exp_name', exp_dir.name),
            'path': str(exp_dir),
            'constrained_classes': str(ccs),
            'accuracy': metrics['accuracy'],
            'precision_macro': metrics['precision_macro'],
            'recall_macro': metrics['recall_macro'],
            'f1_macro': metrics['f1_macro'],
            'f1_weighted': metrics.get('f1_weighted', 0),
            'constrained_pred': total_constrained_pred,
            'constrained_tp': total_constrained_tp,
            'constrained_fp': total_constrained_fp,
            'ece': metrics.get('ece'),
            'brier_score': metrics.get('brier_score'),
            'training_time': res.get('training_time', 0),
            'samples_adjusted': res.get('samples_adjusted', 0),
        }
        for c in range(num_classes):
            record[f'prec_c{c}'] = metrics['precision_per_class'][c]
            record[f'rec_c{c}'] = metrics['recall_per_class'][c]
            record[f'f1_c{c}'] = metrics['f1_per_class'][c]
            record[f'support_c{c}'] = int(metrics['support_per_class'][c])
        records.append(record)

    return pd.DataFrame(records)


def generate_comparison_charts(results_dir='results'):
    output_dir = Path(results_dir) / 'figures' / 'comparison'
    output_dir.mkdir(parents=True, exist_ok=True)

    df = collect_all_experiments(results_dir)
    if len(df) == 0:
        log.warning("No completed experiments found in %s", results_dir)
        return df

    n_opt = len(df[df['method'] == 'our_approach'])
    n_heu = len(df[df['method'] == 'heuristic'])
    log.info("Found %d experiments (%d optimization, %d heuristic)", len(df), n_opt, n_heu)

    df.to_csv(Path(results_dir) / 'all_metrics.csv', index=False)
    log.info("Saved: %s/all_metrics.csv", results_dir)

    return df
