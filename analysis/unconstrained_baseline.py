"""Evaluate unconstrained warmup models (no constraint enforcement).

For each (model, dataset, slice) combo, loads the cached warmup model,
runs inference on the test set, and computes accuracy/F1/precision/recall.
Saves results to analysis/output/unconstrained_baseline.csv and a LaTeX table.

Usage:
    python -m analysis.unconstrained_baseline
"""

import json
import logging
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from src.models import get_model
from src.training.model_cache import load_from_cache
from src.training.metrics import compute_metrics, get_predictions_with_probabilities
from src.utils.data_loader import load_experiment_data
from src.config_generators.generate_configs import (
    HYPERPARAMS, MODEL_NAMES, compute_base_model_id)

logging.basicConfig(level=logging.INFO, format='%(message)s')
log = logging.getLogger(__name__)

DATASETS = {
    'dermmnist': {
        'dataset_mode': 'dermmnist',
        'num_classes': 7,
        'target_column': 'label',
        'group_column': 'loc_group',
        'image_size': 224,
        'constrained_class': 4,  # arbitrary, not used for unconstrained
    },
    'tissuemnist': {
        'dataset_mode': 'tissuemnist',
        'num_classes': 8,
        'target_column': 'label',
        'group_column': 'synth_group',
        'image_size': 224,
        'constrained_class': 4,
    },
}

NUM_SLICES = 5


def run_unconstrained_baseline():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log.info("Device: %s", device)

    rows = []

    for ds_name, ds_info in DATASETS.items():
        num_classes = ds_info['num_classes']
        for model_name in MODEL_NAMES:
            for slice_idx in range(1, NUM_SLICES + 1):
                data_dir = f"data/{ds_name}/slice_{slice_idx}"
                if not Path(data_dir).exists():
                    log.warning("SKIP: %s not found", data_dir)
                    continue

                # Build a minimal config to load the cached warmup model
                config = {
                    'model_name': model_name,
                    'hyperparams': HYPERPARAMS.copy(),
                    'dataset_mode': ds_name,
                    'dataset_config': {
                        'data_dir': data_dir,
                        'target_column': ds_info['target_column'],
                        'group_column': ds_info['group_column'],
                        'num_classes': num_classes,
                        'image_size': ds_info['image_size'],
                        'constrained_class': ds_info['constrained_class'],
                    },
                    'methodology': 'our_approach',
                    'constraint': [0.5, 0.5],  # dummy, needed by load_experiment_data
                }
                base_model_id = compute_base_model_id(
                    model_name, HYPERPARAMS,
                    dataset_mode=ds_name, data_dir=data_dir)

                log.info("--- %s / %s / slice_%d (base_id=%s) ---",
                         ds_name, model_name, slice_idx, base_model_id)

                # Load data
                try:
                    data = load_experiment_data(config)
                    X_train, X_test, y_train, y_test, groups_test = (
                        data[0], data[1], data[2], data[3], data[4])
                except Exception as e:
                    log.error("Failed to load data: %s", e)
                    continue

                # Determine input_dim
                if len(X_test.shape) == 4:
                    input_dim = X_test.shape[1]
                else:
                    input_dim = X_test.shape[1]

                # Load cached warmup model
                model = load_from_cache(
                    base_model_id, config, input_dim, num_classes, device)
                if model is None:
                    log.warning("No cached model for %s, skipping", base_model_id)
                    continue

                # Run inference
                model.eval()
                X_test_t = torch.FloatTensor(X_test).to(device)
                with torch.no_grad():
                    logits = model(X_test_t)
                    proba = F.softmax(logits, dim=1).cpu().numpy()
                    preds = logits.argmax(dim=1).cpu().numpy()

                y_np = y_test if isinstance(y_test, np.ndarray) else y_test.numpy()

                metrics = compute_metrics(y_np, preds, y_proba=proba)

                row = {
                    'dataset': ds_name,
                    'model': model_name,
                    'slice': slice_idx,
                    'accuracy': metrics['accuracy'],
                    'f1_macro': metrics['f1_macro'],
                    'precision_macro': metrics['precision_macro'],
                    'recall_macro': metrics['recall_macro'],
                }
                rows.append(row)
                log.info("  acc=%.4f  f1=%.4f  prec=%.4f  rec=%.4f",
                         row['accuracy'], row['f1_macro'],
                         row['precision_macro'], row['recall_macro'])

    if not rows:
        log.error("No results generated!")
        return

    # Save CSV
    import csv
    import statistics

    output_dir = Path('analysis/output')
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / 'unconstrained_baseline.csv'
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    log.info("\nWrote: %s (%d rows)", csv_path, len(rows))

    # Print summary
    print("\n=== UNCONSTRAINED BASELINE SUMMARY ===")
    by_ds_model = defaultdict(list)
    for r in rows:
        by_ds_model[(r['dataset'], r['model'])].append(r)

    for (ds, model), rs in sorted(by_ds_model.items()):
        acc = statistics.mean([r['accuracy'] for r in rs])
        f1 = statistics.mean([r['f1_macro'] for r in rs])
        std = statistics.stdev([r['f1_macro'] for r in rs]) if len(rs) > 1 else 0
        print(f"  {ds:15s} {model:15s}  acc={acc:.4f}  f1={f1:.4f}±{std:.4f}  (n={len(rs)})")

    # LaTeX table
    latex_lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\small",
        r"\caption{Unconstrained baseline: warmup model performance with no "
        r"constraint enforcement (mean{\scriptsize$\pm$}std over 5 slices).}",
        r"\label{tab:unconstrained_baseline}",
        r"\begin{tabular}{l l c c c c}",
        r"\toprule",
        r"Dataset & Model & Accuracy & F1 Macro & Precision & Recall \\",
        r"\midrule",
    ]
    for ds in ['dermmnist', 'tissuemnist']:
        for model in MODEL_NAMES:
            rs = by_ds_model.get((ds, model), [])
            if not rs:
                continue
            acc = statistics.mean([r['accuracy'] for r in rs])
            f1 = statistics.mean([r['f1_macro'] for r in rs])
            f1_std = statistics.stdev([r['f1_macro'] for r in rs]) if len(rs) > 1 else 0
            prec = statistics.mean([r['precision_macro'] for r in rs])
            rec = statistics.mean([r['recall_macro'] for r in rs])
            model_short = {'MobileNetV3': 'MV3', 'EfficientNetB0': 'EffB0',
                           'ConvNeXtTiny': 'CNxT'}.get(model, model)
            f1_str = f"{f1:.3f}" + r"{\scriptsize$\pm$" + f"{f1_std:.3f}" + "}"
            latex_lines.append(
                f"{ds} & {model_short} & {acc:.3f} & {f1_str} "
                f"& {prec:.3f} & {rec:.3f} \\\\")
        latex_lines.append(r"\midrule")
    latex_lines[-1] = r"\bottomrule"
    latex_lines.extend([
        r"\end{tabular}",
        r"\end{table}",
    ])

    tex_path = output_dir / 'unconstrained_baseline.tex'
    tex_path.write_text("\n".join(latex_lines), encoding='utf-8')
    log.info("Wrote: %s", tex_path)


if __name__ == '__main__':
    run_unconstrained_baseline()
