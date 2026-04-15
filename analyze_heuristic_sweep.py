"""Sweep all models x constraint configs with heuristic to find where models struggle.

Uses the actual pipeline components (data loader, trainer, heuristic) to ensure
results match real experiments. For each model:
  1. Train warmup via ConstraintTrainer (or load from cache) — pretrained + fine-tuned
  2. Get probabilities on test set
  3. Apply heuristic across many constraint configurations
  4. Report per-class precision/recall to find optimization opportunities
"""

import logging
import hashlib
import json
import time

import numpy as np
import pandas as pd
import torch

from src.models import get_model
from src.training.trainer import ConstraintTrainer
from src.training.constraints import compute_global_constraints, compute_local_constraints
from src.experiments.run_heuristic import apply_allocation_heuristic, _build_hierarchy
from src.utils.inference import chunked_forward

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
log = logging.getLogger(__name__)

UNLIMITED = 1e10
MODELS = ['ResNet18', 'MobileNetV3', 'EfficientNetB0', 'ConvNeXtTiny']
CLASS_NAMES = ['AKIEC', 'BCC', 'BKL', 'DF', 'MEL', 'NV', 'VASC']
NUM_CLASSES = 7
DATA_DIR = 'data/dermmnist/slice_1'

HYPERPARAMS = {
    'lr': 0.0001, 'lr_constraint': 5e-6, 'dropout': 0.3, 'batch_size': 64,
    'warmup_epochs': 50, 'constraint_epochs': 300, 'lambda_global': 0.01,
    'lambda_local': 0.01, 'lambda_step': 0.002, 'use_sum_loss': True,
    'initial_rho': 5.0, 'rho_target': 100.0, 'alpha_kl': 0.5,
    'kl_temperature': 1.0, 'pretrained': True, 'class_weighted_ce': False,
    'constraint_chunk_size': 64,
}

CONSTRAINED_CLASS_SETS = {
    # All 7 single-class
    'single_AKIEC':        [0],
    'single_BCC':          [1],
    'single_BKL':          [2],
    'single_DF':           [3],
    'single_MEL':          [4],
    'single_NV':           [5],
    'single_VASC':         [6],
    # All 2-class combinations (21 pairs)
    'multi_AKIEC_BCC':     [0, 1],
    'multi_AKIEC_BKL':     [0, 2],
    'multi_AKIEC_DF':      [0, 3],
    'multi_AKIEC_MEL':     [0, 4],
    'multi_AKIEC_NV':      [0, 5],
    'multi_AKIEC_VASC':    [0, 6],
    'multi_BCC_BKL':       [1, 2],
    'multi_BCC_DF':        [1, 3],
    'multi_BCC_MEL':       [1, 4],
    'multi_BCC_NV':        [1, 5],
    'multi_BCC_VASC':      [1, 6],
    'multi_BKL_DF':        [2, 3],
    'multi_BKL_MEL':       [2, 4],
    'multi_BKL_NV':        [2, 5],
    'multi_BKL_VASC':      [2, 6],
    'multi_DF_MEL':        [3, 4],
    'multi_DF_NV':         [3, 5],
    'multi_DF_VASC':       [3, 6],
    'multi_MEL_NV':        [4, 5],
    'multi_MEL_VASC':      [4, 6],
    'multi_NV_VASC':       [5, 6],
    # 3-class combinations (selection of interesting ones)
    'multi_MEL_BCC_VASC':  [4, 1, 6],
    'multi_MEL_BKL_BCC':   [4, 2, 1],
    'multi_AKIEC_MEL_BCC': [0, 4, 1],
    'multi_AKIEC_BKL_MEL': [0, 2, 4],
    'multi_BCC_BKL_VASC':  [1, 2, 6],
    'multi_DF_MEL_VASC':   [3, 4, 6],
    'multi_AKIEC_DF_VASC': [0, 3, 6],
    # 4+ class combinations
    'multi_AKIEC_BCC_MEL_VASC':     [0, 1, 4, 6],
    'multi_AKIEC_BCC_BKL_MEL':      [0, 1, 2, 4],
    'multi_BCC_BKL_MEL_VASC':       [1, 2, 4, 6],
    'multi_all_except_NV':          [0, 1, 2, 3, 4, 6],
}

# Asymmetric (local_pct, global_pct) pairs — matching real experiment configs
CONSTRAINT_PAIRS = [
    (0.2, 0.2),
    (0.3, 0.3),
    (0.5, 0.5),
    (0.8, 0.8),
    (0.5, 0.3),   # global tighter
    (0.8, 0.3),   # global much tighter
    (0.8, 0.5),   # global tighter
    (0.3, 0.8),   # local tighter
    (0.3, 0.5),   # local tighter
]


def compute_base_model_id(model_name):
    key = {
        'model_name': model_name, 'lr': HYPERPARAMS['lr'],
        'dropout': HYPERPARAMS['dropout'], 'batch_size': HYPERPARAMS['batch_size'],
        'warmup_epochs': HYPERPARAMS['warmup_epochs'],
        'pretrained': HYPERPARAMS.get('pretrained', False),
        'class_weighted_ce': HYPERPARAMS.get('class_weighted_ce', False),
        'dataset_mode': 'dermmnist', 'data_dir': DATA_DIR,
    }
    h = hashlib.md5(json.dumps(key, sort_keys=True).encode()).hexdigest()[:12]
    return f"{model_name}_dermmnist_{h}"


def load_data():
    """Load data using the same normalization as the pipeline."""
    from src.utils.data_loader import _ensure_3channel, _apply_imagenet_normalization
    import os

    X_train = _ensure_3channel(np.load(os.path.join(DATA_DIR, 'train_images.npy')))
    y_train = np.load(os.path.join(DATA_DIR, 'train_labels.npy')).ravel()
    X_test = _ensure_3channel(np.load(os.path.join(DATA_DIR, 'test_images.npy')))
    y_test = np.load(os.path.join(DATA_DIR, 'test_labels.npy')).ravel()
    X_train = _apply_imagenet_normalization(X_train)
    X_test = _apply_imagenet_normalization(X_test)

    test_meta = pd.read_csv(os.path.join(DATA_DIR, 'test_meta.csv'))
    groups = test_meta['loc_group'].values.astype(np.int64)

    return X_train, y_train, X_test, y_test, groups


def train_or_load_warmup(model_name, X_train, y_train, device):
    """Use ConstraintTrainer to train warmup (identical to real experiments)."""
    base_id = compute_base_model_id(model_name)
    config = {
        'model_name': model_name,
        'hyperparams': HYPERPARAMS.copy(),
        'base_model_id': base_id,
        'dataset_mode': 'dermmnist',
    }

    import tempfile, os
    tmp_dir = tempfile.mkdtemp(prefix=f'sweep_{model_name}_')
    trainer = ConstraintTrainer(config, tmp_dir, device, num_classes=NUM_CLASSES)
    trainer.setup_model(input_dim=None, base_model_id=base_id)

    X_train_t = torch.FloatTensor(X_train)
    y_train_t = torch.LongTensor(y_train)
    actual_epochs = trainer.train_warmup(X_train_t, y_train_t, base_id)
    log.info("Warmup for %s: %d epochs (cached=%s)", model_name, actual_epochs, trainer.from_cache)

    return trainer.model


def get_probabilities(model, X_test, device):
    """Get softmax probabilities using chunked inference."""
    model.eval()
    X_tensor = torch.FloatTensor(X_test).to(device)
    with torch.no_grad():
        logits = chunked_forward(model, X_tensor)
        proba = torch.softmax(logits.float(), dim=1).cpu().numpy()
    return proba


def compute_constraints_for_scenario(y_test, groups, constrained_classes,
                                      local_pct, global_pct):
    """Compute constraints using the pipeline's constraint functions."""
    test_df = pd.DataFrame({'label': y_test, 'group': groups})
    global_con = compute_global_constraints(
        test_df, 'label', global_pct,
        constrained_class=constrained_classes, num_classes=NUM_CLASSES)
    local_con = compute_local_constraints(
        test_df, 'label', local_pct, 'group',
        constrained_class=constrained_classes, num_classes=NUM_CLASSES)
    return global_con, local_con


def per_class_metrics(y_true, y_pred):
    results = {}
    for c in range(NUM_CLASSES):
        tp = int(((y_pred == c) & (y_true == c)).sum())
        fp = int(((y_pred == c) & (y_true != c)).sum())
        fn = int(((y_pred != c) & (y_true == c)).sum())
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        results[c] = {'precision': prec, 'recall': rec, 'f1': f1,
                       'tp': tp, 'fp': fp, 'fn': fn,
                       'n_pred': int((y_pred == c).sum()),
                       'n_true': int((y_true == c).sum())}
    return results


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log.info("Device: %s", device)

    X_train, y_train, X_test, y_test, groups = load_data()
    log.info("Data loaded: train=%s test=%s groups=%d",
             X_train.shape, X_test.shape, len(np.unique(groups)))

    # Ground truth
    print("\n" + "=" * 80)
    print("GROUND TRUTH CLASS DISTRIBUTION")
    print("=" * 80)
    for c in range(NUM_CLASSES):
        n = int((y_test == c).sum())
        print(f"  {CLASS_NAMES[c]:>6} (class {c}): {n:>5} ({n / len(y_test) * 100:>5.1f}%)")
    print(f"  Total: {len(y_test)}")

    # === Phase 1: Train warmup for each model, get predictions ===
    model_proba = {}
    print("\n" + "=" * 80)
    print("WARMUP MODEL PERFORMANCE (unconstrained argmax)")
    print("=" * 80)

    for model_name in MODELS:
        t0 = time.time()
        model = train_or_load_warmup(model_name, X_train, y_train, device)
        proba = get_probabilities(model, X_test, device)
        model_proba[model_name] = proba
        argmax = np.argmax(proba, axis=1)
        acc = (argmax == y_test).mean()
        metrics = per_class_metrics(y_test, argmax)
        f1_macro = np.mean([metrics[c]['f1'] for c in range(NUM_CLASSES)])
        elapsed = time.time() - t0

        print(f"\n  {model_name} (acc={acc:.4f}, F1-macro={f1_macro:.4f}, time={elapsed:.1f}s)")
        print(f"    {'Class':>8} {'N_true':>7} {'N_pred':>7} {'Prec':>7} {'Rec':>7} {'F1':>7}")
        for c in range(NUM_CLASSES):
            m = metrics[c]
            print(f"    {CLASS_NAMES[c]:>8} {m['n_true']:>7} {m['n_pred']:>7} "
                  f"{m['precision']:>7.3f} {m['recall']:>7.3f} {m['f1']:>7.3f}")

        # Free GPU memory
        del model
        torch.cuda.empty_cache()

    # === Phase 2: Heuristic sweep ===
    print("\n" + "=" * 80)
    print("HEURISTIC SWEEP")
    print("=" * 80)

    all_results = []
    n_configs = len(CONSTRAINED_CLASS_SETS) * len(CONSTRAINT_PAIRS) * len(MODELS)
    log.info("Running %d configurations (%d scenarios x %d pairs x %d models)",
             n_configs, len(CONSTRAINED_CLASS_SETS), len(CONSTRAINT_PAIRS), len(MODELS))

    for scenario_name, constrained_classes in CONSTRAINED_CLASS_SETS.items():
        for local_pct, global_pct in CONSTRAINT_PAIRS:
            pair_tag = f"L{int(local_pct*100):02d}_G{int(global_pct*100):02d}"
            global_con, local_con = compute_constraints_for_scenario(
                y_test, groups, constrained_classes, local_pct, global_pct)

            # Skip if any limit is 0 (infeasible)
            skip = False
            for c in constrained_classes:
                if global_con[c] < 1:
                    skip = True
            if skip:
                continue

            limits_str = ", ".join(
                f"{CLASS_NAMES[c]}={int(global_con[c])}"
                for c in constrained_classes)
            print(f"\n--- {scenario_name} | {pair_tag} | limits: {limits_str} ---")

            header = f"{'Model':<16}"
            for c in constrained_classes:
                header += f" | {CLASS_NAMES[c]:>5} prec   rec    f1 pred/lim"
            header += f" | {'F1-mac':>7} {'Acc':>7}"
            print(header)

            for model_name in MODELS:
                proba = model_proba[model_name]
                hierarchy = _build_hierarchy(NUM_CLASSES, global_con, constrained_classes)
                y_pred, _ = apply_allocation_heuristic(
                    proba, groups, hierarchy, global_con, local_con, NUM_CLASSES)

                metrics = per_class_metrics(y_test, y_pred)
                f1_macro = np.mean([metrics[c]['f1'] for c in range(NUM_CLASSES)])
                acc = (y_pred == y_test).mean()

                row = f"{model_name:<16}"
                for c in constrained_classes:
                    m = metrics[c]
                    lim = int(global_con[c])
                    row += (f" | {m['precision']:>5.3f} {m['recall']:>5.3f} "
                            f"{m['f1']:>5.3f} {m['n_pred']:>4}/{lim:>4}")
                row += f" | {f1_macro:>7.4f} {acc:>7.4f}"
                print(row)

                for c in constrained_classes:
                    all_results.append({
                        'model': model_name, 'scenario': scenario_name,
                        'constraint_pair': pair_tag,
                        'local_pct': local_pct, 'global_pct': global_pct,
                        'class': c, 'class_name': CLASS_NAMES[c],
                        'precision': metrics[c]['precision'],
                        'recall': metrics[c]['recall'],
                        'f1': metrics[c]['f1'],
                        'n_pred': metrics[c]['n_pred'],
                        'limit': int(global_con[c]),
                        'n_true': metrics[c]['n_true'],
                        'f1_macro': f1_macro, 'accuracy': acc,
                    })

    # === Phase 3: Summary ===
    print("\n" + "=" * 80)
    print(f"SWEEP COMPLETE: {len(all_results)} data points")
    print("=" * 80)

    meaningful = [r for r in all_results if r['limit'] > 3]

    print("\n" + "=" * 80)
    print("WORST HEURISTIC CONSTRAINED-CLASS PRECISION (optimization opportunities)")
    print("Filtered to limit > 3")
    print("=" * 80)
    sorted_results = sorted(meaningful, key=lambda x: x['precision'])
    print(f"{'Model':<16} {'Scenario':<28} {'Pair':>7} {'Class':>6} "
          f"{'Prec':>6} {'Rec':>6} {'F1':>6} {'Pred/Lim':>9} {'True':>5} {'F1-mac':>7}")
    for r in sorted_results[:50]:
        print(f"{r['model']:<16} {r['scenario']:<28} {r['constraint_pair']:>7} "
              f"{r['class_name']:>6} {r['precision']:>6.3f} {r['recall']:>6.3f} "
              f"{r['f1']:>6.3f} {r['n_pred']:>4}/{r['limit']:>4} {r['n_true']:>5} "
              f"{r['f1_macro']:>7.4f}")

    # Per-class average precision
    print("\n" + "=" * 80)
    print("PER-CLASS AVERAGE HEURISTIC PRECISION (across all scenarios and pairs)")
    print("=" * 80)
    print(f"{'Model':<16}", end="")
    for c in range(NUM_CLASSES):
        print(f" {CLASS_NAMES[c]:>8}", end="")
    print()
    for model_name in MODELS:
        print(f"{model_name:<16}", end="")
        for c in range(NUM_CLASSES):
            class_results = [r for r in meaningful
                             if r['model'] == model_name and r['class'] == c]
            if class_results:
                avg_prec = np.mean([r['precision'] for r in class_results])
                print(f" {avg_prec:>8.3f}", end="")
            else:
                print(f" {'N/A':>8}", end="")
        print()

    # Per-class worst precision
    print("\n" + "=" * 80)
    print("PER-CLASS WORST HEURISTIC PRECISION (min across all scenarios)")
    print("=" * 80)
    print(f"{'Model':<16}", end="")
    for c in range(NUM_CLASSES):
        print(f" {CLASS_NAMES[c]:>8}", end="")
    print()
    for model_name in MODELS:
        print(f"{model_name:<16}", end="")
        for c in range(NUM_CLASSES):
            class_results = [r for r in meaningful
                             if r['model'] == model_name and r['class'] == c]
            if class_results:
                min_prec = min(r['precision'] for r in class_results)
                print(f" {min_prec:>8.3f}", end="")
            else:
                print(f" {'N/A':>8}", end="")
        print()

    print("\n" + "=" * 80)
    print("BEST HEURISTIC PERFORMANCE (highest constrained-class precision)")
    print("=" * 80)
    for r in sorted_results[-20:]:
        print(f"{r['model']:<16} {r['scenario']:<28} {r['constraint_pair']:>7} "
              f"{r['class_name']:>6} {r['precision']:>6.3f} {r['recall']:>6.3f} "
              f"{r['f1']:>6.3f} {r['n_pred']:>4}/{r['limit']:>4} {r['n_true']:>5} "
              f"{r['f1_macro']:>7.4f}")

    # Save CSV
    import csv
    out_path = 'heuristic_sweep_results.csv'
    with open(out_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=all_results[0].keys())
        writer.writeheader()
        writer.writerows(all_results)
    print(f"\nFull results saved to {out_path}")


if __name__ == '__main__':
    main()
