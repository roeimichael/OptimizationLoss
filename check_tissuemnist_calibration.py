"""Check model calibration on TissueMNIST.
Trains warmup one model at a time (frees memory between models) then
evaluates predictions and runs heuristic sweep across constraint configs.

Run: python check_tissuemnist_calibration.py
"""
import gc
import hashlib
import json
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.utils.data_loader import _ensure_3channel, _apply_imagenet_normalization
from src.models import get_model
from src.training.metrics import compute_train_accuracy
from src.training.model_cache import load_from_cache, save_to_cache
from src.utils.inference import chunked_forward
from src.training.constraints import compute_global_constraints, compute_local_constraints
from src.experiments.run_heuristic import apply_allocation_heuristic, _build_hierarchy

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODELS = ['ResNet18', 'MobileNetV3', 'EfficientNetB0', 'ConvNeXtTiny']
NUM_CLASSES = 8
CLASS_NAMES = ['CDI', 'CDS', 'CST', 'EPI', 'GE', 'PTC', 'STR', 'TUB']
DATA_DIR = 'data/tissuemnist'
UNLIMITED = 1e10

HP = {'lr': 0.0001, 'dropout': 0.3, 'batch_size': 64, 'warmup_epochs': 50,
      'pretrained': True, 'class_weighted_ce': False}


def base_id(model_name):
    key = {'model_name': model_name, 'lr': 0.0001, 'dropout': 0.3, 'batch_size': 64,
           'warmup_epochs': 50, 'pretrained': True, 'class_weighted_ce': False,
           'dataset_mode': 'tissuemnist', 'data_dir': DATA_DIR}
    h = hashlib.md5(json.dumps(key, sort_keys=True).encode()).hexdigest()[:12]
    return '%s_tissuemnist_%s' % (model_name, h)


def train_one_model(model_name):
    """Train warmup for one model, save to cache, free all memory."""
    bid = base_id(model_name)
    config = {'model_name': model_name, 'hyperparams': HP}
    model = load_from_cache(bid, config, None, NUM_CLASSES, device)
    if model is not None:
        print("  Cached: %s" % bid)
        del model
        torch.cuda.empty_cache()
        return

    print("  Training: %s ..." % model_name, flush=True)
    X_train = _apply_imagenet_normalization(
        _ensure_3channel(np.load('%s/train_images.npy' % DATA_DIR)))
    y_train = np.load('%s/train_labels.npy' % DATA_DIR).ravel()

    model = get_model(model_name, n_classes=NUM_CLASSES, dropout=0.3, pretrained=True).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    criterion = nn.CrossEntropyLoss()
    loader = DataLoader(
        TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train)),
        batch_size=64, shuffle=True, pin_memory=True, num_workers=0)

    use_amp = device.type == 'cuda'
    amp_dtype = torch.bfloat16 if (use_amp and torch.cuda.get_device_capability(0)[0] >= 8) else torch.float16
    t0 = time.time()
    for epoch in range(50):
        model.train()
        for bx, by in loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda', dtype=amp_dtype, enabled=use_amp):
                loss = criterion(model(bx), by)
            loss.backward()
            optimizer.step()
        if epoch < 3 or (epoch + 1) % 10 == 0:
            acc = compute_train_accuracy(model, loader, device)
            print("    epoch %d/50 acc=%.4f [%.1fs]" % (epoch + 1, acc, time.time() - t0), flush=True)
    save_to_cache(model, bid, config)
    print("  Done: %s in %.1fs" % (model_name, time.time() - t0), flush=True)

    del model, optimizer, criterion, loader, X_train, y_train
    torch.cuda.empty_cache()
    gc.collect()


def evaluate_model(model_name, X_test):
    """Load cached model, return probabilities."""
    bid = base_id(model_name)
    config = {'model_name': model_name, 'hyperparams': HP}
    model = load_from_cache(bid, config, None, NUM_CLASSES, device)
    model.eval()
    with torch.no_grad():
        logits = chunked_forward(model, torch.FloatTensor(X_test).to(device))
        proba = torch.softmax(logits.float(), dim=1).cpu().numpy()
    del model
    torch.cuda.empty_cache()
    return proba


def per_class_metrics(y_true, y_pred):
    results = {}
    for c in range(NUM_CLASSES):
        tp = int(((y_pred == c) & (y_true == c)).sum())
        n_pred = int((y_pred == c).sum())
        n_true = int((y_true == c).sum())
        prec = tp / n_pred if n_pred > 0 else 0
        rec = tp / n_true if n_true > 0 else 0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
        results[c] = {'precision': prec, 'recall': rec, 'f1': f1,
                       'n_pred': n_pred, 'n_true': n_true}
    return results


def main():
    print("Device: %s" % device, flush=True)
    if device.type == 'cuda':
        print("GPU: %s" % torch.cuda.get_device_name(0), flush=True)

    # ── Phase 1: Train all warmup models (one at a time) ──
    print("\n=== Training warmup models ===", flush=True)
    for model_name in MODELS:
        train_one_model(model_name)

    # ── Phase 2: Load test data only ──
    print("\n=== Loading test data ===", flush=True)
    X_test = _apply_imagenet_normalization(
        _ensure_3channel(np.load('%s/test_images.npy' % DATA_DIR)))
    y_test = np.load('%s/test_labels.npy' % DATA_DIR).ravel()
    meta = pd.read_csv('%s/test_meta.csv' % DATA_DIR)
    groups = meta['synth_group'].values.astype(np.int64)
    print("Test: %s (%.2f GB)" % (str(X_test.shape), X_test.nbytes / 1e9), flush=True)

    print("\n" + "=" * 80)
    print("GROUND TRUTH")
    print("=" * 80)
    for c in range(NUM_CLASSES):
        n = int((y_test == c).sum())
        print("  %4s (class %d): %5d (%5.1f%%)" % (CLASS_NAMES[c], c, n, n / len(y_test) * 100))
    print("  Total: %d  |  Groups: %s" % (len(y_test), [int((groups == g).sum()) for g in np.unique(groups)]))

    # ── Phase 3: Calibration analysis per model ──
    print("\n" + "=" * 80)
    print("MODEL CALIBRATION (unconstrained argmax)")
    print("=" * 80)

    model_proba = {}
    for model_name in MODELS:
        proba = evaluate_model(model_name, X_test)
        model_proba[model_name] = proba
        argmax = np.argmax(proba, axis=1)
        acc = (argmax == y_test).mean()
        max_probs = proba.max(axis=1)
        sorted_probs = np.sort(proba, axis=1)[:, ::-1]
        margin = sorted_probs[:, 0] - sorted_probs[:, 1]

        metrics = per_class_metrics(y_test, argmax)
        f1_macro = np.mean([metrics[c]['f1'] for c in range(NUM_CLASSES)])

        print("\n  %s (acc=%.4f, F1-macro=%.4f)" % (model_name, acc, f1_macro))
        print("  Confidence: P>=0.9: %.1f%%  P>=0.7: %.1f%%  P>=0.5: %.1f%%" % (
            (max_probs >= 0.9).mean() * 100,
            (max_probs >= 0.7).mean() * 100,
            (max_probs >= 0.5).mean() * 100))
        print("  Borderline: margin<0.1: %d (%.1f%%)  margin<0.2: %d (%.1f%%)" % (
            (margin < 0.1).sum(), (margin < 0.1).mean() * 100,
            (margin < 0.2).sum(), (margin < 0.2).mean() * 100))
        print("  %8s %5s %5s %6s %6s %6s" % ('Class', 'True', 'Pred', 'Prec', 'Rec', 'F1'))
        for c in range(NUM_CLASSES):
            m = metrics[c]
            print("  %8s %5d %5d %6.3f %6.3f %6.3f" % (
                CLASS_NAMES[c], m['n_true'], m['n_pred'], m['precision'], m['recall'], m['f1']))

    # ── Phase 4: Comparison with DermMNIST ──
    print("\n" + "=" * 80)
    print("COMPARISON: DermMNIST vs TissueMNIST")
    print("=" * 80)
    print("DermMNIST MobileNetV3:  acc=0.854, F1=0.728, P>=0.9: 89%, borderline: 0.9%")
    print("TissueMNIST: see above")

    # ── Phase 5: Quick heuristic sweep on constrained classes ──
    print("\n" + "=" * 80)
    print("HEURISTIC SWEEP (selected scenarios)")
    print("=" * 80)

    # GE (class 4) is the designated constrained class per CLAUDE.md
    # Also test others to find weakness
    scenarios = {
        'single_GE':       [4],
        'single_CDI':      [0],
        'single_EPI':      [3],
        'single_STR':      [6],
        'single_TUB':      [7],
        'single_CDS':      [1],
        'single_CST':      [2],
        'single_PTC':      [5],
        'multi_GE_EPI':    [4, 3],
        'multi_GE_CDS':    [4, 1],
        'multi_CDI_STR':   [0, 6],
        'multi_GE_TUB':    [4, 7],
        'multi_CDS_CST_PTC': [1, 2, 5],
    }
    pairs = [(0.3, 0.3), (0.5, 0.5), (0.8, 0.8), (0.2, 0.8), (0.8, 0.3)]

    all_results = []
    test_df = pd.DataFrame({'label': y_test, 'group': groups})

    for scenario_name, cc_list in scenarios.items():
        for local_pct, global_pct in pairs:
            pair_tag = 'L%02d_G%02d' % (int(local_pct * 100), int(global_pct * 100))
            global_con = compute_global_constraints(test_df, 'label', global_pct,
                                                     constrained_class=cc_list, num_classes=NUM_CLASSES)
            local_con = compute_local_constraints(test_df, 'label', local_pct, 'group',
                                                    constrained_class=cc_list, num_classes=NUM_CLASSES)

            skip = any(global_con[c] < 1 for c in cc_list)
            if skip:
                continue

            limits_str = ", ".join("%s=%d" % (CLASS_NAMES[c], int(global_con[c])) for c in cc_list)
            print("\n--- %s | %s | %s ---" % (scenario_name, pair_tag, limits_str))

            header = "%-16s" % 'Model'
            for c in cc_list:
                header += " | %5s prec   rec    f1 pred/lim" % CLASS_NAMES[c]
            header += " | %7s %7s" % ('F1-mac', 'Acc')
            print(header)

            for model_name in MODELS:
                proba = model_proba[model_name]
                hierarchy = _build_hierarchy(NUM_CLASSES, global_con, cc_list)
                y_pred, _ = apply_allocation_heuristic(
                    proba, groups, hierarchy, global_con, local_con, NUM_CLASSES)

                metrics = per_class_metrics(y_test, y_pred)
                f1_macro = np.mean([metrics[c]['f1'] for c in range(NUM_CLASSES)])
                acc = (y_pred == y_test).mean()

                row = "%-16s" % model_name
                for c in cc_list:
                    m = metrics[c]
                    lim = int(global_con[c])
                    row += " | %5.3f %5.3f %5.3f %4d/%4d" % (
                        m['precision'], m['recall'], m['f1'], m['n_pred'], lim)
                row += " | %7.4f %7.4f" % (f1_macro, acc)
                print(row)

                for c in cc_list:
                    all_results.append({
                        'model': model_name, 'scenario': scenario_name,
                        'pair': pair_tag, 'class': c, 'class_name': CLASS_NAMES[c],
                        'precision': metrics[c]['precision'], 'recall': metrics[c]['recall'],
                        'f1': metrics[c]['f1'], 'n_pred': metrics[c]['n_pred'],
                        'limit': int(global_con[c]), 'n_true': metrics[c]['n_true'],
                        'f1_macro': f1_macro, 'accuracy': acc,
                    })

    # ── Phase 6: Summary ──
    meaningful = [r for r in all_results if r['limit'] > 3]

    print("\n" + "=" * 80)
    print("WORST HEURISTIC PRECISION (optimization opportunities)")
    print("=" * 80)
    sorted_results = sorted(meaningful, key=lambda x: x['precision'])
    print("%-16s %-20s %7s %6s %6s %6s %6s %9s %5s %7s" % (
        'Model', 'Scenario', 'Pair', 'Class', 'Prec', 'Rec', 'F1', 'Pred/Lim', 'True', 'F1-mac'))
    for r in sorted_results[:40]:
        print("%-16s %-20s %7s %6s %6.3f %6.3f %6.3f %4d/%4d %5d %7.4f" % (
            r['model'], r['scenario'], r['pair'],
            r['class_name'], r['precision'], r['recall'],
            r['f1'], r['n_pred'], r['limit'], r['n_true'], r['f1_macro']))

    print("\n" + "=" * 80)
    print("PER-CLASS AVERAGE HEURISTIC PRECISION")
    print("=" * 80)
    print("%-16s" % 'Model', end="")
    for c in range(NUM_CLASSES):
        print(" %8s" % CLASS_NAMES[c], end="")
    print()
    for model_name in MODELS:
        print("%-16s" % model_name, end="")
        for c in range(NUM_CLASSES):
            cr = [r for r in meaningful if r['model'] == model_name and r['class'] == c]
            if cr:
                print(" %8.3f" % np.mean([r['precision'] for r in cr]), end="")
            else:
                print(" %8s" % 'N/A', end="")
        print()

    print("\n" + "=" * 80)
    print("PER-CLASS WORST HEURISTIC PRECISION")
    print("=" * 80)
    print("%-16s" % 'Model', end="")
    for c in range(NUM_CLASSES):
        print(" %8s" % CLASS_NAMES[c], end="")
    print()
    for model_name in MODELS:
        print("%-16s" % model_name, end="")
        for c in range(NUM_CLASSES):
            cr = [r for r in meaningful if r['model'] == model_name and r['class'] == c]
            if cr:
                print(" %8.3f" % min(r['precision'] for r in cr), end="")
            else:
                print(" %8s" % 'N/A', end="")
        print()

    # Save CSV
    import csv
    out_path = 'tissuemnist_calibration_results.csv'
    if all_results:
        with open(out_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=all_results[0].keys())
            writer.writeheader()
            writer.writerows(all_results)
        print("\nFull results saved to %s" % out_path)


if __name__ == '__main__':
    main()
