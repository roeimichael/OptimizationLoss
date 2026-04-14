"""Compare all methods on mar27 replay, focus on constrained class precision."""
import torch, json, pandas as pd, numpy as np
from pathlib import Path
from sklearn.metrics import precision_recall_fscore_support
from src.utils.data_loader import load_experiment_data
from src.training.model_cache import load_from_cache

base = Path('results/pending_runs/mar27_single_GE/L50_G50/MobileNetV3')
config = json.load(open(base / 'heuristic/slice_1/config.json'))
data = load_experiment_data(config)
X_train, X_test, y_train, y_test, groups_test, global_con, local_con, num_classes = data
device = torch.device('cuda')
model = load_from_cache(config['base_model_id'], config, None, num_classes, device)
model.eval()
with torch.no_grad():
    X = torch.FloatTensor(X_test).to(device)
    logits = torch.cat([model(X[i:i+256]) for i in range(0, len(X), 256)])
    probs = torch.softmax(logits, dim=1).cpu().numpy()
y_argmax = probs.argmax(axis=1)
y_true = y_test

# Load predictions from all methods
results = {}
results['argmax'] = y_argmax
for meth in ['heuristic', 'danits_lp', 'our_approach']:
    f = base / meth / 'slice_1' / 'final_predictions.csv'
    df = pd.read_csv(f)
    results[meth] = df['Predicted_Label'].values

# Also raw (pre-posthoc) for our_approach
raw_f = base / 'our_approach/slice_1/final_predictions_raw.csv'
if raw_f.exists():
    df_raw = pd.read_csv(raw_f)
    results['our_approach_RAW'] = df_raw['Predicted_Label'].values

print('=== MARCH 27 REPLAY: single_GE / L50_G50 / MobileNetV3 ===')
print('Constrained: GE(class 4) limit=86, Local group limits: 60/68')
print()
print('{:<22} {:>7} {:>8} {:>8} {:>7} {:>7} {:>7} {:>5}'.format(
    'Method', 'Acc', 'F1_mac', 'F1_wt', 'GE_P', 'GE_R', 'GE_F1', 'GE_N'))
print('-' * 72)
for name, y_pred in results.items():
    acc = (y_pred == y_true).mean()
    _, _, f1m, _ = precision_recall_fscore_support(y_true, y_pred, average='macro', zero_division=0)
    _, _, f1w, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted', zero_division=0)
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average=None, zero_division=0)
    ge_n = int((y_pred == 4).sum())
    print('{:<22} {:>7.4f} {:>8.4f} {:>8.4f} {:>7.4f} {:>7.4f} {:>7.4f} {:>5}'.format(
        name, acc, f1m, f1w, p[4], r[4], f1[4], ge_n))

# Sample differences
print()
print('Sample overlap vs heuristic:')
for name, y in results.items():
    if name == 'heuristic':
        continue
    diff = (y != results['heuristic']).sum()
    print('  {}: {} differing samples (of {})'.format(name, diff, len(y)))
