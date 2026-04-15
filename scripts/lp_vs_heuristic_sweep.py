"""Quick sweep: compare LP vs heuristic across many constraint configs."""
import json
import numpy as np
import torch
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support
from src.utils.data_loader import load_experiment_data
from src.training.model_cache import load_from_cache
from src.experiments.run_heuristic import apply_allocation_heuristic, _build_hierarchy
from src.training.constraints import compute_global_constraints, compute_local_constraints
from danits_research import solve_lp_assignment

UNLIMITED = 1e10

# Load data using existing config as template
config = json.load(open('results/pending_runs/multi_GE_CST/L30_G30/MobileNetV3/heuristic/slice_1/config.json'))
data = load_experiment_data(config)
X_train, X_test, y_train, y_test, groups_test, _, _, num_classes = data

device = torch.device('cuda')
model = load_from_cache(config['base_model_id'], config, None, num_classes, device)
model.eval()
with torch.no_grad():
    X = torch.FloatTensor(X_test).to(device)
    logits = torch.cat([model(X[i:i+256]) for i in range(0, len(X), 256)])
    probs = torch.softmax(logits, dim=1).cpu().numpy()

y_true = y_test
test_df = pd.DataFrame({'label': y_test, 'synth_group': groups_test})

configs_to_test = [
    ('2class_L30_G30', [2, 4], 0.3, 0.3),
    ('2class_L50_G50', [2, 4], 0.5, 0.5),
    ('2class_L80_G80', [2, 4], 0.8, 0.8),
    ('4class_L50_G50', [0, 2, 4, 6], 0.5, 0.5),
    ('4class_L30_G30', [0, 2, 4, 6], 0.3, 0.3),
    ('4class_L80_G80', [0, 2, 4, 6], 0.8, 0.8),
    ('6class_L50_G50', [0, 1, 2, 4, 5, 6], 0.5, 0.5),
    ('6class_L30_G30', [0, 1, 2, 4, 5, 6], 0.3, 0.3),
    ('ALL8_L50_G50', list(range(8)), 0.5, 0.5),
    ('ALL8_L30_G30', list(range(8)), 0.3, 0.3),
    ('ALL8_L80_G80', list(range(8)), 0.8, 0.8),
]

header = "{:<20} {:>7} {:>7} {:>8} {:>7} {:>7} {:>8} {:>6}".format(
    'Config', 'H_acc', 'LP_acc', 'Diff', 'H_F1m', 'LP_F1m', 'DiffF1', '#diff')
print(header)
print('-' * len(header))

for name, cc, g_pct, l_pct in configs_to_test:
    gc = compute_global_constraints(test_df, 'label', g_pct, constrained_class=cc, num_classes=8)
    lc = compute_local_constraints(test_df, 'label', l_pct, 'synth_group', constrained_class=cc, num_classes=8)

    # Heuristic
    hierarchy = _build_hierarchy(8, gc, cc)
    y_h, _ = apply_allocation_heuristic(probs, groups_test, hierarchy, gc, lc, 8)

    # LP
    omega = np.ones((8, 8)) - np.eye(8)
    psi = [int(v) if v < UNLIMITED else None for v in gc]
    phi = {}
    for g, bounds in lc.items():
        phi[g] = [int(v) if v < UNLIMITED else None for v in bounds]
    lp_res = solve_lp_assignment(probs, groups_test, omega, psi, phi, verbose=False)
    y_lp = lp_res.y_pred

    # Metrics
    acc_h = (y_h == y_true).mean()
    acc_lp = (y_lp == y_true).mean()
    _, _, f1m_h, _ = precision_recall_fscore_support(y_true, y_h, average='macro', zero_division=0)
    _, _, f1m_lp, _ = precision_recall_fscore_support(y_true, y_lp, average='macro', zero_division=0)
    n_diff = (y_h != y_lp).sum()

    print("{:<20} {:>7.4f} {:>7.4f} {:>+7.2f}pp {:>7.4f} {:>7.4f} {:>+7.2f}pp {:>5}".format(
        name, acc_h, acc_lp, (acc_lp - acc_h) * 100,
        f1m_h, f1m_lp, (f1m_lp - f1m_h) * 100, n_diff))

    # If there are differences, show constrained class details
    if n_diff > 0:
        p_h, r_h, f1_h, _ = precision_recall_fscore_support(y_true, y_h, average=None, zero_division=0)
        p_lp, r_lp, f1_lp, _ = precision_recall_fscore_support(y_true, y_lp, average=None, zero_division=0)
        for c in cc:
            h_count = (y_h == c).sum()
            lp_count = (y_lp == c).sum()
            limit = int(gc[c])
            print("  class {}: H pred={}/{} prec={:.3f} f1={:.3f} | LP pred={}/{} prec={:.3f} f1={:.3f}".format(
                c, h_count, limit, p_h[c], f1_h[c], lp_count, limit, p_lp[c], f1_lp[c]))
