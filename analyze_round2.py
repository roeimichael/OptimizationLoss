"""Analyze round 2 experiment results (conflicting local/global constraints)."""
import json, glob, os
import pandas as pd
import numpy as np
from src.training.constraints import compute_global_constraints, compute_local_constraints

experiments = sorted(glob.glob('results/pending_runs/exp*/'))
class_names = ['AKIEC','BCC','BKL','DF','MEL','NV','VASC']
summary_rows = []

for exp_dir in experiments:
    exp_name = os.path.basename(os.path.normpath(exp_dir))
    oa_cfg_path = os.path.join(exp_dir, 'our_approach', 'config.json')
    h_cfg_path = os.path.join(exp_dir, 'heuristic', 'config.json')

    if not os.path.exists(oa_cfg_path) or not os.path.exists(h_cfg_path):
        print('MISSING config in', exp_name)
        continue

    oa = json.load(open(oa_cfg_path))
    h = json.load(open(h_cfg_path))

    if oa.get('status') != 'completed' or h.get('status') != 'completed':
        print('%s: oa=%s heur=%s' % (exp_name, oa.get('status'), h.get('status')))
        continue

    r_oa = oa['results']
    r_h = h['results']

    cc = oa['dataset_config']['constrained_class']
    cc_list = cc if isinstance(cc, list) else [cc]
    cc_names = [class_names[c] for c in cc_list]

    print('=' * 90)
    print('%s  |  %s  |  L%d/G%d  |  classes: %s' % (
        exp_name, oa['model_name'],
        oa['constraint'][0]*100, oa['constraint'][1]*100, cc_names))
    print('=' * 90)

    print('%-25s %15s %15s %15s' % ('Metric', 'Our Approach', 'Heuristic', 'Delta'))
    print('-' * 70)
    for key in ['f1_macro', 'accuracy', 'precision_macro', 'recall_macro']:
        v_oa = r_oa.get(key, 0)
        v_h = r_h.get(key, 0)
        d = v_oa - v_h
        marker = ' <<<' if abs(d) > 0.005 else ''
        print('%-25s %15.4f %15.4f %+14.4f%s' % (key, v_oa, v_h, d, marker))
    print('%-25s %15d %15d' % ('samples_adjusted',
          r_oa['samples_adjusted'], r_h.get('samples_adjusted', 0)))
    print('%-25s %15s' % ('checkpoint_source', r_oa.get('checkpoint_source', '')))
    print('%-25s %15s' % ('lp_fallback_used', r_oa.get('lp_fallback_used', '')))

    oa_csv = os.path.join(exp_dir, 'our_approach', 'final_predictions.csv')
    h_csv = os.path.join(exp_dir, 'heuristic', 'final_predictions.csv')

    if os.path.exists(oa_csv) and os.path.exists(h_csv):
        oa_df = pd.read_csv(oa_csv)
        h_df = pd.read_csv(h_csv)
        y_true = oa_df['True_Label'].values
        y_oa = oa_df['Predicted_Label'].values
        y_h = h_df['Predicted_Label'].values
        groups = oa_df['Group_ID'].values

        print()
        print('  Per-class (constrained = *):')
        print('  %-8s %5s | %6s %6s %6s %5s | %6s %6s %6s %5s | %7s' % (
            'Class', 'True', 'O.Prec', 'O.Rec', 'O.F1', 'O.N',
            'H.Prec', 'H.Rec', 'H.F1', 'H.N', 'P.Delta'))

        for c in range(7):
            n_true = int((y_true == c).sum())
            tp_o = int(((y_oa == c) & (y_true == c)).sum())
            n_o = int((y_oa == c).sum())
            prec_o = tp_o / n_o if n_o > 0 else 0
            rec_o = tp_o / n_true if n_true > 0 else 0
            f1_o = 2*prec_o*rec_o/(prec_o+rec_o) if (prec_o+rec_o) > 0 else 0

            tp_h = int(((y_h == c) & (y_true == c)).sum())
            n_h = int((y_h == c).sum())
            prec_h = tp_h / n_h if n_h > 0 else 0
            rec_h = tp_h / n_true if n_true > 0 else 0
            f1_h = 2*prec_h*rec_h/(prec_h+rec_h) if (prec_h+rec_h) > 0 else 0

            marker = ' *' if c in cc_list else '  '
            p_delta = prec_o - prec_h
            flag = ' <<<' if c in cc_list and abs(p_delta) > 0.01 else ''

            print('  %-6s%s %5d | %6.3f %6.3f %6.3f %5d | %6.3f %6.3f %6.3f %5d | %+7.3f%s' % (
                class_names[c], marker, n_true,
                prec_o, rec_o, f1_o, n_o,
                prec_h, rec_h, f1_h, n_h,
                p_delta, flag))

        # Constraint satisfaction detail
        print()
        test_df = pd.DataFrame({'label': y_true, 'group': groups})
        global_con = compute_global_constraints(test_df, 'label', oa['constraint'][1],
                                                constrained_class=cc_list, num_classes=7)
        local_con_cfg = compute_local_constraints(test_df, 'label', oa['constraint'][0], 'group',
                                                   constrained_class=cc_list, num_classes=7)

        print('  Constraint satisfaction:')
        for c in cc_list:
            g_lim = int(global_con[c])
            n_oa_c = int((y_oa == c).sum())
            n_h_c = int((y_h == c).sum())
            local_sum = sum(int(local_con_cfg[gid][c]) for gid in local_con_cfg)
            wasted = g_lim - n_h_c
            utilized = n_oa_c - n_h_c
            print('    %s: global=%d, local_sum=%d | ours=%d, heur=%d | heur wastes %d, ours recovers %d' % (
                class_names[c], g_lim, local_sum, n_oa_c, n_h_c, wasted, max(0, utilized)))

            for gid in sorted(local_con_cfg):
                l_lim = int(local_con_cfg[gid][c])
                g_mask = groups == gid
                oa_g = int((y_oa[g_mask] == c).sum())
                h_g = int((y_h[g_mask] == c).sum())
                print('      group_%d: lim=%d | ours=%d %s | heur=%d %s' % (
                    gid, l_lim, oa_g,
                    'OK' if oa_g <= l_lim else 'VIOL',
                    h_g, 'OK' if h_g <= l_lim else 'VIOL'))

    if 'results_comparison' in oa:
        print()
        print('  Checkpoints:')
        for src, vals in oa['results_comparison'].items():
            print('    %-20s f1=%.4f  acc=%.4f  adj=%d  lp=%s' % (
                src, vals['f1_macro'], vals['accuracy'], vals['adjusted'],
                vals.get('lp_fallback_used', '')))

    summary_rows.append({
        'exp': exp_name, 'model': oa['model_name'],
        'pair': 'L%d/G%d' % (oa['constraint'][0]*100, oa['constraint'][1]*100),
        'classes': '+'.join(cc_names),
        'f1_ours': r_oa['f1_macro'], 'f1_heur': r_h['f1_macro'],
        'f1_delta': r_oa['f1_macro'] - r_h['f1_macro'],
        'acc_ours': r_oa['accuracy'], 'acc_heur': r_h['accuracy'],
        'acc_delta': r_oa['accuracy'] - r_h['accuracy'],
    })
    print()

# Summary table
print('=' * 95)
print('SUMMARY: Round 2 (Conflicting Local/Global Constraints)')
print('=' * 95)
print('%-42s %7s %7s %8s | %7s %7s %8s' % (
    'Experiment', 'F1_Our', 'F1_Heu', 'F1_Delta', 'Acc_Our', 'Acc_Heu', 'Acc_Del'))
print('-' * 95)
for r in summary_rows:
    f1_flag = ' <<<' if r['f1_delta'] > 0.01 else ''
    print('%-42s %7.4f %7.4f %+8.4f%s | %7.4f %7.4f %+8.4f' % (
        '%s %s %s' % (r['model'][:10], r['classes'], r['pair']),
        r['f1_ours'], r['f1_heur'], r['f1_delta'], f1_flag,
        r['acc_ours'], r['acc_heur'], r['acc_delta']))

avg_f1 = np.mean([r['f1_delta'] for r in summary_rows])
avg_acc = np.mean([r['acc_delta'] for r in summary_rows])
print('-' * 95)
print('%-42s %7s %7s %+8.4f   | %7s %7s %+8.4f' % (
    'AVERAGE', '', '', avg_f1, '', '', avg_acc))

print()
print('Comparison with Round 1 (symmetric L80/G80, L50/G30):')
print('  Round 1 avg F1 delta: +0.0068 (range +0.0044 to +0.0102)')
print('  Round 2 avg F1 delta: %+.4f' % avg_f1)
if avg_f1 > 0.0068:
    print('  Improvement: %.1fx larger margin than Round 1' % (avg_f1 / 0.0068))
