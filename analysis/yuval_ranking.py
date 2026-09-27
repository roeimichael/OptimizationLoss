"""Does PAO change the grade-3 RANKING, or only the argmax COUNT? (post hoc, development labels, offline)

Usage: python analysis/yuval_ranking.py RUN_ROOT

capped_first fills the 76 grade-3 slots with the top 76 items by p3, so under the cap only the order
of p3 can matter; the argmax count, which PAO's outer loop steers, is fixed by the cut. Per seed of a
tralo.knee_yuval run, for pto, pao, tralo_final and sham_final:
  AUC of p3 for grade 3 vs the rest; precision of the top 76 by p3 (the capped slots);
  and, with NO cap, the raw argmax grade-3 count, grade-3 F1 and accuracy.
"""

import json
import math
from pathlib import Path
import sys

import numpy as np
from scipy import stats
import torch

CAP, K = 76, 3
ARMS = ('pto', 'pao', 'tralo_final', 'sham_final')


def auc(score, positive):
    ranks = stats.rankdata(score)
    n1 = int(positive.sum())
    return float((ranks[positive].sum() - n1 * (n1 + 1) / 2) / (n1 * (len(score) - n1)))


def describe(probs, labels):
    p3 = probs[:, K].numpy().astype(np.float64)
    positive = labels == K
    top = np.argsort(-p3, kind='stable')[:CAP]
    pred = probs.argmax(1).numpy()
    tp = int(((pred == K) & positive).sum())
    return dict(auc=auc(p3, positive), prec_at_cap=float(positive[top].mean()), raw_count=int((pred == K).sum()),
                raw_f1=2 * tp / (int((pred == K).sum()) + int(positive.sum())), raw_acc=float((pred == labels).mean()))


def ci(d):
    d = np.asarray(d, float)
    h = stats.t.ppf(0.975, len(d) - 1) * d.std(ddof=1) / math.sqrt(len(d))
    return f'{d.mean():+.4f} [{d.mean() - h:+.4f}, {d.mean() + h:+.4f}] p {stats.ttest_1samp(d, 0).pvalue:.3f}'


def main(root):
    rows = []
    for d in sorted(Path(root).glob('seed*')):
        if not (d / 'summary.json').exists():
            continue
        s = json.loads((d / 'summary.json').read_text())
        labels = np.array([r['label'] for r in json.loads((d / 'manifest.json').read_text())['rows'] if r['split'] == 'val'])
        files = dict(pto=d / 'retrain1' / 'final_probabilities.pt', tralo_final=d / 'tralo_final' / 'final_probabilities.pt',
                     sham_final=d / 'sham_final' / 'final_probabilities.pt', pao=d / s['arms']['pao']['file'])
        rows.append(dict(seed=s['seed'], retrains=len(s['retrains']),
                         **{a: describe(torch.load(f, weights_only=True), labels) for a, f in files.items()}))
    print(f'{len(rows)} seeds; {sum(r["retrains"] > 1 for r in rows)} where PAO retrained at least once\n')
    for key in ('auc', 'prec_at_cap', 'raw_count', 'raw_f1', 'raw_acc'):
        print(f'{key:12s} ' + '  '.join(f'{a} {np.mean([r[a][key] for r in rows]):.4f}' for a in ARMS))
    print('\npaired, all seeds:')
    for key in ('auc', 'prec_at_cap', 'raw_count', 'raw_f1', 'raw_acc'):
        print(f'  pao - pto          {key:12s} {ci([r["pao"][key] - r["pto"][key] for r in rows])}')
    for key in ('auc', 'prec_at_cap'):
        print(f'  tralo_final - sham {key:12s} {ci([r["tralo_final"][key] - r["sham_final"][key] for r in rows])}')
    moved = [r for r in rows if r['retrains'] > 1]
    print(f'\npaired, seeds where PAO retrained (n {len(moved)}):')
    for key in ('auc', 'prec_at_cap', 'raw_count', 'raw_f1', 'raw_acc'):
        print(f'  pao - pto          {key:12s} {ci([r["pao"][key] - r["pto"][key] for r in moved])}')


if __name__ == '__main__':
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    main(sys.argv[1])
