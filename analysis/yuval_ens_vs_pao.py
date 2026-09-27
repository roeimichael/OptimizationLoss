"""EXPLORATORY: does Yuval's PAO beat the post-hoc bar, PTO's snapshot ensemble? (post hoc, development labels, offline)

Usage: python analysis/yuval_ens_vs_pao.py RUN_ROOT

Per seed of a tralo.knee_yuval run: capped_first cc-F1, accuracy and macro-F1 of the snapshot ensemble
(the rule of analysis/yuval_ensemble.py: mean of retrain 1's snapshots from epoch max(1, best - 2) to
the last epoch run) against pao and tralo_final, paired by seed, 95% t intervals.
"""

import json
import math
from pathlib import Path
import sys

import numpy as np
import torch
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tralo.global_report import evaluate_global  # noqa: E402

CAPS = [None, None, None, 76, None]
METRICS = ('cc_f1', 'accuracy', 'macro_f1')


def main(root):
    rows = []
    for d in sorted(Path(root).glob('seed*')):
        if not (d / 'summary.json').exists():
            continue
        s = json.loads((d / 'summary.json').read_text())
        val = [r for r in json.loads((d / 'manifest.json').read_text())['rows'] if r['split'] == 'val']
        labels, ids = [r['label'] for r in val], [r['sample_id'] for r in val]
        r1 = s['retrains'][0]
        window = range(max(1, r1['best_epoch'] - 2), r1['epochs_run'] + 1)
        probs = {'ens': torch.stack([torch.load(d / 'retrain1' / f'epoch{e:02d}.pt', weights_only=True) for e in window]).mean(0)}
        for arm in ('pao', 'tralo_final'):
            probs[arm] = torch.load(d / s['arms'][arm]['file'], weights_only=True)
        score = {}
        for name, p in probs.items():
            m = evaluate_global(p.tolist(), labels, CAPS, ids)['capped_first']['metrics']
            score[name] = [m[k] for k in METRICS]
        for arm in ('pao', 'tralo_final'):      # the stored scores must agree with this recomputation
            stored = s['arms'][arm]['scores']['capped_first']
            assert all(abs(stored[k] - score[arm][i]) < 1e-12 for i, k in enumerate(METRICS)), (s['seed'], arm)
        rows.append(score)
    print(f'{len(rows)} seeds')
    for arm in ('pao', 'tralo_final'):
        for i, metric in enumerate(METRICS):
            d = np.array([r['ens'][i] - r[arm][i] for r in rows])
            h = stats.t.ppf(0.975, len(d) - 1) * d.std(ddof=1) / math.sqrt(len(d))
            print(f'  ENS - {arm:11s} {metric:9s} {100 * d.mean():+.2f} [{100 * (d.mean() - h):+.2f}, '
                  f'{100 * (d.mean() + h):+.2f}] p {stats.ttest_1samp(d, 0).pvalue:.4f}; '
                  f'ENS better in {int((d > 0).sum())}, worse in {int((d < 0).sum())} of {len(d)}')


if __name__ == '__main__':
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    main(sys.argv[1])
