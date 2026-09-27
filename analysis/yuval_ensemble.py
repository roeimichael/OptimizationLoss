"""EXPLORATORY: does the snapshot ensemble still help once the recipe early-stops? (not preregistered)

Usage: python analysis/yuval_ensemble.py RUN_ROOT

For each seed of a tralo.knee_yuval run: capped_first on the mean of retrain 1's saved development
snapshots from epoch max(1, best - 2) to the last epoch run, against capped_first on the restored
best epoch (the pto arm). The v3 study found +2-3 slots from the same move on a memorised model.
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
        snaps = [torch.load(d / 'retrain1' / f'epoch{e:02d}.pt', weights_only=True) for e in window]
        best = torch.load(d / 'retrain1' / 'final_probabilities.pt', weights_only=True)
        score = {}
        for name, probs in (('best', best), ('ens', torch.stack(snaps).mean(0))):
            m = evaluate_global(probs.tolist(), labels, CAPS, ids)['capped_first']['metrics']
            score[name] = (m['cc_f1'], m['accuracy'], m['macro_f1'])
        rows.append((s['seed'], len(snaps), score))
    print(f'{len(rows)} seeds; window = epochs max(1, best-2)..last, mean {np.mean([r[1] for r in rows]):.1f} snapshots')
    for i, metric in enumerate(('cc_f1', 'accuracy', 'macro_f1')):
        d = np.array([r[2]['ens'][i] - r[2]['best'][i] for r in rows])
        if len(d) < 2:
            continue
        h = stats.t.ppf(0.975, len(d) - 1) * d.std(ddof=1) / math.sqrt(len(d))
        p = stats.ttest_1samp(d, 0).pvalue
        print(f'  ENS - best {metric:9s} {100 * d.mean():+.2f} [{100 * (d.mean() - h):+.2f}, {100 * (d.mean() + h):+.2f}] p {p:.3f}')


if __name__ == '__main__':
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    main(sys.argv[1])
