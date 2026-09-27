"""EXPLORATORY: how far does averaging independently trained seeds go, and PAO at equal compute? (post hoc)

Usage: python analysis/yuval_deep_ensemble.py RUN_ROOT

For a tralo.knee_yuval run: the 24 seeds are cut into disjoint groups of k consecutive seeds, and each
group's mean of development probabilities is deployed with capped_first. This is done for PTO's restored
best epoch (DE-k) and for each seed's snapshot ensemble (the rule of analysis/yuval_ensemble.py, ENS-DE-k).
Each line reports the mean over the groups. PAO costs one training per retrain, so its equal-compute rival
is DE-k with k near its mean number of retrains.
"""

import json
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tralo.global_report import evaluate_global  # noqa: E402

CAPS = [None, None, None, 76, None]
KEYS = ('cc_f1', 'accuracy', 'macro_f1')


def main(root):
    seeds = []
    for d in sorted(Path(root).glob('seed*')):
        if not (d / 'summary.json').exists():
            continue
        s = json.loads((d / 'summary.json').read_text())
        r1 = s['retrains'][0]
        window = range(max(1, r1['best_epoch'] - 2), r1['epochs_run'] + 1)
        seeds.append(dict(
            best=torch.load(d / 'retrain1' / 'final_probabilities.pt', weights_only=True),
            ens=torch.stack([torch.load(d / 'retrain1' / f'epoch{e:02d}.pt', weights_only=True) for e in window]).mean(0),
            pao=s['arms'].get('pao', {}).get('scores', {}).get('capped_first'), retrains=len(s['retrains'])))
        manifest = d / 'manifest.json'
    val = [r for r in json.loads(manifest.read_text())['rows'] if r['split'] == 'val']
    labels, ids = [r['label'] for r in val], [r['sample_id'] for r in val]
    score = lambda p: [evaluate_global(p.tolist(), labels, CAPS, ids)['capped_first']['metrics'][k] for k in KEYS]
    print(f'{len(seeds)} seeds; groups of k consecutive seeds; mean [min, max] over groups, capped_first, %')
    for k in (1, 2, 3, 4, 6, 8, 12, 24):
        groups = [seeds[i:i + k] for i in range(0, len(seeds) - k + 1, k)]
        for name in ('best', 'ens'):
            v = np.array([score(torch.stack([s[name] for s in g]).mean(0)) for g in groups]) * 100
            print(f"  {'DE' if name == 'best' else 'ENS-DE'}-{k:<2d} ({len(groups):2d} groups)  " +
                  '  '.join(f'{key} {v[:, i].mean():.2f} [{v[:, i].min():.2f}, {v[:, i].max():.2f}]' for i, key in enumerate(KEYS)))
    if all(s['pao'] for s in seeds):
        v = np.array([[s['pao'][key] for key in KEYS] for s in seeds]) * 100
        print(f"  PAO, one seed each, {np.mean([s['retrains'] for s in seeds]):.2f} trainings per seed on average  " +
              '  '.join(f'{key} {v[:, i].mean():.2f}' for i, key in enumerate(KEYS)))


if __name__ == '__main__':
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    main(sys.argv[1])
