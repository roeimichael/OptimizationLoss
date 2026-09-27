"""Do PAO's re-weighted retrains learn more slowly than PTO? (post hoc, development labels, offline)

Usage: python analysis/yuval_retrain_speed.py RUN_ROOT

Per seed of a tralo.knee_yuval run with at least two retrains, at each epoch both runs reached, compare
retrain 2 (the first with C > 1) against retrain 1 (PTO). Both start from the same weights and see the
same sampler order and augmentation draws, so the only difference is PAO's weights, the LR they set,
and the tanh(5e7) gate's gradient spikes. Reported: development argmax accuracy, and live training
false positives of grade 3 (logged by the runner).
"""

import json
import math
from pathlib import Path
import sys

import numpy as np
from scipy import stats
import torch

K = 3


def ci(d):
    d = np.asarray(d, float)
    h = stats.t.ppf(0.975, len(d) - 1) * d.std(ddof=1) / math.sqrt(len(d))
    return f'{d.mean():+.4f} [{d.mean() - h:+.4f}, {d.mean() + h:+.4f}] (n {len(d)})'


def main(root):
    acc, fps = {}, {}
    for d in sorted(Path(root).glob('seed*')):
        if not (d / 'summary.json').exists() or len(json.loads((d / 'summary.json').read_text())['retrains']) < 2:
            continue
        labels = torch.tensor([r['label'] for r in json.loads((d / 'manifest.json').read_text())['rows'] if r['split'] == 'val'])
        runs = []
        for r in (1, 2):
            rows = [json.loads(x) for x in (d / f'retrain{r}' / 'events.jsonl').read_text().splitlines()]
            epochs = {x['epoch']: x for x in rows if x['event'] == 'epoch'}
            runs.append({e: (float((torch.load(d / f'retrain{r}' / f'epoch{e:02d}.pt', weights_only=True).argmax(1) == labels)
                                   .float().mean()), x['live_false_positives']) for e, x in epochs.items()})
        for e in sorted(set(runs[0]) & set(runs[1])):
            acc.setdefault(e, []).append(runs[1][e][0] - runs[0][e][0])
            fps.setdefault(e, []).append(runs[1][e][1] - runs[0][e][1])
    print('retrain 2 - retrain 1 at the same epoch (seeds with >= 2 retrains):')
    for e in sorted(acc):
        if len(acc[e]) >= 3:
            print(f'  epoch {e:2d}: dev accuracy {ci(acc[e])}   live training FPs {ci(fps[e])}')


if __name__ == '__main__':
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    main(sys.argv[1])
