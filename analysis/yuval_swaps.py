"""Which items does TraLO's step swap into and out of the capped slots? (post hoc, development labels, offline)

Usage: python analysis/yuval_swaps.py RUN_ROOT

Per seed of a tralo.knee_yuval run: the top-76 sets by p3 of pto, tralo_final and sham_final. For
each arm against pto: the number of swapped slots, and how many of the items brought in and pushed
out are true grade 3. Also P2 (tralo_final - sham_final cc-F1, exact) against pto's excess over
the cap, over the binding seeds.
"""

import json
from pathlib import Path
import sys

import numpy as np
from scipy import stats
import torch

CAP, K = 76, 3


def top(p3):
    return set(torch.argsort(-p3, stable=True)[:CAP].tolist())


def main(root):
    swaps = {'tralo_final': [0, 0, 0], 'sham_final': [0, 0, 0]}     # swapped slots, in true-3, out true-3
    net, excess, p2 = [], [], []
    for d in sorted(Path(root).glob('seed*')):
        if not (d / 'summary.json').exists():
            continue
        s = json.loads((d / 'summary.json').read_text())
        y = torch.tensor([r['label'] for r in json.loads((d / 'manifest.json').read_text())['rows'] if r['split'] == 'val'])
        base = top(torch.load(d / 'retrain1' / 'final_probabilities.pt', weights_only=True)[:, K])
        for arm in swaps:
            moved = top(torch.load(d / arm / 'final_probabilities.pt', weights_only=True)[:, K])
            ins, outs = moved - base, base - moved
            swaps[arm][0] += len(ins)
            swaps[arm][1] += sum(int(y[i] == K) for i in ins)
            swaps[arm][2] += sum(int(y[i] == K) for i in outs)
            if arm == 'tralo_final':
                net.append(sum(int(y[i] == K) for i in ins) - sum(int(y[i] == K) for i in outs))
        excess.append(s['steps']['tralo_final']['hard_before'] - CAP)
        p2.append(100 * (s['arms']['tralo_final']['scores']['capped_first']['cc_f1']
                         - s['arms']['sham_final']['scores']['capped_first']['cc_f1']))
    print(f'{len(net)} seeds')
    for arm, (n, i, o) in swaps.items():
        if n:
            print(f'  {arm:11s} swapped {n} slots; in true-3 {i}/{n} = {i / n:.3f}; out true-3 {o}/{n} = {o / n:.3f}; '
                  f'correct-direction share {(i + n - o) / (2 * n):.3f}; net {i - o:+d}')
        else:
            print(f'  {arm:11s} swapped 0 slots')
    excess, p2 = np.array(excess), np.array(p2)
    b = excess > 0
    print(f'  net slots per seed {np.mean(net):+.2f}; P2 vs excess over the cap (binding, n {b.sum()}): '
          f'Spearman {stats.spearmanr(excess[b], p2[b]).statistic:+.2f} p {stats.spearmanr(excess[b], p2[b]).pvalue:.3f}')


if __name__ == '__main__':
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    main(sys.argv[1])
