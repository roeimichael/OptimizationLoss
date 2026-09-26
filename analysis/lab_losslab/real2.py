"""Offline, finished knee runs, dev 'val' labels only: does the per-step WHO gain SURVIVE the next epoch of CE?

For each seed and arm (tralo_target, sham_target): correct grade-3 items in the top-cap (capped_first slots)
at every checkpoint: before_k, after_k, before_{k+1}, ..., final. Reports
  step gain   = correct(after_k) - correct(before_k)
  epoch decay = correct(before_{k+1}) - correct(after_k)
  target - sham at every checkpoint and at final (paired by seed).
"""
import json, sys
from pathlib import Path
import numpy as np
import torch
import lab

G = 3


def correct(p, y, cap):
    s = np.lexsort((np.arange(len(p)), -p[:, G].double().numpy()))[:cap]
    return int((y[s] == G).sum())


for root in sys.argv[1:]:
    per = {}
    for d in sorted(Path(root).glob('seed*')):
        if not (d / 'summary.json').exists():
            continue
        cap = json.loads((d / 'config.json').read_text())['caps'][G]
        val = [r for r in json.loads((d / 'manifest.json').read_text())['rows'] if r['split'] == 'val']
        y = np.array([r['label'] for r in val])
        for arm in ('tralo_target', 'sham_target', 'clipper', 'tralo_null'):
            seq = []
            for ep in range(1, 31):
                for tag in ('before', 'after'):
                    f = d / arm / f'epoch{ep:02d}_{tag}_constraint.pt'
                    if f.exists():
                        seq.append(('%s%02d' % (tag[0], ep), correct(torch.load(f, weights_only=True), y, cap)))
            f = d / arm / 'final_probabilities.pt'
            if f.exists():
                fp = torch.load(f, weights_only=True)
                if isinstance(fp, dict):
                    fp = fp.get('val', fp.get('development', None))
                if fp is not None and fp.shape[0] == len(y):
                    seq.append(('final', correct(fp, y, cap)))
            per.setdefault(arm, {})[d.name] = dict(seq)
    print('===', root, {a: len(v) for a, v in per.items()})
    T, S = per.get('tralo_target', {}), per.get('sham_target', {})
    steps, decays = [], []
    for s, seq in T.items():
        for ep in range(1, 31):
            b, a, nb = seq.get('b%02d' % ep), seq.get('a%02d' % ep), seq.get('b%02d' % (ep + 1))
            if b is not None and a is not None:
                steps.append(a - b)
                if nb is not None:
                    decays.append(nb - a)
    print('  target per-step gain  ', lab.mean_ci(steps))
    print('  target next-epoch decay', lab.mean_ci(decays))
    common = sorted(set(T) & set(S))
    keys = sorted({k for s in common for k in T[s]} & {k for s in common for k in S[s]}, key=lambda k: (k == 'final', k[1:], k[0]))
    for k in keys:
        v = lab.paired([T[s].get(k, np.nan) for s in common], [S[s].get(k, np.nan) for s in common])
        if 'mean' in v:
            print('  target-sham %-6s %+.3f [%+.3f,%+.3f] n=%d W/L/T %d/%d/%d' % (k, v['mean'], v['lo'], v['hi'], v['n'], v['wins'], v['losses'], v['ties']))
    for arm in ('clipper', 'tralo_null'):
        if arm in per and common:
            ks = [k for k in keys if k == 'final']
            for k in ks:
                v = lab.paired([T[s].get(k, np.nan) for s in common], [per[arm].get(s, {}).get(k, np.nan) for s in common])
                if 'mean' in v:
                    print('  target-%s %-6s %+.3f [%+.3f,%+.3f] n=%d' % (arm, k, v['mean'], v['lo'], v['hi'], v['n']))
