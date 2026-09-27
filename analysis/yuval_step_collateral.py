"""Where does TraLO's step move the non-capped predictions? (post hoc, development labels, offline)

Usage: python analysis/yuval_step_collateral.py RUN_ROOT

Per arm of a tralo.knee_yuval run: per-class capped_first F1 averaged over seeds, and for tralo_final and
sham_final the predictions that differ from pto, counted by (pto class -> new class).
"""

import json
from pathlib import Path
import sys

import numpy as np

ARMS = ('pto', 'tralo_final', 'sham_final')


def main(root):
    f1 = {a: [] for a in ARMS}
    moves = {a: np.zeros((5, 5), int) for a in ARMS[1:]}
    for d in sorted(Path(root).glob('seed*')):
        if not (d / 'summary.json').exists():
            continue
        y = np.array([r['label'] for r in json.loads((d / 'manifest.json').read_text())['rows'] if r['split'] == 'val'])
        preds = {a: np.array(json.loads((d / a / 'report.json').read_text())['capped_first']['predictions']) for a in ARMS}
        for a, p in preds.items():
            f1[a].append([2 * ((p == c) & (y == c)).sum() / max(1, (p == c).sum() + (y == c).sum()) for c in range(5)])
        for a in moves:
            changed = preds[a] != preds['pto']
            np.add.at(moves[a], (preds['pto'][changed], preds[a][changed]), 1)
    mean = {a: 100 * np.mean(v, 0) for a, v in f1.items()}
    print(f"{len(f1['pto'])} seeds; per-class capped_first F1 (%), grades 0-4")
    for a in ARMS:
        print(f'  {a:12s} ' + ' '.join(f'{x:6.2f}' for x in mean[a]))
    print('  tralo - sham ' + ' '.join(f'{x:+6.2f}' for x in mean['tralo_final'] - mean['sham_final']))
    for a, t in moves.items():
        top = sorted(np.ndindex(5, 5), key=lambda ij: -t[ij])[:6]
        print(f'  {a}: {t.sum()} predictions differ from pto ({t.sum() / len(f1["pto"]):.1f} per seed); largest moves: '
              + ', '.join(f'{i}->{j} {t[i, j]}' for i, j in top if t[i, j]))


if __name__ == '__main__':
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    main(sys.argv[1])
