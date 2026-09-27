"""How noisy is the count Yuval's outer loop steers on? (descriptive, labels not read)

Usage: python analysis/yuval_count_noise.py RUN_ROOT

Per seed of a tralo.knee_yuval run: the development pool's argmax grade-3 count at every saved epoch of
every retrain, its spread, and the loop's trajectory (count and C per retrain).
"""

import json
from pathlib import Path
import sys

import numpy as np

CAP = 76


def main(root):
    spreads, jumps, lengths = [], [], []
    for d in sorted(Path(root).glob('seed*')):
        if not (d / 'summary.json').exists():
            continue
        s = json.loads((d / 'summary.json').read_text())
        line = []
        for r in s['retrains']:
            rows = [json.loads(x) for x in (d / f"retrain{r['retrain']}" / 'events.jsonl').read_text().splitlines()]
            counts = [x['hard_counts'][3] for x in rows if x['event'] == 'epoch']
            late = counts[2:] if len(counts) > 3 else counts
            spreads.append(float(np.std(late, ddof=1)))
            line.append(f"r{r['retrain']} C {r['C'][0]:.2f} best {r['best_epoch']}/{r['epochs_run']} count {r['hard_count']} "
                        f"(epochs {min(counts)}-{max(counts)})")
        counts = [r['hard_count'] for r in s['retrains']]
        jumps += [abs(b - a) for a, b in zip(counts, counts[1:])]
        lengths.append(len(counts))
        print(f"{s['seed']}: converged {s['converged']}; " + ' | '.join(line))
    print(f'\nwithin-retrain sd of the pool grade-3 count over epochs 3+: median {np.median(spreads):.1f} '
          f'(range {min(spreads):.1f}-{max(spreads):.1f}); cap {CAP}')
    if jumps:
        print(f'|count change| between consecutive retrains: median {np.median(jumps):.0f}, max {max(jumps)}')
    print(f'retrains per seed: {dict(zip(*np.unique(lengths, return_counts=True)))}')


if __name__ == '__main__':
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    main(sys.argv[1])
