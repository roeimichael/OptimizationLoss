"""Arm-vs-arm contrasts on the snapshot ensemble (mean of epoch 06-10 after_constraint dev probabilities).

python analysis/ens_contrasts.py RUN_ROOT NAME=a:b [NAME=a:b ...]

Correct capped_first slots and grade-3 F1 (points) per contrast, paired t over seeds, Holm over the
listed contrasts. Offline, development labels only.
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tralo.global_clipper import allocate  # noqa: E402

GRADE = 3


def main():
    root = Path(sys.argv[1])
    contrasts = [(c.split('=')[0], *c.split('=')[1].split(':')) for c in sys.argv[2:]]
    arms = {a for _, x, y in contrasts for a in (x, y)}
    slots, seeds, cap, n3 = {a: [] for a in arms}, [], None, None
    for d in sorted(root.glob('seed*')):
        if not d.is_dir() or not (d / 'summary.json').exists():
            continue
        caps = json.loads((d / 'config.json').read_text())['caps']
        cap = caps[GRADE]
        rows = [r for r in json.loads((d / 'manifest.json').read_text())['rows'] if r['split'] == 'val']
        labels, ids = np.array([r['label'] for r in rows]), [r['sample_id'] for r in rows]
        n3 = int((labels == GRADE).sum())
        seeds.append(int(d.name[4:]))
        for arm in arms:
            snaps = [torch.load(d / arm / ('epoch%02d_after_constraint.pt' % e), weights_only=True).double()
                     for e in range(6, 11)]
            pred = np.array(allocate(torch.stack(snaps).mean(0).tolist(), caps, ids, 'capped_first'))
            slots[arm].append(int(((pred == GRADE) & (labels == GRADE)).sum()))
    print('seeds %d, cap %s: %s' % (len(seeds), cap, seeds))
    rows = []
    for name, a, b in contrasts:
        v = np.array(slots[a], float) - np.array(slots[b], float)
        h = stats.t.ppf(.975, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v))
        rows.append((name, a, b, v, h, stats.ttest_1samp(v, 0).pvalue))
    order = sorted(range(len(rows)), key=lambda i: rows[i][5])
    holm, running = [0.0] * len(rows), 0.0
    for k, i in enumerate(order):
        running = max(running, min(1.0, (len(rows) - k) * rows[i][5]))
        holm[i] = running
    f1 = 200.0 / (n3 + cap)
    for (name, a, b, v, h, p), ph in zip(rows, holm):
        print('  %-7s %-13s - %-13s %+5.2f [%+5.2f, %+5.2f] slots = %+5.2f F1  p=%.4f Holm=%.4f  W/T/L %d/%d/%d'
              % (name, a, b, v.mean(), v.mean() - h, v.mean() + h, f1 * v.mean(), p, ph,
                 (v > 0).sum(), (v == 0).sum(), (v < 0).sum()))


if __name__ == '__main__':
    main()
