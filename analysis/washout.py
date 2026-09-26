"""Offline (development labels, finished runs only): does the next task epoch wash out a constraint step?

For each targeted-step run root and arm, correct capped slots (capped_first grade-3 true positives) at:
  B_e = before the constraint step of epoch e, A_e = right after it.
The step's own effect is A_e - B_e; the task epoch that follows moves the model from A_e to B_{e+1}.
Reports, per arm and per epoch, the paired-with-null difference at B and A, and how much of the step's
gain survives to the next pre-step snapshot: (B_{e+1}(arm) - B_{e+1}(null)) vs (A_e(arm) - A_e(null)).

python analysis/washout.py RUN_ROOT
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(sys.argv[0]).resolve().parent))
from tralo.global_clipper import allocate  # noqa: E402

GRADE = 3


def correct(probs, labels, caps, ids):
    pred = np.array(allocate(probs.tolist(), caps, ids, 'capped_first'))
    return int(((pred == GRADE) & (labels == GRADE)).sum())


def main():
    root = Path(sys.argv[1])
    table = {}
    for d in sorted(root.glob('seed*')):
        if not (d / 'summary.json').exists():
            continue
        caps = json.loads((d / 'config.json').read_text())['caps']
        rows = [r for r in json.loads((d / 'manifest.json').read_text())['rows'] if r['split'] == 'val']
        labels, ids = np.array([r['label'] for r in rows]), [r['sample_id'] for r in rows]
        for arm in ('tralo_null', 'tralo_target', 'sham_target', 'tralo_adam'):
            for e in range(6, 11):
                for phase, key in (('before', 'B'), ('after', 'A')):
                    p = torch.load(d / arm / f'epoch{e:02d}_{phase}_constraint.pt', weights_only=True)
                    table.setdefault((arm, key, e), []).append(correct(p, labels, caps, ids))
    n = len(table[('tralo_null', 'B', 6)])
    print('seeds:', n)
    for arm in ('tralo_target', 'sham_target', 'tralo_adam'):
        print('\n%s minus null, correct capped slots (mean [95%% CI])' % arm)
        for e in range(6, 11):
            out = []
            for key in ('B', 'A'):
                diff = np.array(table[(arm, key, e)]) - np.array(table[('tralo_null', key, e)])
                h = 1.96 * diff.std(ddof=1) / np.sqrt(n)
                out.append('%s %+5.2f [%+5.2f,%+5.2f]' % (key, diff.mean(), diff.mean() - h, diff.mean() + h))
            step = np.array(table[(arm, 'A', e)]) - np.array(table[(arm, 'B', e)])
            out.append('own step %+5.2f' % step.mean())
            print('  epoch %d  %s' % (e, '   '.join(out)))


if __name__ == '__main__':
    main()
