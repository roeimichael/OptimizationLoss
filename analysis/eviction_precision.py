"""Offline (development labels, after training): WHO does a constraint step evict?

For every applied targeted step (tralo_target and sham_target), compare the grade-3 items the
step removes from the raw argmax with the items capped_first would have removed from the SAME
pre-step probabilities. That post-hoc cut is the baseline any step has to beat. It reports
the fraction of evicted items that were NOT truly grade 3 (eviction precision, higher is
better) and the net change in correct grade-3 slots under capped_first across the step.

python analysis/eviction_precision.py RUN_ROOT
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tralo.global_clipper import allocate  # noqa: E402

GRADE = 3


def main():
    root = Path(sys.argv[1])
    rows = {}
    for d in sorted(root.glob('seed*')):
        if not (d / 'summary.json').exists():
            continue
        config = json.loads((d / 'config.json').read_text())
        caps = config['caps']
        manifest = json.loads((d / 'manifest.json').read_text())
        val = [r for r in manifest['rows'] if r['split'] == 'val']
        labels = np.array([r['label'] for r in val])
        ids = [r['sample_id'] for r in val]
        for arm in ('tralo_target', 'sham_target', 'tralo_adam'):
            for line in open(d / arm / 'events.jsonl'):
                e = json.loads(line)
                if e.get('event') != 'epoch' or not e.get('constraint_applied'):
                    continue
                ep = e['epoch']
                pb = torch.load(d / arm / f'epoch{ep:02d}_before_constraint.pt', weights_only=True)
                pa = torch.load(d / arm / f'epoch{ep:02d}_after_constraint.pt', weights_only=True)
                rb, ra = pb.argmax(1).numpy(), pa.argmax(1).numpy()
                step_out = np.flatnonzero((rb == GRADE) & (ra != GRADE))
                step_in = np.flatnonzero((rb != GRADE) & (ra == GRADE))
                cb = np.array(allocate(pb.tolist(), caps, ids, 'capped_first'))
                ca = np.array(allocate(pa.tolist(), caps, ids, 'capped_first'))
                post_out = np.flatnonzero((rb == GRADE) & (cb != GRADE))
                r = rows.setdefault(arm, dict(steps=0, step_out=0, step_out_wrong=0, step_in=0, step_in_right=0,
                                              post_out=0, post_out_wrong=0, overlap=0, d_correct_capped=[]))
                r['steps'] += 1
                r['step_out'] += len(step_out)
                r['step_out_wrong'] += int((labels[step_out] != GRADE).sum())
                r['step_in'] += len(step_in)
                r['step_in_right'] += int((labels[step_in] == GRADE).sum())
                r['post_out'] += len(post_out)
                r['post_out_wrong'] += int((labels[post_out] != GRADE).sum())
                r['overlap'] += len(set(step_out) & set(post_out))
                r['d_correct_capped'].append(int(((ca == GRADE) & (labels == GRADE)).sum())
                                             - int(((cb == GRADE) & (labels == GRADE)).sum()))
    base = None
    for arm, r in rows.items():
        d = np.array(r['d_correct_capped'])
        print('%-12s steps=%d | step evicts %d (%.1f%% wrong-grade), admits %d (%.1f%% true 3) | '
              'post-hoc cut evicts %d (%.1f%% wrong-grade) | overlap %.1f%% | capped correct slots per step %+.2f (sd %.2f)'
              % (arm, r['steps'], r['step_out'], 100 * r['step_out_wrong'] / max(r['step_out'], 1),
                 r['step_in'], 100 * r['step_in_right'] / max(r['step_in'], 1),
                 r['post_out'], 100 * r['post_out_wrong'] / max(r['post_out'], 1),
                 100 * r['overlap'] / max(r['step_out'], 1), d.mean(), d.std(ddof=1)))
        if base is None:
            base = labels
    if base is not None:
        print('base rate: %.1f%% of development items are not grade 3' % (100 * (base != GRADE).mean()))


if __name__ == '__main__':
    main()
