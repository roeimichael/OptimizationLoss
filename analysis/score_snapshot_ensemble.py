"""Score the preregistered snapshot-ensemble confirmation (experiments/claude_snapshot_ensemble_prereg_20260926.md).

python analysis/score_snapshot_ensemble.py RUN_ROOT [--exclude SEED ...]

ENS  = capped_first on the mean of epoch06..10 after_constraint dev probabilities.
BASE = capped_first on final_probabilities.pt (equal to the epoch-10 snapshot).
Primary: correct capped slots ENS - BASE for clipper and tralo_null, paired t, Holm over the two.
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
PRIMARY = ('clipper', 'tralo_null')


def correct(p, caps, ids, labels):
    pred = np.array(allocate(p.tolist(), caps, ids, 'capped_first'))
    return int(((pred == GRADE) & (labels == GRADE)).sum())


def main():
    root, exclude = Path(sys.argv[1]), set()
    if '--exclude' in sys.argv:
        exclude = {int(s) for s in sys.argv[sys.argv.index('--exclude') + 1:]}
    diffs, seeds, cap = {}, [], None
    for d in sorted(root.glob('seed*')):
        if not d.is_dir() or not (d / 'summary.json').exists() or int(d.name[4:]) in exclude:
            continue
        caps = json.loads((d / 'config.json').read_text())['caps']
        cap = caps[GRADE]
        rows = [r for r in json.loads((d / 'manifest.json').read_text())['rows'] if r['split'] == 'val']
        labels, ids = np.array([r['label'] for r in rows]), [r['sample_id'] for r in rows]
        seeds.append(int(d.name[4:]))
        for arm in [r['arm'] for r in json.loads((d / 'summary.json').read_text())]:
            a = d / arm
            final = torch.load(a / 'final_probabilities.pt', weights_only=True).double()
            snaps = [torch.load(a / ('epoch%02d_after_constraint.pt' % e), weights_only=True).double() for e in range(6, 11)]
            if not torch.equal(snaps[-1], final):
                raise ValueError('%s/%s: final is not the epoch-10 snapshot' % (d.name, arm))
            ens = torch.stack(snaps).mean(0)
            diffs.setdefault(arm, []).append(correct(ens, caps, ids, labels) - correct(final, caps, ids, labels))
    print('seeds %d (cap %s): %s' % (len(seeds), cap, seeds))
    if exclude:
        print('excluded: %s' % sorted(exclude))
    rows = []
    for arm, v in diffs.items():
        v = np.array(v, float)
        h = stats.t.ppf(.975, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v))
        rows.append((arm, v.mean(), v.mean() - h, v.mean() + h, stats.ttest_1samp(v, 0).pvalue,
                     (v > 0).sum(), (v == 0).sum(), (v < 0).sum()))
    prim = [r for r in rows if r[0] in PRIMARY]
    ps = sorted((r[4], r[0]) for r in prim)
    holm, running = {}, 0.0
    for k, (p, arm) in enumerate(ps):
        running = max(running, min(1.0, (len(ps) - k) * p))
        holm[arm] = running
    for arm, m, lo, hi, p, w, t, l in rows:
        tag = 'PRIMARY Holm=%.4f' % holm[arm] if arm in holm else 'secondary'
        print('  %-18s ENS-BASE %+5.2f [%+5.2f, %+5.2f] slots  p=%.4f  W/T/L %d/%d/%d  %s' % (arm, m, lo, hi, p, w, t, l, tag))


if __name__ == '__main__':
    main()
