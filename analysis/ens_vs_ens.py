import json, sys, glob
from pathlib import Path
import numpy as np, torch
from scipy import stats
from tralo.global_clipper import allocate
def corr(p, caps, ids, lab):
    pred = np.array(allocate(p.tolist(), caps, ids, 'capped_first')); return int(((pred == 3) & (lab == 3)).sum())
for root, pairs in [('claude-target-20260925', [('tralo_target','sham_target'),('tralo_target','clipper'),('tralo_null','clipper'),('tralo_adam','clipper')]),
                    ('claude-target50-20260926', [('tralo_target','sham_target'),('tralo_target','clipper'),('tralo_null','clipper'),('tralo_adam','clipper')])]:
    E, B = {}, {}
    for d in sorted(Path.home().glob('tralo-rebuild/runs/%s/seed*' % root)):
        if not d.is_dir() or not (d / 'summary.json').exists(): continue
        caps = json.loads((d / 'config.json').read_text())['caps']
        rows = [r for r in json.loads((d / 'manifest.json').read_text())['rows'] if r['split'] == 'val']
        lab, ids = np.array([r['label'] for r in rows]), [r['sample_id'] for r in rows]
        for arm in {a for p in pairs for a in p}:
            s = [torch.load(d / arm / ('epoch%02d_after_constraint.pt' % e), weights_only=True).double() for e in range(6, 11)]
            E.setdefault(arm, []).append(corr(torch.stack(s).mean(0), caps, ids, lab)); B.setdefault(arm, []).append(corr(s[-1], caps, ids, lab))
    print(root)
    for a, b in pairs:
        for name, X in (('ENS', E), ('BASE', B)):
            v = np.array(X[a], float) - np.array(X[b], float); h = stats.t.ppf(.975, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v))
            print('  %-4s %-12s - %-11s %+5.2f [%+5.2f, %+5.2f] slots p=%.3f' % (name, a, b, v.mean(), v.mean() - h, v.mean() + h, stats.ttest_1samp(v, 0).pvalue))
