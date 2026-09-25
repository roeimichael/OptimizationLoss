"""Slot-level instrument on augfin (local + global policy caps, fmow2, deployed allocations).

Slot set of an arm for capped class c = items whose DEPLOYED label is c (the allocator
already enforced every per-group and global ceiling). Compare arms at the same seed and
against the reseed floor of each arm's own seed-to-seed turnover.
"""
import glob, os, itertools, collections
import numpy as np
import pandas as pd
from scipy import stats

BASE = os.path.expanduser('~/optloss-rank')
CAP = [1, 2, 7]
runs = {}
for f in glob.glob(BASE + '/augfin_[ab]/*/*/*/*/*/final_predictions.csv'):
    p = f.split('/')
    cap, arm, seed = p[-4], p[-3], int(p[-2].split('_')[1])
    d = pd.read_csv(f)
    runs[(cap, arm, seed)] = (d['True_Label'].to_numpy(), d['Predicted_Label'].to_numpy())

def sets(y_pred):
    return {c: set(np.flatnonzero(y_pred == c).tolist()) for c in CAP}

def turnover(a, b):
    sa, sb = sets(a[1]), sets(b[1])
    k = sum(len(sa[c]) for c in CAP)
    return 1 - sum(len(sa[c] & sb[c]) for c in CAP) / k

def correct(a):
    y, yp = a
    return sum(int(((yp == c) & (y == c)).sum()) for c in CAP)

for cap in sorted({k[0] for k in runs}):
    arms = sorted({k[1] for k in runs if k[0] == cap})
    seeds = sorted({k[2] for k in runs if k[0] == cap})
    print('\n' + '=' * 90)
    print(cap, '| seeds', len(seeds), '| slots filled per run:', sum(len(v) for v in sets(runs[(cap, arms[0], seeds[0])][1]).values()))
    floors = {}
    for a in arms:
        floors[a] = np.mean([turnover(runs[(cap, a, s1)], runs[(cap, a, s2)]) for s1, s2 in itertools.combinations(seeds, 2)])
    print('  reseed floor per arm: ' + ', '.join('%s %.3f' % (a, floors[a]) for a in arms))
    ref_floor = floors['aug_tralo_null']
    print('  %-16s %-16s %9s %8s %10s %24s %8s' % ('arm', 'vs', 'turnover', 'x floor', 'd_correct', '95% t CI', 'W/T/L'))
    for a, b in [('aug_tralo', 'aug_tralo_null'), ('aug_tralo', 'aug_clip'), ('aug_tralo_null', 'aug_clip'),
                 ('aug_tralo_stab', 'aug_tralo_null'), ('aug_clip', 'clip'), ('tralo_null', 'clip')]:
        tv = [turnover(runs[(cap, a, s)], runs[(cap, b, s)]) for s in seeds]
        dc = np.array([correct(runs[(cap, a, s)]) - correct(runs[(cap, b, s)]) for s in seeds], float)
        m, sd = dc.mean(), dc.std(ddof=1)
        h = stats.t.ppf(.975, len(dc) - 1) * sd / np.sqrt(len(dc))
        print('  %-16s %-16s %9.3f %8.2f %+10.2f   [%+7.2f, %+7.2f]   %d/%d/%d'
              % (a, b, np.mean(tv), np.mean(tv) / ref_floor, m, m - h, m + h, (dc > 0).sum(), (dc == 0).sum(), (dc < 0).sum()))
