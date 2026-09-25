"""Does the constraint change WHICH items occupy the capped slots, beyond what a reseed changes?

Allocator-free, score-level measure: for each capped class c, the slot set S_A(c) is the
K_c development items with the highest p(c) under arm A. This is what any exact-K
allocator starts from, and it does not depend on the rebuild's allocator code.

For every (dataset, setting, seed) against the SAME-SEED common null:
  turnover  = 1 - |S_arm & S_null| / K        (fraction of slots that changed occupant)
  d_correct = #true-c items in S_arm - #true-c items in S_null   (net correct slots gained)
Reference floors, all computed the same way:
  reseed    = null(seed s) vs null(seed s')   (pure seed noise, no intervention)
  schedule  = clipper vs null, same seed      (the phase-boundary optimizer reset alone)
Also: count / soft-count against the cap, and where exiting/entering items sat in the
null ranking (near the cut = a re-ordering at the margin; far = a real re-ranking).
"""
import os, json, sys, itertools, collections
import numpy as np
from scipy import stats

ROOT = sys.argv[1]
CAPS = {'knee': {3: 82, 4: 16}, 'cifar': {c: 10 for c in range(10)}}

runs = {}
for dp, dn, fn in os.walk(ROOT):
    if 'predictions.json' not in fn:
        continue
    rel = os.path.relpath(dp, ROOT).replace(os.sep, '/')
    grp, leaf = rel.split('/')[0], rel.split('/')[-1]
    seed, arm = leaf.split('_', 1)
    ds = 'knee' if '_knee_' in grp else 'cifar'
    setting = grp.split('_' + ds + '_', 1)[1]
    if setting == 'controls':
        setting = arm
    p = json.load(open(os.path.join(dp, 'predictions.json')))
    order = np.argsort(np.array(p['sample_ids']))
    runs[(ds, setting, int(seed))] = (np.array(p['labels'])[order], np.array(p['probabilities'], dtype=np.float64)[order])


def slots(prob, c, k):
    # stable: ties broken by sample index
    return set(np.lexsort((np.arange(len(prob)), -prob[:, c]))[:k].tolist())


def compare(a, b, ds):
    ya, pa = a
    yb, pb = b
    assert (ya == yb).all()
    out = []
    for c, k in CAPS[ds].items():
        sa, sb = slots(pa, c, k), slots(pb, c, k)
        ca = sum(ya[i] == c for i in sa)
        cb = sum(yb[i] == c for i in sb)
        out.append((c, 1 - len(sa & sb) / k, ca - cb))
    return out


def summarise(rows):
    t = np.array([r[1] for r in rows]); d = np.array([r[2] for r in rows])
    return t.mean(), d.mean(), d


report = collections.OrderedDict()
for ds in ('knee', 'cifar'):
    seeds = sorted({s for (d, st, s) in runs if d == ds and st == 'tralo_null'})
    settings = sorted({st for (d, st, s) in runs if d == ds})
    null = {s: runs[(ds, 'tralo_null', s)] for s in seeds}
    # reseed floor
    rs = [r for s1, s2 in itertools.combinations(seeds, 2) for r in compare(null[s1], null[s2], ds)]
    rs_t, rs_d, _ = summarise(rs)
    print('\n' + '=' * 96)
    print('%s  | %d seeds | caps %s' % (ds.upper(), len(seeds), CAPS[ds]))
    y = null[seeds[0]][0]
    for c, k in CAPS[ds].items():
        hard = np.mean([(null[s][1].argmax(1) == c).sum() for s in seeds])
        soft = np.mean([null[s][1][:, c].sum() for s in seeds])
        print('  class %d: cap %3d | true %3d | null hard count %.1f | null soft count %.1f' % (c, k, (y == c).sum(), hard, soft))
    print('  RESEED FLOOR (null s vs null s\'): slot turnover %.3f' % rs_t)
    print('  %-14s %9s %9s %12s %22s %9s' % ('arm', 'turnover', 'x floor', 'd_correct', '95% t CI (per-seed sum)', 'W/T/L'))
    for st in ['clipper'] + [x for x in settings if x not in ('clipper', 'tralo_null')]:
        per_seed = []
        tv = []
        for s in seeds:
            rows = compare(runs[(ds, st, s)], null[s], ds)
            tv += [r[1] for r in rows]
            per_seed.append(sum(r[2] for r in rows))
        per_seed = np.array(per_seed, dtype=float)
        m = per_seed.mean(); sd = per_seed.std(ddof=1)
        h = stats.t.ppf(0.975, len(per_seed) - 1) * sd / np.sqrt(len(per_seed)) if sd > 0 else 0.0
        w, ti, l = (per_seed > 0).sum(), (per_seed == 0).sum(), (per_seed < 0).sum()
        print('  %-14s %9.3f %9.2f %+12.2f   [%+7.2f, %+7.2f]      %d/%d/%d'
              % (st, np.mean(tv), np.mean(tv) / rs_t, m, m - h, m + h, w, ti, l))
        report[(ds, st)] = dict(turnover=float(np.mean(tv)), floor=float(rs_t), d_correct=per_seed.tolist())

json.dump({'%s|%s' % k: v for k, v in report.items()}, open(os.path.join(os.path.dirname(__file__), 'slot_analysis_sweep.json'), 'w'), indent=1)
