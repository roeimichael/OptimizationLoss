"""Score the preregistered controller/sham study exactly as fixed in
experiments/claude_controller_sham_protocol_20260925.md.

python analysis/score_sham.py RUN_ROOT [--json OUT]
"""
import itertools
import json
import math
import sys
from pathlib import Path

import numpy as np
from scipy import stats

ARMS = ('clipper', 'tralo_null', 'tralo_adam', 'tralo_sgd', 'sham_sgd')
PRIMARY = [('C1', 'tralo_sgd', 'tralo_null'), ('C2', 'tralo_sgd', 'sham_sgd')]
SECONDARY = [('tralo_adam', 'tralo_null'), ('sham_sgd', 'tralo_null'), ('clipper', 'tralo_null'),
             ('tralo_adam', 'sham_sgd'), ('tralo_sgd', 'tralo_adam')]
GRADE, CAP = 3, 76


def load(root):
    seeds = {}
    for d in sorted(Path(root).glob('seed*')):
        if not d.is_dir() or not (d / 'summary.json').exists():
            continue
        summary = {r['arm']: r for r in json.loads((d / 'summary.json').read_text())}
        if set(summary) != set(ARMS):
            continue
        arms = {}
        for arm in ARMS:
            rep = json.loads((d / arm / 'report.json').read_text())['capped_first']
            pred = np.array(rep['predictions'])
            cm = np.array(rep['metrics']['confusion'])
            arms[arm] = dict(f1=100 * rep['metrics']['cc_f1'], tp=int(cm[GRADE, GRADE]),
                             slots=set(np.flatnonzero(pred == GRADE).tolist()), summary=summary[arm])
        seeds[int(d.name[4:])] = arms
    return seeds


def paired(seeds, a, b, key):
    d = np.array([seeds[s][a][key] - seeds[s][b][key] for s in sorted(seeds)], float)
    n = len(d)
    m = d.mean()
    sd = d.std(ddof=1) if n > 1 else float('nan')
    if n > 1 and sd > 0:
        h = stats.t.ppf(.975, n - 1) * sd / math.sqrt(n)
        p = float(2 * stats.t.sf(abs(m) / (sd / math.sqrt(n)), n - 1))
    else:
        h, p = float('nan'), float('nan')
    return dict(n=n, mean=m, sd=sd, lo=m - h, hi=m + h, p=p,
                wins=int((d > 0).sum()), ties=int((d == 0).sum()), losses=int((d < 0).sum()), deltas=d.tolist())


def holm(ps):
    order = sorted(range(len(ps)), key=lambda i: ps[i])
    adj, running = [0.0] * len(ps), 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(ps) - rank) * ps[i]))
        adj[i] = running
    return adj


def turnover(a, b):
    return 1 - len(a & b) / CAP


def main():
    root = sys.argv[1]
    seeds = load(root)
    out = {'n_seeds': len(seeds), 'seeds': sorted(seeds)}
    print('complete seeds: %d  %s' % (len(seeds), sorted(seeds)))
    if len(seeds) < 2:
        print('need >= 2 complete seeds')
        return
    print('\nINTEGRITY (per seed)')
    integ = []
    for s in sorted(seeds):
        a = seeds[s]
        ids = {(a[x]['summary']['warmup_sha256'], a[x]['summary']['batch_sha256']) for x in ARMS}
        disp = {x: a[x]['summary']['parameter_displacements'] for x in ('tralo_adam', 'tralo_sgd', 'sham_sgd')}
        row = dict(seed=s, matched=len(ids) == 1,
                   binds=all(a[x]['summary']['cap_binds_on_hard_count'] for x in ARMS),
                   null_first_hard=a['tralo_null']['summary']['first_constraint_hard_counts'][GRADE],
                   steps={x: a[x]['summary']['constraint_updates_applied'] for x in disp},
                   sham_eq_sgd=abs(disp['sham_sgd'][0] - disp['tralo_sgd'][0]) / disp['tralo_sgd'][0] < 1e-4,
                   sgd_eq_adam=abs(disp['tralo_sgd'][0] - disp['tralo_adam'][0]) / disp['tralo_adam'][0] < 1e-3,
                   first_disp={x: round(v[0], 6) for x, v in disp.items()},
                   total_disp={x: round(sum(v), 5) for x, v in disp.items()})
        integ.append(row)
        print('  %d matched=%s binds=%s null_hard=%d steps=%s sham==sgd:%s sgd==adam:%s total_disp=%s'
              % (s, row['matched'], row['binds'], row['null_first_hard'], row['steps'],
                 row['sham_eq_sgd'], row['sgd_eq_adam'], row['total_disp']))
    out['integrity'] = integ

    print('\nPRIMARY (Holm over C1, C2) -- capped_first grade-3 F1, points; correct slots of %d' % CAP)
    prim = [paired(seeds, a, b, 'f1') for _, a, b in PRIMARY]
    adj = holm([r['p'] for r in prim])
    out['primary'] = {}
    for (name, a, b), r, pa in zip(PRIMARY, prim, adj):
        tp = paired(seeds, a, b, 'tp')
        print('  %s %-10s - %-10s  %+6.2f [%+6.2f, %+6.2f]  p=%.4f  Holm=%.4f  W/T/L %d/%d/%d | slots %+5.2f [%+5.2f, %+5.2f]'
              % (name, a, b, r['mean'], r['lo'], r['hi'], r['p'], pa, r['wins'], r['ties'], r['losses'],
                 tp['mean'], tp['lo'], tp['hi']))
        out['primary'][name] = dict(a=a, b=b, f1=r, slots=tp, holm_p=pa)

    print('\nSECONDARY (t intervals, not in the family)')
    out['secondary'] = {}
    for a, b in SECONDARY:
        r = paired(seeds, a, b, 'f1')
        print('  %-10s - %-10s  %+6.2f [%+6.2f, %+6.2f]  p=%.4f  W/T/L %d/%d/%d'
              % (a, b, r['mean'], r['lo'], r['hi'], r['p'], r['wins'], r['ties'], r['losses']))
        out['secondary']['%s-%s' % (a, b)] = r

    print('\nARM MEANS (capped_first grade-3 F1)')
    for arm in ARMS:
        v = [seeds[s][arm]['f1'] for s in sorted(seeds)]
        print('  %-10s %6.2f  (sd %.2f)' % (arm, np.mean(v), np.std(v, ddof=1)))

    print('\nSLOT TURNOVER vs the null (same seed), against the null reseed floor')
    ss = sorted(seeds)
    floor = np.mean([turnover(seeds[x]['tralo_null']['slots'], seeds[y]['tralo_null']['slots'])
                     for x, y in itertools.combinations(ss, 2)])
    print('  reseed floor (null s vs null s\'): %.3f' % floor)
    out['turnover'] = {'floor': floor}
    for arm in ARMS:
        if arm == 'tralo_null':
            continue
        t = np.mean([turnover(seeds[s][arm]['slots'], seeds[s]['tralo_null']['slots']) for s in ss])
        print('  %-10s %.3f  (%.2fx floor)' % (arm, t, t / floor))
        out['turnover'][arm] = t
    if len(sys.argv) > 3 and sys.argv[2] == '--json':
        Path(sys.argv[3]).write_text(json.dumps(out, indent=1, default=float))


if __name__ == '__main__':
    main()
