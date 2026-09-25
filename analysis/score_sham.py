"""Score the preregistered controller/sham studies exactly as fixed in
experiments/claude_controller_sham_protocol_20260925.md (v1, arms *_sgd) and
experiments/claude_targeted_step_protocol_20260925.md (v3, arms *_target).
The study is detected from the arm names in summary.json.

python analysis/score_sham.py RUN_ROOT [--json OUT]
"""
import itertools
import json
import math
import sys
from pathlib import Path

import numpy as np
from scipy import stats

STUDIES = {
    'sgd': dict(arms=('clipper', 'tralo_null', 'tralo_adam', 'tralo_sgd', 'sham_sgd'),
                primary=[('C1', 'tralo_sgd', 'tralo_null'), ('C2', 'tralo_sgd', 'sham_sgd')],
                secondary=[('tralo_adam', 'tralo_null'), ('sham_sgd', 'tralo_null'), ('clipper', 'tralo_null'),
                           ('tralo_adam', 'sham_sgd'), ('tralo_sgd', 'tralo_adam')]),
    'target': dict(arms=('clipper', 'tralo_null', 'tralo_adam', 'tralo_target', 'sham_target'),
                   primary=[('C1', 'tralo_target', 'tralo_null'), ('C2', 'tralo_target', 'sham_target')],
                   secondary=[('tralo_adam', 'tralo_null'), ('sham_target', 'tralo_null'), ('clipper', 'tralo_null'),
                              ('tralo_target', 'tralo_adam')]),
}
GRADE, CAP = 3, 76


def load(root):
    seeds, study = {}, None
    for d in sorted(Path(root).glob('seed*')):
        if not d.is_dir() or not (d / 'summary.json').exists():
            continue
        summary = {r['arm']: r for r in json.loads((d / 'summary.json').read_text())}
        if study is None:
            study = STUDIES['target' if 'tralo_target' in summary else 'sgd']
        if set(summary) != set(study['arms']):
            continue
        arms = {}
        for arm in study['arms']:
            report = json.loads((d / arm / 'report.json').read_text())
            rep = report['capped_first']
            pred = np.array(rep['predictions'])
            cm = np.array(rep['metrics']['confusion'])
            arms[arm] = dict(f1=100 * rep['metrics']['cc_f1'], tp=int(cm[GRADE, GRADE]),
                             raw_f1=100 * report['raw']['metrics']['cc_f1'],
                             slots=set(np.flatnonzero(pred == GRADE).tolist()), summary=summary[arm])
        seeds[int(d.name[4:])] = arms
    return seeds, study


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


def integrity(s, a, arms):
    ids = {(a[x]['summary']['warmup_sha256'], a[x]['summary']['batch_sha256']) for x in arms}
    row = dict(seed=s, matched=len(ids) == 1,
               task_updates={a[x]['summary']['task_updates_applied'] for x in arms} == {1810},
               null_first_hard=a['tralo_null']['summary']['first_constraint_hard_counts'][GRADE])
    if 'tralo_sgd' in arms:
        disp = {x: a[x]['summary']['parameter_displacements'] for x in ('tralo_adam', 'tralo_sgd', 'sham_sgd')}
        row.update(binds=all(a[x]['summary']['cap_binds_on_hard_count'] for x in arms),
                   sham_eq_sgd=abs(disp['sham_sgd'][0] - disp['tralo_sgd'][0]) / disp['tralo_sgd'][0] < 1e-4,
                   sgd_eq_adam=abs(disp['tralo_sgd'][0] - disp['tralo_adam'][0]) / disp['tralo_adam'][0] < 1e-3)
    else:
        t, h = a['tralo_target']['summary']['targeted_steps'], a['sham_target']['summary']['targeted_steps']
        row.update(target_steps=sum(x['applied'] for x in t), sham_steps=sum(x['applied'] for x in h),
                   target_lands=all(x['hard_before'] > CAP and x['hard_after'] <= CAP for x in t if x['applied']),
                   sham_triggers=all(x['hard_before'] > CAP for x in h if x['applied']),
                   first_radius_equal=(not (t[0]['applied'] and h[0]['applied'])) or t[0]['radius'] == h[0]['radius'],
                   target_radii=[round(x.get('radius', 0.0), 5) for x in t],
                   sham_radii=[round(x.get('radius', 0.0), 5) for x in h],
                   sham_hard=[(x['hard_before'], x.get('hard_after')) for x in h])
    return row


def main():
    root = sys.argv[1]
    seeds, study = load(root)
    out = {'n_seeds': len(seeds), 'seeds': sorted(seeds)}
    print('complete seeds: %d  %s' % (len(seeds), sorted(seeds)))
    if not seeds:
        return
    arms, primary, secondary = study['arms'], study['primary'], study['secondary']
    print('\nINTEGRITY (per seed)')
    out['integrity'] = []
    for s in sorted(seeds):
        row = integrity(s, seeds[s], arms)
        out['integrity'].append(row)
        print('  ' + ' '.join('%s=%s' % kv for kv in row.items()))

    print('\nARM MEANS (grade-3 F1: capped_first / raw)')
    for arm in arms:
        v = [seeds[s][arm]['f1'] for s in sorted(seeds)]
        w = [seeds[s][arm]['raw_f1'] for s in sorted(seeds)]
        print('  %-12s %6.2f (sd %.2f)  raw %6.2f' % (arm, np.mean(v), np.std(v, ddof=1) if len(v) > 1 else 0, np.mean(w)))
    if len(seeds) < 2:
        print('need >= 2 complete seeds for contrasts')
        return

    print('\nPRIMARY (Holm over C1, C2) -- capped_first grade-3 F1, points; correct slots of %d' % CAP)
    prim = [paired(seeds, a, b, 'f1') for _, a, b in primary]
    adj = holm([r['p'] for r in prim])
    out['primary'] = {}
    for (name, a, b), r, pa in zip(primary, prim, adj):
        tp = paired(seeds, a, b, 'tp')
        print('  %s %-12s - %-12s  %+6.2f [%+6.2f, %+6.2f]  p=%.4f  Holm=%.4f  W/T/L %d/%d/%d | slots %+5.2f [%+5.2f, %+5.2f]'
              % (name, a, b, r['mean'], r['lo'], r['hi'], r['p'], pa, r['wins'], r['ties'], r['losses'],
                 tp['mean'], tp['lo'], tp['hi']))
        out['primary'][name] = dict(a=a, b=b, f1=r, slots=tp, holm_p=pa)

    print('\nRAW (unallocated) grade-3 F1, secondary')
    for name, a, b in primary:
        r = paired(seeds, a, b, 'raw_f1')
        print('  %s %-12s - %-12s  %+6.2f [%+6.2f, %+6.2f]  p=%.4f' % (name, a, b, r['mean'], r['lo'], r['hi'], r['p']))
        out['raw_' + name] = r

    print('\nSECONDARY (t intervals, not in the family)')
    out['secondary'] = {}
    for a, b in secondary:
        r = paired(seeds, a, b, 'f1')
        print('  %-12s - %-12s  %+6.2f [%+6.2f, %+6.2f]  p=%.4f  W/T/L %d/%d/%d'
              % (a, b, r['mean'], r['lo'], r['hi'], r['p'], r['wins'], r['ties'], r['losses']))
        out['secondary']['%s-%s' % (a, b)] = r

    print('\nSLOT TURNOVER vs the null (same seed), against the null reseed floor')
    ss = sorted(seeds)
    floor = np.mean([turnover(seeds[x]['tralo_null']['slots'], seeds[y]['tralo_null']['slots'])
                     for x, y in itertools.combinations(ss, 2)])
    print("  reseed floor (null s vs null s'): %.3f" % floor)
    out['turnover'] = {'floor': floor}
    for arm in arms:
        if arm == 'tralo_null':
            continue
        t = np.mean([turnover(seeds[s][arm]['slots'], seeds[s]['tralo_null']['slots']) for s in ss])
        print('  %-12s %.3f  (%.2fx floor)' % (arm, t, t / floor))
        out['turnover'][arm] = t
    if len(sys.argv) > 3 and sys.argv[2] == '--json':
        Path(sys.argv[3]).write_text(json.dumps(out, indent=1, default=float))


if __name__ == '__main__':
    main()
