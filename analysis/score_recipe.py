"""Score the recipe factorial (experiments/claude_recipe_factorial_prereg_20260927.md).

Usage: python analysis/score_recipe.py RUN_ROOT [V3_ROOT R18_ROOT]
       python analysis/score_recipe.py --gate RUN_ROOT SEED     (pilot gate: integrity only, no score)

RUN_ROOT holds seed<seed>_a<A>s<S>e<E>/ directories written by tralo.knee_yuval. Primary, cc-F1 of
grade 3 under capped_first, Holm over four: the main effects of augmentation (F-A), the balanced
sampler (F-S) and early stopping (F-E) on pto, and P2-pooled, the per-seed mean over the eight cells
of tralo_final - sham_final. Contrasts are formed within a seed and tested across the 24 seeds.
Development labels are read here, offline, and nowhere else.
"""

import hashlib
import itertools
import json
from pathlib import Path
import re
import sys

import numpy as np
from scipy import stats
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from score_yuval import CAP, METRICS, fmt, holm, load as load_yuval, metrics, paired  # noqa: E402
from tralo.global_report import evaluate_global  # noqa: E402

CELLS = tuple(itertools.product((1, 0), repeat=3))           # (augment, balanced, early_stop)
SWITCHES = ('A augment', 'S balanced', 'E early_stop')
ARMS = ('pto', 'tralo_final', 'sham_final')
STUDY = range(4200, 4224)
JOB = re.compile(r'seed(\d+)_a([01])s([01])e([01])$')


def load(d):
    summary = json.loads((d / 'summary.json').read_text())
    if len(summary['retrains']) != 1 or (d / 'pao').exists() or (d / 'retrain2').exists():
        raise RuntimeError(f'{d.name}: the factorial trains PTO once')
    events = [json.loads(x) for x in (d / 'events.jsonl').read_text().splitlines()]
    initial = [e['initial_sha256'] for e in events if e['event'] == 'model_initialized']
    rows = [r for r in json.loads((d / 'manifest.json').read_text())['rows'] if r['split'] == 'val']
    labels, ids = [r['label'] for r in rows], [r['sample_id'] for r in rows]
    out = dict(retrain=summary['retrains'][0], steps=summary['steps'], initial=initial[0], arms={})
    for arm in ARMS:
        report = json.loads((d / arm / 'report.json').read_text())['capped_first']
        m = metrics(labels, report['predictions'])
        for key in ('cc_f1', 'accuracy', 'macro_f1'):
            if abs(m[key] - report['metrics'][key]) > 1e-12:
                raise RuntimeError(f'{d.name} {arm} {key} differs from report.json')
        if report['counts'][3] != CAP:
            raise RuntimeError(f'{d.name} {arm} does not fill exactly {CAP} grade-3 slots')
        out['arms'][arm] = dict(m, hash=hashlib.sha256(json.dumps(report['predictions']).encode()).hexdigest())
    r = out['retrain']
    window = range(max(1, r['best_epoch'] - 2), r['epochs_run'] + 1)
    ens = torch.stack([torch.load(d / 'retrain1' / f'epoch{e:02d}.pt', weights_only=True) for e in window]).mean(0)
    out['ens'] = metrics(labels, evaluate_global(ens.tolist(), labels, [None, None, None, CAP, None], ids)
                         ['capped_first']['predictions'])
    return out


def gather(root, seeds):
    jobs = {}
    for d in Path(root).glob('seed*'):
        m = JOB.match(d.name)
        if m and int(m[1]) in seeds and (d / 'summary.json').exists():
            jobs.setdefault(int(m[1]), {})[tuple(int(x) for x in m.groups()[1:])] = load(d)
    return jobs


def check(seed, cells):
    """The pilot gate's integrity items, applied to every seed; raises on the first failure."""
    if set(cells) != set(CELLS):
        raise RuntimeError(f'{seed}: cells missing {sorted(set(CELLS) - set(cells))}')
    if len({c['initial'] for c in cells.values()}) != 1:
        raise RuntimeError(f'{seed}: the cells do not share one initialisation')
    for key, groups in (('first_order_sha256', lambda c: c[1]), ('first_batch_sha256', lambda c: c[:2])):
        by = {}
        for cell, c in cells.items():
            by.setdefault(groups(cell), set()).add(c['retrain'][key])
        if any(len(v) != 1 for v in by.values()) or len(set.union(*by.values())) != len(by):
            raise RuntimeError(f'{seed}: {key} is not shared exactly by the matching cells')
    for cell, c in cells.items():
        r = c['retrain']
        if cell[2] == 0 and not r['epochs_run'] == r['best_epoch'] == 10:
            raise RuntimeError(f'{seed} {cell}: early_stop off must keep epoch 10')
        if cell[2] == 1 and not r['best_epoch'] <= r['epochs_run'] <= 75:
            raise RuntimeError(f'{seed} {cell}: bad early-stopping record')
        t, s = c['steps']['tralo_final'], c['steps']['sham_final']
        if t.get('radius') != s.get('radius') or (t['applied'] and t['hard_after'] > CAP):
            raise RuntimeError(f'{seed} {cell}: step or sham out of spec')


def effect(values, switch):
    """Per seed: mean over the cells with the switch on minus the mean with it off."""
    on = [v for cell, v in values.items() if cell[switch] == 1]
    off = [v for cell, v in values.items() if cell[switch] == 0]
    return np.mean(on) - np.mean(off)


def main(root, v3_root=None, r18_root=None):
    jobs = gather(root, STUDY)
    complete = {s: c for s, c in sorted(jobs.items()) if set(c) == set(CELLS)}
    print(f'{len(complete)} complete seeds of {len(STUDY)}; incomplete: '
          f'{ {s: len(c) for s, c in jobs.items() if s not in complete} or "none"}')
    if len(complete) < 2:
        raise SystemExit('fewer than two complete seeds')
    for seed, cells in complete.items():
        check(seed, cells)
    hashes = [c['arms']['pto']['hash'] for cells in complete.values() for c in cells.values()]
    if len(set(hashes)) != len(hashes):
        raise RuntimeError('duplicate pto predictions across jobs')
    print(f'integrity: {len(hashes)} jobs pass every gate item; all pto prediction vectors distinct\n')
    seeds = list(complete.values())
    print('cell means, % (A S E: pto cc-F1 / acc; tralo_final - sham_final; ENS - best; best epoch, epochs run):')
    for cell in CELLS:
        cs = [s[cell] for s in seeds]
        print(f"  {cell}: {100 * np.mean([c['arms']['pto']['cc_f1'] for c in cs]):.2f} / "
              f"{100 * np.mean([c['arms']['pto']['accuracy'] for c in cs]):.2f};  "
              f"P2 {100 * np.mean([c['arms']['tralo_final']['cc_f1'] - c['arms']['sham_final']['cc_f1'] for c in cs]):+.2f};  "
              f"ENS {100 * np.mean([c['ens']['cc_f1'] - c['arms']['pto']['cc_f1'] for c in cs]):+.2f};  "
              f"{np.mean([c['retrain']['best_epoch'] for c in cs]):.1f}, {np.mean([c['retrain']['epochs_run'] for c in cs]):.1f}")
    names = [f'F-{x}' for x in SWITCHES] + ['P2-pooled tralo_final - sham_final']

    def primaries(metric):
        rows = [[effect({cell: c['arms']['pto'][metric] for cell, c in s.items()}, i) for s in seeds] for i in range(3)]
        rows.append([np.mean([c['arms']['tralo_final'][metric] - c['arms']['sham_final'][metric] for c in s.values()])
                     for s in seeds])
        return [paired(r) for r in rows]

    for metric in METRICS:
        results = primaries(metric)
        adjusted = holm([r['p'] for r in results])
        print(f"\n{metric} ({'PRIMARY' if metric == 'cc_f1' else 'secondary'}), n = {len(seeds)}:")
        for name, r, h in zip(names, results, adjusted):
            print(f'  {name:36s} {fmt(r)}  Holm {h:.3f}')
    print('\nsecondary, cc-F1 (no family claim):')
    for i, j in ((0, 1), (0, 2), (1, 2)):
        d = [effect({c: v['arms']['pto']['cc_f1'] for c, v in s.items() if c[j] == 1}, i)
             - effect({c: v['arms']['pto']['cc_f1'] for c, v in s.items() if c[j] == 0}, i) for s in seeds]
        print(f"  interaction: effect of {SWITCHES[i][0]} with {SWITCHES[j][0]} on minus off  {fmt(paired(d))}")
    for i in range(3):
        d = [effect({c: v['arms']['tralo_final']['cc_f1'] - v['arms']['sham_final']['cc_f1'] for c, v in s.items()}, i)
             for s in seeds]
        print(f'  P2 with {SWITCHES[i]} on minus off  {fmt(paired(d))}')
    for cell in CELLS:
        p2 = paired([s[cell]['arms']['tralo_final']['cc_f1'] - s[cell]['arms']['sham_final']['cc_f1'] for s in seeds])
        ens = paired([s[cell]['ens']['cc_f1'] - s[cell]['arms']['pto']['cc_f1'] for s in seeds])
        print(f'  cell {cell}: P2 {fmt(p2)} | ENS - best {fmt(ens)}')
    d = [s[(1, 1, 1)]['arms']['pto']['cc_f1'] - s[(0, 0, 0)]['arms']['pto']['cc_f1'] for s in seeds]
    print(f'  all on - all off, pto  {fmt(paired(d))}')

    def dose(cells):   # (excess of pto's argmax count over the cap, P2) for the binding jobs
        pts = [(c['steps']['tralo_final']['hard_before'] - CAP,
                c['arms']['tralo_final']['cc_f1'] - c['arms']['sham_final']['cc_f1']) for c in cells]
        return [(x, y) for x, y in pts if x > 0]

    pooled = dose([c for s in seeds for c in s.values()])
    r = stats.spearmanr(*zip(*pooled))
    print(f'  dose (amendment 1): P2 vs excess over the cap, binding jobs (n {len(pooled)}): '
          f'Spearman {r.statistic:+.3f} p {r.pvalue:.4f}')
    within = [stats.spearmanr(*zip(*pts)).statistic for pts in (dose(s.values()) for s in seeds)
              if len(pts) >= 4 and len({x for x, _ in pts}) > 1 and len({y for _, y in pts}) > 1]
    if len(within) >= 3:
        print(f'  dose within seed: Spearman {fmt(paired(within), 1)} over {len(within)} seeds')
    references = []
    if r18_root:
        r18 = [load_yuval(d)['arms']['pto'] for d in sorted(Path(r18_root).glob('seed*')) if (d / 'summary.json').exists()]
        references.append(('all-on cell vs the ResNet18 block pto', (1, 1, 1), r18))
    if v3_root:
        v3 = []
        for d in sorted(Path(v3_root).glob('seed*')):
            if (d / 'clipper' / 'report.json').exists():
                rows = [r for r in json.loads((d / 'manifest.json').read_text())['rows'] if r['split'] == 'val']
                v3.append(metrics([r['label'] for r in rows],
                                  json.loads((d / 'clipper' / 'report.json').read_text())['capped_first']['predictions']))
        references.append(('all-off cell vs the v3 clipper', (0, 0, 0), v3))
    for name, cell, ref in references:
        print(f'\n{name} (unpaired Welch, n {len(seeds)} vs {len(ref)}):')
        for metric in METRICS:
            a, b = np.array([s[cell]['arms']['pto'][metric] for s in seeds]), np.array([r[metric] for r in ref])
            print(f'  {metric:12s} {100 * a.mean():.2f} vs {100 * b.mean():.2f}  diff {100 * (a.mean() - b.mean()):+.2f}  '
                  f'p {stats.ttest_ind(a, b, equal_var=False).pvalue:.4f}')


def gate(root, seed):
    cells = gather(root, {int(seed)}).get(int(seed), {})
    check(int(seed), cells)
    for cell, c in sorted(cells.items(), reverse=True):
        r, t = c['retrain'], c['steps']['tralo_final']
        print(f"  {cell}: best {r['best_epoch']}/{r['epochs_run']}, order {r['first_order_sha256'][:8]}, "
              f"batch {r['first_batch_sha256'][:8]}, step {'applied ' + str(t['hard_before']) + ' -> ' + str(t['hard_after']) if t['applied'] else 'not needed'}")
    print(f'PILOT GATE PASSED for {seed}: 8 cells, one initialisation, shared order and batches, E-off keeps epoch 10')


if __name__ == '__main__':
    if len(sys.argv) == 4 and sys.argv[1] == '--gate':
        gate(sys.argv[2], sys.argv[3])
    elif len(sys.argv) in (2, 4):
        main(*sys.argv[1:])
    else:
        raise SystemExit(__doc__)
