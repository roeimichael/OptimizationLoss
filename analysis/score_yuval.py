"""Score the Yuval-pipeline study (experiments/claude_yuval_pipeline_prereg_20260927.md).

Usage: python analysis/score_yuval.py RUN_ROOT [V3_ROOT]

RUN_ROOT holds seedNNNN/ directories written by tralo.knee_yuval. Primary: cc-F1 under capped_first,
P1 pao-pto, P2 tralo_final-sham_final, P3 tralo_final-pto, Holm over the three. Secondary: Yuval's
accuracy, macro-F1 and weighted-F1. V3_ROOT (optional): the v3 ResNet18 study, whose clipper gives
the unpaired recipe contrast. Development labels are read here, offline, and nowhere else.
"""

import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np
from scipy import stats

ARMS = ('pto', 'tralo_final', 'sham_final', 'pao')
PRIMARY = (('P1 pao - pto', 'pao', 'pto'), ('P2 tralo_final - sham_final', 'tralo_final', 'sham_final'),
           ('P3 tralo_final - pto', 'tralo_final', 'pto'))
METRICS = ('cc_f1', 'accuracy', 'macro_f1', 'weighted_f1')
CAP = 76


def metrics(labels, preds, k=3, classes=5):
    labels, preds = np.asarray(labels), np.asarray(preds)
    f1, support = [], []
    for c in range(classes):
        tp = int(((preds == c) & (labels == c)).sum())
        den = int((preds == c).sum()) + int((labels == c).sum())
        f1.append(2 * tp / den if den else 0.0)
        support.append(int((labels == c).sum()))
    return dict(cc_f1=f1[k], accuracy=float((preds == labels).mean()), macro_f1=float(np.mean(f1)),
                weighted_f1=float(np.dot(f1, support) / sum(support)))


def paired(d):
    d = np.asarray(d, float)
    n = len(d)
    m, sd = d.mean(), d.std(ddof=1)
    if sd == 0:
        return dict(n=n, mean=m, sd=0.0, lo=m, hi=m, p=1.0 if m == 0 else 0.0)
    h = stats.t.ppf(0.975, n - 1) * sd / math.sqrt(n)
    return dict(n=n, mean=m, sd=sd, lo=m - h, hi=m + h, p=float(2 * stats.t.sf(abs(m) / (sd / math.sqrt(n)), n - 1)))


def holm(ps):
    order = sorted(range(len(ps)), key=ps.__getitem__)
    adjusted, running = [0.0] * len(ps), 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(ps) - rank) * ps[i]))
        adjusted[i] = running
    return adjusted


def load(seed_dir):
    summary = json.loads((seed_dir / 'summary.json').read_text())
    rows = [r for r in json.loads((seed_dir / 'manifest.json').read_text())['rows'] if r['split'] == 'val']
    labels = [r['label'] for r in rows]
    out = dict(seed=summary['seed'], converged=summary['converged'], retrains=summary['retrains'],
               steps=summary['steps'], arms={})
    for arm in ARMS:
        report = json.loads((seed_dir / arm / 'report.json').read_text())['capped_first']
        preds = report['predictions']
        m = metrics(labels, preds)
        for key in ('cc_f1', 'accuracy', 'macro_f1'):
            if abs(m[key] - report['metrics'][key]) > 1e-12:
                raise RuntimeError(f'{seed_dir.name} {arm} {key} differs from report.json')
        if report['counts'][3] != CAP:
            raise RuntimeError(f'{seed_dir.name} {arm} does not fill exactly {CAP} grade-3 slots')
        out['arms'][arm] = dict(m, hash=hashlib.sha256(json.dumps(preds).encode()).hexdigest())
    return out


def fmt(r, scale=100):
    return (f"{r['mean'] * scale:+.2f} [{r['lo'] * scale:+.2f}, {r['hi'] * scale:+.2f}] "
            f"sd {r['sd'] * scale:.2f} p {r['p']:.3f}")


def main(root, v3_root=None):
    seeds = [load(d) for d in sorted(Path(root).glob('seed*')) if (d / 'summary.json').exists()]
    seen, kept = set(), []
    for s in seeds:
        key = tuple(s['arms'][a]['hash'] for a in ARMS)
        if key in seen:
            print(f"DUPLICATE predictions: seed {s['seed']} dropped")
            continue
        seen.add(key)
        kept.append(s)
    print(f'{len(kept)} seeds scored ({[s["seed"] for s in kept]})\n')
    binds = [s for s in kept if s['retrains'][0]['hard_count'] > CAP]
    print(f"cap binds on pto's hard count: {len(binds)}/{len(kept)}; PAO converged {sum(s['converged'] for s in kept)}/{len(kept)}; "
          f"retrains mean {np.mean([len(s['retrains']) for s in kept]):.2f} (max {max(len(s['retrains']) for s in kept)})")
    print(f"pto hard count mean {np.mean([s['retrains'][0]['hard_count'] for s in kept]):.1f}; "
          f"pao final hard count mean {np.mean([s['retrains'][-1]['hard_count'] for s in kept]):.1f}; "
          f"pao final C (non-capped) mean {np.mean([s['retrains'][-1]['C'][0] for s in kept]):.2f}")
    print(f"pto best epoch mean {np.mean([s['retrains'][0]['best_epoch'] for s in kept]):.1f}, epochs run "
          f"{np.mean([s['retrains'][0]['epochs_run'] for s in kept]):.1f}\n")
    print('arm means (capped_first, %):')
    for arm in ARMS:
        print(f'  {arm:12s} ' + '  '.join(f"{k} {100 * np.mean([s['arms'][arm][k] for s in kept]):.2f}" for k in METRICS))
    for subset_name, subset in (('ALL seeds (intent to treat)', kept), ('BINDING seeds only', binds)):
        if len(subset) < 3:
            continue
        print(f'\n=== {subset_name}, n = {len(subset)} ===')
        for metric in METRICS:
            results = [paired([s['arms'][a][metric] - s['arms'][b][metric] for s in subset]) for _, a, b in PRIMARY]
            adjusted = holm([r['p'] for r in results])
            label = 'PRIMARY' if metric == 'cc_f1' and subset is kept else 'secondary'
            print(f'  {metric} ({label}):')
            for (name, _, _), r, h in zip(PRIMARY, results, adjusted):
                print(f'    {name:28s} {fmt(r)}  Holm {h:.3f}')
            r = paired([s['arms']['pao'][metric] - s['arms']['tralo_final'][metric] for s in subset])
            print(f"    {'pao - tralo_final':28s} {fmt(r)}")
    if v3_root:
        v3 = []
        for d in sorted(Path(v3_root).glob('seed*')):
            path = d / 'clipper' / 'report.json'
            if not path.exists():
                continue
            rows = [r for r in json.loads((d / 'manifest.json').read_text())['rows'] if r['split'] == 'val']
            v3.append(metrics([r['label'] for r in rows], json.loads(path.read_text())['capped_first']['predictions']))
        print(f'\n=== recipe contrast (unpaired Welch): this study pto (n {len(kept)}) vs v3 clipper (n {len(v3)}) ===')
        for metric in METRICS:
            a = np.array([s['arms']['pto'][metric] for s in kept])
            b = np.array([r[metric] for r in v3])
            t = stats.ttest_ind(a, b, equal_var=False)
            print(f'  {metric:12s} yuval-recipe {100 * a.mean():.2f} (sd {100 * a.std(ddof=1):.2f})  v3 {100 * b.mean():.2f} '
                  f'(sd {100 * b.std(ddof=1):.2f})  diff {100 * (a.mean() - b.mean()):+.2f}  p {t.pvalue:.4f}')
    print('\nper seed (cc-F1 %, pto / tralo_final / sham_final / pao; pto hard; retrains; step radius):')
    for s in kept:
        a = s['arms']
        print(f"  {s['seed']}  " + ' / '.join(f"{100 * a[x]['cc_f1']:.2f}" for x in ARMS) +
              f"  hard {s['retrains'][0]['hard_count']}  retrains {len(s['retrains'])}  "
              f"radius {s['steps']['tralo_final'].get('radius', 0):.4g}")


if __name__ == '__main__':
    if len(sys.argv) not in (2, 3):
        raise SystemExit(__doc__)
    main(*sys.argv[1:])
