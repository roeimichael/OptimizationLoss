"""Score the small-backbone blocks in Yuval's pipeline (experiments/claude_yuval_smallbb_prereg_20260927.md).

Usage: python analysis/score_smallbb.py MN3_ROOT RGY_ROOT [V3_MN3_ROOT V3_RGY_ROOT]
       python analysis/score_smallbb.py --gate PILOT_ROOT

Each ROOT holds seedNNNN/ directories written by tralo.knee_yuval with max_retrains 1. Primary: P2
tralo_final - sham_final on capped_first cc-F1, Holm over the two blocks. Secondary: P3 tralo_final - pto,
Yuval's metrics, and the unpaired recipe contrast against the v3 clipper of the same backbone. The
snapshot-ensemble confirmation and the swaps are analysis/yuval_ensemble.py and analysis/yuval_swaps.py,
run unchanged on each root. --gate checks the pilots' integrity and prints no score. Development labels
are read here, offline, and nowhere else.
"""

import json
from pathlib import Path
import sys

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from score_yuval import CAP, METRICS, fmt, holm, load, metrics, paired  # noqa: E402

ARMS = ('pto', 'tralo_final', 'sham_final')
BLOCKS = (('MobileNetV3', 'mobilenet_v3_large', 'MobileNetV3', range(4300, 4324)),
          ('RegNetY', 'regnet_y_400mf', 'RegNet', range(4400, 4424)))
PILOTS = {4399: ('mobilenet_v3_large', 'MobileNetV3'), 4499: ('regnet_y_400mf', 'RegNet')}
CONTRASTS = (('P2 tralo_final - sham_final', 'tralo_final', 'sham_final'), ('P3 tralo_final - pto', 'tralo_final', 'pto'))


def model_class(seed_dir):
    for line in (seed_dir / 'events.jsonl').read_text().splitlines():
        row = json.loads(line)
        if row.get('event') == 'model_initialized':
            return row.get('architecture'), row.get('model_class')
    raise RuntimeError(f'{seed_dir.name}: no model_initialized event')


def block(root, backbone, cls, seeds):
    kept, seen = [], set()
    for d in sorted(Path(root).glob('seed*')):
        if not (d.is_dir() and (d / 'summary.json').exists()):
            continue
        s = load(d, ARMS)
        if s['seed'] not in seeds:
            raise RuntimeError(f'{d.name} is not a study seed of this block')
        if model_class(d) != (backbone, cls):
            raise RuntimeError(f'{d.name} was not trained on {backbone}: {model_class(d)}')
        key = tuple(s['arms'][a]['hash'] for a in ARMS)
        if key in seen:
            print(f"DUPLICATE predictions: seed {s['seed']} dropped")
            continue
        seen.add(key)
        kept.append(s)
    return kept


def v3_clipper(v3_root):
    out = []
    for d in sorted(Path(v3_root).glob('seed*')):
        path = d / 'clipper' / 'report.json'
        if path.exists():
            rows = [r for r in json.loads((d / 'manifest.json').read_text())['rows'] if r['split'] == 'val']
            out.append(metrics([r['label'] for r in rows], json.loads(path.read_text())['capped_first']['predictions']))
    return out


def main(roots, v3_roots=(None, None)):
    blocks = [(spec[0], block(root, *spec[1:])) for spec, root in zip(BLOCKS, roots)]
    for name, kept in blocks:
        if len(kept) < 2:
            raise SystemExit(f'{name}: fewer than 2 complete seeds, nothing to score')
    primary = [paired([s['arms']['tralo_final']['cc_f1'] - s['arms']['sham_final']['cc_f1'] for s in kept])
               for _, kept in blocks]
    print('PRIMARY: P2 tralo_final - sham_final, capped_first cc-F1 (points), Holm over the two blocks')
    for (name, kept), r, h in zip(blocks, primary, holm([r['p'] for r in primary])):
        print(f'  {name:12s} n {len(kept):2d}  {fmt(r)}  Holm {h:.3f}')
    for (name, kept), v3_root in zip(blocks, v3_roots):
        binds = [s for s in kept if s['retrains'][0]['hard_count'] > CAP]
        r1 = [s['retrains'][0] for s in kept]
        print(f"\n=== {name}: seeds {[s['seed'] for s in kept]}")
        print(f"  cap binds on pto's hard count in {len(binds)}/{len(kept)}; pto hard count mean "
              f"{np.mean([r['hard_count'] for r in r1]):.1f}; best epoch mean {np.mean([r['best_epoch'] for r in r1]):.1f}, "
              f"epochs run mean {np.mean([r['epochs_run'] for r in r1]):.1f}")
        for arm in ARMS:
            print(f'  {arm:12s} ' + '  '.join(f"{k} {100 * np.mean([s['arms'][arm][k] for s in kept]):.2f}" for k in METRICS))
        for subset_name, subset in (('all seeds (intent to treat)', kept), ('binding seeds only', binds)):
            if len(subset) < 3:
                continue
            print(f'  {subset_name}, n = {len(subset)} (secondary unless P2 cc_f1 on all seeds):')
            for metric in METRICS:
                for label, a, b in CONTRASTS:
                    r = paired([s['arms'][a][metric] - s['arms'][b][metric] for s in subset])
                    print(f'    {metric:12s} {label:28s} {fmt(r)}')
        if v3_root:
            v3 = v3_clipper(v3_root)
            print(f'  recipe contrast (unpaired Welch): this pto (n {len(kept)}) vs the v3 clipper, same backbone (n {len(v3)})')
            for metric in METRICS:
                a, b = np.array([s['arms']['pto'][metric] for s in kept]), np.array([r[metric] for r in v3])
                print(f'    {metric:12s} yuval-recipe {100 * a.mean():.2f} (sd {100 * a.std(ddof=1):.2f})  v3 {100 * b.mean():.2f} '
                      f'(sd {100 * b.std(ddof=1):.2f})  diff {100 * (a.mean() - b.mean()):+.2f}  '
                      f'p {stats.ttest_ind(a, b, equal_var=False).pvalue:.4f}')
        print('  per seed (cc-F1 %, pto / tralo_final / sham_final; pto hard; step radius):')
        for s in kept:
            print(f"    {s['seed']}  " + ' / '.join(f"{100 * s['arms'][x]['cc_f1']:.2f}" for x in ARMS) +
                  f"  hard {s['retrains'][0]['hard_count']}  radius {s['steps']['tralo_final'].get('radius', 0):.4g}")


def gate(root):
    problems, seen = [], set()
    for d in sorted(Path(root).glob('seed*')):
        if not d.is_dir():
            continue
        seed = int(d.name[4:])
        if seed not in PILOTS:
            problems.append(f'{d.name}: not a pilot seed')
            continue
        seen.add(seed)
        if not (d / 'summary.json').exists():
            problems.append(f'{d.name}: no summary.json (the run did not complete)')
            continue
        s = json.loads((d / 'summary.json').read_text())
        if model_class(d) != PILOTS[seed]:
            problems.append(f'{d.name}: trained on {model_class(d)}, expected {PILOTS[seed]}')
        t, sham = s['steps']['tralo_final'], s['steps']['sham_final']
        if t.get('applied') and t.get('hard_after') != CAP:
            problems.append(f"{d.name}: the step landed on {t.get('hard_after')}, not {CAP}")
        if t.get('radius') != sham.get('radius'):
            problems.append(f'{d.name}: the sham radius differs from the target radius')
        r = s['retrains'][0]
        step = f"applied {t['hard_before']} -> {t['hard_after']}" if t.get('applied') else 'not applied (cap does not bind)'
        print(f"  {d.name}: {model_class(d)[1]}, best {r['best_epoch']}/{r['epochs_run']}, step {step}")
    problems += [f'pilot {p} missing' for p in sorted(set(PILOTS) - seen)]
    if problems:
        print('PILOT GATE FAILED:\n  ' + '\n  '.join(problems))
        raise SystemExit(1)
    print('PILOT GATE PASSED: both pilots complete, on their backbones, steps on the cap, sham radius matched')


if __name__ == '__main__':
    if len(sys.argv) == 3 and sys.argv[1] == '--gate':
        gate(sys.argv[2])
    elif len(sys.argv) in (3, 5):
        main(sys.argv[1:3], tuple(sys.argv[3:5]) if len(sys.argv) == 5 else (None, None))
    else:
        raise SystemExit(__doc__)
