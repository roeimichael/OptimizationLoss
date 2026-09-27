"""Score the fmow2 step-ensemble study (experiments/claude_fmow_stepens_prereg_20260928.md).

Usage: python analysis/score_fmow_stepens.py RUN_ROOT
       python analysis/score_fmow_stepens.py --gate PILOT_ROOT REFERENCE_ROOT   (pilot gate: integrity only, no score)

The design is analysis/score_stepens.py's, for class 1 (crop_field) of fmow2's 8 classes at the cap in each run's
manifest (development pool size // cap_divisor). RUN_ROOT holds seed<seed>/ directories written by tralo.fmow_yuval.
Each ensemble averages epochNN.pt, epochNN_tralo.pt or epochNN_sham.pt over the window max(1, best - 2)..last and is
deployed with capped_first. Primary, cc-F1 of class 1, Holm over two:
  E1  ens_tralo - ens_sham   does TraLO's who-signal survive snapshot ensembling?
  E2  ens_tralo - ens_pto    TraLO against the ensembled clipper, the thesis bar
Development labels are read here, offline, and nowhere else; the reserved test countries are never read.
"""

import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from scipy import stats
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from score_stepens import PRIMARY, SECONDARY, SUFFIX, initialised, window  # noqa: E402
from score_yuval import METRICS, fmt, holm, metrics, paired  # noqa: E402
from tralo.global_report import evaluate_global  # noqa: E402

K, CLASSES = 1, 8
STUDY, PILOT = range(5000, 5048), 5099
ARCH = ('mobilenet_v3_large', 'MobileNetV3')


def load(d):
    """One seed: the single-model arms from report.json and the three snapshot ensembles."""
    summary = json.loads((d / 'summary.json').read_text())
    manifest = json.loads((d / 'manifest.json').read_text())
    cap, rows = manifest['cap'], manifest['rows']
    if summary['cap'] != cap or summary['capped_class'] != K or len(summary['retrains']) != 1 or (d / 'retrain2').exists():
        raise RuntimeError(f'{d.name}: not one PTO run at the manifest cap on class {K}')
    labels, ids = [r['label'] for r in rows], [r['sample_id'] for r in rows]
    caps = [None] * CLASSES
    caps[K] = cap
    out = dict(seed=summary['seed'], cap=cap, n_true=labels.count(K), retrains=summary['retrains'], arms={})
    for arm in ('pto', 'tralo_final', 'sham_final'):
        report = json.loads((d / arm / 'report.json').read_text())['capped_first']
        m = metrics(labels, report['predictions'], K, CLASSES)
        if any(abs(m[k] - report['metrics'][k]) > 1e-12 for k in ('cc_f1', 'accuracy', 'macro_f1')):
            raise RuntimeError(f'{d.name} {arm} metrics differ from report.json')
        if report['counts'][K] != cap:
            raise RuntimeError(f'{d.name} {arm} does not fill exactly {cap} class-{K} slots')
        out['arms'][arm] = dict(m, hash=hashlib.sha256(json.dumps(report['predictions']).encode()).hexdigest())
    retrain = summary['retrains'][0]
    steps = retrain.get('snapshot_steps') or {}
    if set(steps) != {str(e) for e in range(1, retrain['epochs_run'] + 1)}:
        raise RuntimeError(f'{d.name}: snapshot steps missing for some epochs')
    for e, s in steps.items():
        t, h = s['tralo'], s['sham']
        if t.get('radius') != h.get('radius') or t['applied'] != h['applied'] or (t['applied'] and t['hard_after'] > cap):
            raise RuntimeError(f'{d.name} epoch {e}: step or sham out of spec')
    for name, suffix in SUFFIX.items():
        mean = torch.stack([torch.load(d / 'retrain1' / f'epoch{e:02d}{suffix}.pt', weights_only=True)
                            for e in window(retrain)]).mean(0)
        report = evaluate_global(mean.tolist(), labels, caps, ids)['capped_first']
        if report['counts'][K] != cap:
            raise RuntimeError(f'{d.name} {name} does not fill exactly {cap} class-{K} slots')
        out['arms'][name] = dict(metrics(labels, report['predictions'], K, CLASSES),
                                 hash=hashlib.sha256(json.dumps(report['predictions']).encode()).hexdigest())
    out['excess'] = float(np.mean([steps[str(e)]['tralo']['hard_before'] - cap for e in window(retrain)]))
    out['applied'] = sum(steps[str(e)]['tralo']['applied'] for e in window(retrain))
    out['window'] = len(window(retrain))
    return out


def main(root):
    dirs = [d for d in sorted(Path(root).glob('seed*')) if d.is_dir()]
    seeds = {}
    for d in dirs:
        if not d.name[4:].isdigit() or int(d.name[4:]) not in STUDY:
            raise RuntimeError(f'{d.name} is not a study seed')
        if (d / 'summary.json').exists():
            row = initialised(d)
            if (row.get('architecture'), row.get('model_class')) != ARCH:
                raise RuntimeError(f"{d.name} was not trained on {ARCH[0]}: {row.get('architecture')}")
            seeds[int(d.name[4:])] = load(d)
    missing = sorted(set(STUDY) - set(seeds))
    print(f'fmow2 {ARCH[0]}: {len(seeds)} complete seeds of {len(STUDY)}; missing or incomplete: {missing or "none"}')
    if len({(s['cap'], s['n_true']) for s in seeds.values()}) > 1:
        raise RuntimeError('seeds differ in the cap or the development pool')
    owner = {}
    for seed, s in seeds.items():
        other = owner.setdefault(s['arms']['pto']['hash'], seed)
        if other != seed:
            raise RuntimeError(f'duplicate pto predictions: seeds {other} and {seed}')
    kept = list(seeds.values())
    if len(kept) < 2:
        raise SystemExit('fewer than two complete seeds')
    cap, n_true = kept[0]['cap'], kept[0]['n_true']
    slot = 2 / (cap + n_true)
    print(f"class {K} cap {cap} on a pool with {n_true} true class-{K} items: one slot = {100 * slot:.4f} cc-F1 points")
    print(f"integrity: every epoch's step and sham share one radius and land on the cap; window "
          f"{np.mean([s['window'] for s in kept]):.1f} snapshots, {np.mean([s['applied'] for s in kept]):.1f} "
          f"of them stepped (min {min(s['applied'] for s in kept)}); no pto prediction vector repeats\n")

    def contrast(metric, a, b):
        return paired([s['arms'][a][metric] - s['arms'][b][metric] for s in kept])

    for metric in METRICS:
        results = [contrast(metric, a, b) for _, a, b in PRIMARY]
        adjusted = holm([r['p'] for r in results])
        print(f"{metric} ({'PRIMARY' if metric == 'cc_f1' else 'secondary'}), n = {len(kept)}:")
        for (name, _, _), r, h in zip(PRIMARY, results, adjusted):
            slots = f"  = {r['mean'] / slot:+.2f} [{r['lo'] / slot:+.2f}, {r['hi'] / slot:+.2f}] slots" if metric == 'cc_f1' else ''
            print(f'  {name:26s} {fmt(r)}  Holm {h:.3f}{slots}')
    print('\nsecondary, cc-F1 (no family claim):')
    for name, a, b in SECONDARY:
        print(f'  {name:46s} {fmt(contrast("cc_f1", a, b))}')
    e1 = [s['arms']['ens_tralo']['cc_f1'] - s['arms']['ens_sham']['cc_f1'] for s in kept]
    r = stats.spearmanr([s['excess'] for s in kept], e1)
    print(f"  dose: E1 vs mean excess of the window's hard count over the cap: Spearman {r.statistic:+.3f} p {r.pvalue:.4f}")
    print('\nper seed (cc-F1 %: ens_pto / ens_tralo / ens_sham | pto / tralo_final / sham_final; window, stepped):')
    for seed, s in seeds.items():
        a = s['arms']
        print(f"  {seed}  {100 * a['ens_pto']['cc_f1']:.2f} / {100 * a['ens_tralo']['cc_f1']:.2f} / "
              f"{100 * a['ens_sham']['cc_f1']:.2f} | {100 * a['pto']['cc_f1']:.2f} / {100 * a['tralo_final']['cc_f1']:.2f} / "
              f"{100 * a['sham_final']['cc_f1']:.2f}; {s['window']}, {s['applied']}")


def gate(pilot_root, reference_root):
    """The pilot 5099: complete, every step in spec, and PTO byte-identical to its steps-off reference at every epoch."""
    jobs = [x for x in sorted(Path(pilot_root).glob('seed*')) if x.is_dir()]
    if [x.name for x in jobs] != [f'seed{PILOT}_stepens']:
        raise SystemExit(f'PILOT GATE FAILED: unexpected seed directories {[x.name for x in jobs]}')
    d, ref = jobs[0], Path(reference_root) / f'seed{PILOT}_ref'
    if not (ref / 'summary.json').exists():
        raise SystemExit(f'PILOT GATE FAILED: no reference run at {ref}')
    row = initialised(d)
    if (row.get('architecture'), row.get('model_class')) != ARCH or initialised(ref).get('initial_sha256') != row.get('initial_sha256'):
        raise SystemExit(f'PILOT GATE FAILED: {d.name} is not {ARCH[0]} from the initial weights of the reference')
    s = load(d)
    run, stored = s['retrains'][0], json.loads((ref / 'summary.json').read_text())['retrains'][0]
    problems = []
    if (run['best_epoch'], run['epochs_run']) != (stored['best_epoch'], stored['epochs_run']):
        problems.append(f"best/run epochs {run['best_epoch']}/{run['epochs_run']} vs reference "
                        f"{stored['best_epoch']}/{stored['epochs_run']}")
    for name in [f'epoch{e:02d}.pt' for e in range(1, run['epochs_run'] + 1)] + ['final_probabilities.pt']:
        a, b = d / 'retrain1' / name, ref / 'retrain1' / name
        if not b.exists() or not torch.equal(torch.load(a, weights_only=True), torch.load(b, weights_only=True)):
            problems.append(f'{name} differs from the reference run')
    if problems:
        print('PILOT GATE FAILED:\n  ' + '\n  '.join(problems))
        raise SystemExit(1)
    stepped = sum(x['tralo']['applied'] for x in run['snapshot_steps'].values())
    print(f"PILOT GATE PASSED for {PILOT} (fmow2, {ARCH[0]}): PTO byte-identical to the reference at all {run['epochs_run']} "
          f"epochs and the restored best; {stepped} of {run['epochs_run']} epochs stepped (hard count above cap {s['cap']}), "
          f"every sham on the same radius; PTO's restored best predicts {run['hard_count']} class-{K} items")


if __name__ == '__main__':
    if len(sys.argv) == 4 and sys.argv[1] == '--gate':
        gate(sys.argv[2], sys.argv[3])
    elif len(sys.argv) == 2:
        main(sys.argv[1])
    else:
        raise SystemExit(__doc__)
