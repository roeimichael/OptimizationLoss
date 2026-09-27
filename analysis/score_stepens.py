"""Score the step-ensemble study (experiments/claude_stepens_prereg_20260927.md, ResNet18) or its RegNetY
replication (experiments/claude_stepens_rgy_prereg_20260927.md).

Usage: python analysis/score_stepens.py RUN_ROOT
       python analysis/score_stepens.py --gate PILOT_ROOT REFERENCE_ROOT   (pilot gate: integrity only, no score)

RUN_ROOT holds seed<seed>/ directories of one block, written by tralo.knee_yuval with snapshot_steps. Every epoch e of
retrain1 has epoch<e>.pt (PTO's pool probabilities) and, from side copies of that epoch's model,
epoch<e>_tralo.pt (TraLO's targeted step) and epoch<e>_sham.pt (the same radius, a seeded random
direction). Each ensemble averages one of the three over the window max(1, best - 2)..last
(analysis/yuval_ensemble.py's rule) and is deployed with capped_first at cap 76. Primary, cc-F1 of
grade 3, Holm over two:
  E1  ens_tralo - ens_sham   does TraLO's who-signal survive snapshot ensembling?
  E2  ens_tralo - ens_pto    TraLO against the ensembled clipper, the thesis bar
Development labels are read here, offline, and nowhere else.
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
from score_yuval import CAP, METRICS, fmt, holm, load as load_yuval, metrics, paired  # noqa: E402
from tralo.global_report import evaluate_global  # noqa: E402

BLOCKS = (('resnet18', 'ResNet', range(4500, 4572), 4000),          # backbone, model class, study seeds, pilot;
          ('regnet_y_400mf', 'RegNet', range(4600, 4672), 4400))    # the pilot job reruns that stored seed
SUFFIX = dict(ens_pto='', ens_tralo='_tralo', ens_sham='_sham')
PRIMARY = (('E1 ens_tralo - ens_sham', 'ens_tralo', 'ens_sham'), ('E2 ens_tralo - ens_pto', 'ens_tralo', 'ens_pto'))
SECONDARY = (('P2 tralo_final - sham_final (single model)', 'tralo_final', 'sham_final'),
             ('ens_pto - pto (ensemble confirmation, set 8)', 'ens_pto', 'pto'),
             ('ens_sham - ens_pto (a random move, ensembled)', 'ens_sham', 'ens_pto'))
SLOT = 2 / (CAP + 106)                 # one grade-3 slot in cc-F1: F1 = 2 TP / (76 + 106 true grade-3 knees)


def initialised(seed_dir):
    """The run's model_initialized event: its architecture, model class and initial weights' hash."""
    for line in (seed_dir / 'events.jsonl').read_text().splitlines():
        row = json.loads(line)
        if row.get('event') == 'model_initialized':
            return row
    raise RuntimeError(f'{seed_dir.name}: no model_initialized event')


def window(retrain):
    return range(max(1, retrain['best_epoch'] - 2), retrain['epochs_run'] + 1)


def load(d):
    """One seed: the single-model arms from report.json and the three snapshot ensembles."""
    out = load_yuval(d, ('pto', 'tralo_final', 'sham_final'))
    if len(out['retrains']) != 1 or (d / 'pao').exists() or (d / 'retrain2').exists():
        raise RuntimeError(f'{d.name}: the step-ensemble study trains PTO once')
    retrain = out['retrains'][0]
    steps = retrain.get('snapshot_steps') or {}
    if set(steps) != {str(e) for e in range(1, retrain['epochs_run'] + 1)}:
        raise RuntimeError(f'{d.name}: snapshot steps missing for some epochs')
    for e, s in steps.items():
        t, h = s['tralo'], s['sham']
        if t.get('radius') != h.get('radius') or t['applied'] != h['applied'] or (t['applied'] and t['hard_after'] > CAP):
            raise RuntimeError(f'{d.name} epoch {e}: step or sham out of spec')
    rows = [r for r in json.loads((d / 'manifest.json').read_text())['rows'] if r['split'] == 'val']
    labels, ids = [r['label'] for r in rows], [r['sample_id'] for r in rows]
    for name, suffix in SUFFIX.items():
        mean = torch.stack([torch.load(d / 'retrain1' / f'epoch{e:02d}{suffix}.pt', weights_only=True)
                            for e in window(retrain)]).mean(0)
        report = evaluate_global(mean.tolist(), labels, [None, None, None, CAP, None], ids)['capped_first']
        if report['counts'][3] != CAP:
            raise RuntimeError(f'{d.name} {name} does not fill exactly {CAP} grade-3 slots')
        out['arms'][name] = dict(metrics(labels, report['predictions']),
                                 hash=hashlib.sha256(json.dumps(report['predictions']).encode()).hexdigest())
    out['excess'] = float(np.mean([steps[str(e)]['tralo']['hard_before'] - CAP for e in window(retrain)]))
    out['applied'] = sum(steps[str(e)]['tralo']['applied'] for e in window(retrain))
    out['window'] = len(window(retrain))
    return out


def main(root):
    dirs = [d for d in sorted(Path(root).glob('seed*')) if d.is_dir()]
    block = next((b for b in BLOCKS if dirs and int(dirs[0].name[4:]) in b[2]), None)
    seeds = {}
    for d in dirs:
        if block is None or int(d.name[4:]) not in block[2]:
            raise RuntimeError(f'{d.name} is not a study seed of one block')
        if (d / 'summary.json').exists():
            row = initialised(d)
            if (row.get('architecture'), row.get('model_class')) != block[:2]:
                raise RuntimeError(f"{d.name} was not trained on {block[0]}: {row.get('architecture')}")
            seeds[int(d.name[4:])] = load(d)
    study = block[2] if block else ()
    missing = sorted(set(study) - set(seeds))
    print(f'{block[0] if block else "no block"}: {len(seeds)} complete seeds of {len(study)}; '
          f'missing or incomplete: {missing or "none"}')
    owner = {}
    for seed, s in seeds.items():
        other = owner.setdefault(s['arms']['pto']['hash'], seed)
        if other != seed:
            raise RuntimeError(f'duplicate pto predictions: seeds {other} and {seed}')
    kept = list(seeds.values())
    if len(kept) < 2:
        raise SystemExit('fewer than two complete seeds')
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
            slots = f"  = {r['mean'] / SLOT:+.2f} [{r['lo'] / SLOT:+.2f}, {r['hi'] / SLOT:+.2f}] slots" if metric == 'cc_f1' else ''
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
    """The pilot (4000 or 4400): complete, every step in spec, and PTO byte-identical to the stored run at every epoch."""
    jobs = [x for x in sorted(Path(pilot_root).glob('seed*')) if x.is_dir()]
    block = next((b for b in BLOCKS if jobs and jobs[0].name == f'seed{b[3]}_stepens'), None)
    if block is None or len(jobs) != 1:
        raise SystemExit(f'PILOT GATE FAILED: unexpected seed directories {[x.name for x in jobs]}')
    d, ref = jobs[0], Path(reference_root) / f'seed{block[3]}'
    row = initialised(d)       # the stored runs predate model_class: they are matched by their initial weights
    if ((row.get('architecture'), row.get('model_class')) != block[:2]
            or initialised(ref).get('initial_sha256') != row.get('initial_sha256')):
        raise SystemExit(f'PILOT GATE FAILED: {d.name} is not {block[0]} from the initial weights of the stored run')
    s = load(d)
    run, stored = s['retrains'][0], json.loads((ref / 'summary.json').read_text())['retrains'][0]
    problems = []
    if (run['best_epoch'], run['epochs_run']) != (stored['best_epoch'], stored['epochs_run']):
        problems.append(f"best/run epochs {run['best_epoch']}/{run['epochs_run']} vs stored "
                        f"{stored['best_epoch']}/{stored['epochs_run']}")
    names = [f'epoch{e:02d}.pt' for e in range(1, run['epochs_run'] + 1)] + ['final_probabilities.pt']
    for name in names:
        a, b = d / 'retrain1' / name, ref / 'retrain1' / name
        if not b.exists() or not torch.equal(torch.load(a, weights_only=True), torch.load(b, weights_only=True)):
            problems.append(f'{name} differs from the stored run')
    if problems:
        print('PILOT GATE FAILED:\n  ' + '\n  '.join(problems))
        raise SystemExit(1)
    stepped = sum(x['tralo']['applied'] for x in run['snapshot_steps'].values())
    print(f"PILOT GATE PASSED for {block[3]} ({block[0]}): PTO byte-identical to the stored run at all {run['epochs_run']} epochs "
          f"and the restored best; {stepped} of {run['epochs_run']} epochs stepped, every sham on the same radius")


if __name__ == '__main__':
    if len(sys.argv) == 4 and sys.argv[1] == '--gate':
        gate(sys.argv[2], sys.argv[3])
    elif len(sys.argv) == 2:
        main(sys.argv[1])
    else:
        raise SystemExit(__doc__)
