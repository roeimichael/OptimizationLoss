"""Post hoc (not preregistered): who the ensembled step swaps, and which grades pay, in a knee step-ensemble study.

Usage: python analysis/stepens_swaps.py RUN_ROOT

For every complete seed of one block (analysis/score_stepens.py's blocks), rebuilds ens_pto and ens_tralo exactly as
the scorer does (mean over the window max(1, best - 2)..last, capped_first at 76) and reports:
  * swaps: the knees ens_tralo puts into its 76 grade-3 slots that ens_pto does not ("in") and the reverse ("out"),
    with the share of each that are truly grade 3, and the net change in correct slots;
  * where the knees leaving the grade-3 slots go, by true grade;
  * per-grade F1, ens_tralo - ens_pto, with paired 95% t intervals over seeds (E2 split by grade).
Development labels are read here, offline, and nowhere else.
"""

import json
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from score_stepens import BLOCKS, SUFFIX, initialised, window  # noqa: E402
from score_yuval import CAP, fmt, paired  # noqa: E402
from tralo.global_report import evaluate_global  # noqa: E402

K = 3


def ensembles(d):
    summary = json.loads((d / 'summary.json').read_text())
    retrain = summary['retrains'][0]
    rows = [r for r in json.loads((d / 'manifest.json').read_text())['rows'] if r['split'] == 'val']
    labels, ids = np.array([r['label'] for r in rows]), [r['sample_id'] for r in rows]
    preds = {}
    for name in ('ens_pto', 'ens_tralo'):
        mean = torch.stack([torch.load(d / 'retrain1' / f'epoch{e:02d}{SUFFIX[name]}.pt', weights_only=True)
                            for e in window(retrain)]).mean(0)
        report = evaluate_global(mean.tolist(), labels.tolist(), [None, None, None, CAP, None], ids)['capped_first']
        if report['counts'][K] != CAP:
            raise RuntimeError(f'{d.name} {name} does not fill {CAP} slots')
        preds[name] = np.array(report['predictions'])
    return labels, preds


def f1(labels, preds, c):
    tp = int(((preds == c) & (labels == c)).sum())
    den = int((preds == c).sum()) + int((labels == c).sum())
    return 2 * tp / den if den else 0.0


def main(root):
    dirs = [d for d in sorted(Path(root).glob('seed*')) if d.is_dir() and (d / 'summary.json').exists()]
    block = next(b for b in BLOCKS if int(dirs[0].name[4:]) in b[2])
    n_in = n_out = true_in = true_out = 0
    moved = np.zeros(5, int)
    per_seed_net, grade_deltas = [], {c: [] for c in range(5)}
    for d in dirs:
        if int(d.name[4:]) not in block[2] or (initialised(d).get('architecture'), initialised(d).get('model_class')) != block[:2]:
            raise RuntimeError(f'{d.name} is not a {block[0]} study seed')
        labels, p = ensembles(d)
        a, b = p['ens_pto'] == K, p['ens_tralo'] == K
        into, out = b & ~a, a & ~b
        n_in, n_out = n_in + int(into.sum()), n_out + int(out.sum())
        true_in, true_out = true_in + int((labels[into] == K).sum()), true_out + int((labels[out] == K).sum())
        per_seed_net.append(int((labels[into] == K).sum()) - int((labels[out] == K).sum()))
        for c in range(5):
            moved[c] += int((labels[out] == c).sum())
            grade_deltas[c].append(f1(labels, p['ens_tralo'], c) - f1(labels, p['ens_pto'], c))
    n = len(per_seed_net)
    print(f'{block[0]}: {n} seeds (post hoc, not preregistered)')
    print(f'swaps per seed: {n_in / n:.2f} knees in, {n_out / n:.2f} out (the slot count is fixed at {CAP})')
    print(f'  truly grade 3: {100 * true_in / max(n_in, 1):.1f}% of those in, {100 * true_out / max(n_out, 1):.1f}% of those out')
    print(f'  net correct slots per seed: {fmt(paired(per_seed_net), scale=1)}')
    print(f'  seeds with a net gain / no change / a net loss: {sum(x > 0 for x in per_seed_net)} / '
          f'{sum(x == 0 for x in per_seed_net)} / {sum(x < 0 for x in per_seed_net)}')
    print('  true grades of the knees leaving the slots: ' + ', '.join(f'grade {c}: {moved[c]}' for c in range(5)))
    print('per-grade F1, ens_tralo - ens_pto (points; paired 95% t over seeds):')
    for c in range(5):
        print(f'  grade {c}: {fmt(paired(grade_deltas[c]))}')


if __name__ == '__main__':
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    main(sys.argv[1])
