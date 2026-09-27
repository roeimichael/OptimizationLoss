"""Synthetic pre-screen: does TraLO's targeted step help once its DIRECTION carries training labels?

Usage: python fp_lab.py [SEEDS] [PROCS]      (reuses pao_lab.py's data, model, training and scoring)

Early-stopped regime only (the one where training false positives exist). From the PTO model:
  tralo     label-free: -grad sum_pool p_k                       (as tralo.targeted_step)
  fp        label-aware: -grad sum over TRAINING false positives of p_k (argmax k, label != k)
  trainall  label-free control on train: -grad sum over training items predicted k of p_k
  fpsham    random direction, fp's radius and per-tensor norms
Every step has the smallest radius that brings the POOL hard count to the cap.
"""
import copy
import json
import math
import sys
from multiprocessing import Pool

import numpy as np
import torch
from scipy import stats

import pao_lab as L


def targeted(m, direction_loss, P, sham_seed=None):
    params = list(m.parameters())
    if int((L.soft(m, P).argmax(1) == L.k).sum()) <= L.CAP:
        return None
    m.zero_grad()
    m.eval()
    direction_loss(m).backward()
    g = [p.grad.detach().clone() for p in params]
    norm = math.sqrt(sum(float(x.double().square().sum()) for x in g))
    if not norm > 0:
        return None
    unit = [-x / norm for x in g]
    origin = [p.detach().clone() for p in params]

    def place(r, dirs):
        with torch.no_grad():
            for p, o, u in zip(params, origin, dirs):
                p.copy_(o + r * u)

    def hard():
        return int((L.soft(m, P).argmax(1) == L.k).sum())
    lo, hi = 0.0, 1e-3
    for _ in range(60):
        place(hi, unit)
        if hard() <= L.CAP:
            break
        lo, hi = hi, 2 * hi
    else:
        place(0.0, unit)
        return None
    for _ in range(30):
        mid = 0.5 * (lo + hi)
        place(mid, unit)
        if hard() <= L.CAP:
            hi = mid
        else:
            lo = mid
    dirs = unit
    if sham_seed is not None:
        gen = torch.Generator().manual_seed(sham_seed)
        dirs = []
        for u in unit:
            z = torch.randn(u.shape, generator=gen)
            dirs.append(z * (float(u.norm()) / float(z.norm())))
    place(hi, dirs)
    return hi


def run(seed):
    torch.set_num_threads(1)
    d = L.data(seed)
    X, y, P, yp = d[0], d[1], d[4], d[5]
    torch.manual_seed(seed)
    init = copy.deepcopy(L.mlp().state_dict())
    pto, info = L.train(d, init, torch.ones(L.K), 'es', seed)
    pr = L.soft(pto, P)
    tr = L.soft(pto, X).argmax(1)
    fp_mask = (tr == L.k) & (y != L.k)
    pred_mask = tr == L.k
    out = dict(seed=seed, binds=int((pr.argmax(1) == L.k).sum()) > L.CAP, n_fp=int(fp_mask.sum()),
               n_pred=int(pred_mask.sum()), scores=dict(pto=L.score(pr, yp)), radius={})
    directions = {
        'tralo': (lambda m: m(P).softmax(1)[:, L.k].sum(), None),
        'fp': (lambda m: m(X[fp_mask]).softmax(1)[:, L.k].sum(), None),
        'trainall': (lambda m: m(X[pred_mask]).softmax(1)[:, L.k].sum(), None),
        'fpsham': (lambda m: m(X[fp_mask]).softmax(1)[:, L.k].sum(), seed + 7),
        # ORACLE bounds (pool labels, never deployable): demote pool false positives only, or demote
        # pool false positives while promoting pool true positives (a ranking direction)
        'oracle_fp': (lambda m: m(P[(pr.argmax(1) == L.k) & (yp != L.k)]).softmax(1)[:, L.k].sum(), None),
        'oracle_rank': (lambda m: m(P[yp != L.k]).softmax(1)[:, L.k].sum() - m(P[yp == L.k]).softmax(1)[:, L.k].sum(), None),
    }
    for name, (loss, sham) in directions.items():
        m = copy.deepcopy(pto)
        out['radius'][name] = targeted(m, loss, P, sham)
        if out['radius'][name] is None and name.startswith('oracle') and out['binds']:
            out.setdefault('oracle_unsized', []).append(name)
        after = L.soft(m, P)
        out['scores'][name] = L.score(after, yp)
        if name in ('tralo', 'fp', 'trainall', 'oracle_fp', 'oracle_rank') and out['radius'][name] is not None:
            demote = (pr[:, L.k] - after[:, L.k]).clamp(min=0)
            out.setdefault('frac_true', {})[name] = float(demote[yp == L.k].sum() / demote.sum())
    return out


def ci(x):
    x = np.asarray(x)
    h = stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / math.sqrt(len(x))
    return f'{100 * x.mean():+.2f} [{100 * (x.mean() - h):+.2f}, {100 * (x.mean() + h):+.2f}] p {stats.ttest_1samp(x, 0).pvalue:.4f}'


if __name__ == '__main__':
    seeds = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    procs = int(sys.argv[2]) if len(sys.argv) > 2 else 12
    with Pool(procs) as pool:
        rows = pool.map(run, range(seeds))
    json.dump(rows, open(__file__.replace('.py', '_results.json'), 'w'), indent=1)
    print(f'cap {L.CAP}; {seeds} seeds, early-stopped; binds {sum(r["binds"] for r in rows)}; '
          f'train FP at PTO {np.mean([r["n_fp"] for r in rows]):.1f}, train predicted-k {np.mean([r["n_pred"] for r in rows]):.1f}')
    for name in ('tralo', 'fp', 'trainall', 'oracle_fp', 'oracle_rank'):
        v = [r['frac_true'][name] for r in rows if 'frac_true' in r and name in r['frac_true']]
        print(f'  {name:9s} demotion mass on true-k {np.mean(v):.3f} (n {len(v)})')
    for metric in ('cc_f1', 'accuracy', 'macro_f1'):
        print(f'{metric}: means ' + '  '.join(f'{a} {100 * np.mean([r["scores"][a][metric] for r in rows]):.2f}'
                                          for a in ('pto', 'tralo', 'fp', 'trainall', 'fpsham', 'oracle_fp', 'oracle_rank')))
        for a, b in (('fp', 'pto'), ('fp', 'fpsham'), ('fp', 'tralo'), ('fp', 'trainall'), ('tralo', 'pto'), ('trainall', 'pto'),
                     ('oracle_fp', 'pto'), ('oracle_rank', 'pto')):
            print(f'   {a}-{b:9s} {ci([r["scores"][a][metric] - r["scores"][b][metric] for r in rows])}')
