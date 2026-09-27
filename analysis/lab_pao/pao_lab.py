"""Synthetic CPU lab: Yuval Kassif's PAO (CustomLoss + outer loop) vs TraLO's targeted count step.

Usage: python pao_lab.py [SEEDS] [PROCS]

5 ordinal classes (KL-like prevalence), constrained class k=3 with confusers 2 and 4, 1 informative
latent + 15 noise dims (so an MLP can memorise), randomly rotated. Class-balanced sampler (Yuval's
recipe) makes k over-predicted, so the cap binds. Cap = round(0.7 * expected pool k-count) = 76.
Pool labels are read only by `score`. Constant LR in both regimes, so the regimes differ only in
when training stops: 'es' = early stopping (patience 5, train-carved split, criterion = the
training loss, best state restored, as train.train_model); 'mem' = train until 100% train accuracy.
Methods, all deployed with capped_first at the cap:
  pto    the first outer-loop model (CustomLoss with C = 1, i.e. CE up to 1e-7)
  pao    the model after the outer loop (mu 8/600, b 100, F/dF closed form, reseed per iteration)
  tralo  pto + ONE step along -grad(sum_pool p_k), smallest radius with hard count <= cap
  sham   pto + the same radius, seeded random direction with the real step's per-tensor norms
"""
import copy
import json
import math
import os
import sys
from multiprocessing import Pool

import numpy as np
import torch
import torch.nn.functional as F
from scipy import stats

K, k = 5, 3
PRIOR = np.array([2286, 1046, 1516, 757, 173], float) / 5778
D, N_TRAIN, N_ES, N_POOL = 16, 1350, 150, 826
CAP = int(round(0.7 * N_POOL * PRIOR[k]))
LR, WD, BATCH, PATIENCE = 1e-3, 1e-4, 64, 5
MAX_EPOCHS = {'es': 300, 'mem': 4000}
SCALE = float(os.environ.get('PAO_MU_SCALE', 1))   # sensitivity only; 1 = Yuval's mu
MU, B, MAX_IT = 8 / 600 * SCALE, 100.0, 8
OUT = __file__.replace('.py', f'_results_mu{SCALE:g}.json')


def data(seed):
    r = np.random.RandomState(seed)
    Q = np.linalg.qr(r.randn(D, D))[0]

    def draw(n):
        y = r.choice(K, n, p=PRIOR)
        z = r.randn(n, D)
        z[:, 0] = y + 0.6 * z[:, 0]
        return torch.tensor(z @ Q, dtype=torch.float32), torch.tensor(y, dtype=torch.long)
    X, y = draw(N_TRAIN + N_ES)
    P, yp = draw(N_POOL)
    return X[:N_TRAIN], y[:N_TRAIN], X[N_TRAIN:], y[N_TRAIN:], P, yp


def mlp():
    return torch.nn.Sequential(torch.nn.Linear(D, 256), torch.nn.ReLU(), torch.nn.Linear(256, 256),
                               torch.nn.ReLU(), torch.nn.Linear(256, K))


def pao_loss(logits, y, C):
    """losses.CustomLoss verbatim: CE weighted by C[y] where argmax is k, plain CE elsewhere."""
    p = logits.softmax(1)
    yt = F.one_hot(y, K).float()
    t = torch.tanh(50000000 * (p.max(1)[0] - p[:, k])).unsqueeze(1)
    l1 = -(yt * (C.unsqueeze(0) * torch.log(1e-7 + torch.relu(1 + (p - 1) * (1 - t))))).sum(1)
    l2 = -(yt * torch.log(1e-7 + torch.relu(1 + (p - 1) * t))).sum(1)
    return (l1 + l2).mean(), float(t.mean().clamp(0, 1))


def F_and_dF(N, n, b):
    """update_weights.calculate_F_and_derivative in closed form."""
    t = math.tanh(b * (n - N))
    return (N - n) ** 2 * (t + 1), -2 * (N - n) * (t + 1) + (N - n) ** 2 * b * (1 - t * t)


def soft(m, X):
    m.eval()
    with torch.no_grad():
        return m(X).softmax(1)


def train(d, init, C, regime, seed):
    X, y, Xe, ye = d[:4]
    m = mlp()
    m.load_state_dict(init)
    opt = torch.optim.Adam(m.parameters(), lr=LR, weight_decay=WD)
    g = torch.Generator().manual_seed(seed + 1)          # same draws every outer iteration
    w = (1 / torch.bincount(y, minlength=K).float())[y]   # WeightedRandomSampler weights
    best, state, bad, live, memorised = math.inf, None, 0, [], False
    for epoch in range(MAX_EPOCHS[regime]):
        idx = torch.multinomial(w, len(y), replacement=True, generator=g)
        m.train()
        fp = 0
        for s in range(0, len(idx), BATCH):
            b = idx[s:s + BATCH]
            out = m(X[b])
            loss, t = pao_loss(out, y[b], C)
            fp += int(((out.argmax(1) == k) & (y[b] != k)).sum())   # samples weighted by C[y] != 1
            for pg in opt.param_groups:                            # train.py dynamic LR
                pg['lr'] = (1 - t) * LR / float(C.mean()) + t * LR
            opt.zero_grad()
            loss.backward()
            opt.step()
        live.append(fp)
        if regime == 'es':
            with torch.no_grad():
                m.eval()
                v = float(pao_loss(m(Xe), ye, C)[0])
            if v < best:
                best, state, bad = v, copy.deepcopy(m.state_dict()), 0
            else:
                bad += 1
            if bad >= PATIENCE:
                break
        elif bool((soft(m, X).argmax(1) == y).all()):
            memorised = True
            break
    if regime == 'es':
        m.load_state_dict(state)
    pr = soft(m, X).argmax(1)
    return m, dict(epochs=epoch + 1, live=float(np.mean(live)), live_last=float(np.mean(live[-5:])),
                   train_acc=float((pr == y).float().mean()), train_fp=int(((pr == k) & (y != k)).sum()),
                   memorised=memorised)


def step(m, P, sham_seed=None):
    """tralo.targeted_step on a small model: bracket by doubling, then bisection on the hard count."""
    params = list(m.parameters())
    if int((soft(m, P).argmax(1) == k).sum()) <= CAP:
        return None
    m.zero_grad()
    m.eval()
    m(P).softmax(1)[:, k].sum().backward()
    g = [p.grad.detach().clone() for p in params]
    norm = math.sqrt(sum(float(x.double().square().sum()) for x in g))
    unit = [-x / norm for x in g]
    origin = [p.detach().clone() for p in params]

    def place(r, dirs):
        with torch.no_grad():
            for p, o, u in zip(params, origin, dirs):
                p.copy_(o + r * u)

    def hard():
        return int((soft(m, P).argmax(1) == k).sum())
    lo, hi = 0.0, 1e-3
    for _ in range(60):
        place(hi, unit)
        if hard() <= CAP:
            break
        lo, hi = hi, 2 * hi
    else:
        raise RuntimeError('no radius meets the cap')
    for _ in range(30):
        mid = 0.5 * (lo + hi)
        place(mid, unit)
        if hard() <= CAP:
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


def capped_first(pr):
    other = pr.clone()
    other[:, k] = -1
    pred = other.argmax(1)
    pred[torch.argsort(-pr[:, k], stable=True)[:CAP]] = k
    return pred


def score(pr, y):
    pred = capped_first(pr)
    f1 = []
    for c in range(K):
        tp = int(((pred == c) & (y == c)).sum())
        den = int((pred == c).sum()) + int((y == c).sum())
        f1.append(2 * tp / den if den else 0.0)
    return dict(cc_f1=f1[k], accuracy=float((pred == y).float().mean()), macro_f1=float(np.mean(f1)))


def run(job):
    seed, regime = job
    torch.set_num_threads(1)
    d = data(seed)
    P, yp = d[4], d[5]
    torch.manual_seed(seed)
    init = copy.deepcopy(mlp().state_dict())
    C, its = torch.ones(K), []
    for it in range(1, MAX_IT + 1):
        m, info = train(d, init, C, regime, seed)
        pr = soft(m, P)
        n = int((pr.argmax(1) == k).sum())
        Fv, dF = F_and_dF(CAP, n, B)
        its.append(dict(info, pool_hard=n, C_other=float(C[0])))
        if it == 1:
            pto, pto_pr = m, pr
        if Fv < 1e-5:
            break
        C = C + MU * dF
        C[k] = 1
    out = dict(seed=seed, regime=regime, binds=its[0]['pool_hard'] > CAP, iterations=its,
               converged=its[-1]['pool_hard'] <= CAP, scores=dict(pto=score(pto_pr, yp), pao=score(pr, yp)))
    for name, sham in (('tralo', None), ('sham', seed + 7)):
        m = copy.deepcopy(pto)
        r = step(m, P, sham)
        after = soft(m, P)
        out['scores'][name] = score(after, yp)
        if name == 'tralo' and r is not None:
            demote = (pto_pr[:, k] - after[:, k]).clamp(min=0)
            out['step'] = dict(radius=r, frac_true=float(demote[yp == k].sum() / demote.sum()),
                               share_true=float(pto_pr[yp == k, k].sum() / pto_pr[:, k].sum()),
                               prec_cap_before=float((yp[torch.argsort(-pto_pr[:, k])[:CAP]] == k).float().mean()))
    return out


def ci(x):
    x = np.asarray(x)
    h = stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / math.sqrt(len(x))
    return f'{x.mean():+.4f} [{x.mean() - h:+.4f}, {x.mean() + h:+.4f}]'


if __name__ == '__main__':
    seeds = int(sys.argv[1]) if len(sys.argv) > 1 else 40
    procs = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    jobs = [(s, r) for r in os.environ.get('PAO_REGIMES', 'es,mem').split(',') for s in range(seeds)]
    with Pool(procs) as pool:
        rows = pool.map(run, jobs)
    json.dump(rows, open(OUT, 'w'), indent=1)
    print(f'cap {CAP}; pool {N_POOL}; mu {MU:.5f} (x{SCALE:g}); seeds {seeds} per regime; results {OUT}')
    for regime in sorted({r['regime'] for r in rows}):
        R = [r for r in rows if r['regime'] == regime]
        print(f'\n=== regime {regime}: binds {sum(r["binds"] for r in R)}/{len(R)}, PAO converged '
              f'{sum(r["converged"] for r in R)}/{len(R)}, mean iterations {np.mean([len(r["iterations"]) for r in R]):.2f}')
        first = [r['iterations'][0] for r in R]
        last = [r['iterations'][-1] for r in R]
        print(f'PTO: epochs {np.mean([i["epochs"] for i in first]):.1f}, train acc {np.mean([i["train_acc"] for i in first]):.3f}, '
              f'memorised {sum(i["memorised"] for i in first)}/{len(R)}, train FP at deploy {np.mean([i["train_fp"] for i in first]):.1f}, '
              f'live FP/epoch {np.mean([i["live"] for i in first]):.1f} (last5 {np.mean([i["live_last"] for i in first]):.1f}), pool hard {np.mean([i["pool_hard"] for i in first]):.1f}')
        print(f'PAO final: epochs {np.mean([i["epochs"] for i in last]):.1f}, C_other {np.mean([i["C_other"] for i in last]):.2f}, '
              f'train acc {np.mean([i["train_acc"] for i in last]):.3f}, train FP at deploy {np.mean([i["train_fp"] for i in last]):.1f}, '
              f'live FP/epoch (C>1 weighted) {np.mean([i["live"] for i in last]):.1f} (last5 {np.mean([i["live_last"] for i in last]):.1f}), pool hard {np.mean([i["pool_hard"] for i in last]):.1f}')
        st = [r['step'] for r in R if 'step' in r]
        if st:
            print(f'TraLO step (n={len(st)}): demotion mass on true-k {np.mean([s["frac_true"] for s in st]):.3f} vs true-k share of '
                  f'soft count {np.mean([s["share_true"] for s in st]):.3f}; PTO precision@cap {np.mean([s["prec_cap_before"] for s in st]):.3f}')
        for metric in ('cc_f1', 'accuracy', 'macro_f1'):
            means = '  '.join(f'{m} {np.mean([r["scores"][m][metric] for r in R]):.4f}' for m in ('pto', 'pao', 'tralo', 'sham'))
            print(f'{metric:9s} means: {means}')
            for a, b in (('pao', 'pto'), ('tralo', 'pto'), ('sham', 'pto'), ('tralo', 'sham'), ('pao', 'tralo')):
                print(f'   {a}-{b:6s} {ci([r["scores"][a][metric] - r["scores"][b][metric] for r in R])}')
