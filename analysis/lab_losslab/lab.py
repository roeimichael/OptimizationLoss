"""losslab: synthetic testbed for WHEN a transductive count constraint carries WHO information.

CPU only. Gaussian mixtures, 5 classes, class 3 is capped. For every seed:
  base model  = MLP after E CE steps on the labelled train set
  clip        = base, post-hoc capped_first (top-cap by p3 over the pool)
  target      = base + tralo.targeted_step on the pool (bisection to the hard cap), then top-cap by p3
  sham        = same radius, seeded random direction with the real step's per-tensor norms
  ce_more     = base + K more CE steps, then top-cap by p3            (matched compute for joint)
  joint       = base + K steps of CE + bounded_count_penalty(pool) with advance_controller, then top-cap
  bayes       = true posterior under POOL priors/distribution (headroom oracle, not an arm)
Metric: precision@cap on the pool against ground truth. Pool labels are used only for scoring.
"""
import argparse, copy, json, math, sys, time
import numpy as np
import torch
import torch.nn as nn

from tralo.targeted_step import targeted_step
from tralo.global_constraint import bounded_count_penalty, advance_controller
from tralo.streamed_constraint import count_logit_gradient

C3 = 3
K = 5


# ----------------------------------------------------------------------------- data
def log_gauss(x, mu, sigma):
    d = x.shape[1]
    return -0.5 * ((x[:, None, :] - mu[None]) ** 2).sum(-1) / sigma ** 2 - d * math.log(sigma)


def make(cfg, seed):
    rng = np.random.default_rng(seed)
    D, sigma = cfg['D'], cfg.get('sigma', 1.0)
    mu = rng.normal(0, cfg['sep'] / math.sqrt(2 * D), size=(K, D)) * math.sqrt(2)  # E|mu_a-mu_b| ~ sep*sqrt(2)
    out = dict(mu=mu)
    # optional sub-cluster R of class `r_class`, placed near class 3
    R = cfg.get('R')
    if R:
        u = rng.normal(size=D); u /= np.linalg.norm(u)
        muR = mu[C3] + R['dist'] * u
        out['muR'] = muR

    def sample(n, priors, r_frac=0.0, shift=None):
        y = rng.choice(K, size=n, p=priors)
        x = mu[y] + sigma * rng.normal(size=(n, D))
        inR = np.zeros(n, bool)
        if R:
            cand = np.where(y == R['cls'])[0]
            m = rng.random(len(cand)) < r_frac
            idx = cand[m]
            x[idx] = muR + sigma * R.get('rsig', 1.0) * rng.normal(size=(len(idx), D))
            inR[idx] = True
        if shift is not None:
            x = x + shift
        return x, y, inR

    ptr = np.array(cfg.get('train_priors', [0.2] * K))
    xtr, ytr, rtr = sample(cfg['n_train'], ptr, R['r_frac_train'] if R else 0.0)
    # training label noise
    xu, _, _ = sample(cfg.get('n_pool', 800), ptr, R['r_frac_train'] if R else 0.0)
    ytr_obs = ytr.copy()
    if R:
        idx = np.where(rtr)[0]
        flip = rng.random(len(idx)) < R['flip']
        ytr_obs[idx[flip]] = R['flip_to']
    if cfg.get('noise', 0) > 0:
        f = rng.random(len(ytr)) < cfg['noise']
        ytr_obs[f] = rng.integers(0, K, f.sum())
    groups = cfg.get('groups')
    if groups:
        xs, ys, gs, rs = [], [], [], []
        for g, G in enumerate(groups):
            shift = None
            if G.get('shift', 0):
                v = mu[C3] - mu.mean(0); v /= np.linalg.norm(v)
                shift = G['shift'] * v
            x, y, r = sample(G['n'], np.array(G['priors']), shift=shift)
            xs.append(x); ys.append(y); gs.append(np.full(len(y), g)); rs.append(r)
        xp, yp, gp, rp = np.concatenate(xs), np.concatenate(ys), np.concatenate(gs), np.concatenate(rs)
        # bayes posterior per group (knows the group's prior and shift)
        post = []
        for g, G in enumerate(groups):
            m = gp == g
            sh = np.zeros(D)
            if G.get('shift', 0):
                v = mu[C3] - mu.mean(0); v /= np.linalg.norm(v); sh = G['shift'] * v
            lp = log_gauss(xp[m] - sh, mu, sigma) + np.log(np.array(G['priors']))
            post.append((m, lp))
        bayes = np.zeros(len(yp))
        for m, lp in post:
            lp = lp - lp.max(1, keepdims=True); p = np.exp(lp); p /= p.sum(1, keepdims=True)
            bayes[m] = p[:, C3]
    else:
        pp = np.array(cfg.get('pool_priors', [0.2] * K))
        xp, yp, rp = sample(cfg['n_pool'], pp, R['r_frac_pool'] if R else 0.0)
        if R and R.get('pool_truth') is not None:
            yp = yp.copy(); yp[rp] = R['pool_truth']
        gp = np.zeros(len(yp), int)
        # bayes: mixture incl. R component
        lp = log_gauss(xp, mu, sigma) + np.log(pp)
        if R:
            # class r_class density = (1-f) N(mu) + f N(muR); in pool, R truth may be relabelled
            f = R['r_frac_pool']
            lR = log_gauss(xp, muR[None], sigma * R.get('rsig', 1.0))[:, 0] - (D * math.log(R.get('rsig', 1.0)))*0
            c = R['cls']
            truthR = R.get('pool_truth', c)
            base_c = lp[:, c] + np.log(1 - f)
            rterm = lR + np.log(pp[c]) + np.log(f)
            lp[:, c] = base_c
            # add R mass to the class it truly belongs to in the pool
            lp[:, truthR] = np.logaddexp(lp[:, truthR], rterm)
        lp = lp - lp.max(1, keepdims=True); p = np.exp(lp); p /= p.sum(1, keepdims=True)
        bayes = p[:, C3]
    out.update(xu=xu, xtr=xtr, ytr=ytr_obs, ytr_clean=ytr, xp=xp, yp=yp, gp=gp, rp=rp, bayes=bayes)
    return out


# ----------------------------------------------------------------------------- model
class MLP(nn.Module):
    def __init__(self, D, H, depth=2):
        super().__init__()
        layers, d = [], D
        for _ in range(depth):
            layers += [nn.Linear(d, H), nn.ReLU()]; d = H
        self.body = nn.Sequential(*layers)
        self.head = nn.Linear(d, K)

    def forward(self, x):
        return self.head(self.body(x))


def ce_steps(model, opt, x, y, steps):
    lossf = nn.CrossEntropyLoss()
    model.train()
    for _ in range(steps):
        opt.zero_grad(); lossf(model(x), y).backward(); opt.step()


def probs(model, x):
    model.eval()
    with torch.no_grad():
        return model(x).softmax(1)


def topcap(p3, cap):
    order = np.lexsort((np.arange(len(p3)), -p3))
    return order[:cap]


def local_top(p3, groups, caps):
    sel = []
    for g, cap in enumerate(caps):
        idx = np.where(groups == g)[0]
        sel.extend(idx[topcap(p3[idx], cap)])
    return np.array(sel)


def prec(sel, y):
    return float((y[sel] == C3).mean()) if len(sel) else float('nan')


def who_auc(margin_delta, y, band):
    """AUC of -delta(margin) for 'not class 3' within the band: >0.5 = the step pushed non-3 items down more."""
    d = -margin_delta[band]; t = (y[band] != C3)
    if t.all() or (~t).all():
        return float('nan')
    r = d.argsort().argsort().astype(float)
    n1 = t.sum(); n0 = (~t).sum()
    return float((r[t].sum() - n1 * (n1 - 1) / 2) / (n1 * n0))


def margin(p):
    p = p.clamp_min(1e-30)
    l = p.log()
    other = torch.cat([l[:, :C3], l[:, C3 + 1:]], 1).logsumexp(1)
    return (l[:, C3] - other).numpy()


# ----------------------------------------------------------------------------- multi-group targeted step (S4)
def targeted_multi(model, x, groups, caps, sham_gen=None, r0=1e-3, iters=20):
    """Minimal r along -grad(sum of violating groups' soft counts) so every group's hard count <= its cap."""
    params = [p for p in model.parameters() if p.requires_grad]

    def hard():
        a = probs(model, x).argmax(1).numpy()
        return [int(((a == C3) & (groups == g)).sum()) for g in range(len(caps))]

    h0 = hard()
    viol = [g for g in range(len(caps)) if h0[g] > caps[g]]
    if not viol:
        return dict(applied=False)
    model.eval()
    model.zero_grad()
    pr = model(x).softmax(1)
    mask = torch.tensor(np.isin(groups, viol))
    pr[mask, C3].sum().backward()
    grads = [p.grad.detach().clone() for p in params]
    norm = math.sqrt(sum(float(g.square().sum()) for g in grads))
    unit = [-g / norm for g in grads]
    origin = [p.detach().clone() for p in params]

    def place(d, r):
        with torch.no_grad():
            for p, o, u in zip(params, origin, d):
                p.copy_(o + r * u)

    ok = lambda: all(h <= c for h, c in zip(hard(), caps))
    lo, hi = 0.0, r0
    for _ in range(40):
        place(unit, hi)
        if ok():
            break
        lo, hi = hi, 2 * hi
    else:
        place(unit, 0.0); return dict(applied=False, failed=True)
    for _ in range(iters):
        mid = (lo + hi) / 2; place(unit, mid)
        if ok(): hi = mid
        else: lo = mid
    step = unit
    if sham_gen is not None:
        step = []
        for u in unit:
            n = torch.randn(u.shape, generator=sham_gen, dtype=torch.float64)
            step.append((n * (float(u.norm()) / float(n.norm()))).to(u.dtype))
    place(step, hi)
    return dict(applied=True, radius=hi, hard_after=hard())


def targeted_seq(model, x, groups, caps, sham_gen=None, rounds=2):
    """Per-group targeted steps: direction = -grad(this group's soft count), bisection on THIS group's hard count."""
    info = []
    for _ in range(rounds):
        for g in range(len(caps)):
            sub = [cap if h == g else 10 ** 9 for h, cap in enumerate(caps)]
            info.append(targeted_multi(model, x, groups, sub, sham_gen))
    return info


# ----------------------------------------------------------------------------- one seed
def run_seed(cfg, seed):
    torch.manual_seed(seed)
    d = make(cfg, seed)
    X = torch.tensor(d['xtr'], dtype=torch.float32); Y = torch.tensor(d['ytr'])
    XP = torch.tensor(d['xp'], dtype=torch.float32); yp = d['yp']; gp = d['gp']
    model = MLP(cfg['D'], cfg['H'], cfg.get('depth', 2))
    opt = torch.optim.Adam(model.parameters(), lr=cfg.get('lr', 3e-3), weight_decay=cfg.get('wd', 1e-4))
    ce_steps(model, opt, X, Y, cfg['E'])
    base_state = {k: v.clone() for k, v in model.state_dict().items()}
    opt_state = copy.deepcopy(opt.state_dict())  # Adam state must not be shared across arms
    p0 = probs(model, XP)
    tr_acc = float((probs(model, X).argmax(1) == Y).float().mean())
    tr_acc_clean = float((probs(model, X).argmax(1).numpy() == d['ytr_clean']).mean())
    p3_0 = p0[:, C3].numpy().astype(np.float64)
    n_true = int((yp == C3).sum()); hard0 = int((p0.argmax(1) == C3).sum())
    res = dict(seed=seed, train_acc=tr_acc, train_acc_clean=tr_acc_clean, n_true3=n_true, hard0=hard0,
               soft0=float(p3_0.sum()), pool_acc=float((p0.argmax(1).numpy() == yp).mean()))
    groups = cfg.get('groups')
    if groups:
        caps = [int(G['cap']) for G in groups]
        gcap = sum(caps)
        res['hard0_g'] = [int(((p0.argmax(1).numpy() == C3) & (gp == g)).sum()) for g in range(len(caps))]
        res['clip_local'] = prec(local_top(p3_0, gp, caps), yp)
        res['clip_global'] = prec(topcap(p3_0, gcap), yp)
        res['bayes_local'] = prec(local_top(d['bayes'], gp, caps), yp)
        res['bayes_global'] = prec(topcap(d['bayes'], gcap), yp)
        for arm, gen in (('target', None), ('sham', torch.Generator().manual_seed(10**6 + seed))):
            model.load_state_dict(base_state)
            info = targeted_multi(model, XP, gp, caps, gen)
            p = probs(model, XP)[:, C3].numpy().astype(np.float64)
            res[arm + '_local'] = prec(local_top(p, gp, caps), yp)
            res[arm + '_global'] = prec(topcap(p, gcap), yp)
            res[arm + '_applied'] = bool(info.get('applied'))
            res[arm + '_hard_after_g'] = info.get('hard_after')
            res[arm + '_radius'] = info.get('radius')
        for arm, gen in (('tseq', None), ('tseqsham', torch.Generator().manual_seed(2 * 10**6 + seed))):
            model.load_state_dict(base_state)
            targeted_seq(model, XP, gp, caps, gen)
            pa = probs(model, XP)
            p = pa[:, C3].numpy().astype(np.float64)
            res[arm + '_local'] = prec(local_top(p, gp, caps), yp)
            res[arm + '_global'] = prec(topcap(p, gcap), yp)
            res[arm + '_hard_after_g'] = [int(((pa.argmax(1).numpy() == C3) & (gp == g)).sum()) for g in range(len(caps))]
        # ce_more vs joint with LOCAL penalty (sum over groups) -- and with a GLOBAL penalty
        for arm in ('ce_more', 'joint_local', 'joint_global'):
            model.load_state_dict(base_state)
            o2 = torch.optim.Adam(model.parameters(), lr=cfg.get('lr', 3e-3), weight_decay=cfg.get('wd', 1e-4))
            o2.load_state_dict(copy.deepcopy(opt_state))
            run_joint(model, o2, X, Y, XP, gp, caps if arm != 'joint_global' else [gcap],
                      arm, cfg, res, glob=(arm == 'joint_global'))
            p = probs(model, XP)[:, C3].numpy().astype(np.float64)
            res[arm + '_local'] = prec(local_top(p, gp, caps), yp)
            res[arm + '_global'] = prec(topcap(p, gcap), yp)
        return res
    cap = int(cfg['cap']) if 'cap' in cfg else int(round(cfg['cap_frac'] * n_true))
    res['cap'] = cap
    sel0 = topcap(p3_0, cap)
    res['clip'] = prec(sel0, yp)
    res['bayes'] = prec(topcap(d['bayes'], cap), yp)
    res['frac_p3_gt_0999'] = float((p3_0 > 0.999).mean())
    # post-hoc prior correction (Saerens-Latinne-Decaestecker EM) on the pool, then top-cap: label-free
    ptr_ = np.array(cfg.get('train_priors', [0.2] * K)); P_ = p0.numpy().astype(np.float64); w_ = ptr_.copy()
    for _ in range(200):
        Q_ = P_ * (w_ / ptr_); Q_ /= Q_.sum(1, keepdims=True); w_ = Q_.mean(0)
    res['clip_em'] = prec(topcap(Q_[:, C3], cap), yp)
    res['em_prior3'] = float(w_[C3])
    if d['rp'].any():
        res['R_in_clip'] = int(d['rp'][sel0].sum()); res['R_pool'] = int(d['rp'].sum())
    m0 = margin(p0)
    rank0 = np.empty(len(p3_0), int); rank0[np.lexsort((np.arange(len(p3_0)), -p3_0))] = np.arange(len(p3_0))
    band = (rank0 >= max(0, cap - 60)) & (rank0 < cap + 60)
    for arm, gen in (('target', None), ('sham', torch.Generator().manual_seed(10**6 + seed))):
        model.load_state_dict(base_state)
        caps = [None] * K; caps[C3] = cap
        try:
            info = targeted_step(model, [XP], caps, sham_generator=gen)
        except RuntimeError as e:
            res[arm + '_error'] = str(e); info = dict(applied=False)
        p = probs(model, XP)
        p3 = p[:, C3].numpy().astype(np.float64)
        sel = topcap(p3, cap)
        res[arm] = prec(sel, yp)
        res[arm + '_applied'] = bool(info.get('applied'))
        res[arm + '_hard_after'] = int(info.get('hard_after', hard0))
        res[arm + '_overlap'] = len(set(sel) & set(sel0)) / cap if cap else float('nan')
        admitted = np.setdiff1d(sel, sel0); evicted = np.setdiff1d(sel0, sel)
        res[arm + '_n_swap'] = int(len(admitted))
        res[arm + '_adm_true3'] = int((yp[admitted] == C3).sum()); res[arm + '_evi_true3'] = int((yp[evicted] == C3).sum())
        res[arm + '_who_auc'] = who_auc(margin(p) - m0, yp, band)
        if d['rp'].any():
            res[arm + '_R_in'] = int(d['rp'][sel].sum())
    # persistence: interleave CE blocks and targeted steps (the real protocol's shape), 5 rounds
    R_ = cfg.get('rounds', 5); blk = cfg.get('Kjoint', 100) // R_
    for arm, sg in (('trep', None), ('srep', torch.Generator().manual_seed(3 * 10**6 + seed)), ('crep', 'none')):
        model.load_state_dict(base_state)
        o2 = torch.optim.Adam(model.parameters(), lr=cfg.get('lr', 3e-3), weight_decay=cfg.get('wd', 1e-4))
        o2.load_state_dict(copy.deepcopy(opt_state))
        gains, decays, last = [], [], None
        for r_ in range(R_):
            ce_steps(model, o2, X, Y, blk)
            pb_ = probs(model, XP)[:, C3].numpy().astype(np.float64); cb = prec(topcap(pb_, cap), yp)
            if last is not None:
                decays.append(cb - last)
            if sg != 'none':
                cs = [None] * K; cs[C3] = cap
                try:
                    targeted_step(model, [XP], cs, sham_generator=sg)
                except RuntimeError:
                    pass
            pa_ = probs(model, XP)[:, C3].numpy().astype(np.float64); ca = prec(topcap(pa_, cap), yp)
            gains.append(ca - cb); last = ca
        res[arm] = last
        res[arm + '_gain_per_step'] = float(np.mean(gains)); res[arm + '_decay_per_block'] = float(np.mean(decays))
    XU = torch.tensor(d['xu'], dtype=torch.float32)
    pce = None
    for w in [None] + list(cfg.get('pen_ws', [1.0])):
        for src in (['pool'] if w is None else ['pool', 'trainpool']):
            arm = 'ce_more' if w is None else ('joint' if src == 'pool' else 'jtrain') + '_w%g' % w
            model.load_state_dict(base_state)
            o2 = torch.optim.Adam(model.parameters(), lr=cfg.get('lr', 3e-3), weight_decay=cfg.get('wd', 1e-4))
            o2.load_state_dict(copy.deepcopy(opt_state))
            c2 = dict(cfg); c2['pen_w'] = w if w is not None else 0.0
            if src == 'pool':
                run_joint(model, o2, X, Y, XP, gp, [cap], 'ce_more' if w is None else 'joint', c2, res, glob=True)
            else:
                # same RELATIVE cut on a fresh unlabeled draw from the training distribution
                hu = int((probs(model, XU).argmax(1) == C3).sum())
                capu = int(round(hu * cap / max(hard0, 1)))
                run_joint(model, o2, X, Y, XU, np.zeros(len(XU), int), [capu], 'joint', c2, res, glob=True)
            p = probs(model, XP); p3 = p[:, C3].numpy().astype(np.float64)
            sel = topcap(p3, cap)
            res[arm] = prec(sel, yp)
            res[arm + '_hard'] = int((p.argmax(1) == C3).sum())
            if d['rp'].any():
                res[arm + '_R_in'] = int(d['rp'][sel].sum())
            if w is None:
                pce, selce = p, sel
                pr = p3; rk = np.empty(len(pr), int); rk[np.lexsort((np.arange(len(pr)), -pr))] = np.arange(len(pr))
                bandce = (rk >= max(0, cap - 60)) & (rk < cap + 60)
            else:
                res[arm + '_who_auc'] = who_auc(margin(p) - margin(pce), yp, bandce)
                res[arm + '_overlap_ce'] = len(set(sel) & set(selce)) / cap
    return res


def run_joint(model, opt, X, Y, XP, gp, caps, arm, cfg, res, glob):
    """K steps of CE (+ bounded_count_penalty on the pool for joint arms), controller on hard counts."""
    lossf = nn.CrossEntropyLoss()
    ng = len(caps)
    lam = [cfg.get('lam0', 1.0)] * ng; rho = cfg.get('rho0', 1.0); frozen = False
    Kst = cfg.get('Kjoint', 100)
    for t in range(Kst):
        model.train(); opt.zero_grad()
        loss = lossf(model(X), Y)
        if arm != 'ce_more':
            lp = model(XP)
            for g in range(ng):
                sub = lp if glob else lp[torch.tensor(gp == g)]
                cs = [None] * K; cs[C3] = int(caps[g])
                mult = torch.zeros(K); mult[C3] = lam[g]
                loss = loss + cfg.get('pen_w', 10.0) * bounded_count_penalty(sub, cs, mult, rho)
        loss.backward(); opt.step()
        if arm != 'ce_more':
            a = probs(model, XP).argmax(1).numpy()
            hard = [int(((a == C3) & (True if glob else gp == g)).sum()) for g in range(ng)]
            # one scalar controller per group
            new = []
            for g in range(ng):
                l2, r2, _ = advance_controller([hard[g]], [int(caps[g])], [lam[g]], rho,
                                              cfg.get('rho_step', 0.5), cfg.get('lam_step', 0.5), False)
                new.append(l2[0])
            if any(h > c for h, c in zip(hard, caps)):
                rho += cfg.get('rho_step', 0.5)
            lam = new
    if arm != 'ce_more':
        a = probs(model, XP).argmax(1).numpy()
        res[arm + '_final_lam'] = lam


# ----------------------------------------------------------------------------- stats
def paired(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    m = ~(np.isnan(a) | np.isnan(b)); dlt = a[m] - b[m]; n = len(dlt)
    if n < 2:
        return dict(n=n)
    from scipy import stats
    se = dlt.std(ddof=1) / math.sqrt(n); t = stats.t.ppf(0.975, n - 1)
    return dict(n=n, mean=float(dlt.mean()), lo=float(dlt.mean() - t * se), hi=float(dlt.mean() + t * se),
                wins=int((dlt > 1e-12).sum()), losses=int((dlt < -1e-12).sum()), ties=int((np.abs(dlt) <= 1e-12).sum()))


def mean_ci(a):
    a = np.asarray(a, float); a = a[~np.isnan(a)]; n = len(a)
    if n < 2:
        return dict(n=n)
    se = a.std(ddof=1) / math.sqrt(n)
    return dict(n=n, mean=float(a.mean()), lo=float(a.mean() - 2.07 * se), hi=float(a.mean() + 2.07 * se))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('config'); ap.add_argument('name'); ap.add_argument('--seeds', type=int, default=24)
    ap.add_argument('--out', required=True); ap.add_argument('--start', type=int, default=0)
    a = ap.parse_args()
    cfgs = json.load(open(a.config))
    cfg = cfgs[a.name]
    t0 = time.time()
    rows = [run_seed(cfg, 5000 + s) for s in range(a.start, a.start + a.seeds)]
    json.dump(dict(name=a.name, cfg=cfg, seconds=time.time() - t0, rows=rows), open(a.out, 'w'))
    print(a.name, a.start, 'sec %.0f' % (time.time() - t0))
