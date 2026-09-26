"""Tiny CPU check: (1) an exponential tilt of class c (the posterior-regularization
projection of a count constraint) never changes the top-K set by p_c; (2) one
free-logit count-gradient step g_i = p_ic (e_c - p_i) is order preserving for small
steps and reorders only past a step bound; (3) a kernel-coupled step (shared
parameters) reorders at any step size, by feature similarity."""
import numpy as np
rng = np.random.default_rng(0)
N, C, c, K = 826, 5, 3, 76
def softmax(z):
    z = z - z.max(1, keepdims=True); e = np.exp(z); return e / e.sum(1, keepdims=True)
res = {}
# (1) tilt
fails = 0
for t in range(200):
    p = softmax(rng.normal(0, 3, (N, C)))
    lam = rng.uniform(0, 10)
    q = p.copy(); q[:, c] *= np.exp(-lam); q /= q.sum(1, keepdims=True)
    fails += set(np.argsort(-p[:, c])[:K]) != set(np.argsort(-q[:, c])[:K])
res['tilt_topK_changes_of_200'] = int(fails)
# (2) free-logit step; smallest step that changes the top-K set / any pair order
def first_reorder(p, z, kind):
    g = p[:, [c]] * (np.eye(C)[c] - p)
    base_top = set(np.argsort(-p[:, c])[:K]); base_rank = np.argsort(np.argsort(-p[:, c]))
    for eta in np.geomspace(1e-3, 1e3, 600):
        q = softmax(z - eta * g)[:, c]
        if kind == 'topK' and set(np.argsort(-q)[:K]) != base_top: return eta
        if kind == 'pair' and (np.argsort(np.argsort(-q)) != base_rank).any(): return eta
    return np.inf
etas_pair, etas_top = [], []
for t in range(30):
    z = rng.normal(0, 3, (N, C)); p = softmax(z)
    etas_pair.append(first_reorder(p, z, 'pair')); etas_top.append(first_reorder(p, z, 'topK'))
res['free_step_first_pair_reorder_eta_median_min'] = [float(np.median(etas_pair)), float(np.min(etas_pair))]
res['free_step_first_topK_change_eta_median_min'] = [float(np.median(etas_top)), float(np.min(etas_top))]
# (3) kernel-coupled step: dz_j = -eta * sum_i K(x_j,x_i) g_i  (NTK-style linearisation)
changed = []
for t in range(30):
    z = rng.normal(0, 3, (N, C)); p = softmax(z)
    x = rng.normal(0, 1, (N, 16)); Kmat = np.exp(-((x[:, None] - x[None]) ** 2).sum(-1) / 16)
    g = p[:, [c]] * (np.eye(C)[c] - p)
    base = set(np.argsort(-p[:, c])[:K])
    eta = 1e-2 / Kmat.sum(1).mean()   # tiny step, 1e-2 in per-item units
    q = softmax(z - eta * Kmat @ g)[:, c]
    changed.append(len(base - set(np.argsort(-q)[:K])))
res['kernel_step_topK_items_changed_at_eta_1e-2_mean_max'] = [float(np.mean(changed)), int(np.max(changed))]
print(res)
