"""Compare top-K membership change (items evicted from the top-76 by p_3) for
(a) free-logit count step, (b) the same step routed through a feature kernel
(row-normalised, diagonal weight 1/2), at equal eta; and check the closed form
dp3 = -eta p3^2 [(1-p3)^2 + sum_{k!=3} p_k^2] to first order."""
import numpy as np
rng = np.random.default_rng(1)
N, C, c, K = 826, 5, 3, 76
def softmax(z):
    z = z - z.max(1, keepdims=True); e = np.exp(z); return e / e.sum(1, keepdims=True)
out = {}
z = rng.normal(0, 3, (N, C)); p = softmax(z)
g = p[:, [c]] * (np.eye(C)[c] - p)
eta = 1e-4
num = (softmax(z - eta * g)[:, c] - p[:, c]) / eta
ana = -p[:, c] ** 2 * ((1 - p[:, c]) ** 2 + (np.delete(p, c, 1) ** 2).sum(1))
out['closed_form_max_abs_err'] = float(np.abs(num - ana).max())
# spearman between first-order push and p3 alone
from scipy.stats import spearmanr
out['spearman(push, p3*(1-p3)) and (push,p3)'] = [float(spearmanr(-ana, (p[:, c]*(1-p[:, c]))**2).correlation), float(spearmanr(-ana, p[:, c]).correlation)]
rows = []
for eta in [0.3, 1, 3, 10]:
    fr, kr = [], []
    for t in range(20):
        z = rng.normal(0, 3, (N, C)); p = softmax(z); g = p[:, [c]] * (np.eye(C)[c] - p)
        x = rng.normal(0, 1, (N, 16)); Km = np.exp(-((x[:, None] - x[None]) ** 2).sum(-1) / 16)
        np.fill_diagonal(Km, 0); Km = 0.5 * Km / Km.sum(1, keepdims=True) + 0.5 * np.eye(N)
        base = set(np.argsort(-p[:, c])[:K])
        fr.append(len(base - set(np.argsort(-softmax(z - eta * g)[:, c])[:K])))
        kr.append(len(base - set(np.argsort(-softmax(z - eta * Km @ g)[:, c])[:K])))
    rows.append((eta, float(np.mean(fr)), float(np.std(fr)), float(np.mean(kr)), float(np.std(kr))))
out['eta, free_changed_mean,sd, kernel_changed_mean,sd (n=20 pools, K=76)'] = rows
print(out)
