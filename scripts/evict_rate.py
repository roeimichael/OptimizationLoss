"""What does one unit of constraint displacement BUY, per surrogate?

constraint_step.py rescales the constraint gradient to a fixed norm, so every
step spends the same budget of logit displacement. The only question that
matters is what that budget buys: items actually evicted from the capped class,
and accuracy destroyed on the way. Same probabilities, same step size, two
surrogates for the same hard count.
"""
import numpy as np

def make(seed, n=4000, K=8, sharp=2.0):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, K, size=n)
    z = rng.normal(size=(n, K)); z[np.arange(n), y] += sharp
    return z, y

def softmax(z):
    e = np.exp(z - z.max(1, keepdims=True)); return e / e.sum(1, keepdims=True)

def grad_soft(p, c):
    """d/dZ of sum_i p_i(c): p_c(delta_cc' - p_c')."""
    g = -p * p[:, [c]]
    g[:, c] += p[:, c]
    return g

def grad_margin(z, p, c, tau=1.0):
    """d/dZ of sum_i sigmoid((z_c - max_{c'!=c} z_c')/tau), straight-through max."""
    zz = z.copy(); zz[:, c] = -np.inf
    rival = zz.argmax(1)
    m = z[:, c] - z[np.arange(len(z)), rival]
    s = 1.0 / (1.0 + np.exp(-m / tau))
    w = s * (1 - s) / tau
    g = np.zeros_like(z)
    g[:, c] += w
    g[np.arange(len(z)), rival] -= w
    return g

for sharp in (4.0, 2.0, 1.0):
    out = []
    for seed in range(1, 6):
        z, y = make(seed, sharp=sharp); p = softmax(z); c = 0
        base_in = (p.argmax(1) == c)
        base_acc = (p.argmax(1) == y).mean()
        row = []
        for name, g in (("soft", grad_soft(p, c)), ("margin", grad_margin(z, p, c))):
            g = g / np.linalg.norm(g)
            z2 = z - 20.0 * g                    # identical displacement budget
            a2 = softmax(z2).argmax(1)
            row.append(((base_in.sum() - (a2 == c).sum()),
                        base_acc - (a2 == y).mean()))
        out.append(row)
    a = np.array(out)                            # seeds x 2 x 2
    m = a.mean(0)
    print("sharp %.1f  |  soft: evicted %5.1f, accuracy cost %+.4f   "
          "|  margin: evicted %5.1f, accuracy cost %+.4f"
          % (sharp, m[0][0], m[0][1], m[1][0], m[1][1]))


def at_equal_evictions(sharp, target_frac=0.20, c=0):
    """Bisect the step size so BOTH surrogates evict the same number of items,
    then compare what that cost. Equal displacement is the wrong control --
    constraint_step rescales, and the dual keeps stepping until the cap is met,
    so the budget is set by the EVICTION TARGET, not by the step norm."""
    res = {}
    for name in ("soft", "margin"):
        costs, hits = [], []
        for seed in range(1, 6):
            z, y = make(seed, sharp=sharp); p = softmax(z)
            base_in = int((p.argmax(1) == c).sum())
            base_acc = (p.argmax(1) == y).mean()
            target = int(round(base_in * target_frac))
            g = grad_soft(p, c) if name == "soft" else grad_margin(z, p, c)
            g = g / np.linalg.norm(g)
            lo, hi = 0.0, 1.0
            while int((softmax(z - hi * g).argmax(1) == c).sum()) > base_in - target:
                hi *= 2.0
                if hi > 1e6:
                    break
            for _ in range(40):
                mid = 0.5 * (lo + hi)
                if int((softmax(z - mid * g).argmax(1) == c).sum()) > base_in - target:
                    lo = mid
                else:
                    hi = mid
            a2 = softmax(z - hi * g).argmax(1)
            hits.append(base_in - int((a2 == c).sum()))
            costs.append(base_acc - (a2 == y).mean())
        res[name] = (np.mean(hits), np.mean(costs))
    return res


print("")
print("EQUAL EVICTIONS (evict 20% of the capped class), 5 seeds:")
for sharp in (4.0, 2.0, 1.0):
    r = at_equal_evictions(sharp)
    s_n, s_c = r["soft"]; m_n, m_c = r["margin"]
    print("  sharp %.1f | soft evicted %5.1f cost %+.4f | margin evicted %5.1f "
          "cost %+.4f | margin is %+.0f%% of the damage"
          % (sharp, s_n, s_c, m_n, m_c, 100 * (m_c / s_c - 1)))
