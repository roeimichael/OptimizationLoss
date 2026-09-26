import json, glob, os
import numpy as np, torch
rng = np.random.default_rng(0)
def ci(x):
    x = np.asarray(x, float); b = rng.choice(x, (10000, len(x))).mean(1)
    return '%+.2f [%+.2f, %+.2f] W/T/L %d/%d/%d' % (x.mean(), *np.percentile(b, [2.5, 97.5]), (x > 0).sum(), (x == 0).sum(), (x < 0).sum())
def lo(P):
    p3 = P[:, 3].clamp(1e-12, 1 - 1e-12); return (p3.log() - (1 - p3).log()).numpy()
def top(score, cap):
    return set(np.argsort(-score, kind='stable')[:cap].tolist())
for root in ['claude-target-20260925', 'claude-target50-20260926']:
    for arm in ['tralo_null', 'clipper']:
        d = {k: [] for k in ['meanp3', 'meanlogit', 'p2b', 'last2']}
        for sd in sorted(glob.glob(os.path.expanduser('~/tralo-rebuild/runs/%s/seed*' % root))):
            if not os.path.exists(sd + '/summary.json'): continue
            cap = json.load(open(sd + '/config.json'))['caps'][3]
            lab = np.array([r['label'] for r in json.load(open(sd + '/manifest.json'))['rows'] if r['split'] == 'val'])
            Ps = [torch.load('%s/%s/epoch%02d_after_constraint.pt' % (sd, arm, e), weights_only=True).double() for e in range(6, 11)]
            L = np.stack([lo(P) for P in Ps])
            good = lambda S: sum(lab[i] == 3 for i in S)
            base = good(top(L[-1], cap))
            d['meanp3'].append(good(top(np.mean([P[:, 3].numpy() for P in Ps], 0), cap)) - base)
            d['meanlogit'].append(good(top(L.mean(0), cap)) - base)
            d['last2'].append(good(top(L[-2:].mean(0), cap)) - base)
            memb = np.zeros(len(lab))
            for e in range(5):
                memb[list(top(L[e], cap))] += 1
            prop = 5 - memb
            A = np.vstack([L[-1], np.ones(len(lab))]).T
            res = prop - A @ np.linalg.lstsq(A, prop, rcond=None)[0]
            d['p2b'].append(good(top(L[-1] - (res - res.mean()) / res.std(), cap)) - base)
        print(root, arm, 'n=%d' % len(d['p2b']))
        for k, v in d.items(): print('   %-10s %s' % (k, ci(v)))
