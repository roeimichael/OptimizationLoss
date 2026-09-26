"""Why does the targeted step push TRUE class-3 items down harder than non-3 items at the cut?

Per seed: the same base model as lab.run_seed; targeted step restricted to parameter subsets
(bias of the class-3 logit only / whole head / full network) plus diagnostics on the band
(clip ranks cap-60 .. cap+60): push = -(margin_after - margin_before).
"""
import json, math, sys
import numpy as np
import torch
from scipy import stats
from tralo.targeted_step import targeted_step
import lab

C3, K = 3, 5


def partial_spearman(a, b, ctrl):
    ra, rb, rc = (stats.rankdata(v) for v in (a, b, ctrl))
    res = lambda y: y - np.polyval(np.polyfit(rc, y, 1), rc)
    return float(stats.pearsonr(res(ra), res(rb))[0])


def main(name, nseeds):
    cfg = json.load(open('configs.json'))[name]
    out = []
    for s in range(nseeds):
        seed = 5000 + s
        torch.manual_seed(seed)
        d = lab.make(cfg, seed)
        X = torch.tensor(d['xtr'], dtype=torch.float32); Y = torch.tensor(d['ytr'])
        XP = torch.tensor(d['xp'], dtype=torch.float32); yp = d['yp']
        model = lab.MLP(cfg['D'], cfg['H'], cfg.get('depth', 2))
        opt = torch.optim.Adam(model.parameters(), lr=cfg.get('lr', 3e-3), weight_decay=cfg.get('wd', 1e-4))
        lab.ce_steps(model, opt, X, Y, cfg['E'])
        base = {k: v.clone() for k, v in model.state_dict().items()}
        p0 = lab.probs(model, XP); p3 = p0[:, C3].numpy().astype(np.float64); m0 = lab.margin(p0)
        n3 = int((yp == C3).sum()); cap = int(cfg['cap']) if 'cap' in cfg else int(round(cfg['cap_frac'] * n3))
        sel0 = lab.topcap(p3, cap)
        rank0 = np.empty(len(p3), int); rank0[np.lexsort((np.arange(len(p3)), -p3))] = np.arange(len(p3))
        band = (rank0 >= max(0, cap - 60)) & (rank0 < cap + 60)
        dist3 = np.linalg.norm(d['xp'] - d['mu'][C3], axis=1)
        # label-free density of uncertain pool mass around each item (kNN count of items with 0.05<p3<0.95)
        unc = (p3 > 0.05) & (p3 < 0.95)
        D2 = ((d['xp'][:, None, :] - d['xp'][None, unc, :]) ** 2).sum(-1)
        dens = -np.sort(D2, axis=1)[:, 10]  # minus the 10th-NN squared distance to the uncertain set
        row = dict(seed=seed, cap=cap, clip=lab.prec(sel0, yp),
                   band_dist3_true3=float(dist3[band & (yp == C3)].mean()),
                   band_dist3_non3=float(dist3[band & (yp != C3)].mean()))
        subsets = {
            'bias3': lambda n, p: n == 'head.bias',
            'head': lambda n, p: n.startswith('head.'),
            'full': lambda n, p: True,
        }
        for key, keep in subsets.items():
            model.load_state_dict(base)
            for n, p in model.named_parameters():
                p.requires_grad_(bool(keep(n, p)))
            caps = [None] * K; caps[C3] = cap
            try:
                info = targeted_step(model, [XP], caps)
            except RuntimeError as e:
                row[key + '_error'] = str(e); continue
            for p in model.parameters():
                p.requires_grad_(True)
            p1 = lab.probs(model, XP); q3 = p1[:, C3].numpy().astype(np.float64)
            sel = lab.topcap(q3, cap)
            push = -(lab.margin(p1) - m0)
            row[key + '_prec'] = lab.prec(sel, yp)
            row[key + '_overlap'] = len(set(sel) & set(sel0)) / cap
            row[key + '_who_auc'] = lab.who_auc(lab.margin(p1) - m0, yp, band)
            row[key + '_pcorr_push_dist3'] = partial_spearman(push[band], dist3[band], m0[band])
            row[key + '_pcorr_push_dens'] = partial_spearman(push[band], dens[band], m0[band])
            row[key + '_pcorr_push_bayes'] = partial_spearman(push[band], d['bayes'][band], m0[band])
        row['pcorr_dens_bayes'] = partial_spearman(dens[band], d['bayes'][band], m0[band])
        row['pcorr_dist3_bayes'] = partial_spearman(dist3[band], d['bayes'][band], m0[band])
        out.append(row)
    keys = [k for k in out[0] if k != 'seed' and isinstance(out[0][k], float)]
    summ = {k: lab.mean_ci([r.get(k, float('nan')) for r in out]) for k in keys}
    pairs = {'%s-clip' % k: lab.paired([r.get(k + '_prec', float('nan')) for r in out], [r['clip'] for r in out]) for k in ('bias3', 'head', 'full')}
    json.dump(dict(name=name, rows=out, summary=summ, paired=pairs), open('mech_%s.json' % name, 'w'), indent=1)
    print('===', name, 'n=%d' % len(out))
    for k, v in summ.items():
        print('  %-28s %+.4f [%+.4f,%+.4f]' % (k, v['mean'], v['lo'], v['hi']))
    for k, v in pairs.items():
        print('  PAIR %-12s %+.4f [%+.4f,%+.4f] W/L/T %d/%d/%d' % (k, v['mean'], v['lo'], v['hi'], v['wins'], v['losses'], v['ties']))


if __name__ == '__main__':
    main(sys.argv[1], int(sys.argv[2]))
