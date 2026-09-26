"""EXPLORATORY summary of ens_window.py output.  python summarise.py raw.json summary.json"""
import itertools
import json
import sys
from collections import defaultdict

import numpy as np
from scipy import stats

WINDOWS = ['win2', 'win3', 'win4', 'win5', 'all10']
SPACES = ['win5', 'geo', 'logodds', 'rank']  # win5 == arithmetic mean of probabilities, epochs 6-10
EPOCHS = ['ep06', 'ep07', 'ep08', 'ep09', 'ep10']


def ci(v):
    v = np.asarray(v, float)
    h = stats.t.ppf(.975, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v)) if v.std() > 0 else 0.0
    p = stats.ttest_1samp(v, 0).pvalue if v.std() > 0 else 1.0
    return {'mean': v.mean(), 'lo': v.mean() - h, 'hi': v.mean() + h, 'p': float(p),
            'W': int((v > 0).sum()), 'T': int((v == 0).sum()), 'L': int((v < 0).sum()), 'n': len(v)}


def fmt(c):
    return '%+5.2f [%+5.2f,%+5.2f] %2d/%2d/%2d' % (c['mean'], c['lo'], c['hi'], c['W'], c['T'], c['L'])


def main():
    recs = json.load(open(sys.argv[1]))
    by = defaultdict(list)
    for r in recs:
        by[(r['root'], r['arm'])].append(r)
    roots = list(dict.fromkeys(r['root'] for r in recs))
    out = {'q1': {}, 'q2': {}, 'q3': {}, 'q4': {}, 'q0_before_equals_after': {}}

    print('Q1 window: correct slots minus BASE, mean [95% t-CI] W/T/L')
    for (root, arm), rs in by.items():
        d = {w: ci([r['correct'][w] - r['correct']['base'] for r in rs]) for w in WINDOWS}
        out['q1']['%s/%s' % (root, arm)] = d
        out['q0_before_equals_after']['%s/%s' % (root, arm)] = sum(r['before_equals_after'] for r in rs)
        print('%-26s %-18s ' % (root, arm) + ' | '.join('%s %s' % (w, fmt(d[w])) for w in WINDOWS))
    print('\nQ1 arm-averaged per seed (mean over arms within seed, then CI over seeds)')
    for root in roots:
        seeds = defaultdict(list)
        for r in recs:
            if r['root'] == root:
                seeds[r['seed']].append(r)
        d = {w: ci([np.mean([r['correct'][w] - r['correct']['base'] for r in v]) for v in seeds.values()]) for w in WINDOWS}
        out['q1']['%s/ARMAVG' % root] = d
        print('%-26s ' % root + ' | '.join('%s %+5.2f [%+5.2f,%+5.2f]' % (w, d[w]['mean'], d[w]['lo'], d[w]['hi']) for w in WINDOWS))

    print('\nQ2 averaging space (epochs 6-10): gain vs BASE, and paired t vs prob-mean (win5)')
    for key in list(by) + [(root, 'ARMAVG') for root in roots]:
        root, arm = key
        if arm == 'ARMAVG':
            seeds = defaultdict(list)
            for r in recs:
                if r['root'] == root:
                    seeds[r['seed']].append(r)
            val = lambda s: [np.mean([r['correct'][s] - r['correct']['base'] for r in v]) for v in seeds.values()]
        else:
            val = lambda s, rs=by[key]: [r['correct'][s] - r['correct']['base'] for r in rs]
        d = {s: ci(val(s)) for s in SPACES}
        for s in SPACES[1:]:
            diff = np.array(val(s)) - np.array(val('win5'))
            d[s + '_minus_prob'] = ci(diff)
        out['q2']['%s/%s' % key] = d
        print('%-26s %-18s ' % key + ' | '.join('%s %+5.2f' % (s, d[s]['mean']) for s in SPACES)
              + ' || vs prob: ' + ' '.join('%s %+5.2f p=%.3f' % (s, d[s + '_minus_prob']['mean'], d[s + '_minus_prob']['p']) for s in SPACES[1:]))

    print('\nQ3 Spearman(per-seed gain win5-base, near-cut log-odds std); per-epoch correct slots (mean)')
    for root in roots:
        rr = [r for r in recs if r['root'] == root]
        g = [r['correct']['win5'] - r['correct']['base'] for r in rr]
        sd = [r['near_cut_logodds_std'] for r in rr]
        rho, p = stats.spearmanr(g, sd)
        out['q3'][root] = {'pooled_arms_seeds': {'rho': rho, 'p': p, 'n': len(rr), 'mean_std': float(np.mean(sd))}}
        print('%-26s pooled n=%d rho=%+.3f p=%.3f mean_std=%.3f' % (root, len(rr), rho, p, np.mean(sd)))
        for (rt, arm), rs in by.items():
            if rt != root:
                continue
            g = [r['correct']['win5'] - r['correct']['base'] for r in rs]
            sd = [r['near_cut_logodds_std'] for r in rs]
            rho, p = stats.spearmanr(g, sd)
            ep = {e: float(np.mean([r['correct'][e] for r in rs])) for e in EPOCHS + ['win5']}
            ep10_rank = float(np.mean([sorted([r['correct'][e] for e in EPOCHS]).index(r['correct']['ep10']) for r in rs]))
            out['q3'][root][arm] = {'rho': rho, 'p': p, 'mean_std': float(np.mean(sd)), 'epoch_correct': ep,
                                    'ep10_mean_rank_among_5_low0': ep10_rank}
            print('   %-18s rho=%+.3f p=%.3f std=%.3f  ' % (arm, rho, p, np.mean(sd))
                  + ' '.join('%s %.2f' % (e, v) for e, v in ep.items()) + '  ep10 rank(0=worst) %.2f' % ep10_rank)

    print('\nQ4 turnover (1 - |overlap|/cap), mean: ENS vs BASE | ep09 vs ep10 | BASE seed-vs-seed (same val ids only)')
    for (root, arm), rs in by.items():
        cap = rs[0]['cap']
        pairs = [1 - len(set(a['base_slot_ids']) & set(b['base_slot_ids'])) / cap
                 for a, b in itertools.combinations(rs, 2) if a['val_ids_md5'] == b['val_ids_md5']]
        d = {'ens_vs_base': float(np.mean([r['turnover_ens_base'] for r in rs])),
             'ep09_vs_ep10': float(np.mean([r['turnover_ep09_ep10'] for r in rs])),
             'reseed_base': float(np.mean(pairs)) if pairs else None, 'reseed_pairs': len(pairs)}
        out['q4']['%s/%s' % (root, arm)] = d
        print('%-26s %-18s %.3f | %.3f | %s (%d pairs)' % (root, arm, d['ens_vs_base'], d['ep09_vs_ep10'],
              '%.3f' % d['reseed_base'] if pairs else 'n/a', len(pairs)))
    print('\nbefore==after snapshots (seeds of 24):', out['q0_before_equals_after'])
    json.dump(out, open(sys.argv[2], 'w'), indent=1, default=float)


if __name__ == '__main__':
    main()
