"""OFFLINE DIAGNOSTIC (CPU only, dev 'val' split only, labels used only for scoring).
Pre-declared gate: does any label-free per-item property predict which top-cap occupants
are wrong (label != 3) beyond logit(p3)?  CLEARS iff residual-AUC CI low > 0.5 and point >= 0.55
at BOTH caps.  Orientation fixed before looking: larger property value = predicted more likely wrong.
  P1a |p3 - p3_tta|, P1b |L - L_tta|           (only where tta_probabilities.pt exists)
  P2a std_e L_e (e = 6..10 after_constraint), P2b 5 - #epochs inside top-cap
  P3a p2/(1-p3), P3b -entropy(non-3 renormalised)   (concentrated rival = more wrong)
  P4  |L_clipper - L_null|  (final)
  P5  -(L_10 - L_6)  (falling grade-3 log-odds = more wrong)
"""
import json, sys
from pathlib import Path
import numpy as np, torch

RUNS = Path.home() / 'tralo-rebuild/runs'
ROOTS = {76: RUNS / 'claude-target-20260925', 50: RUNS / 'claude-target50-20260926'}
PILOTS = [RUNS / 'claude-bandcons-pilot/seed2000', RUNS / 'claude-bandcons-a1pilot/seed2400']
EPS = 1e-7
B = 10000
rng = np.random.default_rng(0)


def load(p):
    return torch.load(p, map_location='cpu', weights_only=True).double().numpy()


def logit3(P):
    p = np.clip(P[:, 3], EPS, 1 - EPS)
    return np.log(p) - np.log1p(-p)


def topcap(P, cap, ids):
    order = sorted(range(len(P)), key=lambda i: (-P[i, 3], ids[i]))
    return np.array(order)


def auc(score, y):  # y=1 wrong; Mann-Whitney with ties
    pos, neg = score[y == 1], score[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return np.nan
    d = pos[:, None] - neg[None, :]
    return float(((d > 0) + 0.5 * (d == 0)).mean())


def resid(x, L):
    A = np.c_[np.ones_like(L), L]
    beta, *_ = np.linalg.lstsq(A, x, rcond=None)
    return x - A @ beta


def props(arm_dir, clip_dir, cap, ids):
    P = load(arm_dir / 'final_probabilities.pt')
    L = logit3(P)
    E = [load(arm_dir / ('epoch%02d_after_constraint.pt' % e)) for e in range(6, 11)]
    LE = np.stack([logit3(x) for x in E])
    member = np.zeros(len(P))
    for x in E:
        member[topcap(x, cap, ids)[:cap]] += 1
    rest = np.clip(1 - P[:, 3], EPS, None)
    q = np.delete(P, 3, axis=1) / rest[:, None]
    q = np.clip(q, 1e-12, None)
    out = {'P2a_epoch_std': LE.std(0), 'P2b_nonmember_epochs': 5 - member,
           'P3a_p2_share': P[:, 2] / rest, 'P3b_neg_rival_entropy': (q * np.log(q)).sum(1),
           'P5_neg_trend': -(LE[-1] - LE[0])}
    if clip_dir is not None:
        out['P4_cross_arm'] = np.abs(logit3(load(clip_dir / 'final_probabilities.pt')) - L)
    t = arm_dir / 'tta_probabilities.pt'
    if t.exists():
        T = load(t)
        out['P1a_tta_abs_p3'] = np.abs(P[:, 3] - T[:, 3])
        out['P1b_tta_abs_logit'] = np.abs(L - logit3(T))
    return P, L, out, LE, E


def seed_stats(arm_dir, clip_dir, cap, ids, y3):
    P, L, pr, LE, E = props(arm_dir, clip_dir, cap, ids)
    order = topcap(P, cap, ids)
    top, win = order[:cap], order[max(cap - 20, 0):cap + 20]
    inside = np.zeros(len(P), bool); inside[top] = True
    wrong = (y3 != 3).astype(int)
    res = {'neg_p3_auc': auc(-P[top, 3], wrong[top]), 'wrong_in_top': int(wrong[top].sum())}
    for k, x in pr.items():
        rt = resid(x[top], L[top])
        rw = resid(x[win], L[win])
        wi = win[inside[win] & (wrong[win] == 1)]
        co = win[~inside[win] & (wrong[win] == 0)]
        rwmap = dict(zip(win.tolist(), rw))
        sw = auc(np.r_[[rwmap[i] for i in wi], [rwmap[i] for i in co]],
                 np.r_[np.ones(len(wi), int), np.zeros(len(co), int)])
        res[k] = {'raw': auc(x[top], wrong[top]), 'resid': auc(rt, wrong[top]),
                  'window': auc(rw, wrong[win]), 'swap': sw}
        # label-free, untuned re-rank (weight 1), evaluated only for reporting
        ra = resid(x, L)
        z = (ra - ra.mean()) / (ra.std() + 1e-12)
        s = L - z
        new = sorted(range(len(P)), key=lambda i: (-s[i], ids[i]))[:cap]
        res[k]['rerank_delta_correct'] = int((y3[new] == 3).sum() - (y3[top] == 3).sum())
    return res, P, L, E


def boot(v):
    v = np.array([a for a in v if a == a])
    m = rng.integers(0, len(v), (B, len(v)))
    bs = v[m].mean(1)
    return [float(v.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5)), len(v)]


def val_rows(seed_dir):
    m = json.loads((seed_dir / 'manifest.json').read_text())
    rows = [r for r in m['rows'] if r['split'] == 'val']
    assert len(rows) == 826
    return [r['sample_id'] for r in rows], np.array([r['label'] for r in rows])


def verify(arm_dir, cap, ids, P):
    pred = np.array(json.loads((arm_dir / 'report.json').read_text())['capped_first']['predictions'])
    mine = set(topcap(P, cap, ids)[:cap].tolist())
    theirs = set(np.flatnonzero(pred == 3).tolist())
    return mine == theirs, len(mine ^ theirs)


def main():
    out = {'note': 'OFFLINE DIAGNOSTIC; dev val split only; labels used only for scoring; re-rank weight fixed at 1, untuned',
           'gate': 'CLEARS iff resid AUC CI low > 0.5 and point >= 0.55 at both caps', 'caps': {}, 'verify': {}, 'epoch_check': {}}
    for cap, root in ROOTS.items():
        for arm, other in (('tralo_null', 'clipper'), ('clipper', 'tralo_null')):
            per, mism, ep = [], [], []
            for sd in sorted(root.glob('seed*')):
                if not sd.is_dir():
                    continue
                cfg = json.loads((sd / 'config.json').read_text())
                assert cfg['caps'][3] == cap
                ids, y = val_rows(sd)
                r, P, L, E = seed_stats(sd / arm, sd / other, cap, ids, y)
                ok, nd = verify(sd / arm, cap, ids, P)
                mism.append(nd)
                ep.append(float(np.abs(P - E[-1]).max()))
                r['seed'] = sd.name
                per.append(r)
            keys = [k for k in per[0] if isinstance(per[0][k], dict)]
            agg = {'n_seeds': len(per), 'neg_p3_auc': boot([p['neg_p3_auc'] for p in per]),
                   'wrong_in_top_mean': float(np.mean([p['wrong_in_top'] for p in per]))}
            for k in keys:
                d = {m: boot([p[k][m] for p in per]) for m in ('raw', 'resid', 'window', 'swap')}
                dl = np.array([p[k]['rerank_delta_correct'] for p in per])
                d['rerank_delta'] = boot(dl) + [int((dl > 0).sum()), int((dl == 0).sum()), int((dl < 0).sum())]
                agg[k] = d
            out['caps'].setdefault(str(cap), {})[arm] = {'agg': agg, 'per_seed': per}
            out['verify']['%d_%s' % (cap, arm)] = {'symdiff_per_seed': mism}
            out['epoch_check']['%d_%s' % (cap, arm)] = {'max_abs_final_minus_epoch10_after': max(ep)}
    gate = {}
    for k in out['caps']['76']['tralo_null']['agg']:
        if not k.startswith('P'):
            continue
        gate[k] = all(out['caps'][c]['tralo_null']['agg'][k]['resid'][1] > 0.5 and
                      out['caps'][c]['tralo_null']['agg'][k]['resid'][0] >= 0.55 for c in ('76', '50'))
    out['gate_result_tralo_null'] = gate
    pil = {}
    for sd in PILOTS:
        ids, y = val_rows(sd)
        cap = json.loads((sd / 'config.json').read_text())['caps'][3]
        for arm in sorted(p.name for p in sd.iterdir() if p.is_dir()):
            if not (sd / arm / 'tta_probabilities.pt').exists():
                continue
            r, *_ = seed_stats(sd / arm, None, cap, ids, y)
            pil['%s/%s' % (sd.parent.name, arm)] = {k: r[k] for k in r if k.startswith('P1')} | {'cap': cap}
    out['pilots_P1_descriptive_single_seed'] = pil
    dest = Path(sys.argv[1])
    dest.write_text(json.dumps(out, indent=1))
    for c in ('76', '50'):
        for arm in ('tralo_null', 'clipper'):
            a = out['caps'][c][arm]['agg']
            print('cap %s %s n=%d wrong_in_top=%.1f AUC(-p3)=%.3f [%.3f,%.3f]' % (
                c, arm, a['n_seeds'], a['wrong_in_top_mean'], *a['neg_p3_auc'][:3]))
            for k in a:
                if k.startswith('P'):
                    d = a[k]
                    print('  %-24s raw %.3f resid %.3f [%.3f,%.3f] win %.3f swap %.3f rerank %+.2f [%+.2f,%+.2f] W/T/L %d/%d/%d' % (
                        k, d['raw'][0], *d['resid'][:3], d['window'][0], d['swap'][0], *d['rerank_delta'][:3], *d['rerank_delta'][4:]))
    print('verify', out['verify'])
    print('epoch_check', out['epoch_check'])
    print('gate', gate)
    for k, v in pil.items():
        print('pilot', k, {kk: (round(vv['raw'], 3), round(vv['resid'], 3)) for kk, vv in v.items() if kk != 'cap'}, 'cap', v['cap'])


if __name__ == '__main__':
    main()
