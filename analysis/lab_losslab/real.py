"""Offline check on FINISHED knee runs (dev 'val' labels only): does the real targeted step also push
TRUE grade-3 items at the cut down harder than non-3 items (WHO-AUC < 0.5), as in the synthetic lab?
Also: the per-item (free-logit) channel alone, from the same pre-step probabilities (label-free).

python real.py ROOT [ROOT...]
"""
import json, math, sys
from pathlib import Path
import numpy as np
import torch
from tralo.streamed_constraint import count_logit_gradient
import lab

G = 3


def margin(p):
    l = p.double().clamp_min(1e-300).log()
    return (l[:, G] - torch.cat([l[:, :G], l[:, G + 1:]], 1).logsumexp(1)).numpy()


def topcap(p3, cap):
    return np.lexsort((np.arange(len(p3)), -p3))[:cap]


def free_logit(pb, cap):
    z = pb.double().clamp_min(1e-300).log()
    caps = [None] * 5; caps[G] = 0
    m = torch.zeros(5, dtype=torch.float64); m[G] = 1
    for _ in range(100000):
        if int((z.softmax(1).argmax(1) == G).sum()) <= cap:
            break
        g = count_logit_gradient(z.softmax(1), caps, m, 0.0)
        z = z - 0.02 * g / g.norm() * math.sqrt(len(z))
    return z.softmax(1)[:, G].numpy()


rows = {}
for root in sys.argv[1:]:
    for d in sorted(Path(root).glob('seed*')):
        if not (d / 'summary.json').exists():
            continue
        cap = json.loads((d / 'config.json').read_text())['caps'][G]
        val = [r for r in json.loads((d / 'manifest.json').read_text())['rows'] if r['split'] == 'val']
        y = np.array([r['label'] for r in val])
        for arm in ('tralo_target', 'sham_target'):
            per = []
            for line in open(d / arm / 'events.jsonl'):
                e = json.loads(line)
                if e.get('event') != 'epoch' or not e.get('constraint_applied'):
                    continue
                ep = e['epoch']
                pb = torch.load(d / arm / f'epoch{ep:02d}_before_constraint.pt', weights_only=True)
                pa = torch.load(d / arm / f'epoch{ep:02d}_after_constraint.pt', weights_only=True)
                p3b = pb[:, G].double().numpy(); p3a = pa[:, G].double().numpy()
                rk = np.empty(len(p3b), int); rk[np.lexsort((np.arange(len(p3b)), -p3b))] = np.arange(len(p3b))
                band = (rk >= max(0, cap - 60)) & (rk < cap + 60)
                sb, sa = topcap(p3b, cap), topcap(p3a, cap)
                r = dict(epoch=ep, who_auc=lab.who_auc(margin(pa) - margin(pb), y, band),
                         overlap=len(set(sa) & set(sb)) / cap,
                         d_correct=int((y[sa] == G).sum()) - int((y[sb] == G).sum()),
                         frac_p3_gt_0999=float((p3b > 0.999).mean()),
                         hard_before=int((pb.argmax(1) == G).sum()))
                if arm == 'tralo_target':
                    pf = free_logit(pb, cap); sf = topcap(pf, cap)
                    r.update(free_overlap=len(set(sf) & set(sb)) / cap,
                             free_d_correct=int((y[sf] == G).sum()) - int((y[sb] == G).sum()))
                    caps = [None] * 5; caps[G] = 0
                    g = count_logit_gradient(pb.double(), caps, torch.tensor([0, 0, 0, 1.0, 0], dtype=torch.float64), 0.0)
                    gn = g.norm(dim=1).numpy()
                    top = np.zeros(len(p3b), bool); top[sb] = True
                    r.update(grad_share_topcap=float(gn[top].sum() / gn.sum()),
                             grad_share_band=float(gn[band].sum() / gn.sum()),
                             grad_share_p3_gt_099=float(gn[p3b > 0.99].sum() / gn.sum()))
                per.append(r)
            if per:
                agg = {k: float(np.nanmean([p[k] for p in per])) for k in per[0] if k != 'epoch'}
                agg['n_steps'] = len(per)
                rows.setdefault(root, {}).setdefault(arm, {})[d.name] = agg

for root, arms in rows.items():
    print('===', root)
    for arm, seeds in arms.items():
        keys = next(iter(seeds.values())).keys()
        print('  ', arm, 'n_seeds=%d' % len(seeds))
        for k in keys:
            v = lab.mean_ci([s[k] for s in seeds.values()])
            print('     %-22s %+.4f [%+.4f,%+.4f] n=%d' % (k, v['mean'], v['lo'], v['hi'], v['n']))
    if 'tralo_target' in arms and 'sham_target' in arms:
        common = sorted(set(arms['tralo_target']) & set(arms['sham_target']))
        for k in ('who_auc', 'd_correct', 'overlap'):
            v = lab.paired([arms['tralo_target'][s][k] for s in common], [arms['sham_target'][s][k] for s in common])
            print('   PAIR target-sham %-10s %+.4f [%+.4f,%+.4f] n=%d W/L/T %d/%d/%d' % (k, v['mean'], v['lo'], v['hi'], v['n'], v['wins'], v['losses'], v['ties']))
json.dump(rows, open('real_check.json', 'w'), indent=1)
