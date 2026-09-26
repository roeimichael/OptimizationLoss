"""EXPLORATORY: snapshot-ensemble window, averaging space, mechanism, turnover.

python ens_window.py OUT.json ROOT [ROOT ...]
Dev ('val') rows only. Grade 3 is the only capped class, so capped_first's
grade-3 slots are the top-cap items by the grade-3 column (ties by sample_id).
Writes per-(root, seed, arm) raw numbers; summarise.py builds the tables.
"""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

from tralo.global_clipper import allocate

G, EPS = 3, 1e-7


def to_matrix(p3, other):
    """Row-stochastic matrix whose grade-3 column is the monotone score p3 in (0,1)."""
    other = other.clone()
    other[:, G] = 0
    other = other / other.sum(1, keepdim=True) * (1 - p3)[:, None]
    other[:, G] = p3
    return other


def slots(p, caps, ids):
    pred = np.array(allocate(p.tolist(), caps, ids, 'capped_first'))
    return set(np.flatnonzero(pred == G).tolist())


def logit3(s):
    q = s[:, G].clamp(EPS, 1 - EPS)
    return torch.log(q) - torch.log1p(-q)


def main():
    out, recs = Path(sys.argv[1]), []
    for root in map(Path, sys.argv[2:]):
        for d in sorted(root.glob('seed*')):
            if not (d / 'summary.json').exists():
                continue
            caps = json.loads((d / 'config.json').read_text())['caps']
            cap = caps[G]
            rows = [r for r in json.loads((d / 'manifest.json').read_text())['rows'] if r['split'] == 'val']
            y, ids = np.array([r['label'] for r in rows]), [r['sample_id'] for r in rows]
            true3 = set(np.flatnonzero(y == G).tolist())
            for arm in [r['arm'] for r in json.loads((d / 'summary.json').read_text())]:
                a = d / arm
                ld = lambda f: torch.load(a / f, weights_only=True).double()
                final = ld('final_probabilities.pt')
                aft = [ld('epoch%02d_after_constraint.pt' % e) for e in range(6, 11)]
                bef = [ld('epoch%02d_before_constraint.pt' % e) for e in range(6, 11)]
                assert torch.equal(aft[-1], final)
                assert len(y) == final.shape[0]
                S = lambda p: slots(p, caps, ids)
                mean = lambda L: torch.stack(L).mean(0)
                sel = {'base': S(final)}
                for k in (2, 3, 4, 5):
                    sel['win%d' % k] = S(mean(aft[-k:]))
                sel['all10'] = S(mean(aft + bef))
                # Q2 averaging spaces on epochs 6-10
                m = mean(aft)
                g = torch.stack([s.clamp_min(EPS).log() for s in aft]).mean(0).exp()
                sel['geo'] = S(g / g.sum(1, keepdim=True))
                sel['logodds'] = S(to_matrix(torch.sigmoid(torch.stack([logit3(s) for s in aft]).mean(0)), m))
                n = len(ids)
                ranks = []
                for s in aft:  # rank 0 = highest p3, ties by sample_id (as in allocate)
                    order = sorted(range(n), key=lambda i: (-float(s[i, G]), ids[i]))
                    r = np.empty(n)
                    r[order] = np.arange(n)
                    ranks.append(r)
                sel['rank'] = S(to_matrix(torch.tensor(1 - (np.mean(ranks, 0) + 0.5) / n), m))
                for e, s in zip(range(6, 11), aft):
                    sel['ep%02d' % e] = S(s)
                # Q3 between-epoch log-odds std near the cut (ranks cap-20..cap+19 by final p3)
                order = sorted(range(n), key=lambda i: (-float(final[i, G]), ids[i]))
                near = order[cap - 20:cap + 20]
                L = torch.stack([logit3(s) for s in aft])[:, near]
                recs.append({
                    'root': root.name, 'seed': int(d.name[4:]), 'arm': arm, 'cap': cap,
                    'correct': {k: len(v & true3) for k, v in sel.items()},
                    'turnover_ens_base': 1 - len(sel['win5'] & sel['base']) / cap,
                    'turnover_ep09_ep10': 1 - len(sel['ep09'] & sel['ep10']) / cap,
                    'near_cut_logodds_std': float(L.std(0, unbiased=True).mean()),
                    'before_equals_after': all(torch.equal(x, z) for x, z in zip(aft, bef)),
                    'base_slot_ids': sorted(ids[i] for i in sel['base']),
                    'val_ids_md5': hashlib.md5('|'.join(ids).encode()).hexdigest(),
                })
            print(d, flush=True)
    out.write_text(json.dumps(recs))


if __name__ == '__main__':
    main()
