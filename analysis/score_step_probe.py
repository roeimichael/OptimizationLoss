"""Score the targeted-step dose-response probe (experiments/claude_step_probe_20260926.md).

python analysis/score_step_probe.py RUN_ROOT
Offline, development labels. For each push fraction f: correct capped_first slots (at the study cap)
of the TraLO-direction step and of the same-radius sham, both minus the un-stepped state S, paired
over seeds; plus TraLO - sham (the information), and eviction precision of the TraLO step.
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tralo.global_clipper import allocate  # noqa: E402

GRADE = 3


def ci(x):
    x = np.asarray(x, float)
    h = 1.96 * x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else float('nan')
    return '%+5.2f [%+5.2f, %+5.2f]' % (x.mean(), x.mean() - h, x.mean() + h)


def main():
    root = Path(sys.argv[1])
    per = {}
    n = 0
    for d in sorted(root.glob('seed*')):
        if not (d / 'probe.json').exists():
            continue
        n += 1
        probe = json.loads((d / 'probe.json').read_text())
        cap = probe['cap']
        caps = [None, None, None, cap, None]
        rows = [r for r in json.loads((d / 'manifest.json').read_text())['rows'] if r['split'] == 'val']
        labels, ids = np.array([r['label'] for r in rows]), [r['sample_id'] for r in rows]

        def score(p):
            pred = np.array(allocate(p.tolist(), caps, ids, 'capped_first'))
            return int(((pred == GRADE) & (labels == GRADE)).sum()), pred

        base_tp, base_pred = score(torch.load(d / 'state_S.pt', weights_only=True))
        base_raw = torch.load(d / 'state_S.pt', weights_only=True).argmax(1).numpy()
        for r in probe['rows']:
            p = torch.load(d / r['file'], weights_only=True)
            tp, _ = score(p)
            raw = p.argmax(1).numpy()
            evicted = np.flatnonzero((base_raw == GRADE) & (raw != GRADE))
            slot = per.setdefault((r['kind'], r['fraction']), dict(d=[], radius=[], evict=0, evict_wrong=0))
            slot['d'].append(tp - base_tp)
            slot['radius'].append(r.get('radius') or 0.0)
            slot['evict'] += len(evicted)
            slot['evict_wrong'] += int((labels[evicted] != GRADE).sum())
    print('seeds: %d' % n)
    for f in sorted({k[1] for k in per}, reverse=True):
        t, s = per[('tralo', f)], per[('sham', f)]
        print('f=%.1f radius %.4f | tralo-S %s | sham-S %s | tralo-sham %s | tralo evicts %d, %.1f%% not grade 3'
              % (f, np.mean(t['radius']), ci(t['d']), ci(s['d']), ci(np.array(t['d']) - np.array(s['d'])),
                 t['evict'], 100 * t['evict_wrong'] / max(t['evict'], 1)))


if __name__ == '__main__':
    main()
