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
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tralo.global_clipper import allocate  # noqa: E402

GRADE = 3
SEEDS = set(range(2601, 2625))


def ci(x):
    x = np.asarray(x, float)
    h = stats.t.ppf(.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else float('nan')
    return '%+5.2f [%+5.2f, %+5.2f]' % (x.mean(), x.mean() - h, x.mean() + h)


def main():
    root = Path(sys.argv[1])
    per = {}
    found, excluded = set(), []
    for d in sorted(root.glob('seed*')):
        if not d.is_dir() or not (d / 'probe.json').exists():
            continue
        probe = json.loads((d / 'probe.json').read_text())
        cap = probe['cap']
        caps = [None, None, None, cap, None]
        rows = [r for r in json.loads((d / 'manifest.json').read_text())['rows'] if r['split'] == 'val']
        labels, ids = np.array([r['label'] for r in rows]), [r['sample_id'] for r in rows]

        def score(p):
            pred = np.array(allocate(p.tolist(), caps, ids, 'capped_first'))
            return int(((pred == GRADE) & (labels == GRADE)).sum()), pred

        state = torch.load(d / 'state_S.pt', weights_only=True)
        base_tp, _ = score(state)
        base_raw = state.argmax(1).numpy()
        rows_by = {(r['kind'], r['fraction']): r for r in probe['rows']}
        bad = []
        for (kind, f), r in rows_by.items():
            if r['hard_before'] != int((base_raw == GRADE).sum()):
                raise ValueError('%s: hard_before does not match state_S' % d.name)
            if kind == 'tralo':
                sham = rows_by[('sham', f)]
                if not (r['applied'] and r['hard_after'] == r['target']):
                    bad.append((f, r['applied'], r['hard_after'], r['target']))
                elif not sham['applied'] or abs(sham['radius'] - r['radius']) > 1e-12:
                    raise ValueError('%s f=%s: sham radius differs from tralo' % (d.name, f))
        if bad:
            excluded.append((d.name, bad))
            continue
        found.add(int(d.name[4:]))
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
    print('seeds scored: %d' % len(found))
    if found != SEEDS:
        print('MISSING seeds: %s' % sorted(SEEDS - found))
    for name, bad in excluded:
        print('EXCLUDED %s: step not applied or hard_after != target (f, applied, hard_after, target) %s' % (name, bad))
    diffs = {f: np.array(per[('tralo', f)]['d']) - np.array(per[('sham', f)]['d']) for f in sorted({k[1] for k in per})}
    for f in sorted({k[1] for k in per}, reverse=True):
        t, s = per[('tralo', f)], per[('sham', f)]
        print('f=%.1f radius %.4f | tralo-S %s | sham-S %s | tralo-sham %s | tralo evicts %d, %.1f%% not grade 3'
              % (f, np.mean(t['radius']), ci(t['d']), ci(s['d']), ci(np.array(t['d']) - np.array(s['d'])),
                 t['evict'], 100 * t['evict_wrong'] / max(t['evict'], 1)))
    fs = sorted(diffs)
    ps = [stats.ttest_1samp(diffs[f], 0).pvalue for f in fs]
    order = np.argsort(ps)
    adj, running = [0.0] * len(ps), 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(ps) - rank) * ps[i]))
        adj[i] = running
    print('PRIMARY 1 (amendment 1): tralo-sham per f, Holm over %d fractions' % len(fs))
    for f, p, a in zip(fs, ps, adj):
        print('  f=%.1f mean %+.2f p=%.4f Holm=%.4f' % (f, diffs[f].mean(), p, a))
    depth = 1 - np.array(fs)
    slopes = [np.polyfit(depth, [diffs[f][i] for f in fs], 1)[0] for i in range(len(diffs[fs[0]]))]
    print('PRIMARY 2 (amendment 1): per-seed slope of tralo-sham on depth (1-f), slots per unit depth %s' % ci(slopes))


if __name__ == '__main__':
    main()
