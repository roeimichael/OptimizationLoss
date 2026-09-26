"""Score the preregistered BANDCONS study (experiments/claude_bandcons_protocol_20260926.md).

python analysis/score_bandcons.py RUN_ROOT [--json OUT]

Primary: capped_first grade-3 F1 on CLEAN probabilities, Holm over B1-B4 within the block.
Secondary: the same contrasts on TTA probabilities, bandcons_tta - clipper_tta, the offline
swap analysis (development labels, after training) and slot turnover.
"""
import itertools
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from score_sham import holm, paired  # noqa: E402

ARMS = ('clipper', 'tralo_null', 'aug_clip', 'bandcons', 'bandcons_unc', 'bandcons_rand')
PRIMARY = [('B1', 'bandcons', 'bandcons_rand'), ('B2', 'bandcons', 'bandcons_unc'),
           ('B3', 'bandcons', 'tralo_null'), ('B4', 'bandcons', 'aug_clip')]
SECONDARY = [('aug_clip', 'tralo_null'), ('clipper', 'tralo_null'), ('bandcons_rand', 'tralo_null'),
             ('bandcons_unc', 'tralo_null')]
GRADE = 3


def load(root):
    seeds, cap = {}, None
    for d in sorted(Path(root).glob('seed*')):
        if not d.is_dir() or not (d / 'summary.json').exists():
            continue
        summary = {r['arm']: r for r in json.loads((d / 'summary.json').read_text())}
        if set(summary) != set(ARMS):
            continue
        c = json.loads((d / 'config.json').read_text())['caps'][GRADE]
        if cap is None:
            cap = c
        if c != cap:
            raise ValueError('mixed caps under one run root')
        manifest = json.loads((d / 'manifest.json').read_text())
        labels = np.array([r['label'] for r in manifest['rows'] if r['split'] == 'val'])
        arms = {}
        for arm in ARMS:
            row = dict(summary=summary[arm])
            for view, name in (('clean', 'report.json'), ('tta', 'report_tta.json')):
                rep = json.loads((d / arm / name).read_text())['capped_first']
                pred = np.array(rep['predictions'])
                row[view] = 100 * rep['metrics']['cc_f1']
                row[view + '_slots'] = set(np.flatnonzero(pred == GRADE).tolist())
                row[view + '_tp'] = int(((pred == GRADE) & (labels == GRADE)).sum())
            arms[arm] = row
        seeds[int(d.name[4:])] = dict(arms=arms, labels=labels)
    return seeds, cap


def table(seeds, contrasts, key, holm_family):
    rows = [(name, a, b, paired({s: v['arms'] for s, v in seeds.items()}, a, b, key)) for name, a, b in contrasts]
    adj = holm([r[3]['p'] for r in rows]) if holm_family else [float('nan')] * len(rows)
    for (name, a, b, r), pa in zip(rows, adj):
        print('  %-3s %-13s - %-13s %+6.2f [%+6.2f, %+6.2f]  p=%.4f  Holm=%.4f  W/T/L %d/%d/%d'
              % (name, a, b, r['mean'], r['lo'], r['hi'], r['p'], pa, r['wins'], r['ties'], r['losses']))
    return {name: dict(a=a, b=b, holm_p=pa, **r) for (name, a, b, r), pa in zip(rows, adj)}


def main():
    seeds, cap = load(sys.argv[1])
    out = dict(n_seeds=len(seeds), seeds=sorted(seeds), cap=cap)
    print('complete seeds: %d  cap %s  %s' % (len(seeds), cap, sorted(seeds)))
    if not seeds:
        return
    print('\nINTEGRITY')
    out['integrity'] = []
    for s in sorted(seeds):
        a = seeds[s]['arms']
        ids = {(a[x]['summary']['warmup_sha256'], a[x]['summary']['batch_sha256'], a[x]['summary']['tta_draws_sha256'])
               for x in ARMS}
        updates = {a[x]['summary']['task_updates_applied'] for x in ARMS}
        logs = a['bandcons']['summary']['band_logs']
        w = a['bandcons']['summary']['band_half_width']
        row = dict(seed=s, matched=len(ids) == 1, updates=sorted(updates),
                   band_sizes_ok=all(r['band_size'] == 2 * w for r in logs),
                   disagreement_epoch6=logs[0]['band_disagreement_start'] if logs else None)
        out['integrity'].append(row)
        print('  ' + ' '.join('%s=%s' % kv for kv in row.items()))
    print('\nARM MEANS (capped_first grade-3 F1: clean / tta)')
    for arm in ARMS:
        v = [seeds[s]['arms'][arm]['clean'] for s in sorted(seeds)]
        t = [seeds[s]['arms'][arm]['tta'] for s in sorted(seeds)]
        print('  %-13s %6.2f (sd %.2f)  tta %6.2f' % (arm, np.mean(v), np.std(v, ddof=1) if len(v) > 1 else 0, np.mean(t)))
    if len(seeds) < 2:
        print('need >= 2 seeds for contrasts')
        return
    print('\nPRIMARY (clean, Holm over B1-B4)')
    out['primary'] = table(seeds, PRIMARY, 'clean', True)
    print('\nSECONDARY: same contrasts on TTA probabilities')
    out['tta'] = table(seeds, PRIMARY, 'tta', False)
    print('\nSECONDARY: other clean contrasts')
    out['secondary'] = table(seeds, [('S', a, b) for a, b in SECONDARY], 'clean', False)
    print('\nSECONDARY: bandcons_tta - clipper_tta')
    tt = {s: {'x': {'v': v['arms']['bandcons']['tta']}, 'y': {'v': v['arms']['clipper']['tta']}} for s, v in seeds.items()}
    r = paired(tt, 'x', 'y', 'v')
    print('  %+6.2f [%+6.2f, %+6.2f]  p=%.4f' % (r['mean'], r['lo'], r['hi'], r['p']))
    out['bandcons_tta_minus_clipper_tta'] = r
    print('\nSLOT SWAPS vs tralo_null (offline, development labels): correct-direction fraction')
    out['swaps'] = {}
    for arm in ARMS:
        if arm == 'tralo_null':
            continue
        entered = left = good = 0
        for s, v in seeds.items():
            lab = v['labels']
            mine, null = v['arms'][arm]['clean_slots'], v['arms']['tralo_null']['clean_slots']
            inn, out_ = mine - null, null - mine
            entered += len(inn)
            left += len(out_)
            good += sum(lab[i] == GRADE for i in inn) + sum(lab[i] != GRADE for i in out_)
        frac = good / max(entered + left, 1)
        net = np.mean([v['arms'][arm]['clean_tp'] - v['arms']['tralo_null']['clean_tp'] for v in seeds.values()])
        print('  %-13s swapped %d in / %d out, %.1f%% correct-direction, net correct slots %+.2f' % (arm, entered, left, 100 * frac, net))
        out['swaps'][arm] = dict(entered=entered, left=left, correct_fraction=frac, net=net)
    ss = sorted(seeds)
    floor = np.mean([1 - len(seeds[x]['arms']['tralo_null']['clean_slots'] & seeds[y]['arms']['tralo_null']['clean_slots']) / cap
                     for x, y in itertools.combinations(ss, 2)])
    print("\nSLOT TURNOVER vs null, reseed floor %.3f" % floor)
    out['turnover'] = dict(floor=floor)
    for arm in ARMS:
        if arm != 'tralo_null':
            t = np.mean([1 - len(seeds[s]['arms'][arm]['clean_slots'] & seeds[s]['arms']['tralo_null']['clean_slots']) / cap for s in ss])
            print('  %-13s %.3f (%.2fx floor)' % (arm, t, t / floor))
            out['turnover'][arm] = t
    if len(sys.argv) > 3 and sys.argv[2] == '--json':
        Path(sys.argv[3]).write_text(json.dumps(out, indent=1, default=float))


if __name__ == '__main__':
    main()
