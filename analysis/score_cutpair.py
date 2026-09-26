"""Score the CUTPAIR study (tralo/knee_e2e_v5.py; seeds 2701-2724 at cap 76, pilot 2700 excluded by the caller).

python analysis/score_cutpair.py RUN_ROOT [--json OUT]

Primary: capped_first grade-3 F1 on CLEAN probabilities, Holm over P1-P3:
  P1 cutpair_aug - cutpair_aug_shift  (attributable: the anchor, not any margin on grade 3)
  P2 cutpair_aug - aug_clip           (over its null)
  P3 cutpair_aug - clipper            (the bar)
Secondary: the same contrasts on TTA probabilities, tralo_null contrasts, the offline swap
analysis vs aug_clip (development labels, after training), slot turnover, and integrity
(hashes, task updates, dose ratio, active-set sizes).
"""
import itertools
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from score_sham import holm, paired  # noqa: E402

ARMS = ('clipper', 'tralo_null', 'aug_clip', 'cutpair_aug', 'cutpair_aug_shift')
CUTPAIR = ('cutpair_aug', 'cutpair_aug_shift')
PRIMARY = [('P1', 'cutpair_aug', 'cutpair_aug_shift'), ('P2', 'cutpair_aug', 'aug_clip'),
           ('P3', 'cutpair_aug', 'clipper')]
SECONDARY = [('cutpair_aug', 'tralo_null'), ('cutpair_aug_shift', 'tralo_null'), ('aug_clip', 'tralo_null'),
             ('clipper', 'tralo_null')]
NULL = 'aug_clip'
GRADE = 3
RATIO = 0.1


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
        print('  %-3s %-17s - %-17s %+6.2f [%+6.2f, %+6.2f]  p=%.4f  Holm=%.4f  W/T/L %d/%d/%d'
              % (name, a, b, r['mean'], r['lo'], r['hi'], r['p'], pa, r['wins'], r['ties'], r['losses']))
    return {name: dict(a=a, b=b, holm_p=pa, **r) for (name, a, b, r), pa in zip(rows, adj)}


def integrity(seed, arms):
    ids = {(arms[x]['summary']['warmup_sha256'], arms[x]['summary']['batch_sha256'],
            arms[x]['summary']['tta_draws_sha256']) for x in ARMS}
    logs = {x: arms[x]['summary']['cut_logs'] for x in CUTPAIR}
    dosed = [r for x in CUTPAIR for r in logs[x] if r['active_batches']]
    return dict(seed=seed, matched=len(ids) == 1,
                updates=sorted({arms[x]['summary']['task_updates_applied'] for x in ARMS}),
                dose_ok=all(abs(r['realised_ratio_mean'] - RATIO) < 1e-6 and abs(r['realised_ratio_max'] - RATIO) < 1e-6
                            for r in dosed),
                anchor_ranks={x: [r['anchor_rank'] for r in logs[x]] for x in CUTPAIR},
                n_act={x: [r['n_act'] for r in logs[x]] for x in CUTPAIR},
                p_act={x: [r['p_act'] for r in logs[x]] for x in CUTPAIR},
                active_batches={x: [r['active_batches'] for r in logs[x]] for x in CUTPAIR},
                natural_counts={x: arms[x]['summary'].get('natural_counts_start') for x in ARMS})


def main():
    seeds, cap = load(sys.argv[1])
    out = dict(n_seeds=len(seeds), seeds=sorted(seeds), cap=cap)
    print('complete seeds: %d  cap %s  %s' % (len(seeds), cap, sorted(seeds)))
    if not seeds:
        return
    print('\nINTEGRITY')
    out['integrity'] = [integrity(s, seeds[s]['arms']) for s in sorted(seeds)]
    for row in out['integrity']:
        print('  ' + ' '.join('%s=%s' % kv for kv in row.items()))
    means = {x: np.mean([np.mean(r['n_act'][x]) for r in out['integrity']]) for x in CUTPAIR}
    pmeans = {x: np.mean([np.mean(r['p_act'][x]) for r in out['integrity']]) for x in CUTPAIR}
    print('  mean N_act %s  mean P_act %s' % (means, pmeans))
    out['mean_active'] = dict(n_act=means, p_act=pmeans)
    print('\nARM MEANS (capped_first grade-3 F1: clean / tta)')
    for arm in ARMS:
        v = [seeds[s]['arms'][arm]['clean'] for s in sorted(seeds)]
        t = [seeds[s]['arms'][arm]['tta'] for s in sorted(seeds)]
        print('  %-17s %6.2f (sd %.2f)  tta %6.2f' % (arm, np.mean(v), np.std(v, ddof=1) if len(v) > 1 else 0, np.mean(t)))
    if len(seeds) < 2:
        print('need >= 2 seeds for contrasts')
        return
    print('\nPRIMARY (clean, Holm over P1-P3)')
    out['primary'] = table(seeds, PRIMARY, 'clean', True)
    print('\nSECONDARY: same contrasts on TTA probabilities')
    out['tta'] = table(seeds, PRIMARY, 'tta', False)
    print('\nSECONDARY: tralo_null contrasts (clean)')
    out['secondary'] = table(seeds, [('S%d' % (k + 1), a, b) for k, (a, b) in enumerate(SECONDARY)], 'clean', False)
    print('\nSLOT SWAPS vs %s (offline, development labels): correct-direction fraction' % NULL)
    out['swaps'] = {}
    for arm in ARMS:
        if arm == NULL:
            continue
        entered = left = good = 0
        for v in seeds.values():
            lab = v['labels']
            mine, null = v['arms'][arm]['clean_slots'], v['arms'][NULL]['clean_slots']
            inn, out_ = mine - null, null - mine
            entered += len(inn)
            left += len(out_)
            good += sum(lab[i] == GRADE for i in inn) + sum(lab[i] != GRADE for i in out_)
        frac = good / max(entered + left, 1)
        net = np.mean([v['arms'][arm]['clean_tp'] - v['arms'][NULL]['clean_tp'] for v in seeds.values()])
        print('  %-17s swapped %d in / %d out, %.1f%% correct-direction, net correct slots %+.2f'
              % (arm, entered, left, 100 * frac, net))
        out['swaps'][arm] = dict(entered=entered, left=left, correct_fraction=frac, net=net)
    ss = sorted(seeds)
    floor = np.mean([1 - len(seeds[x]['arms'][NULL]['clean_slots'] & seeds[y]['arms'][NULL]['clean_slots']) / cap
                     for x, y in itertools.combinations(ss, 2)])
    print('\nSLOT TURNOVER vs %s, reseed floor %.3f' % (NULL, floor))
    out['turnover'] = dict(null=NULL, floor=floor)
    for arm in ARMS:
        if arm != NULL:
            t = np.mean([1 - len(seeds[s]['arms'][arm]['clean_slots'] & seeds[s]['arms'][NULL]['clean_slots']) / cap
                         for s in ss])
            print('  %-17s %.3f (%.2fx floor)' % (arm, t, t / floor))
            out['turnover'][arm] = t
    if len(sys.argv) > 3 and sys.argv[2] == '--json':
        Path(sys.argv[3]).write_text(json.dumps(out, indent=1, default=float))


if __name__ == '__main__':
    main()
