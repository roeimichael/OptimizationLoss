"""Re-score the finished knee studies on Yuval Kassif's metrics (Phase A of the Yuval investigation).

python analysis/yuval_metrics_rescore.py RUNS_DIR

His PTO and PAO deploy exactly our capped_first (optimization.constrained_classification: the top-N_K
items by p_k take class k, every other item the argmax over the remaining classes) and report sklearn
accuracy, macro-F1 and weighted-F1 over all classes. Two views per arm:
  FINAL  final_probabilities.pt
  ENS    mean of the post-warm-up after_constraint snapshots (epochs 06-10)
Every FINAL prediction vector, accuracy, macro-F1 and cc-F1 (raw and capped_first) is checked against
the run's own report.json and the script stops on any mismatch. Seeds are de-duplicated by probability
hash before n is counted. Offline, development labels only; the test split is never read.
"""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tralo.global_clipper import allocate  # noqa: E402
from tralo.metrics import classification_metrics  # noqa: E402

GRADE = 3
STUDIES = ('claude-target-20260925', 'claude-target50-20260926', 'claude-repl-mn3', 'claude-repl-rgy')
FAMILY = (('target-clip', 'tralo_target', 'clipper'), ('target-sham', 'tralo_target', 'sham_target'),
          ('target-null', 'tralo_target', 'tralo_null'))
EXTRA = (('adam-null', 'tralo_adam', 'tralo_null'),)
METRICS = ('acc', 'macroF1', 'wF1', 'ccF1')
POLICIES = ('capped_first', 'raw')
VIEWS = ('FINAL', 'ENS')

try:
    from sklearn.metrics import f1_score
except ImportError:
    f1_score = None


def tensor_hash(t):
    return hashlib.sha256(t.contiguous().numpy().tobytes()).hexdigest()


def predict(p, caps, ids):
    rows = p.tolist()
    return {'raw': [max(range(len(r)), key=r.__getitem__) for r in rows],
            'capped_first': allocate(rows, caps, ids, 'capped_first')}


def score(labels, pred):
    m = classification_metrics(labels, pred, 5, [GRADE])
    weighted = sum(r['support'] * r['f1'] for r in m['per_class']) / len(labels)
    if f1_score is not None:
        for avg, ours in (('macro', m['macro_f1']), ('weighted', weighted)):
            if abs(f1_score(labels, pred, average=avg) - ours) > 1e-12:
                raise ValueError('%s-F1 differs from sklearn' % avg)
    return m, dict(acc=100 * m['accuracy'], macroF1=100 * m['macro_f1'], wF1=100 * weighted, ccF1=100 * m['cc_f1'])


def load(root):
    """seed -> arm -> view -> {policy: (metrics, predictions)}, plus hashes; duplicates dropped."""
    data, dropped, seen = {}, [], {}
    for d in sorted(root.glob('seed*')):
        if not d.is_dir() or not (d / 'summary.json').exists():
            continue
        config = json.loads((d / 'config.json').read_text())
        caps = config['caps']
        window = list(range(config['warmup_epochs'] + 1, config['epochs'] + 1))
        if window != [6, 7, 8, 9, 10]:
            raise ValueError('%s: ensemble window is not epochs 06-10' % d.name)
        rows = [r for r in json.loads((d / 'manifest.json').read_text())['rows'] if r['split'] == 'val']
        labels, ids = [r['label'] for r in rows], [r['sample_id'] for r in rows]
        seed, record = int(d.name[4:]), {}
        for arm in [r['arm'] for r in json.loads((d / 'summary.json').read_text())]:
            a = d / arm
            final = torch.load(a / 'final_probabilities.pt', weights_only=True)
            snaps = [torch.load(a / ('epoch%02d_after_constraint.pt' % e), weights_only=True) for e in window]
            if not torch.equal(snaps[-1], final):
                raise ValueError('%s/%s: final is not the epoch-10 snapshot' % (d.name, arm))
            ens = torch.stack([s.double() for s in snaps]).mean(0)
            report = json.loads((a / 'report.json').read_text())
            views = {}
            for view, p in (('FINAL', final.double()), ('ENS', ens)):
                preds, out = predict(p, caps, ids), {}
                for policy in POLICIES:
                    m, s = score(labels, preds[policy])
                    if view == 'FINAL':
                        if report[policy]['predictions'] != preds[policy]:
                            raise ValueError('%s/%s/%s: predictions differ from report.json' % (d.name, arm, policy))
                        for key in ('accuracy', 'macro_f1', 'cc_f1'):
                            if report[policy]['metrics'][key] != m[key]:
                                raise ValueError('%s/%s/%s: %s differs from report.json' % (d.name, arm, policy, key))
                    out[policy] = (s, preds[policy])
                views[view] = out
            record[arm] = dict(views=views, hash=tensor_hash(final), ens_hash=tensor_hash(ens),
                               final=final.double().numpy())
        duplicate = [arm for arm, r in record.items() if (arm, r['hash']) in seen]
        if duplicate:
            dropped.append((seed, seen[(duplicate[0], record[duplicate[0]]['hash'])], duplicate))
            continue
        for arm, r in record.items():
            seen[(arm, r['hash'])] = seed
        data[seed] = dict(record=record, labels=labels, caps=caps)
    return data, dropped


def paired(a, b):
    v = np.array(a, float) - np.array(b, float)
    n, mean, sd = len(v), float(v.mean()), float(v.std(ddof=1))
    if sd == 0.0:
        return dict(mean=mean, sd=0.0, lo=mean, hi=mean, p=1.0 if mean == 0.0 else 0.0, v=v)
    h = stats.t.ppf(.975, n - 1) * sd / np.sqrt(n)
    return dict(mean=mean, sd=sd, lo=mean - h, hi=mean + h, p=float(stats.ttest_1samp(v, 0).pvalue), v=v)


def holm(ps):
    order, out, running = sorted(range(len(ps)), key=ps.__getitem__), [0.0] * len(ps), 0.0
    for k, i in enumerate(order):
        running = max(running, min(1.0, (len(ps) - k) * ps[i]))
        out[i] = running
    return out


def cell(r, h):
    return '%+5.2f sd%4.2f [%+5.2f,%+5.2f] H%.3f' % (r['mean'], r['sd'], r['lo'], r['hi'], h)


def report_study(name, data, dropped):
    seeds = sorted(data)
    arms = list(data[seeds[0]]['record'])
    print('=' * 110)
    print('%s: %d seeds %d-%d, cap %s; duplicates dropped: %s' % (name, len(seeds), seeds[0], seeds[-1],
          data[seeds[0]]['caps'][GRADE], dropped or 'none'))
    for view in VIEWS:
        for policy in POLICIES:
            print('-- %s / %s   (points; Holm over the 3 tralo_target contrasts; adam-null its own family of 1)'
                  % (view, policy))
            level = {arm: {m: np.mean([data[s]['record'][arm]['views'][view][policy][0][m] for s in seeds])
                           for m in METRICS} for arm in arms}
            print('   levels: ' + ' | '.join('%s %s' % (arm, ' '.join('%s=%.2f' % (m, level[arm][m]) for m in METRICS))
                                             for arm in arms[:3]))
            print('           ' + ' | '.join('%s %s' % (arm, ' '.join('%s=%.2f' % (m, level[arm][m]) for m in METRICS))
                                             for arm in arms[3:]))
            print('   %-12s %s' % ('contrast', ''.join('%-38s' % m for m in METRICS)))
            for family in (FAMILY, EXTRA):
                results = {m: [paired([data[s]['record'][a]['views'][view][policy][0][m] for s in seeds],
                                      [data[s]['record'][b]['views'][view][policy][0][m] for s in seeds])
                               for _, a, b in family] for m in METRICS}
                adjusted = {m: holm([r['p'] for r in results[m]]) for m in METRICS}
                for i, (label, a, b) in enumerate(family):
                    same_pred = sum(data[s]['record'][a]['views'][view][policy][1]
                                    == data[s]['record'][b]['views'][view][policy][1] for s in seeds)
                    key = 'hash' if view == 'FINAL' else 'ens_hash'
                    same_prob = sum(data[s]['record'][a][key] == data[s]['record'][b][key] for s in seeds)
                    wtl = results['ccF1'][i]['v']
                    print('   %-12s %s  identical probs %d, preds %d; ccF1 W/T/L %d/%d/%d'
                          % (label, ''.join('%-38s' % cell(results[m][i], adjusted[m][i]) for m in METRICS),
                             same_prob, same_pred, (wtl > 0).sum(), (wtl == 0).sum(), (wtl < 0).sum()))
    return seeds


def second_choice(data):
    """ResNet18 cap 76, FINAL, capped_first: who tralo_target swaps relative to clipper."""
    t = dict(out=0, out3=0, out_ok=0, out_non3=0, out_non3_ok=0, out_clip2_ok=0,
             inn=0, inn3=0, inn_clip_ok=0)
    per_seed = dict(tp=[], acc_items=[], out=[], inn=[], neither=[])
    for s in sorted(data):
        y = np.array(data[s]['labels'])
        rec = data[s]['record']
        pc = np.array(rec['clipper']['views']['FINAL']['capped_first'][1])
        pt = np.array(rec['tralo_target']['views']['FINAL']['capped_first'][1])
        out, inn = (pc == GRADE) & (pt != GRADE), (pt == GRADE) & (pc != GRADE)
        neither = (pc != GRADE) & (pt != GRADE)
        if out.sum() != inn.sum():
            raise ValueError('seed %d: exact-slot sets differ in size' % s)
        t['out'] += int(out.sum())
        t['out3'] += int((out & (y == GRADE)).sum())
        t['out_ok'] += int((out & (pt == y)).sum())
        t['out_non3'] += int((out & (y != GRADE)).sum())
        t['out_non3_ok'] += int((out & (y != GRADE) & (pt == y)).sum())
        # what clipper itself would have said for those items once grade 3 is excluded
        clip2 = np.where(np.arange(5)[None, :] == GRADE, -1.0, rec['clipper']['final']).argmax(1)
        t['out_clip2_ok'] += int((out & (clip2 == y)).sum())
        t['inn'] += int(inn.sum())
        t['inn3'] += int((inn & (y == GRADE)).sum())
        t['inn_clip_ok'] += int((inn & (pc == y)).sum())
        per_seed['tp'].append(int((inn & (y == GRADE)).sum()) - int((out & (y == GRADE)).sum()))
        per_seed['acc_items'].append(int((pt == y).sum()) - int((pc == y).sum()))
        for region, mask in (('out', out), ('inn', inn), ('neither', neither)):
            per_seed[region].append(int((mask & (pt == y)).sum()) - int((mask & (pc == y)).sum()))
    print('=' * 110)
    print('SECOND CHOICE: claude-target-20260925 (ResNet18, cap 76), FINAL, capped_first, tralo_target vs clipper, %d seeds'
          % len(data))
    print('  evicted by tralo_target (in clipper set, not in tralo_target set): %d items, %.1f%% truly grade 3'
          % (t['out'], 100 * t['out3'] / max(t['out'], 1)))
    print('    tralo_target reassigns them correctly: %d/%d = %.1f%% overall; %d/%d = %.1f%% of the truly non-3'
          % (t['out_ok'], t['out'], 100 * t['out_ok'] / max(t['out'], 1), t['out_non3_ok'], t['out_non3'],
             100 * t['out_non3_ok'] / max(t['out_non3'], 1)))
    print("    clipper's own second choice (argmax excluding 3) on the same items is correct: %d/%d = %.1f%%"
          % (t['out_clip2_ok'], t['out'], 100 * t['out_clip2_ok'] / max(t['out'], 1)))
    print('  admitted by tralo_target (in tralo_target set, not in clipper set): %d items, %.1f%% truly grade 3;'
          ' clipper had %.1f%% of them right' % (t['inn'], 100 * t['inn3'] / max(t['inn'], 1),
                                                 100 * t['inn_clip_ok'] / max(t['inn'], 1)))
    for key, text in (('tp', 'net correct grade-3 slots (tralo_target - clipper)'),
                      ('acc_items', 'net correct items overall (accuracy x 826)'),
                      ('out', '  of which: evicted items'), ('inn', '  of which: admitted items'),
                      ('neither', '  of which: items outside both grade-3 sets (non-3 argmax changes)')):
        r = paired(per_seed[key], [0] * len(per_seed[key]))
        print('  %-70s %+6.2f [%+6.2f, %+6.2f] per seed  p=%.4f' % (text, r['mean'], r['lo'], r['hi'], r['p']))


def main():
    root = Path(sys.argv[1])
    print('sklearn cross-check of macro/weighted F1: %s' % ('on' if f1_score is not None else 'UNAVAILABLE'))
    loaded = {}
    for name in STUDIES:
        data, dropped = load(root / name)
        report_study(name, data, dropped)
        loaded[name] = data
    second_choice(loaded['claude-target-20260925'])


if __name__ == '__main__':
    main()
