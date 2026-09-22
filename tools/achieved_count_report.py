"""Audit completed matched runs and report original and TraLO-derived budgets."""
import hashlib
import json
from pathlib import Path
import statistics
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tralo.achieved_counts import achieved_caps
from tralo.global_report import evaluate_global
from tralo.global_comparison import validate_config


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(root, output):
    from sklearn.metrics import accuracy_score, f1_score
    from scipy.stats import t
    root, output = Path(root), Path(output)
    config = json.loads((root/'config.json').read_text())
    validate_config(config)
    if len(config['seeds']) != 4:
        raise ValueError('this registered report requires four distinct seeds')
    caps = json.loads((root/'caps.json').read_text())['global_caps']
    constrained = [c for c,k in enumerate(caps) if k is not None]
    events = [json.loads(s) for s in (root/'events.jsonl').read_text().splitlines()]
    assert events[-1]['event'] == 'completed', 'training incomplete'
    rows, quotas, inputs = [], {}, {}
    seen_run_identities = set()
    for seed in config['seeds']:
        arms = {}
        identities = []
        for arm in ('clipper','tralo_null','tralo'):
            directory = root/f'{seed}_{arm}'
            logs = [json.loads(s) for s in (directory/'events.jsonl').read_text().splitlines()]
            end = logs[-1]
            assert end['event'] == 'completed'
            assert end['seed'] == seed and end['arm'] == arm, 'run identity mismatch'
            for name, sha in end['artifacts'].items():
                assert digest(directory/name) == sha
            assert end['task_updates_skipped'] == end['constraint_updates_skipped'] == 0
            assert end['task_updates'] == end['task_updates_planned'] == end['task_updates_attempted']
            identities.append((end['warmup_sha256'],end['batch_sha256']))
            arms[arm] = json.loads((directory/'predictions.json').read_text())
            inputs[f'{seed}_{arm}'] = digest(directory/'predictions.json')
        assert len(set(identities)) == 1
        assert identities[0] not in seen_run_identities, 'duplicate seed realization'
        seen_run_identities.add(identities[0])
        reference = arms['tralo']
        derived = achieved_caps(reference['probabilities'],caps,reference['sample_ids'])
        quotas[str(seed)] = derived
        for arm, p in arms.items():
            assert p['sample_ids'] == reference['sample_ids'] and p['labels'] == reference['labels']
            original_raw = None
            for budget_name, budget in [('original',caps),('TraLO-derived',derived)]:
                scores = evaluate_global(p['probabilities'],p['labels'],budget,p['sample_ids'])
                for policy, score in scores.items():
                    if budget_name == 'TraLO-derived' and policy == 'raw':
                        continue
                    if policy == 'raw': original_raw = score['predictions']
                    pred = score['predictions']
                    counts = [pred.count(c) for c in range(len(caps))]
                    assert counts == score['counts']
                    m = score['metrics']
                    for key, value in [('accuracy',accuracy_score(p['labels'],pred)),
                        ('macro_f1',f1_score(p['labels'],pred,labels=list(range(len(caps))),average='macro',zero_division=0)),
                        ('cc_f1',f1_score(p['labels'],pred,labels=constrained,average='macro',zero_division=0))]:
                        assert abs(m[key]-value) < 1e-12, key
                    if budget_name == 'TraLO-derived' and policy == 'capped_first':
                        assert all(counts[c] == derived[c] for c in constrained)
                    if arm == 'tralo' and budget_name == 'TraLO-derived' and policy == 'upper_bound_correction':
                        assert pred == original_raw, 'self-count correction must be identity'
                    rows.append(dict(seed=seed,arm=arm,budget=budget_name,policy=policy,
                        accuracy=m['accuracy'],macro_f1=m['macro_f1'],cc_f1=m['cc_f1'],
                        changed=score['changed'],counts=counts,predictions=pred,per_class=m['per_class'],
                        original_excess=sum(max(0,counts[c]-caps[c]) for c in constrained),
                        derived_excess=sum(max(0,counts[c]-derived[c]) for c in constrained)))
    output.mkdir(parents=True,exist_ok=False)
    source_root = Path(__file__).resolve().parents[1]
    source_paths = [Path(__file__),*sorted((source_root/'tralo').glob('*.py'))]
    provenance = dict(source_sha256={str(p.relative_to(source_root)):digest(p) for p in source_paths},
        config_sha256=digest(root/'config.json'),caps_sha256=digest(root/'caps.json'),
        completion_log_sha256=digest(root/'events.jsonl'))
    report = dict(config=config,original_caps=caps,derived_caps=quotas,inputs_sha256=inputs,rows=rows,provenance=provenance)
    (output/'results.json').write_text(json.dumps(report,allow_nan=False),encoding='utf-8')
    text = ['# Four-seed TraLO-derived budget diagnostic', '',
        'Frozen ResNet18 / CIFAR-100; 10,000 training and 2,000 development images. '
        'Seeds 701–704; 10 head epochs, warm-up 5; unchanged high-rho/shared-Adam recipe. '
        'The previously diagnosed training failure is retained to isolate allocation. '
        'This is not a repaired-TraLO test or a run-until-satisfied experiment.', '',
        'Accuracy is percent correct. Macro-F1 averages all 100 class F1 scores; '
        'constrained F1 averages the same 10 constrained classes, including zero-cap classes. '
        'F1 is displayed on a 0–100 scale. Higher is better for these scores. '
        'Excess is the sum of predictions above individual caps; 0 means feasible. '
        'Changed counts label changes relative to that arm’s raw argmax.', '',
        'Original caps are 10 for each class 0–9; all other classes are uncapped. '
        'Derived caps are that seed’s raw TraLO counts, with no label access. '
        'Upper correction changes excess assignments only. Capped-first rebuilds '
        'assignments using the entire probability matrix. Raw has no allocation.', '',
        '## Budgets actually used', '', '| Seed | ' + ' | '.join('Class '+str(c) for c in constrained)+' | Total |',
        '|---|' + '---:|'*(len(constrained)+1)]
    for seed, budget in quotas.items():
        text.append('| '+seed+' | '+' | '.join(str(budget[c]) for c in constrained)+' | '+str(sum(budget[c] for c in constrained))+' |')
    for seed in config['seeds']:
        text += ['', '## Seed '+str(seed), '',
            '| Training | Budget | Allocation | Accuracy % | Macro-F1 | Constrained F1 | Changed | Excess: original | Excess: derived |',
            '|---|---|---|---:|---:|---:|---:|---:|---:|']
        for r in rows:
            if r['seed'] == seed:
                text.append(f"| {r['arm']} | {r['budget']} | {r['policy']} | {100*r['accuracy']:.2f} | {100*r['macro_f1']:.2f} | {100*r['cc_f1']:.2f} | {r['changed']} | {r['original_excess']} | {r['derived_excess']} |")
    text += ['', '## Means ± seed standard deviation (four seeds)', '',
        '| Training | Budget | Allocation | Accuracy % | Macro-F1 | Constrained F1 |',
        '|---|---|---|---:|---:|---:|']
    for arm in arms:
        for budget, policy in [('original','raw'),('original','upper_bound_correction'),('original','capped_first'),('TraLO-derived','upper_bound_correction'),('TraLO-derived','capped_first')]:
            group = [r for r in rows if (r['arm'],r['budget'],r['policy']) == (arm,budget,policy)]
            values = [f"{statistics.mean(r[k] for r in group)*100:.2f} ± {statistics.stdev(r[k] for r in group)*100:.2f}" for k in ('accuracy','macro_f1','cc_f1')]
            text.append('| '+' | '.join([arm,budget,policy,*values])+' |')
    contrasts = []
    text += ['', '## Paired differences: TraLO minus each control', '',
        'Differences are percentage/F1 points. Positive favors TraLO; negative favors the control. '
        'The interval is a two-sided 95% Student-t interval over four paired seeds, conditional '
        'on this fixed development split. These exploratory intervals are not corrected for '
        'multiple comparisons, and normality at four seeds is unverified.', '',
        '| Budget | Allocation | Control | Metric | Seed deltas 701 / 702 / 703 / 704 | Mean | 95% interval |',
        '|---|---|---|---|---|---:|---|']
    for budget in ('original','TraLO-derived'):
        for policy in ('upper_bound_correction','capped_first'):
            for control in ('clipper','tralo_null'):
                for metric in ('accuracy','macro_f1','cc_f1'):
                    def value(seed,arm):
                        return next(r[metric] for r in rows if
                            (r['seed'],r['arm'],r['budget'],r['policy']) == (seed,arm,budget,policy))
                    delta = [100*(value(s,'tralo')-value(s,control)) for s in config['seeds']]
                    mean, sd = statistics.mean(delta), statistics.stdev(delta)
                    half = float(t.ppf(.975,len(delta)-1))*sd/len(delta)**.5
                    interval = [mean-half,mean+half] if sd else None
                    contrasts.append(dict(budget=budget,policy=policy,control=control,metric=metric,
                        deltas=delta,mean=mean,sd=sd,interval95=interval))
                    ci = f'[{mean-half:+.2f}, {mean+half:+.2f}]' if interval else 'unavailable: zero variance'
                    text.append('| '+' | '.join([budget,policy,control,metric,
                        ' / '.join(f'{d:+.2f}' for d in delta),f'{mean:+.2f}',ci])+' |')
    (output/'paired_contrasts.json').write_text(json.dumps(contrasts,allow_nan=False),encoding='utf-8')
    text += ['', 'Source predictions: `'+str(root)+'`. Full counts, per-class metrics and labels '
        'assigned by every policy are in results.json. Independent scikit-learn metrics, '
        'artifact hashes, warm-up/batch matching, update counts and self-count identity passed. '
        'Four seeds on inspected development data do not establish superiority or absence of bugs.']
    (output/'report.md').write_text('\n'.join(text)+'\n',encoding='utf-8')
    print(json.dumps({'rows':len(rows),'seeds':config['seeds'],'output':str(output)}))


if __name__ == '__main__':
    run(*sys.argv[1:])
