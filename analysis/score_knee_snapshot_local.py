"""Complete-panel CPU scoring for the approved knee snapshot attribution pilot.

Callables support explicit fictitious fixture validation. There is no campaign
scoring CLI until the actual input/provenance/launch gate boundary is certified.
Private development targets are opened only after all four run checks pass.
"""

import json
import math
from pathlib import Path
import statistics

from tralo.knee_snapshot_data import FORMAT, encode, sha256
from tralo.knee_snapshot_run import verify_run

CONTRASTS = [('joint_local','pto'), ('joint_local','joint_sham'),
             ('joint_local','global_at_joint_radius'), ('global_native','pto'),
             ('global_native','global_native_sham')]
POLICIES = ('global_only','global_local')


def metrics(truth, prediction):
    from tralo.metrics import classification_metrics
    row = classification_metrics(truth, prediction, 5, [3])
    row['weighted_f1'] = sum(c['f1']*c['support'] for c in row['per_class'])/len(truth)
    return row


def paired(values):
    """Four-seed paired t(3) interval and two-sided p-value, in F1 units."""
    from scipy.stats import t
    if len(values) != 4 or any(not math.isfinite(v) for v in values):
        raise ValueError('paired pilot contrast requires four finite values')
    mean = statistics.mean(values)
    error = statistics.stdev(values)/2
    radius = float(t.ppf(.975, 3))*error
    probability = float(2*t.sf(abs(mean/error),3)) if error else (1. if mean == 0 else 0.)
    return dict(seed_differences=values, mean=mean, interval95=[mean-radius,mean+radius],
                p_two_sided=probability, df=3)


def holm(rows):
    ordered = sorted(rows, key=lambda key: rows[key]['p_two_sided'])
    floor = 0.
    for rank,key in enumerate(ordered):
        floor = max(floor, min(1., (len(ordered)-rank)*rows[key]['p_two_sided']))
        rows[key]['p_holm'] = floor


def score_panel(runs, private, private_sha256, *, allow_simulation=False):
    """Verify four distinct complete runs, then join authorized private labels.

    Runs is a four-item list of (directory, externally pinned completion hash).
    This reuses saved CPU tensors and allocator code; it never replays a model.
    It is an integrity/scoring component, not a launch/data-access certificate.
    """
    from tralo.knee_snapshot_local import ARMS, SEEDS
    if not isinstance(runs, (list,tuple)) or len(runs) != 4:
        raise ValueError('complete four-run panel required before private target access')
    checked = [verify_run(directory, pin, allow_simulation=allow_simulation) for directory,pin in runs]
    first = checked[0]
    seeds = [row['result']['seed'] for row in checked]
    if len(set(seeds)) != 4 or (not allow_simulation and tuple(seeds) != SEEDS):
        raise ValueError('four distinct ordered prospective seeds required')
    config = {k:v for k,v in first['config'].items() if k != 'seed'}
    if any(row['rows'] != first['rows'] or row['manifest'] != first['manifest']
           or any(row['result'][k] != first['result'][k] for k in
                  ['public_manifest_sha256','source_commit','source_sha256'])
           or {k:v for k,v in row['config'].items() if k != 'seed'} != config
           or row['result']['execution']['mode'] != first['result']['execution']['mode'] for row in checked):
        raise ValueError('panel input/source/recipe/mode identity mismatch')
    private = Path(private)
    if private.is_symlink():
        raise ValueError('private target symlink refused')
    private_data = private.read_bytes()  # FIRST private read, after the complete panel.
    if sha256(private_data) != private_sha256:
        raise ValueError('private target pin mismatch')
    targets = json.loads(private_data)  # Parse the same bytes authenticated above.
    ids = [r['sample_id'] for r in first['rows']['development']]
    if (set(targets) != {'format','public_manifest_sha256','rows','source_sha256_by_id'}
            or targets['format'] != FORMAT
            or targets['public_manifest_sha256'] != first['result']['public_manifest_sha256']
            or not isinstance(targets['rows'], list) or len(targets['rows']) != len(ids)):
        raise ValueError('private target/public identity mismatch')
    if any(set(r) != {'sample_id','label'} or type(r['label']) is not int or not 0 <= r['label'] < 5
           for r in targets['rows']):
        raise ValueError('invalid private development targets')
    if [r['sample_id'] for r in targets['rows']] != ids:
        raise ValueError('private development IDs/order differ or repeat')
    all_ids = {r['sample_id'] for role in first['rows'].values() for r in role}
    import re
    if (set(targets['source_sha256_by_id']) != all_ids
            or any(not isinstance(pin,str) or re.fullmatch(r'[0-9a-f]{64}',pin) is None
                   for pin in targets['source_sha256_by_id'].values())):
        raise ValueError('private source lineage ID/hash mismatch')
    from tralo.global_clipper import allocate, allocate_local_capped_first
    actual = [r['label'] for r in targets['rows']]
    groups = [r['group'] for r in first['rows']['development']]
    caps = [None,None,None,76,None]
    local_caps = first['manifest']['local_caps']
    scores = {}
    for run in checked:
        scores[str(run['result']['seed'])] = {}
        for arm in ARMS:
            probabilities = run['averages'][arm].tolist()
            predictions = dict(global_only=allocate(probabilities,caps,ids,'capped_first'),
                               global_local=allocate_local_capped_first(probabilities,caps,ids,groups,local_caps))
            arm_scores = {}
            for policy,calls in predictions.items():
                local_counts = {g:sum(c==3 for c,name in zip(calls,groups) if name==g) for g in local_caps}
                if sum(c==3 for c in calls) > 76 or (policy == 'global_local' and
                                                       any(local_counts[g] > cap for g,cap in local_caps.items())):
                    raise ValueError('common allocation violated its policy quotas')
                arm_scores[policy] = dict(metrics=metrics(actual,calls),
                    groups={g:metrics([c for c,name in zip(actual,groups) if name==g],
                                      [c for c,name in zip(calls,groups) if name==g]) for g in local_caps},
                    grade3_count=sum(c==3 for c in calls), local_grade3_counts=local_counts,
                    pooled_feasible=sum(c==3 for c in calls)<=76,
                    local_feasible=all(local_counts[g]<=cap for g,cap in local_caps.items()),
                    prediction_sha256=sha256(encode(calls)))
            scores[str(run['result']['seed'])][arm] = arm_scores
    contrasts = {}
    for policy in POLICIES:
        for arm,control in CONTRASTS:
            name = f'{policy}:{arm}-{control}'
            contrasts[name] = paired([scores[str(seed)][arm][policy]['metrics']['cc_f1'] -
                                      scores[str(seed)][control][policy]['metrics']['cc_f1'] for seed in seeds])
    holm(contrasts)
    return dict(status='fictitious_component_check' if allow_simulation else 'complete_exploratory_development_panel',
                scientific_superiority_certified=False, seeds=seeds, contrasts=contrasts, scores=scores,
                source_commit=first['result']['source_commit'], source_sha256=first['result']['source_sha256'],
                public_manifest_sha256=first['result']['public_manifest_sha256'],
                private_development_sha256=private_sha256,
                run_completion_sha256={str(seed):run['completion_sha256'] for seed,run in zip(seeds,checked)},
                limitation='Already-viewed development pilot; artificial groups do not establish clinical subgroup benefit. Scoring does not certify source truth, GPU cost/ownership or campaign readiness.')


def main():
    raise RuntimeError('actual campaign/source/data-access gates are not certified; scientific scoring CLI is disabled')


if __name__ == '__main__':
    main()
