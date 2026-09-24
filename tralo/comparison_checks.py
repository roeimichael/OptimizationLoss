"""Independent scoring and equal-slot checks for the declared method campaign."""
from sklearn.metrics import accuracy_score, f1_score


def check_report(report, labels, caps):
    constrained=[i for i,k in enumerate(caps) if k is not None]
    allocated=report['capped_first']['predictions']
    if any(allocated.count(c)!=caps[c] for c in constrained):
        raise RuntimeError('equal-slot comparison did not fill its declared quotas')
    for score in report.values():
        predictions=score['predictions']
        expected=dict(accuracy=accuracy_score(labels,predictions),
            macro_f1=f1_score(labels,predictions,labels=list(range(len(caps))),average='macro',zero_division=0),
            cc_f1=f1_score(labels,predictions,labels=constrained,average='macro',zero_division=0))
        if any(abs(expected[k]-score['metrics'][k])>1e-12 for k in expected):
            raise RuntimeError('independent metrics disagree')


def check_dose(result, config, arm):
    if result['task_updates']!=result['task_updates_planned'] or result['task_updates_skipped'] or result['constraint_updates_skipped'] or result.get('anchor_updates_skipped',0):
        raise RuntimeError('task budget or applied update mismatch')
    anchor_expected=(config['epochs']-config['warmup_epochs']
                     if config.get('constraint_anchor_weight',0)>0 and arm in ('tralo','tralo_null') else 0)
    if result.get('anchor_updates',0)!=anchor_expected:
        raise RuntimeError('anchor update dose mismatch')
    expected=(config['epochs']-config['warmup_epochs'])*(result['task_updates_planned']//config['epochs']) if arm=='alm' else 0
    if result['joint_constraint_updates']!=expected:
        raise RuntimeError('joint ALM dose mismatch')
    if result['dual_updates']!=(config['epochs']-config['warmup_epochs'] if arm=='alm' else 0):
        raise RuntimeError('dual update dose mismatch')
