import pytest
import json
from scripts import deployed_h2h as report


def test_seed_paired_interval_uses_native_differences_and_refuses_false_certainty():
    result = report.paired_difference({1:.3, 2:.5, 3:.7}, {1:.2, 2:.2, 3:.2})
    assert result['deltas'] == pytest.approx([.1, .3, .5])
    assert result['mean'] == pytest.approx(.3)
    assert result['ci95'] == pytest.approx([-.19682754, .79682754])
    assert result['pilot'] is True
    assert report.paired_difference({1:.3}, {1:.2})['ci95'] is None
    constant = report.paired_difference({1:.5, 2:.5}, {1:.25, 2:.25})
    assert constant['ci95'] is None
    missing = report.paired_difference({1:.4, 2:.7}, {1:.3, 3:.6})
    assert missing['missing_pairs'] == [2, 3]


def test_report_bolds_all_exact_best_means_and_includes_every_contrast():
    records = []
    for arm in ['tralo', 'tralo_null', 'clip', 'focal_clip', 'fioretto', 'hounie', 'alm']:
        records.append(dict(dataset='iwildcam', backbone='MobileNetV2', cap='L50_G50',
                            arm=arm, seed=1, cc_f1=.5, macro_f1=.6,
                            constrained_precision=.7, constrained_recall=.4,
                            collateral_f1=.8, collateral_support=4, feasible=True))
    text = report.markdown_report(records)
    assert text.count('**0.5000**') == 7
    for arm in ['tralo_null', 'clip', 'focal_clip', 'fioretto', 'hounie', 'alm']:
        assert 'TraLO - ' + arm in text
    assert 'not significance' in text
    assert 'not multiplicity-adjusted' in text


def test_report_counts_unique_seed_observations_and_rejects_copies():
    records = [dict(dataset='iwildcam', backbone='MobileNetV2', cap='L50_G50',
                    arm=arm, seed=seed, cc_f1=value, macro_f1=.6,
                    constrained_precision=.7, constrained_recall=.4,
                    collateral_f1=.8, collateral_support=4, feasible=True)
               for arm in ('tralo', 'clip') for seed, value in ((1, .3), (2, .5))]
    text = report.markdown_report(records)
    assert '| tralo | 2 | **0.4000** ± 0.1414' in text
    paired = json.loads(next(line.split(': ', 1)[1] for line in text.splitlines()
                             if line.startswith('TraLO - clip')))
    assert paired['n'] == 2 and paired['seeds'] == [1, 2]
    with pytest.raises(ValueError, match='duplicate observation'):
        report.markdown_report(records + records)
