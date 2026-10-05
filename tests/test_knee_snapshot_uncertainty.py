"""Protocol regression: identical deltas cannot manufacture certainty."""
import json
import math

import pytest

from analysis.score_knee_snapshot_local import holm, paired


@pytest.mark.parametrize('difference', [0., .125, -.125])
def test_zero_variance_reports_unavailable_inference(difference):
    row = paired([difference]*4)
    assert row['mean'] == difference
    assert row['seed_differences'] == [difference]*4
    assert row['interval95'] is None
    assert row['p_two_sided'] is None
    assert row['seed_sd'] == 0.
    assert row['inference_status'] == 'unavailable_zero_empirical_variance'
    json.dumps(row, allow_nan=False)


def test_nonzero_variance_keeps_prespecified_t3_inference():
    row = paired([.01, .02, .03, .04])
    sd = math.sqrt(.0005/3)
    error = sd/2
    t_statistic = .025/error
    x = t_statistic/math.sqrt(3)
    # Closed-form two-sided Student t(3) survival, independent of SciPy.
    probability = 1-2/math.pi*(math.atan(x)+x/(1+x*x))
    radius = 3.182446305284263*error
    assert row['seed_sd'] == pytest.approx(sd)
    assert row['interval95'] == pytest.approx([.025-radius, .025+radius])
    assert row['p_two_sided'] == pytest.approx(probability)
    assert row['inference_status'] == 'available_t3'


def test_small_positive_variance_is_not_reclassified_as_zero():
    row = paired([.125, .125, .125, .125+1e-7])
    assert row['seed_sd'] > 0
    assert row['interval95'] is not None
    assert row['p_two_sided'] is not None
    assert row['inference_status'] == 'available_t3'


def test_holm_retains_full_ten_contrast_family_with_unavailable_tests():
    rows = {str(i):paired([.125]*4) for i in range(10)}
    rows['0']['p_two_sided'] = .004
    rows['1']['p_two_sided'] = .006
    rows['2']['p_two_sided'] = .2
    holm(rows)
    assert rows['0']['p_holm'] == pytest.approx(.04)
    assert rows['1']['p_holm'] == pytest.approx(.054)
    assert rows['2']['p_holm'] == 1.
    assert all(rows[str(i)]['p_holm'] is None for i in range(3,10))
    json.dumps(rows, allow_nan=False)


def test_holm_preserves_unavailable_inference_for_entire_family():
    rows = {str(i):paired([0.]*4) for i in range(10)}
    holm(rows)
    assert all(row['p_holm'] is None for row in rows.values())


@pytest.mark.parametrize('values', [[0.]*3, [0.]*5, [0.,0.,0.,float('nan')],
                                   [0.,0.,0.,float('inf')]])
def test_invalid_contrast_still_refused(values):
    with pytest.raises(ValueError):
        paired(values)
