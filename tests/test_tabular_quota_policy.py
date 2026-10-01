"""Quota policy uses metadata counts, including missing rows, never labels."""

import pytest

from tralo.tabular_quota_policy import caps_for_unlabeled_pool


def test_isic_missing_stays_pooled_but_local_groups_have_support():
    groups = ["female"] * 2110 + ["male"] * 2668 + ["missing"] * 17
    caps = caps_for_unlabeled_pool("isic2020", groups)
    assert caps["level1"]["global_cap"] == 48
    assert caps["level1"]["local_caps"] == {"female": 32, "male": 41}
    assert caps["level2"]["global_cap"] == 96
    assert caps["level2"]["local_caps"] == {"female": 64, "male": 81}
    assert caps["level1"]["group_counts"]["missing"] == 17
    assert "missing" not in caps["level1"]["local_caps"]


def test_celeba_caps_and_unexpected_or_small_groups_rejected():
    groups = ["female"] * 17745 + ["male"] * 12722
    caps = caps_for_unlabeled_pool("celeba", groups)
    assert caps["level1"]["global_cap"] == 7617
    assert caps["level1"]["local_caps"] == {"female": 6211, "male": 4453}
    assert caps["level2"]["global_cap"] == 10664
    assert caps["level2"]["local_caps"] == {"female": 7986, "male": 5725}
    with pytest.raises(ValueError, match="unexpected or unsupported"):
        caps_for_unlabeled_pool("celeba", groups + ["missing"])
    with pytest.raises(ValueError, match="unexpected or unsupported"):
        caps_for_unlabeled_pool("isic2020", ["female"] * 999 + ["male"] * 1000)
