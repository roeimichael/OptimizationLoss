"""A partial block cannot access private development labels."""

import pytest

from analysis import score_tabular_persistent as scorer


def test_partial_fixed_block_stops_before_private_development_labels(
        tmp_path, monkeypatch):
    (tmp_path / "seed6801").mkdir()
    def forbidden(*_args, **_kwargs):
        raise AssertionError("private development label path was opened")
    monkeypatch.setattr(scorer, "_private_labels", forbidden)
    with pytest.raises(RuntimeError, match="complete fixed four-seed block"):
        scorer.score(tmp_path, tmp_path / "prepared", tmp_path / "score.json")


def test_weighted_and_group_metrics_include_zero_support_class():
    report = scorer._metrics([0, 0, 1, 1], [0, 1, 1, 0],
                             ["female", "female", "male", "male"])
    assert report["cc_f1"] == 0.5
    assert report["weighted_f1"] == 0.5
    assert report["groups"]["female"]["confusion"] == [[1, 1], [0, 0]]
    assert report["groups"]["male"]["confusion"] == [[0, 0], [1, 1]]
