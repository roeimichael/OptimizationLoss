"""The global/local Clipper diagnostic measures the policy difference only."""

from analysis.clipper_allocator_diagnostic import compare_allocators


def test_country_ceiling_changes_selected_items_without_changing_pooled_cap():
    probabilities = [[0.05, 0.95], [0.10, 0.90], [0.20, 0.80],
                     [0.30, 0.70], [0.40, 0.60]]
    ids = ["a", "b", "c", "d", "e"]
    groups = ["A", "A", "A", "B", "B"]
    quota = {"global_cap": 3, "local_caps": {"A": 1, "B": 2}}

    result = compare_allocators(probabilities, ids, groups, quota, 1)

    assert result["global_selected"] == result["local_selected"] == 3
    assert result["country_counts"]["A"] == {
        "cap": 1, "global_clipper_count": 3,
        "global_clipper_excess": 2, "local_clipper_count": 1}
    assert result["country_counts"]["B"]["local_clipper_count"] == 2
    assert result["selected_overlap"] == 1
    assert result["global_only_selected"] == result["local_only_selected"] == 2
    assert result["selected_jaccard"] == 0.2


def test_country_ceiling_nonbinding_yields_same_global_and_local_clipper():
    probabilities = [[0.05, 0.95], [0.10, 0.90], [0.30, 0.70]]
    ids = ["a", "b", "c"]
    groups = ["A", "A", "B"]
    quota = {"global_cap": 2, "local_caps": {"A": 2, "B": 1}}

    result = compare_allocators(probabilities, ids, groups, quota, 1)

    assert result["global_country_cap_excess_total"] == 0
    assert result["global_violated_countries"] == []
    assert result["selected_overlap"] == 2
    assert result["global_selected_id_sha256"] == result["local_selected_id_sha256"]
