"""Independent arithmetic and mutation checks for the boundary study gate."""

import copy
import math

import pytest
import torch

from analysis import score_fmow_boundary as score


def _accepted():
    quota = {"global_cap": 1, "local_caps": {"A": 1, "B": 1}}
    before = (2, {"A": 1, "B": 1}, 1.6, {"A": .8, "B": .8})
    after = (1, {"A": 1, "B": 0}, 1.5, {"A": .75, "B": .75})
    probe = {"halving": 0, "radius": .1, "pooled_hard": 1,
             "pooled_soft": 1.5, "local_soft": {"A": .75, "B": .75},
             "local_hard": {"A": 1, "B": 0},
             "positive_violations": {"pooled": .5, "local:A": 0., "local:B": 0.},
             "total_positive_violation": .5, "accepted": True, "rejections": []}
    record = {"applied": True, "radius": .1, "displacement": .1,
              "scope_directional_derivatives": {
                  "pooled": -1., "local:A": -.5, "local:B": -.5},
              "boundary_policy": {"applied": True, "radius": .1,
                                  "initial_radius": .1, "reason": "accepted",
                                  "probes": [probe],
                                  "initial_positive_violations": {
                                      "pooled": .6, "local:A": 0., "local:B": 0.},
                                  "initial_total_positive_violation": .6,
                                  "pooled_hard_floor": 0, "pooled_soft_floor": 0.0}}
    return record, before, after, quota


def test_gate_reconstructs_accepted_probe_and_selected_radius():
    record, before, after, quota = _accepted()
    score._policy(record, before, after, quota, require_local_hard=True)


@pytest.mark.parametrize("mutation,match", [
    (lambda r: r["boundary_policy"]["probes"][0].update(radius=.05), "probe radius"),
    (lambda r: r["boundary_policy"]["probes"][0]["local_soft"].update(A=.9),
     "partition mismatch"),
    (lambda r: r["boundary_policy"]["probes"][0].update(accepted=False),
     "probe decision"),
    (lambda r: r["boundary_policy"]["probes"][0].update(pooled_hard=0),
     "probe local hard"),
    (lambda r: r["boundary_policy"].update(initial_radius=.05), "initial radius"),
    (lambda r: r["boundary_policy"]["probes"][0].update(
        total_positive_violation=.2), "total violation"),
])
def test_gate_rejects_mutated_probe_evidence(mutation, match):
    record, before, after, quota = _accepted()
    mutation(record)
    with pytest.raises(RuntimeError, match=match):
        score._policy(record, before, after, quota, require_local_hard=True)


def test_gate_does_not_accept_conflicting_derivative():
    record, before, after, quota = _accepted()
    record["scope_directional_derivatives"]["pooled"] = .1
    with pytest.raises(RuntimeError, match="conflicting scopes"):
        score._policy(record, before, after, quota, require_local_hard=True)


def test_gate_rejects_nonfirst_accepted_probe():
    record, before, after, quota = _accepted()
    second = copy.deepcopy(record["boundary_policy"]["probes"][0])
    second.update(halving=1, radius=.05)
    record["boundary_policy"]["probes"].append(second)
    with pytest.raises(RuntimeError, match="continued after acceptance"):
        score._policy(record, before, after, quota, require_local_hard=True)


def test_phr_zero_gradient_keeps_pto_but_projects_dual():
    pto = torch.zeros(2, score.base.CLASSES)
    pto[:, 1] = .8
    pto[:, 0] = .2
    quota = {"global_cap": 1, "local_caps": {"A": 1, "B": 1}}
    record = {"applied": False, "radius": 0., "displacement": 0., "rho": .5,
              "gradient_norm": 0., "activation_reason": "zero_phr_gradient",
              "scope_directional_derivatives": {},
              "boundary_policy": {"applied": False, "radius": 0.,
                                  "reason": "zero_phr_gradient", "probes": []},
              "hard_before_global": 2, "hard_after_global": 2,
              "hard_before_local": {"A": 1, "B": 1},
              "hard_after_local": {"A": 1, "B": 1},
              "soft_before_global": 1.6, "soft_after_global": 1.6,
              "soft_before_local": {"A": .8, "B": .8},
              "soft_after_local": {"A": .8, "B": .8},
              "residuals_before": [.6, -.2, -.2],
              "residuals_after": [.6, -.2, -.2],
              "dual_before": [0., 0., 0.], "dual_after": [.3, 0., 0.],
              "penalty_before": .09, "penalty_after": .09}
    next_dual = score._audit_side(record, pto, pto.clone(), ["A", "B"],
                                  quota, "phr_local", [0., 0., 0.])
    assert next_dual == pytest.approx([.3, 0., 0.])
    record["dual_after"][0] = .31
    with pytest.raises(RuntimeError, match="projected dual"):
        score._audit_side(record, pto, pto.clone(), ["A", "B"], quota,
                          "phr_local", [0., 0., 0.])


def test_fixed_full_block_denominator_required_before_labels(tmp_path, monkeypatch):
    monkeypatch.setattr(score.alm, "_manifest_and_labels",
                        lambda *_args: pytest.fail("labels reached before full denominator"))
    with pytest.raises(RuntimeError, match="block incomplete"):
        score.main(tmp_path, tmp_path)


def test_full_contrast_family_uses_all_twelve_paired_seeds(tmp_path, monkeypatch):
    for seed in score.SEEDS:
        directory = tmp_path / f"seed{seed}"
        directory.mkdir()
        (directory / "manifest.json").write_text("same fixed manifest")
    monkeypatch.setattr(score, "_data_bytes", lambda *_args: None)
    monkeypatch.setattr(score, "_receipt", lambda *_args: (None,) * 6 +
                        ({"release_commit": "a" * 40},))
    monkeypatch.setattr(score, "_steps", lambda *_args: None)

    def seed_row(directory, *_args, **_kwargs):
        seed = int(directory.name.removeprefix("seed"))
        offset = (seed - score.SEEDS[0]) * .001
        arms = {}
        for name, shift in (("ens_pto", 0.), ("ens_joint", .02),
                            ("ens_sham", 0.), ("ens_phr_local", -.01)):
            arms[name] = {"allocated": {metric: .4 + offset + shift
                                        for metric in score.base.METRICS},
                          "prediction_sha256": f"{seed}-{name}"}
        return {"manifest_sha256": "same", "release_commit": "a" * 40,
                "caps": {"10": {"arms": arms}, "20": {"arms": arms}}}

    monkeypatch.setattr(score, "load_seed", seed_row)
    report = score.main(tmp_path, tmp_path)
    assert len(report["seeds"]) == 12
    assert len(report["primary_family"]) == 6
    assert set(report["contrasts"]["cc_f1"]) == set(report["primary_family"])
    for contrast in report["contrasts"]["cc_f1"].values():
        assert len(contrast["per_seed"]) == 12
        assert contrast["holm_p"] is not None
    assert report["contrasts"]["cc_f1"]["cap_divisor_10_joint_minus_ens_pto"]["mean"] == pytest.approx(.02)
