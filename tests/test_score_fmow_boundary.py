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


def test_radius_recount_allows_float32_soft_count_roundoff():
    record, _before, _after, quota = _accepted()
    before = (2, {"A": 1, "B": 1}, 1.040000003,
              {"A": .5200000015, "B": .5200000015})
    after = (1, {"A": 1, "B": 0}, 1.0, {"A": .5, "B": .5})
    policy = record["boundary_policy"]
    record["radius"] = policy["radius"] = policy["initial_radius"] = .04
    record["displacement"] = .04
    policy["initial_positive_violations"]["pooled"] = .04
    policy["initial_total_positive_violation"] = .04
    probe = policy["probes"][0]
    probe.update(radius=.04, pooled_soft=1.0, local_soft={"A": .5, "B": .5},
                 positive_violations={"pooled": 0., "local:A": 0., "local:B": 0.},
                 total_positive_violation=0.)
    score._policy(record, before, after, quota, require_local_hard=True)


def test_accepted_probe_replay_allows_one_ulp_count_roundoff_but_not_real_drift():
    record, before, after, quota = _accepted()
    rounded = (after[0], after[1], after[2] + 2e-5,
               {"A": after[3]["A"] + 2e-5, "B": after[3]["B"]})
    score._policy(record, before, rounded, quota, require_local_hard=True)
    drifted = (after[0], after[1], after[2] + 1e-3,
               {"A": after[3]["A"] + 1e-3, "B": after[3]["B"]})
    with pytest.raises(RuntimeError, match="accepted probe pooled soft"):
        score._policy(record, before, drifted, quota, require_local_hard=True)


def test_saved_side_must_preserve_logged_boundary_improvement():
    record, _before, _after, quota = _accepted()
    before = (2, {"A": 1, "B": 1}, 1.5, {"A": .75, "B": .75})
    probe = record["boundary_policy"]["probes"][0]
    probe.update(pooled_soft=1.499998, local_soft={"A": .749999, "B": .749999},
                 positive_violations={"pooled": .499998, "local:A": 0.,
                                      "local:B": 0.},
                 total_positive_violation=.499998)
    record["boundary_policy"]["initial_positive_violations"]["pooled"] = .5
    record["boundary_policy"]["initial_total_positive_violation"] = .5
    final = (1, {"A": 1, "B": 0}, 1.50005, {"A": .750025, "B": .750025})
    with pytest.raises(RuntimeError, match="saved boundary side worsened"):
        score._policy(record, before, final, quota, require_local_hard=True)


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
              "residuals_after": [0.6000000238418579,
                                  -0.19999998807907104, -0.19999998807907104],
              "dual_before": [0., 0., 0.],
              "dual_after": [0.30000001192092896, 0., 0.],
              "penalty_before": .09, "penalty_after": .09}
    next_dual = score._audit_side(record, pto, pto.clone(), ["A", "B"],
                                  quota, "phr_local", [0., 0., 0.])
    assert next_dual == pytest.approx([.3, 0., 0.])
    record["dual_after"][0] += 5e-7
    with pytest.raises(RuntimeError, match="projected dual"):
        score._audit_side(record, pto, pto.clone(), ["A", "B"], quota,
                          "phr_local", [0., 0., 0.])


def test_phr_fp32_dual_replay_avoids_six_epoch_double_drift_and_rejects_mutation():
    # Recorded Netherlands scope of seed 6409, cap divisor 20. The runner's
    # float32 state diverges from a Python-double reconstruction by epoch 6.
    residuals = [5.029261589050293, 4.451845645904541,
                 4.337318420410156, 4.203888893127441,
                 4.481339931488037, 3.7643966674804688]
    recorded = [2.5146307945251465, 4.740553855895996,
                6.909213066101074, 9.011157989501953,
                11.25182819366455, 13.134026527404785]
    state, double_state = [0.], 0.
    for residual, expected in zip(residuals, recorded):
        state = score._project_phr_dual(torch.tensor([residual], dtype=torch.float32),
                                        state)
        double_state = max(0., double_state + score.RHO * residual)
        assert state == [expected]
    assert abs(double_state - recorded[-1]) > 9e-7


def test_phr_side_artifact_replay_carries_fp32_state_across_six_epochs():
    quota = {"global_cap": 1, "local_caps": {"A": 1}}
    groups = ["A"] * 50
    state, double_state = [0., 0.], 0.
    for target in (5.029261589050293, 4.451845645904541,
                   4.337318420410156, 4.203888893127441,
                   4.481339931488037, 3.7643966674804688):
        side = torch.zeros(50, score.base.CLASSES, dtype=torch.float32)
        side[:, score.base.CAPPED] = (1 + target) / 50 + .001
        side[:, 0] = 1 - side[:, score.base.CAPPED]
        counts = score._counts(side, groups, quota)
        q = side[:, score.base.CAPPED]
        residual_fp32 = (torch.stack((q.sum(), q[[True] * len(groups)].sum()))
                         - side.new_tensor([1, 1])) / side.new_tensor([1, 1])
        projected = score._project_phr_dual(residual_fp32, state)
        residual_double = counts[2] - 1
        penalty = sum((max(0., lam + score.RHO * residual_double) ** 2 - lam ** 2)
                      / (2 * score.RHO) for lam in state)
        record = {"applied": False, "radius": 0., "displacement": 0., "rho": score.RHO,
                  "gradient_norm": 0., "activation_reason": "zero_phr_gradient",
                  "scope_directional_derivatives": {},
                  "boundary_policy": {"applied": False, "radius": 0.,
                                      "reason": "zero_phr_gradient", "probes": []},
                  "hard_before_global": counts[0], "hard_after_global": counts[0],
                  "hard_before_local": counts[1], "hard_after_local": counts[1],
                  "soft_before_global": counts[2], "soft_after_global": counts[2],
                  "soft_before_local": counts[3], "soft_after_local": counts[3],
                  "residuals_before": residual_fp32.tolist(),
                  "residuals_after": residual_fp32.tolist(),
                  "dual_before": state[:], "dual_after": projected[:],
                  "penalty_before": penalty, "penalty_after": penalty}
        previous = state
        state = score._audit_side(record, side, side.clone(), groups, quota,
                                  "phr_local", state)
        assert state == projected
        double_state = max(0., double_state + score.RHO * residual_double)
    assert abs(state[0] - double_state) > 1e-6
    corrupted = copy.deepcopy(record)
    corrupted["dual_after"][0] += 5e-7
    with pytest.raises(RuntimeError, match="projected dual"):
        score._audit_side(corrupted, side, side.clone(), groups, quota,
                          "phr_local", previous)
    corrupted = copy.deepcopy(record)
    corrupted["residuals_after"][0] += 5e-7
    with pytest.raises(RuntimeError, match="residuals after"):
        score._audit_side(corrupted, side, side.clone(), groups, quota,
                          "phr_local", previous)


def test_exact_float_vector_rejects_json_bools_and_nonfinite_values():
    assert score._exact_float_vector([0., 1.], [0., 1.])
    assert not score._exact_float_vector([False, True], [0., 1.])
    assert not score._exact_float_vector([float("nan"), 1.], [0., 1.])
    assert not score._exact_float_vector([0.], [0., 1.])


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
