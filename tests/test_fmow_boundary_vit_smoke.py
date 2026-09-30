"""CPU-only refusal tests for the ViT CUDA-smoke receipt and GPU preflight."""

import copy
from pathlib import Path
import subprocess
import sys

import pytest
import torch

from tools import fmow_local_boundary_vit_smoke as smoke


def test_module_cli_is_importable_from_release_root_without_pythonpath():
    release = Path(__file__).resolve().parents[1]
    result = subprocess.run([sys.executable, "-m", "tools.fmow_local_boundary_vit_smoke"],
                            cwd=release, capture_output=True, text=True)
    assert result.returncode != 0
    assert "RELEASE_SHA DATA_ROOT GPU_INDEX NEW_RECEIPT_JSON" in result.stderr


def valid_receipt():
    phase = dict(completed=True, seconds=0.25, peak_allocated_bytes=250)
    country = {name: dict(hard=3, soft=1.25, cap=1)
               for name in smoke.COUNTRIES}
    scopes = dict(pooled=dict(hard=15, soft=6.25, cap=3),
                  countries=country)
    cap = dict(joint_gradient_norm=1.0, phr_gradient_norm=2.0,
               joint_applied=True, phr_applied=True,
               all_four_arms=True, scope_derivatives_finite=True,
               pto_unchanged=True, scopes=scopes)
    return dict(memory_smoke_passed=True, label_free=True, precision="fp32",
                backbone="vit_b_16", batch_size=16, development_batch_size=8,
                weight_sha256=smoke.WEIGHT_SHA, development_pool_count=1673,
                total_memory_bytes=1000, peak_allocated_bytes=250,
                phases=dict(
                    train_backward=dict(phase, input_shape=[16, 3, 224, 224],
                                        optimizer_step=True, loss=0.5,
                                        gradient_norm=1.0),
                    development_inference=dict(phase,
                                               probabilities_shape=[1673, 8],
                                               finite_rows=1673,
                                               row_sum_max_error=1e-7),
                    side_copy_constraint_gradient=dict(phase, pto_unchanged=True,
                                                       fixture=smoke.FIXTURE_NAME,
                                                       pooled_gradient_precheck=dict(
                                                           total_norm=2.0,
                                                           backbone_norm=0.1,
                                                           pto_unchanged=True),
                                                       caps={"10": cap, "20": cap})))


def test_measured_receipt_accepts_all_three_phases_and_both_caps():
    assert smoke.validate_success_receipt(valid_receipt())


@pytest.mark.parametrize("path,value", [
    (("memory_smoke_passed",), False),
    (("label_free",), False),
    (("batch_size",), 32),
    (("development_batch_size",), 16),
    (("weight_sha256",), "0" * 64),
    (("development_pool_count",), 8),
    (("peak_allocated_bytes",), 1000),
    (("phases", "train_backward", "completed"), False),
    (("phases", "train_backward", "seconds"), 0),
    (("phases", "train_backward", "peak_allocated_bytes"), 0),
    (("phases", "train_backward", "input_shape"), [8, 3, 224, 224]),
    (("phases", "train_backward", "gradient_norm"), float("nan")),
    (("phases", "train_backward", "optimizer_step"), False),
    (("phases", "development_inference", "probabilities_shape"), [8, 8]),
    (("phases", "development_inference", "finite_rows"), 1672),
    (("phases", "development_inference", "row_sum_max_error"), 0.01),
    (("phases", "side_copy_constraint_gradient", "pto_unchanged"), False),
    (("phases", "side_copy_constraint_gradient", "fixture"), "unfixed_head"),
    (("phases", "side_copy_constraint_gradient", "caps", "10", "joint_gradient_norm"), 0),
    (("phases", "side_copy_constraint_gradient", "caps", "20", "scope_derivatives_finite"), False),
    (("phases", "side_copy_constraint_gradient", "caps", "20", "all_four_arms"), False),
    (("phases", "side_copy_constraint_gradient", "caps", "10", "joint_applied"), False),
    (("phases", "side_copy_constraint_gradient", "caps", "20", "scopes", "pooled", "soft"), 2.0),
    (("phases", "side_copy_constraint_gradient", "caps", "10", "scopes", "countries", "DZA", "hard"), 0),
    (("phases", "side_copy_constraint_gradient", "pooled_gradient_precheck", "backbone_norm"), 0),
])
def test_receipt_mutations_fail_closed(path, value):
    receipt = copy.deepcopy(valid_receipt())
    field = receipt
    for key in path[:-1]:
        field = field[key]
    field[path[-1]] = value
    with pytest.raises(ValueError):
        smoke.validate_success_receipt(receipt)


def test_phases_cannot_be_replaced_by_a_declarative_pass_flag():
    receipt = valid_receipt()
    receipt["phases"] = {}
    with pytest.raises(ValueError, match="measured smoke phase"):
        smoke.validate_success_receipt(receipt)


def test_device_preflight_refuses_a_foreign_compute_pid(monkeypatch):
    calls = []

    def command(*args):
        calls.append(args)
        if "--query-gpu=uuid" in args:
            return "GPU-TEST"
        if "--query-compute-apps=pid" in args:
            return "12345"
        raise AssertionError(args)

    monkeypatch.setattr(smoke, "_command", command)
    with pytest.raises(RuntimeError, match="compute PIDs"):
        smoke._assert_gpu_free("2", "GPU-TEST")
    assert len(calls) == 2


def test_device_preflight_refuses_uuid_index_remap(monkeypatch):
    monkeypatch.setattr(smoke, "_command", lambda *_: "GPU-OTHER")
    with pytest.raises(RuntimeError, match="index changed"):
        smoke._assert_gpu_free("2", "GPU-TEST")


def test_diagnostic_head_activates_all_scopes_with_nonzero_backbone_gradient():
    torch.manual_seed(7)
    backbone = torch.nn.Linear(4, 4)
    head = torch.nn.Linear(4, 8)
    smoke._install_diagnostic_head(head)
    x = torch.arange(60, dtype=torch.float32).reshape(15, 4) / 60
    probabilities = torch.softmax(head(backbone(x)), 1)
    groups = [name for name in smoke.COUNTRIES for _ in range(3)]
    scopes = smoke._scope_diagnostics(
        probabilities, groups, dict(global_cap=3,
                                    local_caps={name: 0 for name in smoke.COUNTRIES}))
    assert smoke._assert_active_scopes(scopes)
    probabilities[:, 1].sum().backward()
    assert backbone.weight.grad is not None
    assert torch.isfinite(backbone.weight.grad).all()
    assert backbone.weight.grad.norm() > 0


def test_fixed_fixture_activates_both_production_quota_scales():
    from tralo.fmow_local import budgets

    head = torch.nn.Linear(4, 8)
    smoke._install_diagnostic_head(head)
    groups = [name for name, count in zip(smoke.COUNTRIES, (337, 334, 334, 334, 334))
              for _ in range(count)]
    probabilities = torch.softmax(head(torch.zeros(len(groups), 4)), 1)
    for quota in budgets(groups).values():
        assert smoke._assert_active_scopes(
            smoke._scope_diagnostics(probabilities, groups, quota))


def test_zero_backbone_gradient_is_rejected_even_with_active_counts():
    backbone = torch.nn.Linear(4, 4)
    head = torch.nn.Linear(4, 8)
    model = torch.nn.Sequential(backbone, head)
    with torch.no_grad():
        head.weight.zero_()
        head.bias.zero_()
        head.bias[1] = 1.0
    images = torch.zeros(15, 4)
    with pytest.raises(RuntimeError, match="backbone gradient is zero"):
        smoke._pooled_gradient_diagnostic(model, head, [images])


def test_inactive_fixture_fails_before_side_copy():
    head = torch.nn.Linear(4, 8)
    smoke._install_diagnostic_head(head)
    with torch.no_grad():
        head.bias[1] = -10
    values = torch.softmax(head(torch.zeros(15, 4)), 1)
    groups = [name for name in smoke.COUNTRIES for _ in range(3)]
    scopes = smoke._scope_diagnostics(
        values, groups, dict(global_cap=3,
                             local_caps={name: 1 for name in smoke.COUNTRIES}))
    with pytest.raises(RuntimeError, match="inactive diagnostic scope"):
        smoke._assert_active_scopes(scopes)


def test_diagnostic_fixture_exercises_real_side_copy_gradients_on_cpu(tmp_path):
    from tralo.fmow_local import snapshot_side_steps
    from tralo.knee_end_to_end import infer

    torch.manual_seed(11)
    head = torch.nn.Linear(4, 8)
    model = torch.nn.Sequential(torch.nn.Linear(4, 4), head).eval()
    smoke._install_diagnostic_head(head)
    pool = [torch.arange(60, dtype=torch.float32).reshape(15, 4) / 60]
    groups = [name for name in smoke.COUNTRIES for _ in range(3)]
    quota = dict(global_cap=3, local_caps={name: 0 for name in smoke.COUNTRIES})
    pto = infer(model, pool)
    assert smoke._assert_active_scopes(smoke._scope_diagnostics(pto, groups, quota))
    before = [p.detach().clone() for p in model.parameters()]
    records = snapshot_side_steps(model, pool, groups, quota, 6500, 1, tmp_path,
                                  fixed_radius=0.1,
                                  phr_state={"dual": torch.zeros(6)},
                                  phr_rho=0.5, boundary_calibrated=True,
                                  pto_probabilities=pto)
    assert set(records) == {"joint", "global_dose", "sham", "phr_local"}
    assert records["joint"]["gradient_norm"] > 0
    assert records["phr_local"]["gradient_norm"] > 0
    assert records["joint"]["applied"]
    assert records["phr_local"]["applied"]
    assert all(torch.equal(p, old) for p, old in zip(model.parameters(), before))
