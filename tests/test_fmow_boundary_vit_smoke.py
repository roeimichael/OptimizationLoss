"""CPU-only refusal tests for the ViT CUDA-smoke receipt and GPU preflight."""

import copy
from pathlib import Path
import subprocess
import sys

import pytest

from tools import fmow_local_boundary_vit_smoke as smoke


def test_module_cli_is_importable_from_release_root_without_pythonpath():
    release = Path(__file__).resolve().parents[1]
    result = subprocess.run([sys.executable, "-m", "tools.fmow_local_boundary_vit_smoke"],
                            cwd=release, capture_output=True, text=True)
    assert result.returncode != 0
    assert "RELEASE_SHA DATA_ROOT GPU_INDEX NEW_RECEIPT_JSON" in result.stderr


def valid_receipt():
    phase = dict(completed=True, seconds=0.25, peak_allocated_bytes=250)
    cap = dict(joint_gradient_norm=1.0, phr_gradient_norm=2.0,
               joint_applied=False, phr_applied=True,
               all_four_arms=True, scope_derivatives_finite=True,
               pto_unchanged=True)
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
    (("phases", "side_copy_constraint_gradient", "caps", "10", "joint_gradient_norm"), 0),
    (("phases", "side_copy_constraint_gradient", "caps", "20", "scope_derivatives_finite"), False),
    (("phases", "side_copy_constraint_gradient", "caps", "20", "all_four_arms"), False),
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
