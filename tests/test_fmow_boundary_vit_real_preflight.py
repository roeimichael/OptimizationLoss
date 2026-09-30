"""CPU-only validation of ViT real-image numerical gate evidence."""

import copy
from pathlib import Path
import subprocess
import sys

import pytest

from tools import fmow_local_boundary_vit_real_preflight as preflight


def valid_receipt():
    return dict(
        preflight_passed=True, development_labels_accessed=False,
        precision="fp32", backbone="vit_b_16", weight_sha256=preflight.WEIGHT_SHA,
        development_batch_size=8, chunk_sizes=[8, 7],
        images_count=15, country_counts=dict.fromkeys(preflight.COUNTRIES, 3),
        pto_unchanged=True, arms_audited=list(preflight.ARMS),
        max_probability_difference=1e-7,
        gradient_relative_errors={name: .001 for name in preflight.SCOPES},
        finite_differences={f"{name}@{epsilon}": dict(
            analytic=-.2, numeric=-.2, error=0., tolerance=.013)
            for name in preflight.SCOPES for epsilon in preflight.EPSILONS},
        artifact_sha256={f"epoch01_{name}.pt": "a" * 64
                         for name in preflight.ARMS})


def test_module_cli_is_importable_from_release_root_without_pythonpath():
    release = Path(__file__).resolve().parents[1]
    result = subprocess.run([sys.executable, "-m",
                             "tools.fmow_local_boundary_vit_real_preflight"],
                            cwd=release, capture_output=True, text=True)
    assert result.returncode != 0
    assert "RELEASE_SHA DATA_ROOT GPU_INDEX NEW_RECEIPT_JSON" in result.stderr


def test_valid_complete_numerical_receipt():
    assert preflight.validate_success_receipt(valid_receipt())


@pytest.mark.parametrize("path,value", [
    (("preflight_passed",), False),
    (("development_labels_accessed",), True),
    (("weight_sha256",), "0" * 64),
    (("images_count",), 12),
    (("development_batch_size",), 3),
    (("chunk_sizes",), [3, 3, 3, 3, 3]),
    (("country_counts", "DZA"), 2),
    (("pto_unchanged",), False),
    (("arms_audited",), ["joint"]),
    (("max_probability_difference",), 1e-4),
    (("max_probability_difference",), float("nan")),
    (("gradient_relative_errors", "pooled"), .02),
    (("finite_differences", "pooled@0.01", "numeric"), -.1),
    (("artifact_sha256", "epoch01_joint.pt"), "0"),
])
def test_numerical_receipt_mutations_fail_closed(path, value):
    receipt = copy.deepcopy(valid_receipt())
    target = receipt
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    with pytest.raises(ValueError):
        preflight.validate_success_receipt(receipt)


def test_finite_difference_missing_scope_fails_closed():
    receipt = valid_receipt()
    del receipt["finite_differences"]["TUR@0.02"]
    with pytest.raises(ValueError, match="missing finite-difference"):
        preflight.validate_success_receipt(receipt)
