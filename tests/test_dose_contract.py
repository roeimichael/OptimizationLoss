"""Actual generated run records must establish their planned constraint dose."""

import json
import pytest
from test_operational_cli import cli, campaign as generated_campaign

campaign = generated_campaign


def complete_counts(campaign):
    for path in campaign.rglob("config.json"):
        cfg = json.loads(path.read_text())
        steps = 0 if cfg["arm"] in {"tralo_null", "clip", "focal_clip"} else 29
        cfg.update(
            status="completed",
            results={
                "constraint_steps_attempted": steps,
                "constraint_steps_applied": steps,
            },
        )
        path.write_text(json.dumps(cfg))


def test_empty_dose_is_incomplete(tmp_path):
    result = cli("scripts.dose_landed", tmp_path)
    assert result.returncode == 1, result.stdout + result.stderr


@pytest.mark.parametrize(
    "defect",
    [
        "none",
        "missing",
        "short",
        "zero",
        "null_active",
        "posthoc_active",
        "pending",
        "unequal",
    ],
)
def test_actual_dose_contract(campaign, defect):
    complete_counts(campaign)
    arm = {"null_active": "tralo_null", "posthoc_active": "clip"}.get(defect, "tralo")
    path = next(
        p
        for p in campaign.rglob("config.json")
        if json.loads(p.read_text())["arm"] == arm
    )
    cfg = json.loads(path.read_text())
    if defect == "missing":
        cfg["results"] = {}
    elif defect in {"short", "zero", "null_active", "posthoc_active", "unequal"}:
        steps = 0 if defect == "zero" else 1
        cfg["results"] = {
            "constraint_steps_attempted": steps,
            "constraint_steps_applied": steps,
        }
    elif defect == "pending":
        cfg["status"] = "pending"
    path.write_text(json.dumps(cfg))
    if defect == "unequal":
        # The other run offsets the short dose: arm averages alone hide this.
        other = next(
            p
            for p in campaign.rglob("config.json")
            if p != path and json.loads(p.read_text())["arm"] == arm
        )
        sibling = json.loads(other.read_text())
        sibling["results"] = {
            "constraint_steps_attempted": 57,
            "constraint_steps_applied": 57,
        }
        other.write_text(json.dumps(sibling))
    result = cli("scripts.dose_landed", campaign)
    assert result.returncode == (0 if defect == "none" else 1), (
        result.stdout + result.stderr
    )


@pytest.mark.parametrize("arm", ["tralo", "tralo_null", "clip", "focal_clip"])
def test_first_completed_run_has_real_dose_shape(campaign, arm):
    path = next(
        p
        for p in campaign.rglob("config.json")
        if json.loads(p.read_text())["arm"] == arm
    )
    cfg = json.loads(path.read_text())
    steps = 29 if arm == "tralo" else 0
    cfg.update(
        status="completed",
        results={
            "constraint_steps_attempted": steps,
            "constraint_steps_applied": steps,
        },
    )
    if arm in {"clip", "focal_clip"}:
        cfg["results"] = {
            "constraint_steps_attempted": None,
            "constraint_steps_applied": None,
        }
    path.write_text(json.dumps(cfg))
    # A post-hoc or null by itself is a valid zero-dose run. It cannot establish
    # a trained dose for the rest of a still-pending campaign.
    one = cli("scripts.dose_landed", path.parent)
    assert one.returncode == 0, one.stdout + one.stderr
    whole = cli("scripts.dose_landed", campaign)
    assert whole.returncode == (0 if arm == "tralo" else 1), whole.stdout + whole.stderr


def test_unequal_attempted_summary_is_a_failure():
    import io
    from scripts.dose_landed import report

    buf = io.StringIO()
    assert report({"tralo": [29, 29, 1], "fioretto": [1, 1, 1]}, {}, out=buf) > 0
    assert "CROSS-ARM ATTEMPTS" in buf.getvalue()
