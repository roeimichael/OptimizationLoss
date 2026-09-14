"""Real temporary campaign fixtures for maintained command boundaries."""

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

REPO = Path(__file__).resolve().parents[1]


def cli(module, *args):
    env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    return subprocess.run(
        [sys.executable, "-m", module, *map(str, args)],
        cwd=REPO,
        env=env,
        text=True,
        capture_output=True,
        timeout=60,
    )


@pytest.fixture
def campaign(tmp_path):
    root = tmp_path / "campaign"
    result = cli(
        "configs.gen_campaign",
        "--root",
        root,
        "--datasets",
        "iwildcam",
        "--models",
        "MobileNetV3",
        "--arms",
        "all",
        "--seeds",
        "1",
        "--caps",
        "L30_G30",
        "L50_G50",
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert len(list(root.rglob("config.json"))) == 14
    return root


@pytest.mark.parametrize("stages", [("typo",), ("all", "typo")])
def test_preflight_rejects_original_unknown_stage(stages):
    result = cli("scripts.preflight", "--stage", *stages, "--collect-only")
    assert result.returncode == 2
    assert "unknown stage" in result.stderr


def test_preflight_list_and_real_budget_stage():
    result = cli("scripts.preflight", "--list")
    assert result.returncode == 0 and "budget" in result.stdout
    result = cli(
        "scripts.preflight",
        "--stage",
        "budget",
        "-k",
        "two_cap_tags_are_not_two_cap_levels",
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "1 passed" in result.stdout


def test_campaign_list_unknown_step_and_invalid_root(tmp_path):
    result = cli("scripts.run_campaign", "--list")
    assert (
        result.returncode == 0
        and "feasibility_check" in result.stdout
        and "deployed_h2h" in result.stdout
    )
    for args in [
        ("--step", "typo"),
        ("--all", "--step", "typo"),
        ("--step", "verify", "--root", tmp_path / "absent"),
    ]:
        result = cli("scripts.run_campaign", *args)
        assert result.returncode == 2, result.stdout + result.stderr


def test_audit_actual_root_has_real_positive_and_dead_key_negative(campaign):
    result = cli("scripts.audit_config", campaign)
    assert result.returncode == 0, result.stdout + result.stderr
    path = next(
        p
        for p in campaign.rglob("config.json")
        if json.loads(p.read_text())["arm"] == "clip"
    )
    cfg = json.loads(path.read_text())
    cfg["hyperparams"]["constraint_grad_mode"] = "normalize"
    path.write_text(json.dumps(cfg))
    result = cli("scripts.audit_config", campaign)
    assert result.returncode != 0
    assert "constraint_grad_mode" in result.stdout + result.stderr


@pytest.mark.parametrize(
    "field,value",
    [
        ("lr_constraint", 0.000005),
        ("constraint_grad_mode", "clip"),
        ("constraint_fp32", False),
        ("warmup_epochs", 2),
        ("seed", 99),
    ],
)
def test_parity_current_campaign_and_changed_recipe(campaign, field, value):
    result = cli("scripts.check_parity", campaign)
    assert result.returncode == 0, result.stdout + result.stderr
    path = next(
        p
        for p in campaign.rglob("config.json")
        if json.loads(p.read_text())["arm"] == "tralo"
    )
    cfg = json.loads(path.read_text())
    cfg["hyperparams"][field] = value
    path.write_text(json.dumps(cfg))
    result = cli("scripts.check_parity", campaign)
    assert result.returncode != 0, result.stdout + result.stderr


@pytest.mark.parametrize(
    "module",
    ["audit_config", "check_parity", "pred_integrity", "feasibility_check", "headroom"],
)
def test_empty_root_cannot_pass(module, tmp_path):
    result = cli("scripts." + module, tmp_path)
    assert result.returncode != 0
    assert "Traceback" not in result.stderr


@pytest.mark.parametrize("mutation", ["data", "caps", "source", "precision", "warmup"])
def test_parity_rejects_changed_comparison_axes(campaign, mutation):
    paths = sorted(campaign.rglob("config.json"))
    for path in paths:
        cfg = json.loads(path.read_text())
        cfg["results"] = {
            "runtime": {
                "gpu_name": "fixture",
                "amp_dtype": "fp32",
                "grad_scaler": False,
            }
        }
        path.write_text(json.dumps(cfg))
    path = paths[0]
    cfg = json.loads(path.read_text())
    if mutation == "data":
        cfg["dataset_config"]["data_dir"] += "/other"
    elif mutation == "caps":
        cfg["constraint"] = [0.2, 0.2]
    elif mutation == "source":
        cfg["run_code_version"] = "other-source"
    elif mutation == "precision":
        cfg["results"]["runtime"]["amp_dtype"] = "fp16"
    else:
        other = next(
            json.loads(p.read_text())
            for p in paths
            if json.loads(p.read_text())["arm"] == "clip"
        )
        cfg["base_model_id"] = other["base_model_id"]
    path.write_text(json.dumps(cfg))
    result = cli("scripts.check_parity", campaign)
    assert result.returncode == 1, result.stdout + result.stderr


def test_campaign_verify_executes_checks_against_its_actual_root(campaign):
    result = cli("scripts.run_campaign", "--root", campaign, "--step", "verify")
    assert result.returncode == 0, result.stdout + result.stderr
    path = next(
        p
        for p in campaign.rglob("config.json")
        if json.loads(p.read_text())["arm"] == "clip"
    )
    cfg = json.loads(path.read_text())
    cfg["hyperparams"]["dead_option"] = True
    path.write_text(json.dumps(cfg))
    # Run the same table entry that verify dispatches, using the real subprocess.
    from scripts.run_campaign import BY_NAME, run_check

    args = next(
        args for name, args, _, _ in BY_NAME["verify"][1] if name == "audit_config"
    )
    code, _ = run_check("audit_config", args, str(campaign), False)
    assert code == 1
