"""Executable boundary checks for the maintained seven-arm campaign."""

import json
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
ARMS = {"tralo", "tralo_null", "clip", "focal_clip", "fioretto", "hounie", "alm"}


def generate(tmp_path, *options):
    return subprocess.run(
        [
            sys.executable,
            "-m",
            "configs.gen_campaign",
            "--root",
            str(tmp_path),
            "--datasets",
            "iwildcam",
            "--caps",
            "L30_G50",
            "L50_G30",
            *options,
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )


def test_all_generates_only_seven_paired_arms_with_reference_recipe(tmp_path):
    result = generate(tmp_path, "--arms", "all")
    assert result.returncode == 0, result.stdout + result.stderr
    configs = [json.loads(p.read_text()) for p in tmp_path.rglob("config.json")]
    assert len(configs) == 7 * 2 * 4
    assert {c["arm"] for c in configs} == ARMS
    for tag in ("L30_G50", "L50_G30"):
        for seed in (1, 2, 3, 4):
            cell = {
                c["arm"]: c
                for c in configs
                if c["constraint_tag"] == tag and c["hyperparams"]["seed"] == seed
            }
            assert set(cell) == ARMS
            for c in cell.values():
                hp = c["hyperparams"]
                assert hp["warmup_epochs"] + hp["constraint_epochs"] == 30
                if c["arm"] not in ("clip", "focal_clip"):
                    assert hp["constraint_fp32"] is True
                    assert hp["constraint_grad_mode"] == "normalize"
            assert cell["tralo"]["base_model_id"] == cell["tralo_null"]["base_model_id"]
            assert cell["clip"]["base_model_id"] != cell["focal_clip"]["base_model_id"]
            null = cell["tralo_null"]["hyperparams"]
            assert [
                null[k] for k in ("lambda_step", "lambda_global", "lambda_local")
            ] == [0, 0, 0]


@pytest.mark.parametrize(
    "options",
    [
        ("--arms", "tralo_margin"),
        ("--arms", "all", "--soft-count-mode", "sum"),
        ("--arms", "fioretto_null"),
        ("--arms", "all+null"),
    ],
)
def test_retired_cli_choices_fail_without_partial_configs(tmp_path, options):
    result = generate(tmp_path, *options)
    assert result.returncode != 0
    assert (
        "invalid choice" in result.stderr or "unrecognized arguments" in result.stderr
    )
    assert not list(tmp_path.rglob("config.json"))


def test_explicit_seeds_remain_paired_and_duplicate_seeds_fail(tmp_path):
    result = generate(tmp_path / "valid", "--arms", "all", "--seeds", "0", "9")
    assert result.returncode == 0, result.stderr
    configs = [
        json.loads(p.read_text()) for p in (tmp_path / "valid").rglob("config.json")
    ]
    assert len(configs) == 28
    assert {c["hyperparams"]["seed"] for c in configs} == {0, 9}
    for options in (("0", "0"), ("-1", "2")):
        bad = generate(tmp_path / "invalid", "--seeds", *options)
        assert bad.returncode != 0
        assert not list((tmp_path / "invalid").rglob("config.json"))


def test_unknown_hyperparameter_is_rejected_before_training(tmp_path):
    from configs.gen_campaign import build_hyperparams, load_protocol
    from src.experiments.runner import run_experiment

    p = load_protocol()
    hp = build_hyperparams(p, p["arms"]["tralo"], 1)
    hp["unknown_research_option"] = True
    config = {"methodology": "tralo", "status": "pending", "hyperparams": hp}
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="unknown_research_option"):
        run_experiment(str(path))
    assert json.loads(path.read_text())["status"] == "pending"


def test_train_inputs_do_not_expose_held_out_labels():
    from dataclasses import fields
    from src.pipeline.contracts import TrainInputs
    assert "y_test" not in {field.name for field in fields(TrainInputs)}


@pytest.mark.parametrize("arm", ["tralo", "fioretto", "hounie", "alm"])
def test_satisfied_trainer_keeps_its_fixed_epoch_budget(tmp_path, arm):
    import csv
    import torch
    from configs.gen_campaign import load_protocol
    from scripts.smoke_arms import make_inputs
    from src.experiments.runner import TRAIN_FNS
    torch.set_num_threads(1)
    inputs, _, _ = make_inputs(load_protocol(), arm, tmp_path)
    inputs.global_con = [1e10] * inputs.num_classes
    inputs.local_con = {}
    inputs.hyperparams["constraint_epochs"] = 32
    TRAIN_FNS[inputs.config["methodology"]](inputs)
    with inputs.csv_log_path.open() as stream:
        assert len(list(csv.DictReader(stream))) == 32
