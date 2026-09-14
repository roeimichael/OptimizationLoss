"""Legacy CSV fields cannot establish warm-up state or applied gradient norm."""

import json
import pandas as pd
import pytest
from test_operational_cli import cli


def test_observed_warmup_infinity_is_not_hidden_by_empty_constraint_fields(tmp_path):
    (tmp_path / "config.json").write_text(
        json.dumps({"arm": "tralo", "status": "completed"})
    )
    pd.DataFrame(
        {
            "Epoch": [1, 2, 3],
            "L_CE": [float("inf"), 0.1, 0.1],
            "Train_Acc": [0.5, 0.6, 0.7],
            "Limit_Class1": [float("nan"), 2, 2],
        }
    ).to_csv(tmp_path / "training_log.csv", index=False)
    result = cli("scripts.log_health", tmp_path)
    assert result.returncode == 1, result.stdout + result.stderr
    assert "FAIL:" in result.stdout


@pytest.mark.parametrize(
    "loss,acc",
    [
        ([0.1, float("inf"), 0.1], [0.9, 0.9, 0.9]),
        ([float("inf")] * 3, [0.9] * 3),
        ([0.1] * 3, [0.9, 0.9, 0.1]),
    ],
)
def test_invalid_observed_logs_fail_the_real_cli(tmp_path, loss, acc):
    (tmp_path / "config.json").write_text(
        json.dumps({"arm": "fioretto", "status": "completed"})
    )
    pd.DataFrame({"epoch": [0, 1, 2], "train_acc": acc, "ce_loss": loss}).to_csv(
        tmp_path / "training_log.csv", index=False
    )
    result = cli("scripts.log_health", tmp_path)
    assert result.returncode == 1, result.stdout + result.stderr
    assert "FAIL" in result.stdout


def test_untagged_rival_rows_do_not_invent_warmup_or_clipping(tmp_path):
    run = tmp_path / "fioretto"
    run.mkdir()
    (run / "config.json").write_text(
        json.dumps(
            {
                "arm": "fioretto",
                "methodology": "fioretto_ldf",
                "status": "completed",
                "hyperparams": {"warmup_epochs": 1, "constraint_epochs": 29},
                "results": {
                    "constraint_steps_attempted": 3,
                    "constraint_steps_applied": 3,
                },
            }
        )
    )
    pd.DataFrame(
        {
            "epoch": [0, 1, 2],
            "train_acc": [0.95, 0.951, 0.952],
            "ce_loss": [0.1, 0.1, 0.1],
            "grad_norm": [0.005] * 3,
        }
    ).to_csv(run / "training_log.csv", index=False)
    result = cli("scripts.log_health", tmp_path)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "unknown" in result.stdout.lower()
    assert "warm-up epoch accuracy" not in result.stdout
    assert "clip binds every step" not in result.stdout
    assert "raw" in result.stdout.lower()
