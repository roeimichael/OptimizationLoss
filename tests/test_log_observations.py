"""Legacy CSV fields cannot establish warm-up state or applied gradient norm."""

import json
import pandas as pd
from test_operational_cli import cli


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
