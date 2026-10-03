import json
from pathlib import Path

import pytest

from tools.tabular_cell_budget import remaining_cell_seconds


def receipt(path: Path, seed: int, elapsed: int, exit_code: int = 0) -> None:
    path.write_text(json.dumps({"seed": seed, "elapsed_seconds": elapsed,
                                "exit_code": exit_code}))


def test_budget_counts_gpu_time_not_time_since_pilot_launch(tmp_path: Path) -> None:
    receipt(tmp_path / "pilot_6890.complete.json", 6890, 6418)
    assert remaining_cell_seconds(tmp_path, 6890, 86400) == 79982

    receipt(tmp_path / "full_6891.complete.json", 6891, 7200)
    assert remaining_cell_seconds(tmp_path, 6890, 86400) == 72782


def test_budget_rejects_failed_or_misattributed_receipt(tmp_path: Path) -> None:
    receipt(tmp_path / "pilot_6890.complete.json", 6890, 6418)
    receipt(tmp_path / "full_6891.complete.json", 6892, 7200)
    with pytest.raises(ValueError, match="seed"):
        remaining_cell_seconds(tmp_path, 6890, 86400)

    receipt(tmp_path / "full_6891.complete.json", 6891, 7200, exit_code=124)
    with pytest.raises(ValueError, match="exit"):
        remaining_cell_seconds(tmp_path, 6890, 86400)
