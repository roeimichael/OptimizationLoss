"""Distribution measurements must not inherit archived seed-noise decisions."""

from test_operational_cli import cli


def test_distribution_cli_has_no_historical_decision_surface(tmp_path):
    from scripts.dataset_screen import _synthetic

    path = _synthetic(str(tmp_path / "slice"), "live", per=30)
    result = cli("scripts.dataset_screen", path)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "NET ex" in result.stdout and "GLOBAL ex" in result.stdout
    assert not any(
        word in result.stdout
        for word in ("STAGE 1 PASS", "DEAD", "MARGINAL", "seed noise", "dermmnist")
    )
    bad = cli("scripts.dataset_screen", path, "--noise", 27.83)
    assert bad.returncode == 2 and "unrecognized arguments" in bad.stderr
