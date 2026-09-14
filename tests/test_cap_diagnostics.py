import json
import pandas as pd
from test_operational_cli import cli, campaign as generated_campaign

campaign = generated_campaign


def test_headroom_uses_deployed_selection_when_global_competition_changes_cut():
    import numpy as np
    from scripts.headroom import group_headroom

    y = np.array([1, 0, 1, 0])
    groups = np.array([0, 0, 1, 1])
    prob = np.array([[0.1, 0.9], [0.2, 0.8], [0.3, 0.7], [0.4, 0.6]])
    # The deployed allocation spent its single global slot on a wrong group-0 item.
    rows = group_headroom(
        y, groups, prob, [1], {1: 1}, {0: {1: 1}, 1: {1: 1}}, np.array([0, 1, 0, 0])
    )
    assert [
        (r["emitted"], r["selected_tp"], r["selected_errors"], r["outside_tp"])
        for r in rows
    ] == [(1, 0, 1, 1), (0, 0, 0, 1)]


def populated(campaign):
    path = next(
        p
        for p in campaign.rglob("config.json")
        if json.loads(p.read_text())["arm"] == "clip"
    )
    c = json.loads(path.read_text())
    c["status"] = "completed"
    c["dataset_config"].update(
        num_classes=3, constrained_class=[1], group_column="group"
    )
    c["constraint"] = [0.5, 1.0]
    path.write_text(json.dumps(c))
    # Each group has two true class-1 items, so local K=1 and global K=4.
    frame = pd.DataFrame(
        {
            "True_Label": [1, 0, 1, 1, 0, 1],
            "Predicted_Label": [1, 0, 0, 1, 0, 0],
            "Group_ID": [0, 0, 0, 1, 1, 1],
            "Prob_Class_0": [0.1, 0.2, 0.8, 0.1, 0.2, 0.8],
            "Prob_Class_1": [0.8, 0.7, 0.1, 0.8, 0.7, 0.1],
            "Prob_Class_2": [0.1] * 6,
        }
    )
    for name in ("final_predictions.csv", "final_predictions_raw.csv"):
        frame.to_csv(path.parent / name, index=False)
    return path.parent, frame


def test_feasibility_checks_embedded_groups_and_rejects_local_violation(campaign):
    run, frame = populated(campaign)
    ok = cli("scripts.feasibility_check", campaign)
    assert ok.returncode == 0, ok.stdout + ok.stderr
    frame.loc[1, "Predicted_Label"] = 1
    frame.to_csv(run / "final_predictions.csv", index=False)
    bad = cli("scripts.feasibility_check", campaign)
    assert bad.returncode == 1 and "LOCAL" in bad.stdout


def test_actual_per_group_headroom_is_not_global_topk(campaign):
    run, frame = populated(campaign)
    # Both local cuts select a correct item; global top two could select group 0 twice.
    from scripts.headroom import group_headroom

    rows = group_headroom(
        frame.True_Label.to_numpy(),
        frame.Group_ID.to_numpy(),
        frame[["Prob_Class_0", "Prob_Class_1", "Prob_Class_2"]].to_numpy(),
        [1],
        {1: 4},
        {0: {1: 1}, 1: {1: 1}},
        frame.Predicted_Label.to_numpy(),
    )
    assert len(rows) == 2
    assert [
        (r["K"], r["selected_tp"], r["selected_errors"], r["outside_tp"]) for r in rows
    ] == [(1, 1, 0, 1), (1, 1, 0, 1)]
    result = cli("scripts.headroom", campaign)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "group" in result.stdout and "outside_tp" in result.stdout


def test_deployed_cli_refuses_unreceipted_altered_fixture(campaign, tmp_path):
    run, _ = populated(campaign)
    result = cli("scripts.deployed_h2h", "--campaign", campaign)
    assert result.returncode == 1, result.stdout + result.stderr
    assert 'config changed' in result.stdout
    cfg = json.loads((run / "config.json").read_text())
    cfg["status"] = "pending"
    (run / "config.json").write_text(json.dumps(cfg))
    output = tmp_path / "must-not-exist.json"
    bad = cli("scripts.deployed_h2h", "--campaign", campaign, "--json", output)
    assert bad.returncode == 1 and not output.exists()


def test_campaign_does_not_report_after_failed_required_checks(campaign):
    populated(campaign)
    result = cli("scripts.run_campaign", "--root", campaign, "--step", "score")
    assert result.returncode == 2, result.stdout + result.stderr
    assert 'config changed' in result.stderr
