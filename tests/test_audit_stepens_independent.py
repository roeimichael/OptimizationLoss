import json
import math
from pathlib import Path
import sys

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "analysis"))
import audit_stepens_independent as audit  # noqa: E402


def test_single_cap_ties_use_sample_id_and_remaining_class_scores():
    probabilities = [
        [0.1, 0.2, 0.1, 0.5, 0.1],
        [0.1, 0.1, 0.2, 0.5, 0.1],
        [0.6, 0.1, 0.1, 0.1, 0.1],
    ]
    assert audit.capped_first_single(probabilities, ["b", "a", "c"], capped=3, cap=1) == [1, 3, 0]
    assert audit.classification_metrics([1, 3, 0], [1, 3, 0], capped=3, classes=5) == dict(
        cc_f1=1.0, accuracy=1.0, macro_f1=0.6, weighted_f1=1.0)
    with pytest.raises(ValueError, match="unique"):
        audit.capped_first_single(probabilities, ["b", "b", "c"], capped=3, cap=1)


def test_metric_and_paired_statistics_have_hand_calculated_answers():
    measured = audit.classification_metrics([0, 1, 2, 2], [0, 1, 1, 2], capped=1, classes=3)
    assert measured["cc_f1"] == pytest.approx(2 / 3)
    assert measured["accuracy"] == pytest.approx(3 / 4)
    assert measured["macro_f1"] == pytest.approx(7 / 9)
    assert measured["weighted_f1"] == pytest.approx(3 / 4)
    interval = audit.paired_interval([0, 1, 2, 3])
    assert interval["mean"] == 1.5
    assert interval["sd"] == pytest.approx(math.sqrt(5 / 3))
    assert interval["lo"] == pytest.approx(-0.554260, abs=1e-6)
    assert interval["hi"] == pytest.approx(3.554260, abs=1e-6)
    assert audit.holm_adjust([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])


def toy_block(root):
    ids, labels = ["a", "b", "c", "d"], [3, 0, 1, 2]
    good = [[0.1, 0.1, 0.1, 0.6, 0.1], [0.6, 0.1, 0.1, 0.1, 0.1],
            [0.1, 0.6, 0.1, 0.1, 0.1], [0.1, 0.1, 0.6, 0.1, 0.1]]
    bad = [[0.1, 0.1, 0.1, 0.3, 0.4], [0.3, 0.1, 0.1, 0.4, 0.1],
           good[2], good[3]]
    for seed in (1, 2):
        directory = root / f"seed{seed}"
        (directory / "retrain1").mkdir(parents=True)
        for arm in ("pto", "tralo_final", "sham_final"):
            (directory / arm).mkdir()
            (directory / arm / "report.json").write_text("{}")
        rows = [dict(split="val", sample_id=sid, label=label) for sid, label in zip(ids, labels)]
        (directory / "manifest.json").write_text(json.dumps(dict(rows=rows)))
        (directory / "events.jsonl").write_text(json.dumps(dict(
            event="model_initialized", architecture="toy", model_class="Tiny")) + "\n")
        step = dict(applied=True, radius=0.01, hard_after=1)
        steps = {str(e): dict(tralo=dict(step), sham=dict(step)) for e in (1, 2)}
        (directory / "summary.json").write_text(json.dumps(dict(
            seed=seed, retrains=[dict(best_epoch=1, epochs_run=2, snapshot_steps=steps)])))
        for epoch in (1, 2):
            for suffix, values in (("", good if seed == 1 else bad), ("_tralo", good),
                                   ("_sham", good if seed == 1 else bad)):
                torch.save(torch.tensor(values), directory / "retrain1" / f"epoch{epoch:02d}{suffix}.pt")
    return dict(seeds=range(1, 3), model_class="Tiny", cap=1, images=4, positives=1)


def test_complete_toy_block_audits_saved_probabilities_without_primary_scorer(tmp_path):
    root = tmp_path / "runs"
    study = toy_block(root)
    result = audit.audit_block(root, "toy", study)
    assert result["seed_count"] == 2
    assert result["contrasts"]["E1_tralo_minus_sham"]["cc_f1"]["mean"] > 0
    assert result["contrasts"]["E2_tralo_minus_pto"]["cc_f1"]["mean"] > 0
    assert result["per_seed"][0]["arms"]["tralo"]["cc_f1"] == 1
    assert result["per_seed"][1]["arms"]["pto"]["cc_f1"] == 0
    (root / "seed2" / "summary.json").unlink()
    with pytest.raises(RuntimeError, match="no completed summary"):
        audit.audit_block(root, "toy", study)
