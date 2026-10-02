"""Matched image batches and the development-target boundary are executable."""

import copy

import pytest
from PIL import Image
import torch
from torchvision import transforms

from tralo.tabular_image_data import PreparedImageRows
from tralo.tabular_persistent_train import (_orders, _pool_batches,
                                             _run_arm, _train_epoch,
                                             CALIBRATED_STEP_SIZES,
                                             correction_step_size_for_arm,
                                             validate_config)
from tralo.events import EventLog


def test_frozen_config_rejects_unregistered_fields():
    config = {"study": "tabular_persistent_v1", "dataset": "isic2020",
              "backbone": "mobilenet_v3_large", "seed": 6800, "pilot": True}
    validate_config(config)
    with pytest.raises(ValueError, match="unfrozen"):
        validate_config({**config, "learning_rate": 1.0})


def test_calibrated_config_freezes_dataset_arm_dose_and_fresh_seeds():
    config = {"study": "tabular_persistent_dose_calibrated_v2",
              "dataset": "celeba", "backbone": "mobilenet_v3_large",
              "seed": 6880, "pilot": True,
              "correction_step_sizes": CALIBRATED_STEP_SIZES["celeba"].copy()}
    validate_config(config)
    assert correction_step_size_for_arm(config, "level1_tralo") == 0.0045
    assert correction_step_size_for_arm(config, "level1_phr") == 0.0027
    for invalid in ({**config, "seed": 6800},
                    {**config, "correction_step_sizes": {
                        **config["correction_step_sizes"], "level1_tralo": 0.01}},
                    {**config, "pilot": False}):
        with pytest.raises(ValueError, match="unfrozen"):
            validate_config(invalid)


def test_two_arms_use_exact_same_stochastic_images_and_task_updates(tmp_path):
    rows = []
    for i, label in enumerate((0, 1, 0, 1)):
        filename = f"{i}.png"
        Image.new("RGB", (4, 4), (40 + 30 * i, 70, 100)).save(tmp_path / filename)
        rows.append({"sample_id": str(i), "file": filename, "group": "female",
                     "label": label})
    data = PreparedImageRows(tmp_path, rows, transforms.Compose([
        transforms.RandomHorizontalFlip(), transforms.ToTensor()]))
    order = _orders(rows, 6800, "celeba")[0]
    torch.manual_seed(3)
    first = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(48, 2))
    second = copy.deepcopy(first)
    a = _train_epoch(first, torch.optim.Adam(first.parameters(), lr=0.001),
                     data, order, 123, 2, torch.device("cpu"))
    b = _train_epoch(second, torch.optim.Adam(second.parameters(), lr=0.001),
                     data, order, 123, 2, torch.device("cpu"))
    assert a == b
    assert all(torch.equal(x, y) for x, y in zip(first.parameters(), second.parameters()))


def test_constraint_pool_refuses_any_development_target(tmp_path):
    Image.new("RGB", (4, 4)).save(tmp_path / "a.png")
    data = PreparedImageRows(tmp_path, [{"sample_id": "a", "file": "a.png",
                                        "group": "female", "label": 1}],
                             transforms.ToTensor())
    with pytest.raises(RuntimeError, match="development target"):
        list(_pool_batches(data, 1, torch.device("cpu"))())


@pytest.mark.parametrize("calibrated, expected_step", [(False, 0.01),
                                                    (True, 0.0045)])
def test_treated_arm_writes_replayable_selected_artifacts_without_dev_labels(
        tmp_path, monkeypatch, calibrated, expected_step):
    from tralo import tabular_persistent_train as runner
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    for i in range(8):
        Image.new("RGB", (4, 4), (90 + i, 50, 20)).save(image_dir / f"{i}.png")
    def row(i, label=True):
        value = {"sample_id": str(i), "file": f"{i}.png",
                 "group": "female" if i % 2 else "male"}
        if label:
            value["label"] = i % 2
        return value
    rows = {"train": [row(i) for i in range(4)],
            "stop": [row(i) for i in range(4, 6)],
            "development_pool": [row(i, False) for i in range(4, 8)]}
    train_tf = eval_tf = transforms.ToTensor()
    model = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(48, 2))
    with torch.no_grad():
        model[1].weight.zero_()
        model[1].bias.copy_(torch.tensor([0., 2.]))
    monkeypatch.setitem(runner.BACKBONES, "mobilenet_v3_large", (2, 0.001))
    actual_step = []
    original_apply = runner.apply_fixed_correction
    def record_step(model, step, maximum):
        actual_step.append(step)
        return original_apply(model, step, maximum)
    monkeypatch.setattr(runner, "apply_fixed_correction", record_step)
    quotas = {"level1": {"global_cap": 1,
                         "local_caps": {"female": 1, "male": 1}}}
    output = tmp_path / "run"
    output.mkdir()
    with EventLog(output / "events.jsonl") as log:
        config = {"seed": 6880 if calibrated else 6800,
                  "backbone": "mobilenet_v3_large"}
        if calibrated:
            config.update(study="tabular_persistent_dose_calibrated_v2",
                          correction_step_sizes=CALIBRATED_STEP_SIZES["celeba"])
        result = _run_arm("level1_tralo", model, rows, image_dir,
                          [[0, 1, 2, 3], [3, 2, 1, 0]], train_tf, eval_tf,
                          quotas, config,
                          output, log, torch.device("cpu"))
    assert actual_step == [expected_step]
    assert len(result["epochs"]) == 2
    assert result["selected_epoch"] in (1, 2)
    snapshot = torch.load(output / result["probabilities"], weights_only=True)
    assert snapshot["sample_ids"] == ["4", "5", "6", "7"]
    assert tuple(snapshot["probabilities"].shape) == (4, 2)
    assert (output / result["checkpoint"]).is_file()
