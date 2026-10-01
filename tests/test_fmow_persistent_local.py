"""Small independent checks for the proposed persistent fmow2 comparison."""

import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from tralo.fmow_persistent_local import (
    clone_warm_branch, correction, focal_loss, label_free_pool_identity,
    load_label_free_training_data,
    observations, sample_orders, stable_state_hash, train_epoch,
    validate_config, write_snapshot, write_pre_correction_snapshot, run,
)


class TinyData:
    labels = [0, 1, 0, 1]
    indices = [11, 12, 13, 14]

    def weights(self):
        return torch.ones(4, dtype=torch.double)

    def batch(self, positions, transform):
        images = torch.tensor([[float(self.indices[p]), 1.0] for p in positions])
        images += torch.rand_like(images) * 0.01
        return images, torch.tensor([self.labels[p] for p in positions])


def tiny_model():
    return torch.nn.Sequential(torch.nn.Linear(2, 4), torch.nn.ReLU(),
                               torch.nn.Linear(4, 2))


def config(seed=6700, pilot=True, reference=False):
    return dict(study="fmow_persistent_local_v1", seed=seed,
                pilot=pilot, reference=reference,
                pretrained_sha256="0" * 64,
                backbone="mobilenet_v3_large", epochs=7,
                batch_size=32, development_batch_size=16, lr=1e-4,
                weight_decay=0.0, decay_epoch=5, decay_factor=0.8,
                radius=0.1, rho=0.5, focal_alpha=0.25, focal_gamma=2.0)


def test_config_has_no_silent_scientific_defaults():
    validate_config(config())
    with pytest.raises(ValueError):
        validate_config({**config(), "weight_decay": 1e-4})
    with pytest.raises(ValueError):
        validate_config({**config(), "development_labels": True})
    with pytest.raises(ValueError):
        validate_config(config(seed=6700, pilot=False))
    with pytest.raises(ValueError):
        validate_config(config(seed=6701, pilot=False, reference=True))


def test_focal_is_multiclass_and_differs_from_ce_at_first_update():
    logits = torch.tensor([[2.0, -1.0], [0.5, 1.0]], requires_grad=True)
    labels = torch.tensor([0, 1])
    ce = torch.nn.functional.cross_entropy(logits, labels)
    focal = focal_loss(logits, labels, alpha=0.25, gamma=2.0)
    expected = -0.25 * ((1 - logits.softmax(1)[range(2), labels]) ** 2 *
                        logits.log_softmax(1)[range(2), labels]).mean()
    assert torch.allclose(focal, expected)
    ce_grad = torch.autograd.grad(ce, logits, retain_graph=True)[0]
    focal_grad = torch.autograd.grad(focal, logits)[0]
    assert not torch.allclose(ce_grad, focal_grad)


def test_optimizer_clone_follows_cloned_parameters_and_null_replays():
    torch.manual_seed(8)
    model = tiny_model()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=0)
    x = torch.tensor([[0.1, 0.2]])
    loss = model(x).sum()
    loss.backward()
    optimizer.step()
    a, oa = clone_warm_branch(model, optimizer)
    b, ob = clone_warm_branch(model, optimizer)
    assert stable_state_hash(oa.state_dict()) == stable_state_hash(ob.state_dict())
    assert set(oa.state) == set(a.parameters())
    assert set(ob.state) == set(b.parameters())
    assert set(oa.state).isdisjoint(set(model.parameters()))
    orders = sample_orders(TinyData(), 8, 2)
    ar = train_epoch(a, oa, TinyData(), lambda x: x, orders[1], 8, 2, 1,
                     "ce", batch_size=2, base_lr=1e-3)
    br = train_epoch(b, ob, TinyData(), lambda x: x, orders[1], 8, 2, 1,
                     "ce", batch_size=2, base_lr=1e-3)
    assert ar["first_batch_sha256"] == br["first_batch_sha256"]
    assert ar["sample_order_sha256"] == br["sample_order_sha256"]
    assert stable_state_hash(a.state_dict()) == stable_state_hash(b.state_dict())
    assert stable_state_hash(oa.state_dict()) == stable_state_hash(ob.state_dict())


def test_focal_starts_from_initial_state_not_ce_warmup():
    torch.manual_seed(19)
    initial = tiny_model()
    ce, focal = copy.deepcopy(initial), copy.deepcopy(initial)
    oc = torch.optim.Adam(ce.parameters(), lr=1e-3, weight_decay=0)
    of = torch.optim.Adam(focal.parameters(), lr=1e-3, weight_decay=0)
    orders = sample_orders(TinyData(), 19, 1)
    assert stable_state_hash(ce.state_dict()) == stable_state_hash(focal.state_dict())
    train_epoch(ce, oc, TinyData(), lambda x: x, orders[0], 19, 1, 0,
                "ce", batch_size=2, base_lr=1e-3)
    train_epoch(focal, of, TinyData(), lambda x: x, orders[0], 19, 1, 0,
                "focal", batch_size=2, base_lr=1e-3)
    assert stable_state_hash(ce.state_dict()) != stable_state_hash(focal.state_dict())


def test_snapshot_hashes_actual_post_state_and_probabilities(tmp_path):
    model = tiny_model()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=0)
    probs = torch.tensor([[0.8, 0.2], [0.3, 0.7]])
    record = write_snapshot(tmp_path, 5, model, opt, probs, None)
    assert Path(tmp_path / record["checkpoint_file"]).is_file()
    assert Path(tmp_path / record["probability_file"]).is_file()
    assert hashlib.sha256((tmp_path / record["checkpoint_file"]).read_bytes()).hexdigest() == record["checkpoint_sha256"]
    assert hashlib.sha256((tmp_path / record["probability_file"]).read_bytes()).hexdigest() == record["probability_sha256"]
    checkpoint = torch.load(tmp_path / record["checkpoint_file"], weights_only=True)
    assert checkpoint["epoch"] == 5
    assert torch.equal(torch.load(tmp_path / record["probability_file"], weights_only=True), probs)


def test_pilot_pre_correction_evidence_is_exclusive_and_detects_mutation(tmp_path):
    model = tiny_model()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=0)
    pool = [torch.tensor([[0.2, 0.8], [0.7, 0.3]])]
    with torch.no_grad():
        probabilities = model(pool[0]).softmax(1)
    dual = torch.tensor([0.0, 0.1, 0.2])
    evidence = write_pre_correction_snapshot(
        tmp_path, 2, "phr", 10, model, optimizer, dual, probabilities,
        pool, ["test1", "test2"])
    checkpoint_path = tmp_path / evidence["checkpoint_file"]
    prediction_path = tmp_path / evidence["probability_file"]
    checkpoint = torch.load(checkpoint_path, weights_only=True)
    prediction = torch.load(prediction_path, weights_only=True)
    assert (checkpoint["epoch"], checkpoint["arm"], checkpoint["cap_divisor"]) == (2, "phr", 10)
    assert torch.equal(checkpoint["dual"], dual)
    assert checkpoint["rng"]["torch_cpu"].numel() > 0
    assert checkpoint["rng"]["numpy"]["bit_generator"]
    assert checkpoint["rng"]["python"]
    assert prediction["sample_ids"] == ["test1", "test2"]
    assert torch.equal(prediction["probabilities"], probabilities)
    assert torch.equal(prediction["first_batch_probabilities"], probabilities)
    assert hashlib.sha256(checkpoint_path.read_bytes()).hexdigest() == evidence["checkpoint_sha256"]
    assert hashlib.sha256(prediction_path.read_bytes()).hexdigest() == evidence["probability_sha256"]
    with pytest.raises(FileExistsError):
        write_pre_correction_snapshot(tmp_path, 2, "phr", 10, model,
                                      optimizer, dual, probabilities, pool,
                                      ["test1", "test2"])
    with prediction_path.open("ab") as stream:
        stream.write(b"mutation")
    assert hashlib.sha256(prediction_path.read_bytes()).hexdigest() != evidence["probability_sha256"]


def test_dataset_entry_never_requests_development_labels(monkeypatch):
    seen = []

    def fake_load(root, *, include_pool_labels):
        seen.append(include_pool_labels)
        return {}, None, [{"sample_id": "test1", "location": "AAA"}], {}

    monkeypatch.setattr("tralo.fmow_persistent_local.load", fake_load)
    load_label_free_training_data("unused")
    assert seen == [False]

    def bad_load(root, *, include_pool_labels):
        return {}, None, [{"sample_id": "test1", "location": "AAA", "label": 1}], {}

    monkeypatch.setattr("tralo.fmow_persistent_local.load", bad_load)
    with pytest.raises(RuntimeError, match="development labels"):
        load_label_free_training_data("unused")


def test_label_free_pool_identity_requires_dev_role_and_exact_fields():
    pool = [{"split": "val", "sample_id": "test10", "location": "IRQ"}]
    assert label_free_pool_identity(pool) == [
        {"sample_id": "test10", "location": "IRQ"}]
    with pytest.raises(RuntimeError, match="non-development"):
        label_free_pool_identity([{**pool[0], "split": "test"}])
    with pytest.raises(RuntimeError, match="fields differ"):
        label_free_pool_identity([{**pool[0], "label": 1}])


def test_signed_and_positive_residual_and_hard_calls():
    probabilities = torch.tensor([[0.1, 0.9], [0.2, 0.8], [0.9, 0.1]])
    scopes = observations(probabilities, ["A", "A", "B"],
                          {"global_cap": 1, "local_caps": {"A": 1, "B": 1}})
    assert scopes["global"]["hard"] == 2
    assert scopes["global"]["hard_excess"] == 1
    assert scopes["A"]["signed_residual"] == pytest.approx(0.7)
    assert scopes["B"]["signed_residual"] == pytest.approx(-0.9)
    assert scopes["B"]["positive_residual"] == 0


def test_phr_dual_advances_even_when_parameter_step_skips(monkeypatch):
    model = torch.nn.Linear(2, 2)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    chunks = [torch.tensor([[1.0, 0.0], [0.0, 1.0]])]
    groups = ["A", "B"]
    quota = {"global_cap": 1, "local_caps": {"A": 1, "B": 1}}
    dual = torch.zeros(3)
    result_dual = torch.tensor([0.25, 0.0, 0.5])

    def skipped(*args, **kwargs):
        return (dict(applied=False, radius=0.0, displacement=0.0,
                     dual_before=dual.tolist(), dual_after=result_dual.tolist(),
                     residuals_after=[0.5, -0.5, 1.0],
                     boundary_policy={"applied": False, "reason": "conflict", "probes": []}),
                result_dual)

    monkeypatch.setattr("tralo.fmow_persistent_local.snapshot_phr_step", skipped)
    record, next_dual = correction(model, optimizer, chunks, groups, quota,
                                   "phr", dual, config())
    assert not record["controller"]["applied"]
    assert record["skipped_constraint_updates"] == 1
    assert torch.equal(next_dual, result_dual)
    assert record["dual_after"] == result_dual.tolist()
    assert record["model_before_sha256"] == record["model_after_sha256"]


def test_real_phr_correction_applies_and_preserves_optimizer_buffers_rng():
    torch.manual_seed(41)
    model = torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.BatchNorm1d(2),
                                torch.nn.Linear(2, 2))
    model.train()
    with torch.no_grad():
        model[2].bias[:] = torch.tensor([0.0, 1.0])
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    images = [torch.tensor([[1.0, 0.2], [0.3, -0.1]]),
              torch.tensor([[-0.6, 0.4], [0.5, 0.8]])]
    record, dual = correction(model, optimizer, images, ["A", "A", "B", "B"],
                              {"global_cap": 1, "local_caps": {"A": 0, "B": 1}},
                              "phr", torch.zeros(3), config())
    assert record["applied_constraint_updates"] == 1
    assert record["model_before_sha256"] != record["model_after_sha256"]
    assert 0 < record["actual_displacement"] <= 0.1 + 1e-5
    assert record["optimizer_before_sha256"] == record["optimizer_after_sha256"]
    assert record["rng_neutral"]
    assert any(float(x) > 0 for x in dual)
    assert record["controller"]["boundary_policy"]["probes"][-1]["accepted"]


def test_real_tralo_correction_applies_with_audited_dose():
    torch.manual_seed(3)
    model = torch.nn.Linear(4, 3)
    with torch.no_grad():
        model.weight.zero_()
        model.bias[:] = torch.tensor([0.0, 2.0, 0.0])
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    images = [torch.randn(5, 4), torch.randn(5, 4)]
    record, dual = correction(model, optimizer, images, ["A"] * 5 + ["B"] * 5,
                              {"global_cap": 4, "local_caps": {"A": 2, "B": 3}},
                              "tralo", None, config())
    assert dual is None and record["applied_constraint_updates"] == 1
    assert record["model_before_sha256"] != record["model_after_sha256"]
    assert record["actual_displacement"] == pytest.approx(record["applied_radius"], abs=1e-5)
    assert record["applied_radius"] <= 0.1
    assert record["before"]["global"]["hard_excess"] > 0
    assert record["controller"]["boundary_policy"]["probes"][-1]["accepted"]


def test_seven_epoch_cpu_fixture_keeps_focal_and_persistent_branches_separate(
        monkeypatch, tmp_path):
    train_transform, eval_transform = object(), object()

    class FakeArray:
        def __init__(self, array, indices, labels):
            self.indices = list(indices)
            self.labels = [int(labels[index]) for index in indices]

        def weights(self):
            return torch.ones(len(self.labels), dtype=torch.double)

        def batch(self, positions, transform):
            images = torch.tensor([[float(self.indices[p]), 1.0] for p in positions])
            if transform is train_transform:
                images += torch.rand_like(images) * 0.01
            return images, torch.tensor([self.labels[p] for p in positions])

    def fake_load(root, *, include_pool_labels):
        assert include_pool_labels is False
        roles = dict(train=[0, 1, 2, 3], stop=[4, 5], dev=[0, 1, 2],
                     stop_countries=["STOP"], dev_countries=["A", "B"],
                     reserved_countries=["SEALED"])
        rows = [{"sample_id": f"test{i}", "location": group}
                for i, group in enumerate(["A", "A", "B"])]
        return {"train": None, "test": None}, np.array([0, 1, 0, 1, 0, 1]), rows, roles

    def fake_tralo(model, chunks, groups, capped, global_cap, local_caps, **kwargs):
        with torch.no_grad():
            parameter = next(model.parameters()).flatten()[0]
            previous = parameter.clone()
            parameter.add_(0.01)
            radius = float((parameter - previous).abs())
        return dict(applied=True, radius=radius, displacement=radius,
                    gradient_norm=1.0,
                    boundary_policy=dict(applied=True, radius=radius,
                                         reason="synthetic_applied", probes=[]))

    def fake_phr(model, chunks, groups, capped, global_cap, local_caps,
                 dual, **kwargs):
        with torch.no_grad():
            parameter = next(model.parameters()).flatten()[0]
            previous = parameter.clone()
            parameter.sub_(0.01)
            radius = float((parameter - previous).abs())
        return (dict(applied=True, radius=radius, displacement=radius,
                     gradient_norm=1.0, dual_before=dual.tolist(),
                     dual_after=(dual + 0.01).tolist(),
                     boundary_policy=dict(applied=True, radius=radius,
                                          reason="synthetic_applied", probes=[])),
                dual + 0.01)

    monkeypatch.setattr("tralo.fmow_persistent_local.cuda_setup", lambda: None)
    monkeypatch.setattr("tralo.fmow_persistent_local.pretrained_weight_provenance",
                        lambda sha: dict(file="synthetic", sha256=sha))
    monkeypatch.setattr("tralo.fmow_persistent_local.load", fake_load)
    monkeypatch.setattr("tralo.fmow_persistent_local.budgets", lambda groups: {
        str(div): dict(global_cap=1, local_caps={"A": 1, "B": 1}) for div in (10, 20)})
    monkeypatch.setattr("tralo.fmow_persistent_local.ArrayImages", FakeArray)
    monkeypatch.setattr("tralo.fmow_persistent_local.pool_chunks",
                        lambda *args: [torch.tensor([[1.0, 0.0], [2.0, 1.0],
                                                     [3.0, 1.0]])])
    monkeypatch.setattr("tralo.fmow_persistent_local.transforms_for",
                        lambda: (train_transform, eval_transform))
    monkeypatch.setattr("tralo.fmow_persistent_local.make_model",
                        lambda **kwargs: tiny_model())
    monkeypatch.setattr("tralo.fmow_persistent_local.local_targeted_step", fake_tralo)
    monkeypatch.setattr("tralo.fmow_persistent_local.snapshot_phr_step", fake_phr)
    monkeypatch.setattr(torch.nn.Module, "cuda", lambda self: self)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda: "CPU synthetic")
    cfg_path = tmp_path / "input.json"
    cfg_path.write_text(json.dumps(config()) + "\n")
    root = tmp_path / "run"
    run(tmp_path, cfg_path, root)
    summary = json.loads((root / "summary.json").read_text())
    assert set(summary["arms"]) == {"ce_null", "focal_clip", "cap10_tralo",
                                    "cap10_phr", "cap20_tralo", "cap20_phr",
                                    "tralo_null", "alm_null"}
    assert all(len(arm["epochs"]) == 7 for arm in summary["arms"].values())
    assert len(summary["arms"]["cap10_phr"]["corrections"]) == 6
    assert summary["arms"]["cap10_phr"]["corrections"][0]["dual_after"] == pytest.approx([0.01] * 3)
    assert summary["arms"]["focal_clip"]["epochs"][0]["loss_kind"] == "focal"
    assert summary["arms"]["ce_null"]["epochs"][0]["loss_kind"] == "ce"
    assert summary["arms"]["focal_clip"]["epochs"][0]["model_sha256"] != summary["ce_warmup"]["model_sha256"]
    for arm_name in ("cap10_tralo", "cap10_phr", "cap20_tralo", "cap20_phr"):
        arm = summary["arms"][arm_name]
        assert arm["corrections"][0]["applied_constraint_updates"] == 1
        assert arm["epochs"][2]["pre_epoch_model_sha256"] == arm["epochs"][1]["post_epoch_model_sha256"]
        assert arm["epochs"][2]["pre_epoch_model_sha256"] != summary["arms"]["ce_null"]["epochs"][2]["pre_epoch_model_sha256"]
        for correction_record in arm["corrections"]:
            epoch = correction_record["epoch"]
            evidence = correction_record["pre_correction_snapshot"]
            assert evidence["model_state_sha256"] == correction_record["model_before_sha256"]
            assert evidence["optimizer_state_sha256"] == correction_record["optimizer_before_sha256"]
            assert evidence["post_model_state_sha256"] == correction_record["model_after_sha256"]
            assert evidence["post_optimizer_state_sha256"] == correction_record["optimizer_after_sha256"]
            checkpoint_path = root / arm_name / evidence["checkpoint_file"]
            prediction_path = root / arm_name / evidence["probability_file"]
            assert hashlib.sha256(checkpoint_path.read_bytes()).hexdigest() == evidence["checkpoint_sha256"]
            assert hashlib.sha256(prediction_path.read_bytes()).hexdigest() == evidence["probability_sha256"]
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
            prediction = torch.load(prediction_path, map_location="cpu", weights_only=True)
            assert checkpoint["epoch"] == epoch
            assert checkpoint["arm"] == arm_name.split("_")[1]
            assert checkpoint["cap_divisor"] == int(arm_name[3:5])
            assert prediction["sample_ids"] == ["test0", "test1", "test2"]
            replay = tiny_model()
            replay.load_state_dict(checkpoint["model"])
            replay.eval()
            with torch.no_grad():
                independent = replay(torch.tensor([[1.0, 0.0], [2.0, 1.0],
                                                   [3.0, 1.0]])).softmax(1)
            assert torch.equal(independent, prediction["probabilities"])
            if checkpoint["arm"] == "phr":
                assert checkpoint["dual"].tolist() == pytest.approx(
                    [0.01 * (epoch - 2)] * 3)
        for epoch in (5, 6, 7):
            checkpoint = torch.load(root / arm_name / f"epoch{epoch:02d}_post.pt",
                                    map_location="cpu", weights_only=True)
            replay = tiny_model()
            replay.load_state_dict(checkpoint["model"])
            replay.eval()
            pool = torch.tensor([[1.0, 0.0], [2.0, 1.0], [3.0, 1.0]])
            with torch.no_grad():
                independent = replay(pool).softmax(1)
            recorded = torch.load(root / arm_name / f"epoch{epoch:02d}_probabilities.pt",
                                  weights_only=True)
            assert torch.equal(independent, recorded)
    assert (root / "config.json").read_bytes() == cfg_path.read_bytes()
    ref_config = tmp_path / "reference.json"
    ref_config.write_text(json.dumps(config(reference=True)) + "\n")
    ref_root = tmp_path / "reference"
    run(tmp_path, ref_config, ref_root)
    reference = json.loads((ref_root / "summary.json").read_text())
    assert set(reference["arms"]) == {"ce_null"}
    assert reference["ce_warmup"]["model_sha256"] == summary["ce_warmup"]["model_sha256"]
    assert reference["arms"]["ce_null"]["final_model_sha256"] == summary["arms"]["ce_null"]["final_model_sha256"]
    for epoch in (5, 6, 7):
        filename = f"epoch{epoch:02d}_probabilities.pt"
        assert torch.equal(torch.load(root / "ce_null" / filename, weights_only=True),
                           torch.load(ref_root / "ce_null" / filename, weights_only=True))
    with pytest.raises(FileExistsError):
        run(tmp_path, cfg_path, root)
