"""Observe the actual optimizer call, including AMP skips after gradient masks."""

import pytest
import torch
import json

from src.training.constraint_step import finish_constraint_step


@pytest.mark.parametrize("overflow", [False, True])
def test_reported_application_matches_gradscaler_optimizer_call(overflow):
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    scaler = torch.amp.GradScaler("cpu", init_scale=8)
    scaler.scale(model(torch.ones(1, 2)).sum()).backward()
    # unscale_ detects the overflow BEFORE head_ids masks this gradient away.
    if overflow:
        model.weight.grad.fill_(float("inf"))
    before = model.bias.detach().clone()
    diagnostics = {}
    _, applied = finish_constraint_step(
        model, optimizer, scaler, clip=1.0, mode="normalize", diagnostics=diagnostics
    )
    moved = not torch.equal(before, model.bias.detach())
    assert applied == moved == (not overflow)
    assert diagnostics["nonfinite_gradient"] == overflow
    assert diagnostics["amp_overflow_detected"] == overflow
    assert diagnostics["optimizer_step_applied"] == applied


@pytest.mark.parametrize("learning_rate", [0.0, 0.2])
def test_step_diagnostics_distinguish_gradient_from_parameter_displacement(
    learning_rate,
):
    model = torch.nn.Linear(2, 1, bias=False)
    model.weight.grad = torch.tensor([[0.3, 0.4]])
    diagnostics = {}
    raw, applied = finish_constraint_step(
        model,
        torch.optim.SGD(model.parameters(), lr=learning_rate),
        None,
        clip=1.0,
        mode="normalize",
        fp32=True,
        diagnostics=diagnostics,
    )
    assert applied
    assert raw == pytest.approx(0.5)
    assert diagnostics["pre_clip_grad_norm"] == pytest.approx(raw)
    assert diagnostics["transformed_grad_norm"] == pytest.approx(1.0)
    assert diagnostics["parameter_delta_norm"] == pytest.approx(learning_rate, abs=1e-7)
    expected = None if learning_rate == 0 else pytest.approx(1.0)
    assert diagnostics["descent_alignment"] == expected
    assert diagnostics["optimizer_step_applied"] is True


def test_diagnostic_collection_is_observational_for_adam():
    def run(collect):
        torch.manual_seed(14)
        model = torch.nn.Linear(2, 1)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        model(torch.ones(3, 2)).sum().backward()
        optimizer.step()
        optimizer.zero_grad()
        model(-torch.ones(3, 2)).sum().backward()
        diagnostics = {} if collect else None
        finish_constraint_step(
            model,
            optimizer,
            None,
            clip=1.0,
            mode="normalize",
            fp32=True,
            diagnostics=diagnostics,
        )
        return [p.detach().clone() for p in model.parameters()], torch.get_rng_state()

    plain, rng_plain = run(False)
    audited, rng_audited = run(True)
    assert all(torch.equal(a, b) for a, b in zip(plain, audited))
    assert torch.equal(rng_plain, rng_audited)


@pytest.mark.parametrize("satisfied", [False, True])
def test_tralo_events_record_state_timing_and_do_not_drop_the_last_epoch(
    tmp_path, satisfied
):
    from configs.gen_campaign import load_protocol
    from scripts.smoke_arms import make_inputs
    from src.methodologies.tralo.train import train

    inputs, _, _ = make_inputs(load_protocol(), "tralo", tmp_path)
    if satisfied:
        inputs.global_con = [10000] * inputs.num_classes
        inputs.local_con = {g: [10000] * inputs.num_classes for g in inputs.local_con}
    output = train(inputs)
    events = [
        json.loads(line)
        for line in (inputs.experiment_path / "constraint_events.jsonl")
        .read_text()
        .splitlines()
    ]
    assert len(events) == inputs.hyperparams["constraint_epochs"]
    assert len({e["attempt_id"] for e in events}) == 1
    assert (
        sum(e["step"]["optimizer_step_applied"] for e in events)
        == output.summary["constraint_steps_applied"]
    )
    for event in events:
        assert event["schema_version"] == 1
        assert event["counts_state"] == "post_task_pre_constraint"
        assert (
            event["epoch_absolute_1based"]
            == inputs.hyperparams["warmup_epochs"] + event["constraint_epoch_1based"]
        )
        assert event["scopes"]
        assert event["task_updates_attempted"] == event["task_updates_applied"]
        for scope in event["scopes"]:
            assert scope["soft_residual"] == pytest.approx(
                scope["soft_count"] - scope["budget"]
            )
            assert scope["hard_residual"] == scope["hard_count"] - scope["budget"]
            assert scope["multiplier_after"] >= scope["multiplier_before"]
        assert "parameter_delta_norm" in event["step"]


def test_event_writer_marks_nonfinite_values_without_mutating_input(tmp_path):
    from src.training.logging import append_constraint_event

    event = {
        "loss": float("inf"),
        "scopes": [{"a/b~c": float("nan")}],
        "negative": float("-inf"),
        "unmeasured": None,
        "finite": 0.5,
    }
    append_constraint_event(tmp_path, event)
    record = json.loads(
        (tmp_path / "constraint_events.jsonl").read_text(),
        parse_constant=lambda value: pytest.fail(value),
    )
    assert record["loss"] is None
    assert record["scopes"][0]["a/b~c"] is None
    assert record["negative"] is None
    assert record["finite"] == 0.5
    assert record["nonfinite_values"] == {
        "/loss": "inf",
        "/scopes/0/a~1b~0c": "nan",
        "/negative": "-inf",
    }
    assert event["loss"] == float("inf")
    assert "nonfinite_values" not in event


def test_nonfinite_task_batch_is_logged_without_aborting_amp_recovery(
    tmp_path, monkeypatch
):
    import importlib
    from configs.gen_campaign import load_protocol
    from scripts.smoke_arms import make_inputs

    trainer = importlib.import_module("src.methodologies.tralo.train")

    class OneOverflow(torch.nn.CrossEntropyLoss):
        def __init__(self):
            super().__init__()
            self.calls = 0

        def forward(self, logits, targets):
            self.calls += 1
            loss = super().forward(logits, targets)
            return loss * float("inf") if self.calls == 1 else loss

    monkeypatch.setattr(trainer, "make_ce_criterion", lambda *args: OneOverflow())
    monkeypatch.setattr(
        trainer,
        "setup_runtime",
        lambda device: (
            False,
            torch.float32,
            torch.amp.GradScaler("cpu", init_scale=8),
        ),
    )
    inputs, _, _ = make_inputs(load_protocol(), "tralo", tmp_path)
    inputs.hyperparams["constraint_fp32"] = True
    trainer.train(inputs)
    events = [
        json.loads(line)
        for line in (inputs.experiment_path / "constraint_events.jsonl")
        .read_text()
        .splitlines()
    ]
    assert len(events) == inputs.hyperparams["constraint_epochs"]
    assert events[0]["task_updates_skipped"] == 1
    assert events[0]["task_loss_online_mean"] is None
    assert events[0]["nonfinite_values"]["/task_loss_online_mean"] == "inf"
    assert events[1]["task_updates_skipped"] == 0
    assert events[1]["task_loss_online_mean"] is not None
