"""Actual Hounie gradients: normalization cancels scale but can retain direction."""

import importlib

import numpy as np
import pytest
import torch

from configs.gen_campaign import load_protocol
from scripts.smoke_arms import make_inputs


@pytest.mark.parametrize("changing", [False, True])
@pytest.mark.parametrize("vary", ["alpha", "eta"])
def test_hounie_under_normalization_matches_multiscope_gradient(
    tmp_path, monkeypatch, changing, vary
):
    trainer = importlib.import_module("src.methodologies.hounie_rcl.train")
    actual_finish = trainer.finish_constraint_step
    directions = []
    first_directions = []
    settings = (
        ((0.5, 0.1), (4.0, 0.1)) if vary == "alpha" else ((1.0, 0.05), (1.0, 0.2))
    )
    for alpha, eta in settings:
        inputs, _, _ = make_inputs(
            load_protocol(), "hounie", tmp_path / f"{alpha}_{eta}"
        )
        inputs.model = torch.nn.Linear(2, 2, bias=False)
        inputs.X_train = inputs.X_test = torch.eye(2)
        inputs.y_train = torch.tensor([0, 1])
        inputs.group_ids = np.array([0, 1])
        inputs.global_con = [1e10, 1e10]
        inputs.local_con = {0: [0, 1e10], 1: [0, 1e10]}
        inputs.constrained_classes = [0]
        inputs.num_classes = 2
        inputs.hyperparams.update(
            constraint_epochs=3,
            hounie_alpha=alpha,
            hounie_eta_lambda=eta,
            lr_constraint=0.0,
            constraint_fp32=True,
            constraint_grad_mode="normalize",
            batch_size=2,
        )
        states = iter(
            ([0.8, 0.6], [0.7, 0.9], [0.9, 0.7]) if changing else ([[0.8, 0.6]] * 3)
        )
        gradients = []

        def ce_epoch(model, *args):
            p = torch.tensor(next(states))
            with torch.no_grad():
                model.weight.copy_(torch.stack((p.log(), (1 - p).log())))
            return [0.0], 0.0

        def finish(model, optimizer, scaler, **kwargs):
            gradients.append(model.weight.grad.detach().clone())
            return actual_finish(model, optimizer, scaler, **kwargs)

        monkeypatch.setattr(trainer, "ce_epoch", ce_epoch)
        monkeypatch.setattr(trainer, "finish_constraint_step", finish)
        trainer.train(inputs)
        raw = gradients[-1]
        directions.append(raw / raw.norm())
        first_directions.append(gradients[0] / gradients[0].norm())
        if changing:
            # Closed form after three dual updates, eta_u=.1, zero initial
            # lambda/u: eta*((1-.3eta+.01eta²+.02eta*alpha)r1
            #              +(1-.1eta)r2+r3).
            expected_lambdas = eta * (
                (1 - 0.3 * eta + 0.01 * eta**2 + 0.02 * eta * alpha)
                * torch.tensor([0.8, 0.6])
                + (1 - 0.1 * eta) * torch.tensor([0.7, 0.9])
                + torch.tensor([0.9, 0.7])
            )
            # Each one-item local scope differentiates p0 with p0*(1-p0).
            slope = expected_lambdas * torch.tensor([0.09, 0.21])
            expected = torch.stack((slope, -slope))
            torch.testing.assert_close(raw, expected, rtol=2e-6, atol=1e-8)
    # First step: alpha has not fed back yet; eta is only a scalar factor.
    torch.testing.assert_close(
        first_directions[0], first_directions[1], rtol=1e-6, atol=1e-7
    )
    if changing:
        assert torch.max(torch.abs(directions[0] - directions[1])) > 1e-5
    else:
        # With fixed residuals, every scope follows one scalar recurrence.
        torch.testing.assert_close(directions[0], directions[1], rtol=1e-6, atol=1e-7)
