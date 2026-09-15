"""CPU research diagnostics, not a production training arm or quality experiment.

Run from the repository root:
  python docs/research/loss_mechanism_probe.py
Outputs are preserved under a unique receipts directory.
"""
import copy
import hashlib
import importlib
import json
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import torch
from src.losses.transductive_loss import MulticlassTransductiveLoss, uniform_grad_count
from src.training.constraint_step import finish_constraint_step
from src.utils.constants import EPSILON, UNLIMITED, clamp_probability


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def derivative_probe():
    criterion = MulticlassTransductiveLoss([10, UNLIMITED], {}, 2).double()
    criterion.rho.fill_(3.5)
    rows = []
    for budget, soft in [(0., 2.), (10., 9.), (10., 10.1), (10., 16.), (10., 90.)]:
        s = torch.tensor(soft, dtype=torch.float64, requires_grad=True)
        k = torch.tensor(budget, dtype=torch.float64)
        grad = torch.autograd.grad(criterion._penalty(s, k), s)[0].item()
        a = max(budget, 1.)
        e = max(soft - budget, 0.) / (a + EPSILON)
        exact = 0. if soft <= budget else (
            (a + EPSILON) / (soft - budget + a + EPSILON)**2
            + 3.5 * 2 * e * (1 + EPSILON)
            / ((a + EPSILON) * (1 + e*e + EPSILON)**2))
        h = 1e-5
        fd = ((criterion._penalty(s.detach()+h, k)
               - criterion._penalty(s.detach()-h, k)) / (2*h)).item()
        assert abs(grad - exact) < 1e-12
        assert abs(grad - fd) < 1e-8
        rows.append(dict(K=budget, S=soft, autograd=grad, analytic=exact, finite_difference=fd))
    return rows


def chunk_probe():
    # Heterogeneous chunks distinguish a local mean from a whole-set mean.
    logits = torch.tensor([[5., -2., -3.], [4., 0., -2.],
                           [.1, 0., -.1], [.2, -.2, 0.]], dtype=torch.float64)
    out = {}
    for mode in ('sum', 'uniform'):
        z = logits.clone().requires_grad_()
        p = z.softmax(1)
        fn = (lambda x: x) if mode == 'sum' else uniform_grad_count
        full = torch.autograd.grad(fn(p)[:, 0].sum(), z)[0]
        z = logits.clone().requires_grad_()
        chunked_value = sum(fn(z[i:i+2].softmax(1))[:, 0].sum() for i in (0, 2))
        chunked = torch.autograd.grad(chunked_value, z)[0]
        out[mode] = dict(max_gradient_difference=float((full-chunked).abs().max()),
                         full_norm=float(full.norm()), chunked_norm=float(chunked.norm()))
    assert out['sum']['max_gradient_difference'] < 1e-12
    assert out['uniform']['max_gradient_difference'] > .01
    z = logits.clone().requires_grad_()
    full = torch.autograd.grad(uniform_grad_count(z.softmax(1))[:, 0].sum(), z)[0]
    z = logits.clone().requires_grad_()
    p = clamp_probability(z.softmax(1).detach())
    weight = (p*(1-p)).mean(0, keepdim=True)
    value = sum(uniform_grad_count(z[i:i+2].softmax(1), weight=weight)[:, 0].sum()
                for i in (0, 2))
    corrected = torch.autograd.grad(value, z)[0]
    error = float((full-corrected).abs().max())
    assert error < 1e-12
    out['uniform_with_full_population_weight'] = dict(max_gradient_difference=error)
    return out


def optimizer_probe():
    model = torch.nn.Linear(2, 1, bias=False).double()
    with torch.no_grad():
        model.weight.zero_()
    opt = torch.optim.Adam(model.parameters(), lr=.01)
    # Deliberately conflicting old momentum; a counterexample, not a real-data estimate.
    for _ in range(20):
        model.weight.grad = torch.tensor([[-1., 0.]], dtype=torch.float64)
        opt.step()
    state = copy.deepcopy(opt.state_dict())
    before = model.weight.detach().clone()
    task_grad = torch.tensor([[1., 0.]], dtype=torch.float64)
    constraint_grad = torch.tensor([[0., 1.]], dtype=torch.float64)
    model.weight.grad = constraint_grad.clone()
    diagnostic = {}
    finish_constraint_step(model, opt, None, 1., mode='normalize', fp32=True,
                           ortho_ref=[task_grad], diagnostics=diagnostic)
    delta = model.weight.detach() - before
    with torch.no_grad():
        model.weight.copy_(before)
    opt.load_state_dict(state)
    model.weight.grad = torch.zeros_like(model.weight)
    opt.step()  # Counterfactual Adam step, NOT the skipped-step tralo_null.
    zero_delta = model.weight.detach() - before
    assert float((task_grad * constraint_grad).sum()) == 0.
    assert float((task_grad * delta).sum()) > 0.
    return dict(raw_gradient_dot=0., actual_task_directional_change=float((task_grad*delta).sum()),
                constraint_directional_change=float((constraint_grad*delta).sum()),
                actual_delta=delta.tolist(), zero_gradient_adam_delta=zero_delta.tolist(),
                incremental_constraint_delta=(delta-zero_delta).tolist(), diagnostics=diagnostic)


def event_probe(receipts):
    from configs.gen_campaign import load_protocol
    from scripts.smoke_arms import make_inputs
    from unittest.mock import patch
    trainer = importlib.import_module('src.methodologies.tralo.train')
    inputs, _, _ = make_inputs(load_protocol(), 'tralo', receipts)
    capture = []
    original = trainer._scope_events

    def independently_check_counts(criterion, gs, gh, ls, lh):
        with torch.no_grad():
            p = inputs.model(inputs.X_test).softmax(1)
            hard = torch.bincount(p.argmax(1), minlength=inputs.num_classes)
        # First callback per epoch is pre-step; second occurs after the step,
        # intentionally with the cached pre-step counts for the multiplier log.
        if len(capture) % 2 == 0:
            assert torch.allclose(gs, p.sum(0), atol=1e-5)
            assert torch.equal(gh.long(), hard)
        capture.append(dict(actual_soft=p.sum(0).tolist(), logged_soft=gs.tolist()))
        return original(criterion, gs, gh, ls, lh)

    with patch.object(trainer, '_scope_events', independently_check_counts):
        result = trainer.train(inputs)
    events = [json.loads(s) for s in (inputs.experiment_path/'constraint_events.jsonl').read_text().splitlines()]
    recon = []
    for event in events:
        crit = MulticlassTransductiveLoss(None, {}, inputs.num_classes)
        crit.rho.fill_(event['rho_before'])
        total = sum(row['multiplier_before'] * float(crit._penalty(
            torch.tensor(row['soft_count']), torch.tensor(float(row['budget']))))
            for row in event['scopes'])
        error = abs(total-event['constraint_objective_pre_step'])
        assert error < 1e-5
        recon.append(dict(epoch=event['constraint_epoch_1based'], reconstructed=total,
                          logged=event['constraint_objective_pre_step'], absolute_error=error))
    assert sum(e['step']['optimizer_step_applied'] for e in events) == result.summary['constraint_steps_applied']
    return dict(scope_forward_checks=capture, loss_reconstruction=recon,
                event_count=len(events), steps_applied=result.summary['constraint_steps_applied'],
                note='Synthetic fixture only; no real warmup or quality inference.')


if __name__ == '__main__':
    torch.set_num_threads(1)
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    receipts = ROOT/'docs/research/receipts'/stamp
    receipts.mkdir(parents=True)
    paths = [Path(__file__), ROOT/'src/losses/transductive_loss.py',
             ROOT/'src/methodologies/tralo/train.py', ROOT/'src/training/constraint_step.py',
             ROOT/'src/training/logging.py', ROOT/'scripts/smoke_arms.py']
    before = {str(p.relative_to(ROOT)): sha(p) for p in paths}
    report = dict(timestamp_utc=stamp, torch_version=torch.__version__, device='cpu',
                  source_hashes=before,
                  git_head=subprocess.check_output(['rtk','git','rev-parse','HEAD'], cwd=ROOT, text=True).strip(),
                  penalty_derivative=derivative_probe(), chunk_gradient=chunk_probe(),
                  optimizer_counterexample=optimizer_probe(), training_log_audit=event_probe(receipts))
    report['source_unchanged_during_probe'] = before == {str(p.relative_to(ROOT)): sha(p) for p in paths}
    assert report['source_unchanged_during_probe'], 'Concurrent source change: rerun this diagnostic.'
    (receipts/'probe.json').write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps({'receipt': str(receipts/'probe.json'), 'results': report}, indent=2))
