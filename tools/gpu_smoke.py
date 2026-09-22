"""Analytic CUDA/logging smoke; no dataset, training campaign or Clipper claim.

Run from the project root: python tools/gpu_smoke.py OUTPUT_DIRECTORY
Requires the server's existing PyTorch environment. Device selection is explicit
through CUDA_VISIBLE_DEVICES; this script refuses to fall back to CPU.
"""

import argparse
import hashlib
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tralo.events import EventLog


def run(output):
    import torch
    import torch.nn.functional as functional

    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required; no CPU fallback')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    with EventLog(output / 'events.jsonl') as log:
        log.emit('started', scope='infrastructure_smoke_not_research',
                 torch=str(torch.__version__), device=str(torch.cuda.get_device_name(0)),
                 precision='float32', script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
        try:
            # At zero logits both classes have probability 1/2. Mean CE is ln(2).
            # dL/dz=(softmax(z)-one_hot(y))/2, hence the exact +/-1/4 matrix.
            logits = torch.zeros((2, 2), device='cuda', requires_grad=True)
            labels = torch.tensor([0, 1], device='cuda')
            loss = functional.cross_entropy(logits, labels)
            loss.backward()
            expected_gradient = torch.tensor([[-0.25, 0.25], [0.25, -0.25]], device='cuda')
            torch.testing.assert_close(loss, torch.tensor(math.log(2), device='cuda'), rtol=1e-6, atol=1e-7)
            torch.testing.assert_close(logits.grad, expected_gradient, rtol=0, atol=0)
            # These explicit values define a test fixture, not a proposed recipe.
            optimizer = torch.optim.Adam([logits], lr=0.01, betas=(0.9, 0.999),
                                         eps=1e-8, weight_decay=0, amsgrad=False,
                                         foreach=False, fused=False)
            optimizer.step()
            # First-step bias correction gives m_hat=g and v_hat=g^2.
            expected = -0.01 * expected_gradient / (expected_gradient.abs() + 1e-8)
            torch.testing.assert_close(logits, expected, rtol=1e-6, atol=1e-7)
            cpu_rng, gpu_rng = torch.get_rng_state().clone(), torch.cuda.get_rng_state().clone()
            before, grad_before = logits.detach().clone(), logits.grad.clone()
            state_before = {k: v.clone() if isinstance(v, torch.Tensor) else v
                            for k, v in optimizer.state[logits].items()}
            log.emit('analytic_checks_passed', loss=float(loss.detach()), gradient='exact +/-0.25',
                     adam_first_step='matches -lr*g/(abs(g)+eps)')
            if not torch.equal(cpu_rng, torch.get_rng_state()) or not torch.equal(gpu_rng, torch.cuda.get_rng_state()):
                raise AssertionError('logging changed Torch RNG state')
            if not torch.equal(before, logits) or not torch.equal(grad_before, logits.grad):
                raise AssertionError('logging changed supplied parameters/gradients')
            for key, value in state_before.items():
                current = optimizer.state[logits][key]
                if isinstance(value, torch.Tensor):
                    if not torch.equal(value, current): raise AssertionError('logging changed optimizer state')
                elif value != current:
                    raise AssertionError('logging changed optimizer state')
            torch.cuda.synchronize()
            log.emit('completed', checks=['cross_entropy', 'gradient', 'adam_first_step', 'logger_state_neutrality'])
        except Exception as exc:
            log.emit('failed', error_type=type(exc).__name__, message=str(exc))
            raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output')
    run(parser.parse_args().output)
