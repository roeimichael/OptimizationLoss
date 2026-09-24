"""Independent small-model checks for the two-pass global-count update."""
import copy
import unittest

try:
    import torch
except ImportError:
    torch = None

from tralo.global_constraint import bounded_count_penalty


@unittest.skipUnless(torch is not None, "Torch is unavailable")
class StreamedConstraintTests(unittest.TestCase):
    def setUp(self):
        from tralo.streamed_constraint import count_logit_gradient, streamed_step
        self.gradient = count_logit_gradient
        self.step = streamed_step
        torch.manual_seed(19)
        self.x = torch.randn(11, 3, dtype=torch.float64)
        self.model = torch.nn.Sequential(torch.nn.Linear(3, 6), torch.nn.BatchNorm1d(6),
                                         torch.nn.Tanh(), torch.nn.Linear(6, 3)).double()
        self.caps = [None, 2, None]
        self.multipliers = torch.tensor([0., 1.2, 0.], dtype=torch.float64)

    def test_logit_derivative_matches_autograd(self):
        z = self.model.eval()(self.x).detach().requires_grad_()
        expected = torch.autograd.grad(bounded_count_penalty(z, self.caps,
                                               self.multipliers, .4), z)[0]
        actual = self.gradient(z.softmax(1).detach(), self.caps, self.multipliers, .4)
        self.assertTrue(torch.allclose(actual, expected, atol=1e-12, rtol=1e-12))

    def test_streamed_parameter_gradient_and_step_match_unchunked(self):
        reference = copy.deepcopy(self.model).eval()
        reference_optim = torch.optim.Adam(reference.parameters(), lr=1e-3)
        reference_optim.zero_grad(set_to_none=True)
        bounded_count_penalty(reference(self.x), self.caps, self.multipliers, .4).backward()
        expected_grad = [p.grad.clone() for p in reference.parameters()]
        reference_optim.step()
        for size in (2, 4, 11):
            candidate = copy.deepcopy(self.model).train()
            buffers = [b.clone() for b in candidate.buffers()]
            optim = torch.optim.Adam(candidate.parameters(), lr=1e-3)
            chunks = [self.x[i:i+size] for i in range(0, len(self.x), size)]
            result = self.step(candidate, chunks, self.caps, self.multipliers, .4, optim)
            self.assertTrue(result['applied'])
            self.assertTrue(candidate.training)
            for actual, expected in zip(candidate.parameters(), reference.parameters()):
                self.assertTrue(torch.allclose(actual, expected, atol=1e-12, rtol=1e-12))
            for actual, expected in zip(candidate.parameters(), expected_grad):
                self.assertTrue(torch.allclose(actual.grad, expected, atol=1e-12, rtol=1e-12))
            for before, after in zip(buffers, candidate.buffers()):
                self.assertTrue(torch.equal(before, after))

    def test_inactive_penalty_does_not_move_or_initialize_optimizer(self):
        model = copy.deepcopy(self.model)
        before = [p.clone() for p in model.parameters()]
        optim = torch.optim.Adam(model.parameters(), lr=1e-3)
        result = self.step(model, [self.x[:4], self.x[4:]], [None, 99, None],
                           self.multipliers, .4, optim)
        self.assertFalse(result['applied'])
        self.assertEqual(len(optim.state), 0)
        for actual, expected in zip(model.parameters(), before):
            self.assertTrue(torch.equal(actual, expected))


if __name__ == '__main__':
    unittest.main()
