import unittest

try:
    import torch
except ImportError:  # pragma: no cover - exercised on CPU-only minimal installs
    torch = None

from tralo.global_constraint import advance_controller, bounded_count_penalty


class ControllerTests(unittest.TestCase):
    def test_ratchets_violating_multipliers_and_rho_until_freeze(self):
        m, rho, frozen = advance_controller([3, 1, 4], [2, None, 4], [0.0, 7.0, 1.0],
                                             0.5, 0.25, 0.1, False)
        self.assertEqual(m, [0.1, 7.0, 1.0])
        self.assertEqual(rho, 0.75)
        self.assertFalse(frozen)

    def test_frozen_controller_is_constant(self):
        self.assertEqual(advance_controller([99], [0], [2.0], 3.0, 4.0, 5.0, True),
                         ([2.0], 3.0, True))

    def test_feasible_controller_freezes_without_ratchet(self):
        self.assertEqual(advance_controller([5, 1], [None, 1], [2.0, 0.0],
                                                   0.0, 1.0, 0.5, False),
                         ([2.0, 0.0], 0.0, True))

    def test_unconstrained_classes_do_not_ratchet(self):
        self.assertEqual(advance_controller([5, 1], [None, 1], [2.0, 0.0],
                                                   0.0, 1.0, 0.5, False),
                         ([2.0, 0.0], 0.0, True))


@unittest.skipUnless(torch is not None, "Torch is unavailable")
class PenaltyTests(unittest.TestCase):
    def test_penalty_hand_value_and_uncapped_zero_gradient(self):
        logits = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], requires_grad=True)
        loss = bounded_count_penalty(logits, [1, None, 1], torch.tensor([2.0, 3.0, 0.0]), 0.5)
        # soft count is 2/3 per class; both capped classes are inactive.
        self.assertEqual(float(loss), 0.0)
        self.assertTrue(torch.equal(torch.autograd.grad(loss, logits)[0], torch.zeros_like(logits)))

    def test_two_class_analytic_gradient(self):
        logits = torch.tensor([[0.0, 0.0]], dtype=torch.float64, requires_grad=True)
        loss = bounded_count_penalty(logits, [0, None], torch.tensor([1.0, 0.0], dtype=torch.float64), 0.0)
        self.assertAlmostEqual(float(loss), 1.0 / 3.0)
        self.assertTrue(torch.allclose(torch.autograd.grad(loss, logits)[0],
                                       torch.tensor([[1.0 / 9.0, -1.0 / 9.0]], dtype=torch.float64)))

    def test_rejects_empty_population(self):
        with self.assertRaises(ValueError):
            bounded_count_penalty(torch.zeros((0, 2)), [0, None], torch.tensor([1.0, 0.0]), 0.0)

    def test_positive_excess_is_monotone_in_multiplier_and_rho(self):
        logits = torch.tensor([[5.0, 0.0], [5.0, 0.0]], dtype=torch.float64)
        a = bounded_count_penalty(logits, [1, None], torch.tensor([1.0, 0.0], dtype=torch.float64), 0.0)
        b = bounded_count_penalty(logits, [1, None], torch.tensor([2.0, 0.0], dtype=torch.float64), 1.0)
        self.assertGreater(float(b), float(a))

    def test_uncapped_class_competes_through_softmax_normalization(self):
        logits = torch.tensor([[2.0, 0.0]], dtype=torch.float64, requires_grad=True)
        loss = bounded_count_penalty(logits, [0, None], torch.tensor([1.0, 0.0], dtype=torch.float64), 0.0)
        grad = torch.autograd.grad(loss, logits)[0]
        self.assertLess(float(grad[0, 1]), 0.0)

    def test_gradcheck_float64(self):
        logits = torch.tensor([[1.2, -0.4, 0.3], [-0.5, 0.8, 0.1]], dtype=torch.float64,
                              requires_grad=True)
        multipliers = torch.tensor([0.7, 0.0, 0.4], dtype=torch.float64)
        self.assertTrue(torch.autograd.gradcheck(
            lambda x: bounded_count_penalty(x, [1, None, 0], multipliers, 0.3),
            (logits,), eps=1e-6, atol=1e-5, rtol=1e-4))


if __name__ == "__main__":
    unittest.main()
