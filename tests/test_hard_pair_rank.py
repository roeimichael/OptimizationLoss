"""Independent autograd examples for hard-pair quota ranking."""
import unittest

import torch

from tralo.hard_pair_rank import hard_pair_rank_gradient, hard_pair_rank_loss


class HardPairRankTests(unittest.TestCase):
    def test_nonzero_when_training_top_k_is_pure(self):
        # Both strongest predictions belong to true grade 3. The prior
        # wrong-occupant term is exactly inactive here; hard pairs are active.
        logits = torch.tensor([
            [0., 0., 0., 5., 0.], [0., 0., 0., 4., 0.],
            [0., 0., 0., 2., 0.], [3., 0., 0., 0., 0.],
            [2., 0., 0., 0., 0.]], dtype=torch.float64, requires_grad=True)
        labels = [3, 3, 3, 0, 0]
        ids = ['a', 'b', 'c', 'd', 'e']
        loss, details = hard_pair_rank_loss(logits, labels, ids, 2)
        expected = torch.autograd.grad(loss, logits)[0]
        actual, selected = hard_pair_rank_gradient(logits.detach(), labels, ids, 2)
        self.assertEqual(details['active_pairs'], 4)
        self.assertEqual(details, selected)
        self.assertEqual(details['weak_positive_ids'], ['c', 'b'])
        self.assertEqual(details['hard_negative_ids'], ['e', 'd'])
        self.assertGreater(float(loss), 0.)
        self.assertGreater(float(actual.norm()), 0.)
        self.assertTrue(torch.allclose(actual, expected, atol=1e-12, rtol=1e-12))

    def test_empty_class_is_zero_and_ties_are_id_stable(self):
        logits = torch.zeros((4, 5), dtype=torch.float64, requires_grad=True)
        ids = ['z', 'a', 'm', 'b']
        labels = [3, 3, 0, 0]
        _, details = hard_pair_rank_loss(logits, labels, ids, 1)
        self.assertEqual(details['weak_positive_ids'], ['a'])
        self.assertEqual(details['hard_negative_ids'], ['b'])
        gradient, details = hard_pair_rank_gradient(logits.detach(), [0] * 4, ids, 1)
        self.assertEqual(details['active_pairs'], 0)
        self.assertTrue(torch.equal(gradient, torch.zeros_like(gradient)))


if __name__ == '__main__':
    unittest.main()
