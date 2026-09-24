"""Independent fixtures for training-label-only quota-cutoff ranking."""
import unittest

try:
    import torch
except ImportError:
    torch = None


@unittest.skipUnless(torch is not None, 'Torch is unavailable')
class CutoffRankTests(unittest.TestCase):
    def test_wrong_occupant_and_missed_true_case(self):
        from tralo.cutoff_rank import cutoff_rank_gradient, cutoff_rank_loss
        z=torch.tensor([[0.,0.,0.,3.,0.],
                        [0.,0.,0.,0.,0.],
                        [0.,0.,0.,2.,0.],
                        [0.,0.,0.,1.,0.]],dtype=torch.float64,requires_grad=True)
        y=[3,3,0,0]; ids=['a','b','c','d']
        loss,details=cutoff_rank_loss(z,y,ids,2)
        actual,independent=cutoff_rank_gradient(z.detach(),y,ids,2)
        expected=torch.autograd.grad(loss,z)[0]
        self.assertTrue(torch.allclose(actual,expected,atol=1e-12,rtol=1e-12))
        self.assertEqual(details['missed_true_ids'],['b'])
        self.assertEqual(details['wrong_occupant_ids'],['c'])
        self.assertEqual(details['active_pairs'],1)
        self.assertEqual(details,independent)
        self.assertLess(float(actual[1,3]),0.)
        self.assertGreater(float(actual[2,3]),0.)
        self.assertTrue(torch.equal(actual[[0,3]],torch.zeros_like(actual[[0,3]])))

    def test_empty_error_sets_have_zero_gradient(self):
        from tralo.cutoff_rank import cutoff_rank_gradient, cutoff_rank_loss
        z=torch.tensor([[0.,0.,0.,3.,0.],[0.,0.,0.,2.,0.],
                        [0.,0.,0.,1.,0.]],dtype=torch.float64,requires_grad=True)
        y=[3,3,0]; ids=['a','b','c']
        loss,details=cutoff_rank_loss(z,y,ids,2)
        grad,_=cutoff_rank_gradient(z.detach(),y,ids,2)
        self.assertEqual(details['active_pairs'],0)
        self.assertEqual(float(loss),0.)
        self.assertTrue(torch.equal(grad,torch.zeros_like(grad)))

    def test_multiple_pairs_analytic_gradient_matches_autograd(self):
        from tralo.cutoff_rank import cutoff_rank_gradient, cutoff_rank_loss
        z=torch.tensor([[0.,.1,0.,0.,0.], [0.,0.,.1,1.,0.],
                        [.1,0.,0.,2.,0.], [0.,.2,0.,3.,0.],
                        [0.,0.,.2,4.,0.]],dtype=torch.float64,requires_grad=True)
        y=[3,3,3,0,0]; ids=['a','b','c','d','e']
        loss,details=cutoff_rank_loss(z,y,ids,2)
        actual,_=cutoff_rank_gradient(z.detach(),y,ids,2)
        expected=torch.autograd.grad(loss,z)[0]
        self.assertEqual(details['active_pairs'],6)
        self.assertTrue(torch.allclose(actual,expected,atol=1e-12,rtol=1e-12))
        self.assertTrue(torch.allclose(actual.sum(1),torch.zeros(5,dtype=z.dtype),atol=1e-12))

    def test_tie_break_uses_id_and_permutation_preserves_result(self):
        from tralo.cutoff_rank import cutoff_rank_gradient
        z=torch.tensor([[0.,0.,0.,1.,0.],[0.,0.,0.,1.,0.]],dtype=torch.float64)
        y=[0,3]; ids=['b','a']
        _,details=cutoff_rank_gradient(z,y,ids,1)
        self.assertEqual(details['selected_ids'],['a'])
        permutation=[1,0]
        _,reordered=cutoff_rank_gradient(z[permutation],[y[i] for i in permutation],
                                         [ids[i] for i in permutation],1)
        self.assertEqual(reordered['selected_ids'],['a'])

    def test_reject_duplicate_ids(self):
        from tralo.cutoff_rank import cutoff_rank_gradient
        with self.assertRaises(ValueError):
            cutoff_rank_gradient(torch.zeros((2,5)),[3,0],['a','a'],1)


if __name__=='__main__':unittest.main()
