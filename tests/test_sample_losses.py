import math
import unittest
import torch
from tralo.sample_losses import sample_loss


class SampleLossTests(unittest.TestCase):
    def test_false_positive_binary_value_and_gradient(self):
        z=torch.zeros((1,2),dtype=torch.float64,requires_grad=True)
        loss=sample_loss(z,torch.tensor([1]),[2,None],'false_positive')
        self.assertAlmostEqual(loss.item(),math.log(2))
        loss.backward()
        self.assertTrue(torch.allclose(z.grad,torch.tensor([[.5,-.5]],dtype=torch.float64)))

    def test_true_capped_examples_are_not_false_positives(self):
        z=torch.tensor([[2.,-1.]],requires_grad=True)
        loss=sample_loss(z,torch.tensor([0]),[2,None],'false_positive')
        loss.backward()
        self.assertEqual(loss.item(),0.)
        self.assertTrue(torch.equal(z.grad,torch.zeros_like(z)))

    def test_margin_hand_gradient_and_satisfied_zero(self):
        z=torch.tensor([[0.,1.],[3.,0.]],requires_grad=True)
        loss=sample_loss(z,torch.tensor([0,0]),[1,None],'margin',1.)
        self.assertEqual(loss.item(),1.)
        loss.backward()
        self.assertTrue(torch.equal(z.grad,torch.tensor([[-.5,.5],[0.,0.]])))

    def test_both_terms_gradcheck_and_logit_shift_invariant(self):
        z=torch.tensor([[.2,1.3,-.8],[.5,-.7,.1]],dtype=torch.float64,requires_grad=True)
        y=torch.tensor([0,2]); caps=[1,None,0]
        for kind in ('margin','false_positive','far_error'):
            f=lambda values: sample_loss(values,y,caps,kind,1.7)
            self.assertTrue(torch.autograd.gradcheck(f,(z,)))
            self.assertTrue(torch.allclose(f(z),f(z+19.)))

    def test_no_constraints_and_disabled_are_zero(self):
        z=torch.tensor([[.2,.7]],requires_grad=True);y=torch.tensor([1])
        for kind in ('none','false_positive','far_error'):
            self.assertEqual(sample_loss(z,y,[None,None],kind).item(),0.)

    def test_far_errors_receive_quadratic_pressure(self):
        z=torch.tensor([[0.,1.],[0.,2.],[2.,0.]],requires_grad=True)
        loss=sample_loss(z,torch.tensor([0,0,0]),[None,1],'far_error',0.)
        self.assertAlmostEqual(loss.item(),5/3)
        loss.backward()
        self.assertTrue(torch.allclose(z.grad,torch.tensor([[-2/3,2/3],[-4/3,4/3],[0.,0.]])))

    def test_extreme_logits_remain_finite(self):
        z=torch.tensor([[1000.,-1000.]],requires_grad=True)
        loss=sample_loss(z,torch.tensor([1]),[0,None],'false_positive')
        loss.backward()
        self.assertEqual(loss.item(),2000.)
        self.assertTrue(torch.isfinite(z.grad).all())

    def test_invalid_labels_and_names_fail(self):
        z=torch.tensor([[.2,.7]])
        for y,kind in [(torch.tensor([2]),'margin'),(torch.tensor([1]),'unknown')]:
            with self.assertRaises(ValueError): sample_loss(z,y,[0,None],kind)
