import unittest
import torch
from tralo.alm import augmented_penalty, residuals, update_dual
from tralo.global_comparison import train_arm
from test_global_comparison import CFG

class ALMTests(unittest.TestCase):
    def test_hand_value_and_gradient(self):
        g=torch.tensor([.5,-2.],dtype=torch.double,requires_grad=True)
        lam=torch.tensor([1.,1.],dtype=torch.double)
        loss=augmented_penalty(g,lam,2.)
        self.assertAlmostEqual(loss.item(),.5)
        loss.backward()
        self.assertTrue(torch.equal(g.grad,torch.tensor([2.,0.],dtype=torch.double)))
        self.assertTrue(torch.equal(update_dual(g.detach(),lam,2.),torch.tensor([2.,0.],dtype=torch.double)))
    def test_gradient_and_uncapped_columns(self):
        z=torch.tensor([[.2,-.1,.3],[.1,.5,-.4]],dtype=torch.double,requires_grad=True)
        caps=[1,None,0]
        f=lambda x: augmented_penalty(residuals(x,caps),torch.tensor([.2,.1],dtype=torch.double),.7)
        self.assertTrue(torch.autograd.gradcheck(f,(z,)))
        self.assertEqual(residuals(z,caps).shape,(2,))
    def test_null_parity_and_budget(self):
        x=torch.tensor([[1.,0.],[0.,1.],[-1.,0.],[0.,-1.]])
        y=torch.tensor([0,1,1,0]); cfg=dict(CFG,alm_rho=.5,alm_lambda_initial=0.)
        out={a:train_arm(x,y,x[:3],[0,None],cfg,1,a,lambda _:None) for a in ('tralo_null','alm_null','alm')}
        self.assertTrue(torch.equal(out['tralo_null']['probabilities'],out['alm_null']['probabilities']))
        self.assertEqual(len({v['warmup_sha256'] for v in out.values()}),1)
        self.assertEqual(len({v['batch_sha256'] for v in out.values()}),1)
        self.assertEqual(out['alm']['joint_constraint_updates'],4)
        self.assertEqual(out['alm']['dual_updates'],2)
        self.assertEqual(out['alm']['task_updates'],8)
        self.assertFalse(torch.equal(out['alm']['probabilities'],out['alm_null']['probabilities']))
    def test_no_caps_matches_null(self):
        x=torch.eye(2); y=torch.tensor([0,1]); cfg=dict(CFG,alm_rho=.5,alm_lambda_initial=0.)
        a=train_arm(x,y,x,[None,None],cfg,1,'alm',lambda _:None)
        b=train_arm(x,y,x,[None,None],cfg,1,'alm_null',lambda _:None)
        self.assertTrue(torch.equal(a['probabilities'],b['probabilities']))

    def test_equal_slots_check_rejects_underfill(self):
        from tralo.global_report import evaluate_global
        from tralo.comparison_checks import check_report
        report=evaluate_global([[.9,.1],[.8,.2],[.3,.7]],[0,1,1],[1,None],['a','b','c'])
        check_report(report,[0,1,1],[1,None])
        report['capped_first']['predictions']=[1,1,1]
        with self.assertRaises(RuntimeError): check_report(report,[0,1,1],[1,None])
