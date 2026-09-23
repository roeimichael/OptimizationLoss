import tempfile
import unittest
from pathlib import Path
try:
    import torch
except ImportError:
    torch = None

@unittest.skipIf(torch is None, 'native PyTorch required')
class TrajectoryTests(unittest.TestCase):
    def test_predictions_and_parameters_unchanged_with_dense_observation(self):
        from tralo.trajectory import SnapshotObserver
        from tralo.global_comparison import train_arm
        x=torch.tensor([[1.,0.],[0.,1.],[-1.,0.],[0.,-1.]])
        y=torch.tensor([0,1,1,0])
        config=dict(seeds=[5],epochs=3,warmup_epochs=1,batch_size=2,lr=.01,
            lambda_initial=.1,lambda_step=.01,rho_initial=.5,rho_target=.5,
            cache_events_sha256='0'*64,constraint_optimizer='separate')
        for arm in ('tralo_null','tralo'):
            reference=train_arm(x,y,x,[0,None],config,5,arm,lambda r:None)
            with tempfile.TemporaryDirectory() as d:
                observer=SnapshotObserver(x,Path(d))
                traced=train_arm(x,y,x,[0,None],config,5,arm,lambda r:None,observer)
                self.assertTrue(torch.equal(reference['probabilities'],traced['probabilities']))
                for key in reference['state']:
                    self.assertTrue(torch.equal(reference['state'][key],traced['state'][key]))
                self.assertEqual(len(observer.index),9+2*traced['constraint_updates'])
                last=torch.load(Path(d)/observer.index[-1]['file'],weights_only=True)
                self.assertTrue(torch.equal(last['logits'].softmax(1),traced['probabilities']))

    def test_snapshot_preserves_gradient_rng_and_optimizer(self):
        from tralo.trajectory import SnapshotObserver
        model=torch.nn.Linear(2,2);x=torch.ones(3,2)
        optimizer=torch.optim.Adam(model.parameters())
        model(x).sum().backward()
        grads=[p.grad.clone() for p in model.parameters()]
        rng=torch.get_rng_state().clone()
        with tempfile.TemporaryDirectory() as d:
            SnapshotObserver(x,Path(d))('before','constraint',2,None,model,optimizer)
        self.assertTrue(torch.equal(rng,torch.get_rng_state()))
        self.assertEqual(len(optimizer.state),0)
        for p,g in zip(model.parameters(),grads):self.assertTrue(torch.equal(p.grad,g))
