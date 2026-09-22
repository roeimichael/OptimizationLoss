import unittest
from tralo.global_comparison import train_arm, validate_config
try:
    import torch
except ImportError:
    torch = None

CFG = dict(seeds=[1], epochs=4, warmup_epochs=2, batch_size=3, lr=0.01,
           lambda_initial=0.01, lambda_step=0.05, rho_initial=0.5, rho_target=2.0,
           cache_events_sha256='0'*64)

class ConfigTests(unittest.TestCase):
    def test_phase_and_seeds_must_be_valid(self):
        validate_config(CFG)
        for update in ({'warmup_epochs':4}, {'seeds':[1,1]}, {'lr':float('nan')}):
            with self.assertRaises(ValueError): validate_config(dict(CFG, **update))

@unittest.skipIf(torch is None, 'PyTorch required; executed in native server suite')
class MatchedTrainingTests(unittest.TestCase):
    def fixture(self):
        x=torch.tensor([[1.,0.],[0.,1.],[-1.,0.],[0.,-1.]])
        return x,torch.tensor([0,1,1,0]),x[:3]

    def test_zero_constraint_is_exact_null_and_logging_neutral(self):
        x,y,u=self.fixture(); records=[]
        cfg=dict(CFG,lambda_initial=0.,lambda_step=0.)
        null=train_arm(x,y,u,[1,None],cfg,1,'tralo_null',records.append)
        zero=train_arm(x,y,u,[1,None],cfg,1,'tralo',lambda _:None)
        for key in null['state']:
            self.assertTrue(torch.equal(null['state'][key],zero['state'][key]))
        self.assertEqual(null['batch_sha256'],zero['batch_sha256'])
        again=train_arm(x,y,u,[1,None],cfg,1,'tralo_null',lambda _:None)
        self.assertTrue(torch.equal(null['probabilities'],again['probabilities']))
        self.assertEqual(null['constraint_updates'],0)

    def test_arms_share_warmup_and_batches_and_constraint_has_actual_dose(self):
        x,y,u=self.fixture()
        results={a:train_arm(x,y,u,[0,None],CFG,1,a,lambda _:None)
                 for a in ('clipper','tralo_null','tralo')}
        self.assertEqual(len({r['warmup_sha256'] for r in results.values()}),1)
        self.assertEqual(len({r['batch_sha256'] for r in results.values()}),1)
        self.assertEqual({r['task_updates'] for r in results.values()},{8})
        self.assertEqual(results['tralo']['constraint_updates'],2)
        self.assertFalse(torch.equal(results['tralo']['probabilities'],results['tralo_null']['probabilities']))

    def test_read_only_step_observer_does_not_change_training(self):
        x,y,u=self.fixture(); seen=[]
        reference=train_arm(x,y,u,[0,None],CFG,1,'tralo',lambda _:None)
        def observe(stage,phase,epoch,batch,model,optimizer):
            seen.append((stage,phase,epoch,batch,
                         sum(float(p.detach().square().sum()) for p in model.parameters())))
        traced=train_arm(x,y,u,[0,None],CFG,1,'tralo',lambda _:None,observer=observe)
        self.assertTrue(torch.equal(reference['probabilities'],traced['probabilities']))
        for key in reference['state']:
            self.assertTrue(torch.equal(reference['state'][key],traced['state'][key]))
        self.assertEqual(len(seen),2*(reference['task_updates']+reference['constraint_updates']))
