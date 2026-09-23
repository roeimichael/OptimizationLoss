import copy
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
    def test_constraint_optimizer_is_optional_and_named(self):
        validate_config(CFG)
        for mode in ('shared', 'separate'):
            validate_config(dict(CFG, constraint_optimizer=mode))
        for mode in ('adam', None, 1, []):
            with self.assertRaises(ValueError):
                validate_config(dict(CFG, constraint_optimizer=mode))

    def test_phase_and_seeds_must_be_valid(self):
        validate_config(CFG)
        for update in ({'warmup_epochs':4}, {'seeds':[1,1]}, {'lr':float('nan')}):
            with self.assertRaises(ValueError): validate_config(dict(CFG, **update))

@unittest.skipIf(torch is None, 'PyTorch required; executed in native server suite')
class MatchedTrainingTests(unittest.TestCase):
    def test_constraint_lr_changes_only_constraint_first_displacement(self):
        x,y,u=self.fixture()
        first=[]
        for rate in (.01,.001):
            snapshots=[]
            def observe(stage,phase,epoch,batch,model,optimizer):
                if phase=='constraint' and len(snapshots)<2:
                    snapshots.append(torch.cat([p.detach().flatten().clone() for p in model.parameters()]))
            cfg=dict(CFG,constraint_optimizer='separate',constraint_lr=rate)
            train_arm(x,y,u,[0,None],cfg,1,'tralo',lambda _:None,observer=observe)
            first.append(snapshots[1]-snapshots[0])
            null=train_arm(x,y,u,[0,None],cfg,1,'tralo_null',lambda _:None)
            reference=train_arm(x,y,u,[0,None],CFG,1,'tralo_null',lambda _:None)
            self.assertTrue(torch.equal(null['probabilities'],reference['probabilities']))
        self.assertTrue(torch.allclose(first[0],first[1]*10,atol=1e-7,rtol=1e-4))

    def test_auxiliary_uses_same_warmup_and_null_with_no_constraint_is_exact(self):
        x,y,u=self.fixture()
        reference=train_arm(x,y,u,[0,None],CFG,1,'tralo_null',lambda _:None)
        for kind in ('margin','false_positive'):
            cfg=dict(CFG,constraint_optimizer='separate',supervised_auxiliary=kind,
                     auxiliary_weight=.2,lambda_initial=0.,lambda_step=0.)
            records=[]
            null=train_arm(x,y,u,[0,None],cfg,1,'tralo_null',records.append)
            tralo=train_arm(x,y,u,[0,None],cfg,1,'tralo',lambda _:None)
            self.assertTrue(torch.equal(null['probabilities'],tralo['probabilities']))
            self.assertEqual(reference['warmup_sha256'],null['warmup_sha256'])
            self.assertFalse(torch.equal(reference['probabilities'],null['probabilities']))
            epochs=[r for r in records if r['event']=='epoch']
            self.assertEqual(epochs[0]['auxiliary_loss'],0.)
            for r in epochs:
                self.assertAlmostEqual(r['task_loss'],r['task_ce_loss']+.2*r['auxiliary_loss'],places=6)
            clip=train_arm(x,y,u,[0,None],cfg,1,'clipper',lambda _:None)
            base=train_arm(x,y,u,[0,None],CFG,1,'clipper',lambda _:None)
            self.assertTrue(torch.equal(clip['probabilities'],base['probabilities']))

    def test_zero_auxiliary_weight_preserves_exact_trajectory(self):
        x,y,u=self.fixture()
        base=train_arm(x,y,u,[0,None],CFG,1,'tralo',lambda _:None)
        for kind in ('margin','false_positive'):
            cfg=dict(CFG,supervised_auxiliary=kind,auxiliary_weight=0.)
            out=train_arm(x,y,u,[0,None],cfg,1,'tralo',lambda _:None)
            self.assertTrue(torch.equal(base['probabilities'],out['probabilities']))

    def fixture(self):
        x=torch.tensor([[1.,0.],[0.,1.],[-1.,0.],[0.,-1.]])
        return x,torch.tensor([0,1,1,0]),x[:3]

    def test_default_and_explicit_shared_are_exact(self):
        x,y,u=self.fixture()
        for arm in ('clipper','tralo_null','tralo'):
            default=train_arm(x,y,u,[0,None],CFG,1,arm,lambda _:None)
            shared=train_arm(x,y,u,[0,None],dict(CFG,constraint_optimizer='shared'),1,arm,lambda _:None)
            self.assertTrue(torch.equal(default['probabilities'],shared['probabilities']))
            for key in default['state']:
                self.assertTrue(torch.equal(default['state'][key],shared['state'][key]))

    def test_inactive_constraints_are_exact_null_in_both_modes(self):
        x,y,u=self.fixture()
        for mode in ('shared','separate'):
            for caps, updates in (([None,None],{}),([0,None],dict(lambda_initial=0.,lambda_step=0.))):
                cfg=dict(CFG,constraint_optimizer=mode,**updates)
                null=train_arm(x,y,u,caps,cfg,1,'tralo_null',lambda _:None)
                zero=train_arm(x,y,u,caps,cfg,1,'tralo',lambda _:None)
                self.assertEqual(zero['constraint_updates'],0)
                self.assertTrue(torch.equal(null['probabilities'],zero['probabilities']))
                for key in null['state']:
                    self.assertTrue(torch.equal(null['state'][key],zero['state'][key]))

    def test_separate_constraint_updates_preserve_task_adam_state(self):
        x,y,u=self.fixture(); task_optimizers=[]; constraint_optimizers=[]; records=[]
        snapshot=None
        def observe(stage,phase,epoch,batch,model,optimizer):
            nonlocal snapshot
            if phase=='task':
                if not task_optimizers or optimizer is not task_optimizers[-1]:
                    task_optimizers.append(optimizer)
            elif stage=='before':
                self.assertIsNot(optimizer,task_optimizers[-1])
                constraint_optimizers.append(optimizer)
                snapshot=copy.deepcopy(task_optimizers[-1].state_dict())
            else:
                current=task_optimizers[-1].state_dict()
                self.assertEqual(snapshot['param_groups'],current['param_groups'])
                for parameter, state in snapshot['state'].items():
                    for key,value in state.items():
                        self.assertTrue(torch.equal(value,current['state'][parameter][key]))
        result=train_arm(x,y,u,[0,None],dict(CFG,constraint_optimizer='separate'),1,
                         'tralo',records.append,observer=observe)
        self.assertEqual(len(task_optimizers),2)
        self.assertEqual(len(constraint_optimizers),result['constraint_updates'])
        self.assertGreater(len(constraint_optimizers),1)
        self.assertTrue(all(o is constraint_optimizers[0] for o in constraint_optimizers))
        self.assertEqual(records[0]['constraint_optimizer'],'separate')

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
