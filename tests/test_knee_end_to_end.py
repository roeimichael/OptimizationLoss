import copy
from pathlib import Path
import tempfile
import unittest

try:
    import torch
except ImportError:
    torch = None

from tralo.knee_end_to_end import train_one, validate, score_snapshots
from tralo.knee_experiment import digest


@unittest.skipUnless(torch is not None, 'Torch is unavailable')
class EndToEndTests(unittest.TestCase):
    def test_matched_warmup_and_inactive_null_parity(self):
        torch.manual_seed(4)
        base = torch.nn.Sequential(torch.nn.Linear(4, 7), torch.nn.Tanh(),
                                   torch.nn.Linear(7, 5)).double()
        features = torch.randn(12, 4, dtype=torch.float64)
        labels = torch.tensor([0, 1, 2, 3, 4, 3, 2, 1, 0, 4, 3, 2])
        training = torch.utils.data.TensorDataset(features, labels)
        development = [torch.randn(3, 4, dtype=torch.float64),
                       torch.randn(2, 4, dtype=torch.float64)]
        config = dict(seed=33, epochs=4, warmup_epochs=2, batch_size=4,
                      task_lr=.002, constraint_lr=.001,
                      caps=[None, None, None, 99, None],
                      lambda_initial=.01, lambda_step=.05,
                      rho_initial=.5, rho_target=.5)
        output = {}
        for arm in ('clipper', 'tralo_null', 'tralo'):
            model = copy.deepcopy(base)
            events, snapshots = [], []
            result = train_one(model, training, development, config, arm,
                               events.append,
                               lambda epoch, phase, probabilities:
                                   snapshots.append((epoch, phase, probabilities.clone())))
            self.assertEqual(result['task_updates_applied'], 12)
            self.assertEqual(result['task_updates_skipped'], 0)
            self.assertEqual(len(snapshots), 5)
            output[arm] = (result, model)
        self.assertEqual(len({output[arm][0]['warmup_sha256'] for arm in output}), 1)
        self.assertEqual(len({output[arm][0]['batch_sha256'] for arm in output}), 1)
        self.assertEqual(output['tralo'][0]['constraint_updates_applied'], 0)
        self.assertEqual(output['tralo'][0]['constraint_updates_inactive'], 2)
        for a,b in zip(output['tralo'][1].parameters(), output['tralo_null'][1].parameters()):
            self.assertTrue(torch.equal(a,b))

    def test_preregistered_contract_rejects_cap_change(self):
        config = dict(seed=1301, epochs=10, warmup_epochs=5, batch_size=32,
                      development_batch_size=16, task_lr=.0001,
                      constraint_lr=.00003, caps=[None,None,None,76,None],
                      lambda_initial=.01, lambda_step=.05,
                      rho_initial=.5, rho_target=.5)
        validate(config)
        config['caps'][3] = 75
        with self.assertRaises(ValueError):
            validate(config)

    def test_offline_quota_slot_audit_names_correct_entry_and_exit(self):
        before = torch.tensor([[.10,.01,.01,.85,.03], [.60,.01,.01,.35,.03],
                               [.20,.01,.01,.75,.03]], dtype=torch.float32)
        after = torch.tensor([[.60,.01,.01,.35,.03], [.05,.01,.01,.90,.03],
                              [.20,.01,.01,.75,.03]], dtype=torch.float32)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            snapshots = []
            for phase, value in [('before_constraint', before), ('after_constraint', after)]:
                path = root / (phase + '.pt')
                torch.save(value, path)
                snapshots.append(dict(epoch=6, phase=phase, file=path.name, sha256=digest(path)))
            report = score_snapshots(root, snapshots, [3, 3, 0], ['a','b','c'],
                                     [None,None,None,1,None])
            slot = report['transitions'][0]['policies']['capped_first']
            self.assertEqual((slot['entries'], slot['entries_correct'],
                              slot['exits'], slot['exits_correct']), (1,1,1,1))
            self.assertEqual(slot['entering_ids'], ['b'])
            self.assertEqual(slot['leaving_ids'], ['a'])


if __name__ == '__main__':
    unittest.main()
