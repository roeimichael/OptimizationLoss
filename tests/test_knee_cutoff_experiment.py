import copy
import json
from pathlib import Path
import unittest

import torch

from tralo.knee_cutoff_experiment import ARMS, TrainingChunks, train_one, validate


class TinyTrainingImages:
    def __init__(self):
        self.values = [
            (torch.tensor([2., 0.]), 3), (torch.tensor([1., 1.]), 0),
            (torch.tensor([0., 2.]), 3), (torch.tensor([-1., 1.]), 1),
            (torch.tensor([-2., 0.]), 3), (torch.tensor([0., -2.]), 2),
        ]

    def __len__(self):
        return len(self.values)

    def __getitem__(self, index):
        return self.values[index]


class CutoffRunnerTests(unittest.TestCase):
    def test_fixed_configs_validate(self):
        base = Path(__file__).resolve().parents[1] / 'experiments' / 'configs'
        for seed in (1401, 1402, 1403, 1404):
            config = json.loads((base / f'knee_cutoff_{seed}.json').read_text())
            validate(config)
            self.assertEqual(config['train_capacity'], 532)
        config['seed'] = 1301
        with self.assertRaises(ValueError):
            validate(config)

    def test_training_chunks_replay_same_order_without_labels(self):
        cohort = TinyTrainingImages()
        chunks = TrainingChunks(cohort, 2)
        first = list(chunks)
        second = list(chunks)
        self.assertEqual(len(chunks), 3)
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(first, second)))
        self.assertEqual(torch.cat(first).shape, (6, 2))

    def test_five_arms_match_warmup_and_batch_schedule(self):
        torch.manual_seed(10)
        base = torch.nn.Linear(2, 5)
        cohort = TinyTrainingImages()
        ids = [f'train-{i}' for i in range(len(cohort))]
        training_labels = [cohort[i][1] for i in range(len(cohort))]
        # Development has images only; no validation-label argument exists.
        development = [torch.tensor([[.3, .8], [.5, -.2], [-.3, .4]])]
        config = dict(seed=1401, epochs=2, warmup_epochs=1, batch_size=3,
                      auxiliary_batch_size=2, task_lr=.001,
                      auxiliary_lr=.0001, caps=[None, None, None, 1, None],
                      train_capacity=2, lambda_initial=.2, lambda_step=.1,
                      rho_initial=.5, rho_target=.5)
        results = {}
        for arm in ARMS:
            events = []
            snapshots = []
            result = train_one(copy.deepcopy(base), cohort, ids, training_labels,
                               development, config, arm, events.append,
                               lambda epoch, phase, values:
                               snapshots.append((epoch, phase, values.clone())))
            results[arm] = result
            self.assertEqual(result['task_updates_applied'], 4)
            self.assertEqual(result['task_updates_skipped'], 0)
            self.assertEqual(result['auxiliary_checks'], int(arm not in ('clipper', 'tralo_null')))
            self.assertEqual(len(snapshots), 2)
            self.assertTrue(torch.isfinite(result['final_probabilities']).all())
            if arm in ('rank_only', 'rank_count'):
                epoch = next(event for event in events if event['event'] == 'epoch' and event['epoch'] == 2)
                self.assertIn('training_selected_true', epoch)
                self.assertIn('active_pairs', epoch)
        self.assertEqual(len({r['warmup_sha256'] for r in results.values()}), 1)
        self.assertEqual(len({r['batch_sha256'] for r in results.values()}), 1)
        # Clipper retains task Adam moments; the phase Null deliberately resets
        # them at the shared boundary, so endpoint equality is not expected.
        self.assertEqual(results['tralo_null']['auxiliary_updates_applied'], 0)


if __name__ == '__main__':
    unittest.main()
