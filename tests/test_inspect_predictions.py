import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from tralo.inspect_predictions import inspect_file


def fixture():
    return {
        'sample_ids': ['a', 'b', 'c'], 'groups': ['g', 'g', 'h'],
        'probabilities': [[0.8, 0.2], [0.6, 0.4], [0.3, 0.7]],
        'labels': [0, 1, 1], 'allocated_predictions': [0, 1, 1],
        'global_caps': [1, None], 'local_caps': {'g': [1, None], 'h': [1, None]},
        'constrained_classes': [0], 'allocation_policy': 'hand_worked_fixture',
    }


class InspectionTests(unittest.TestCase):
    def test_saved_artifacts_match_hand_example_and_hashes(self):
        with tempfile.TemporaryDirectory() as tmp:
            source, output = Path(tmp) / 'input.json', Path(tmp) / 'run'
            source.write_text(json.dumps(fixture()), encoding='utf-8')
            inspect_file(source, output)
            report = json.loads((output / 'report.json').read_text())
            self.assertAlmostEqual(report['raw_metrics']['accuracy'], 2 / 3)
            self.assertEqual(report['allocated_metrics']['accuracy'], 1)
            self.assertFalse(report['raw_quotas']['feasible'])
            self.assertTrue(report['allocated_quotas']['feasible'])
            self.assertEqual(report['changed_predictions'], 1)
            self.assertFalse(report['allocation_policy_verified'])
            self.assertEqual(report['provenance']['input_sha256'], hashlib.sha256(source.read_bytes()).hexdigest())
            self.assertIn('metrics.py', report['provenance']['source_sha256'])
            self.assertEqual(len(report['provenance']['config_sha256']), 64)
            events = [json.loads(s) for s in (output / 'events.jsonl').read_text().splitlines()]
            self.assertEqual([e['event'] for e in events], ['started', 'completed'])
            self.assertEqual(events[-1]['report_sha256'], hashlib.sha256((output / 'report.json').read_bytes()).hexdigest())
            with self.assertRaises(FileExistsError):
                inspect_file(source, output)

    def test_invalid_inputs_leave_no_successful_run(self):
        for mutate in [
            lambda d: d.update(probabilities=[[0.8, 0.4]] * 3),
            lambda d: d.update(sample_ids=['a', 'a', 'c']),
            lambda d: d.update(constrained_classes=[]),
            lambda d: d.update(allocated_predictions=[0, 0, 1]),
            lambda d: d.update(unknown_option=True),
        ]:
            with self.subTest(mutate=mutate), tempfile.TemporaryDirectory() as tmp:
                data = fixture()
                mutate(data)
                source, output = Path(tmp) / 'input.json', Path(tmp) / 'run'
                source.write_text(json.dumps(data), encoding='utf-8')
                with self.assertRaises(ValueError):
                    inspect_file(source, output)
                self.assertFalse(output.exists())

    def test_duplicate_json_keys_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / 'input.json'
            source.write_text('{"labels": [], "labels": [0]}')
            with self.assertRaises(ValueError):
                inspect_file(source, Path(tmp) / 'run')


if __name__ == '__main__':
    unittest.main()
