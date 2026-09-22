import unittest

from tralo.global_report import evaluate_global


class GlobalReportTests(unittest.TestCase):
    def test_same_probabilities_two_named_policies(self):
        report = evaluate_global([[0.4, 0.6], [0.3, 0.7]], [1, 1], [1, None], ['a', 'b'])
        self.assertEqual(report['raw']['metrics']['accuracy'], 1)
        self.assertEqual(report['upper_bound_correction']['metrics']['accuracy'], 1)
        self.assertEqual(report['upper_bound_correction']['changed'], 0)
        self.assertEqual(report['capped_first']['metrics']['accuracy'], 0.5)
        self.assertEqual(report['capped_first']['counts'], [1, 1])
        self.assertTrue(report['capped_first']['feasible'])

    def test_labels_do_not_change_predictions(self):
        arguments = ([[0.7, 0.3], [0.8, 0.2]], [1, None], ['a', 'b'])
        left = evaluate_global(arguments[0], [0, 1], *arguments[1:])
        right = evaluate_global(arguments[0], [1, 0], *arguments[1:])
        for policy in ('raw', 'upper_bound_correction', 'capped_first'):
            self.assertEqual(left[policy]['predictions'], right[policy]['predictions'])


if __name__ == '__main__':
    unittest.main()
