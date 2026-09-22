import unittest

from tralo.metrics import classification_metrics


class MetricsTests(unittest.TestCase):
    def test_hand_calculated_confusion_and_fixed_class_average(self):
        result = classification_metrics([0, 0, 1, 1], [0, 1, 1, 1], 3, [1])
        self.assertEqual(result['confusion'], [[1, 1, 0], [0, 2, 0], [0, 0, 0]])
        self.assertEqual(result['accuracy'], 0.75)
        self.assertAlmostEqual(result['per_class'][0]['f1'], 2 / 3)
        self.assertAlmostEqual(result['per_class'][1]['precision'], 2 / 3)
        self.assertAlmostEqual(result['cc_f1'], 0.8)
        self.assertAlmostEqual(result['macro_f1'], (2 / 3 + 0.8) / 3)
        self.assertEqual(result['per_class'][2]['f1'], 0)

    def test_missing_predictions_are_zero_not_dropped(self):
        result = classification_metrics([0, 1], [0, 0], 2, [1])
        self.assertEqual(result['cc_f1'], 0)
        self.assertEqual(result['per_class'][1]['recall'], 0)

    def test_no_constraints_has_no_cc_metric(self):
        self.assertIsNone(classification_metrics([0], [0], 1, [])['cc_f1'])

    def test_sample_order_does_not_change_metrics(self):
        actual, predicted = [0, 2, 1, 2, 0], [1, 2, 1, 0, 0]
        expected = classification_metrics(actual, predicted, 3, [0, 2])
        order = [4, 1, 3, 0, 2]
        observed = classification_metrics([actual[i] for i in order],
                                          [predicted[i] for i in order], 3, [0, 2])
        self.assertEqual(observed, expected)

    def test_renaming_classes_preserves_aggregate_metrics(self):
        actual, predicted = [0, 2, 1, 2, 0], [1, 2, 1, 0, 0]
        rename = {0: 2, 1: 0, 2: 1}
        expected = classification_metrics(actual, predicted, 3, [0, 2])
        observed = classification_metrics([rename[c] for c in actual],
                                          [rename[c] for c in predicted], 3, [2, 1])
        for metric in ('accuracy', 'macro_f1', 'cc_f1'):
            self.assertAlmostEqual(observed[metric], expected[metric])
        for c in range(3):
            self.assertEqual(observed['per_class'][rename[c]]['f1'],
                             expected['per_class'][c]['f1'])

    def test_reject_invalid_labels_and_shapes(self):
        for actual, predicted, count, constrained in [
            ([], [], 2, [0]), ([0], [], 2, [0]), ([2], [0], 2, [0]),
            ([0], [-1], 2, [0]), ([True], [0], 2, [0]),
            ([0], [0], 0, []), ([0], [0], 2, [2]), ([0], [0], 2, [0, 0]),
            ([0], [0], 2, None),
        ]:
            with self.subTest(actual=actual, predicted=predicted, constrained=constrained):
                with self.assertRaises(ValueError):
                    classification_metrics(actual, predicted, count, constrained)


if __name__ == '__main__':
    unittest.main()
