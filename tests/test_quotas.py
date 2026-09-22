import unittest

from tralo.quotas import audit_quotas


class QuotaAuditTests(unittest.TestCase):
    def test_counts_global_and_local_caps_and_reports_both_violations(self):
        result = audit_quotas(
            [0, 1, 1, 2, 1],
            ["a", "a", "b", "b", "a"],
            3,
            [2, 1, None],
            {"a": [1, 1, None], "b": [1, 0, None]},
        )
        self.assertEqual(result["global_counts"], [1, 3, 1])
        self.assertEqual(result["local_counts"], {"a": [1, 2, 0], "b": [0, 1, 1]})
        self.assertEqual(
            result["violations"],
            [
                {"scope": "global", "group": None, "class": 1, "count": 3, "cap": 1},
                {"scope": "local", "group": "a", "class": 1, "count": 2, "cap": 1},
                {"scope": "local", "group": "b", "class": 1, "count": 1, "cap": 0},
            ],
        )
        self.assertFalse(result["feasible"])

    def test_local_only_and_global_only_caps_are_supported(self):
        local = audit_quotas([0, 1, 0], ["x", "y", "x"], 2, [None, None],
                             {"x": [2, None], "y": [1, None]})
        self.assertEqual(local["global_counts"], [2, 1])
        self.assertEqual(local["violations"], [])
        self.assertTrue(local["feasible"])

        global_only = audit_quotas([0, 1, 1], ["x", "x", "x"], 2, [1, 2], {"x": [None, None]})
        self.assertEqual(global_only["violations"], [])
        self.assertTrue(global_only["feasible"])

    def test_zero_caps_are_real_caps(self):
        result = audit_quotas([0, 1], ["x", "x"], 2, [0, None], {"x": [0, None]})
        self.assertEqual(result["violations"], [
            {"scope": "global", "group": None, "class": 0, "count": 1, "cap": 0},
            {"scope": "local", "group": "x", "class": 0, "count": 1, "cap": 0},
        ])
        self.assertFalse(result["feasible"])

    def test_rejects_unknown_or_missing_local_groups_before_counting(self):
        cases = [
            ([0], ["x"], {"y": [1]}),
            ([0], ["x"], {}),
            ([0, 1], ["x", "y"], {"x": [1]}),
        ]
        for predictions, groups, local_caps in cases:
            with self.subTest(local_caps=local_caps):
                with self.assertRaises(ValueError):
                    audit_quotas(predictions, groups, 2, [None, None], local_caps)

    def test_rejects_malformed_inputs(self):
        bad = [
            ([0], ["x"], 0, [None], {"x": [None]}),
            ([True], ["x"], 2, [None, None], {"x": [None, None]}),
            ([-1], ["x"], 2, [None, None], {"x": [None, None]}),
            ([2], ["x"], 2, [None, None], {"x": [None, None]}),
            ([0], [""], 2, [None, None], {"": [None, None]}),
            ([0], ["x"], 2, [None], {"x": [None, None]}),
            ([0], ["x"], 2, [-1, None], {"x": [None, None]}),
            ([0], ["x"], 2, [False, None], {"x": [None, None]}),
            ([0], ["x"], 2, [None, None], {"x": [1]}),
        ]
        for args in bad:
            with self.subTest(args=args):
                with self.assertRaises((TypeError, ValueError)):
                    audit_quotas(*args)

    def test_rejects_empty_or_mismatched_samples_and_preserves_inputs(self):
        predictions = [0, 1]
        groups = ["a", "a"]
        global_caps = [1, None]
        local_caps = {"a": [1, None]}
        snapshot = (predictions[:], groups[:], global_caps[:], {"a": local_caps["a"][:]})
        with self.assertRaises(ValueError):
            audit_quotas([], [], 2, [None, None], {})
        with self.assertRaises(ValueError):
            audit_quotas([0], ["a", "a"], 2, [None, None], {"a": [None, None]})
        audit_quotas(predictions, groups, 2, global_caps, local_caps)
        self.assertEqual((predictions, groups, global_caps, local_caps), snapshot)


if __name__ == "__main__":
    unittest.main()
