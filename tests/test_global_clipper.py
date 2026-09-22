import unittest

from tralo.global_clipper import allocate


class GlobalClipperTests(unittest.TestCase):
    def test_upper_bound_keeps_feasible_raw_predictions(self):
        probs = [[.8, .2], [.7, .3], [.2, .8]]
        self.assertEqual(allocate(probs, [2, None], ["a", "b", "c"],
                                  "upper_bound_correction"), [0, 0, 1])

    def test_upper_bound_retains_highest_capped_scores_then_fills(self):
        probs = [[.9, .1], [.8, .2], [.7, .3], [.1, .9]]
        # Class 0 has three raw assignments but only two slots. The cleared
        # item is filled by the best feasible class-1 pair.
        self.assertEqual(allocate(probs, [2, None], ["a", "b", "c", "d"],
                                  "upper_bound_correction"), [0, 0, 1, 1])

    def test_capped_first_can_change_a_raw_argmax(self):
        probs = [[.9, .1], [.8, .2], [.7, .3]]
        self.assertEqual(allocate(probs, [1, None], ["a", "b", "c"],
                                  "capped_first"), [0, 1, 1])

    def test_zero_cap_is_real_and_none_is_unlimited(self):
        probs = [[.9, .1], [.2, .8]]
        self.assertEqual(allocate(probs, [0, None], ["a", "b"],
                                  "upper_bound_correction"), [1, 1])

    def test_multiple_caps_use_one_assignment_per_item(self):
        probs = [[.9, .05, .05], [.1, .85, .05], [.1, .2, .7]]
        self.assertEqual(allocate(probs, [1, 1, None], ["a", "b", "c"],
                                  "capped_first"), [0, 1, 2])

    def test_equal_ties_use_sample_id_then_class(self):
        probs = [[.5, .5], [.5, .5]]
        self.assertEqual(allocate(probs, [1, None], ["z", "a"],
                                  "capped_first"), [1, 0])

    def test_refuses_insufficient_total_capacity(self):
        with self.assertRaises(ValueError):
            allocate([[1.0, 0.0], [1.0, 0.0]], [1, 0], ["a", "b"],
                     "upper_bound_correction")

    def test_rejects_malformed_inputs(self):
        cases = [
            ([[.5, .5]], [True, None], ["a"], "capped_first"),
            ([[.5, .5]], [-1, None], ["a"], "capped_first"),
            ([[.5, .5]], [1, None], ["a", "b"], "capped_first"),
            ([[.5, .4]], [1, None], ["a"], "capped_first"),
            ([[float("nan"), 0.0]], [1, None], ["a"], "capped_first"),
            ([[.5, .5]], [1, None], ["a"], "unknown"),
        ]
        for args in cases:
            with self.subTest(args=args):
                with self.assertRaises((TypeError, ValueError)):
                    allocate(*args)


if __name__ == "__main__":
    unittest.main()
