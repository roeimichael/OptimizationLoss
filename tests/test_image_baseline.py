import hashlib
import json
import unittest

from tralo.image_baseline import validate_config, split_indices, config_digest, sample_ids


CONFIG = {
    "data_root": "data",
    "seed": 7,
    "train_samples": 12,
    "development_samples": 5,
    "image_size": 96,
    "feature_batch_size": 4,
    "head_batch_size": 3,
    "head_epochs": 2,
    "head_lr": 0.01,
    "precision": "fp32",
}


class ImageBaselineContractTests(unittest.TestCase):
    def test_validates_exact_config_and_returns_copy(self):
        checked = validate_config(CONFIG)
        self.assertEqual(checked, CONFIG)
        self.assertIsNot(checked, CONFIG)

    def test_rejects_unknown_missing_and_invalid_config_values(self):
        bad = dict(CONFIG, extra=True)
        with self.assertRaises(ValueError):
            validate_config(bad)
        bad = dict(CONFIG)
        del bad["precision"]
        with self.assertRaises(ValueError):
            validate_config(bad)
        for key, value in (("seed", True), ("train_samples", 0),
                           ("image_size", -1), ("head_lr", 0),
                           ("precision", "tf32")):
            with self.subTest(key=key):
                with self.assertRaises(ValueError):
                    validate_config(dict(CONFIG, **{key: value}))

    def test_split_is_seeded_disjoint_and_covers_requested_indices(self):
        train, development = split_indices(30, 12, 5, 7)
        self.assertEqual(len(train), 12)
        self.assertEqual(len(development), 5)
        self.assertEqual(len(set(train) & set(development)), 0)
        self.assertEqual((train, development), split_indices(30, 12, 5, 7))
        self.assertTrue(all(0 <= i < 30 for i in train + development))

    def test_split_rejects_overflow_and_invalid_counts(self):
        for args in ((10, 6, 5, 1), (10, 0, 1, 1), (10, 1, 0, 1),
                     (10, 1, 1, True), (10, 1, 1, -1)):
            with self.subTest(args=args):
                with self.assertRaises(ValueError):
                    split_indices(*args)

    def test_config_digest_is_canonical_and_sha256(self):
        digest = config_digest(CONFIG)
        expected = hashlib.sha256(json.dumps(CONFIG, sort_keys=True,
                                             separators=(",", ":")).encode()).hexdigest()
        self.assertEqual(digest, expected)
        self.assertEqual(digest, config_digest(dict(reversed(list(CONFIG.items())))))

    def test_sample_ids_are_string_stable_dataset_ids(self):
        self.assertEqual(sample_ids([0, 7, 123]), [
            "cifar100:train:000000", "cifar100:train:000007", "cifar100:train:000123",
        ])


if __name__ == "__main__":
    unittest.main()
