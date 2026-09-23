import tempfile
import unittest
from pathlib import Path
from PIL import Image
from tralo.knee_data import audit


class KneeDataTests(unittest.TestCase):
    def fixture(self, root):
        for i, split in enumerate(('train', 'val', 'test')):
            for c in range(5):
                p = root / split / str(c) / f'{9000000+i*10+c}L.png'
                p.parent.mkdir(parents=True, exist_ok=True)
                Image.new('L', (8, 8), 10+i*5+c).save(p)

    def test_counts_and_identity(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); self.fixture(root)
            result = audit(root)
            self.assertEqual(result['counts'], {'train': 5, 'val': 5, 'test': 5})
            self.assertEqual(result['rows'][0]['subject'], '9000000')

    def test_same_patient_other_knee_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); self.fixture(root)
            Image.new('L', (8, 8), 90).save(root/'val'/'0'/'9000000R.png')
            with self.assertRaisesRegex(ValueError, 'subject overlap'): audit(root)

    def test_identical_pixels_different_id_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); self.fixture(root)
            Image.new('L', (8, 8), 10).save(root/'val'/'0'/'9999999L.png')
            with self.assertRaisesRegex(ValueError, 'pixel overlap'): audit(root)
