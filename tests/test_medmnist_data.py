import hashlib
from pathlib import Path
import tempfile
import unittest

import numpy as np
from tralo.medmnist_data import load_archive


class MedMNISTDataTests(unittest.TestCase):
    def archive(self, **changes):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        path = Path(directory.name)/'fixture.npz'
        data = dict(train_images=np.zeros((3,4,4,3),dtype=np.uint8),
                    train_labels=np.array([[0],[1],[0]],dtype=np.int64))
        data.update(changes)
        np.savez(path,**data)
        return path,hashlib.md5(path.read_bytes()).hexdigest()

    def test_preserves_order_and_labels(self):
        p,h = self.archive()
        images,labels = load_archive(p,h,{'train':3},4,2,3)['train']
        self.assertEqual(images.shape,(3,4,4,3))
        self.assertEqual(labels.tolist(),[0,1,0])

    def test_wrong_release_is_rejected(self):
        p,h = self.archive()
        with self.assertRaises(ValueError): load_archive(p,'0'*32,{'train':3},4,2,3)

    def test_misaligned_labels_and_wrong_resolution_rejected(self):
        p,h = self.archive(train_labels=np.array([[0],[1]]))
        with self.assertRaises(ValueError): load_archive(p,h,{'train':3},4,2,3)
        p,h = self.archive()
        with self.assertRaises(ValueError): load_archive(p,h,{'train':3},8,2,3)

    def test_label_domain_and_unexpected_splits_rejected(self):
        p,h = self.archive(train_labels=np.array([[0],[2],[0]]))
        with self.assertRaises(ValueError): load_archive(p,h,{'train':3},4,2,3)
        p,h = self.archive(other=np.zeros(1))
        with self.assertRaises(ValueError): load_archive(p,h,{'train':3},4,2,3)
