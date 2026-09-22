"""Read an explicitly identified MedMNIST archive without modifying its splits."""
import hashlib
from pathlib import Path


def load_archive(path, expected_md5, expected_samples, resolution, n_classes, channels):
    import numpy as np
    path = Path(path)
    data = path.read_bytes()
    if hashlib.md5(data).hexdigest() != expected_md5:
        raise ValueError('archive does not match the supplied release checksum')
    result = {}
    with np.load(path,allow_pickle=False) as archive:
        expected_keys = {s+'_'+kind for s in expected_samples for kind in ('images','labels')}
        if set(archive.files) != expected_keys:
            raise ValueError('archive split keys differ from the manifest')
        for split, count in expected_samples.items():
            images, labels = archive[split+'_images'], archive[split+'_labels']
            shape = (count,resolution,resolution,channels)
            if channels == 1: shape = shape[:-1]
            if images.shape != shape or images.dtype != np.uint8:
                raise ValueError('image shape/type mismatch: '+split)
            if (labels.shape != (count,1) or not np.issubdtype(labels.dtype,np.integer)
                    or (labels < 0).any() or (labels >= n_classes).any()):
                raise ValueError('label shape/type/range mismatch: '+split)
            result[split] = (images,labels[:,0])
    return result
