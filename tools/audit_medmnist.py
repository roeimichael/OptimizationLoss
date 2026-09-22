"""Audit a staged dataset; no model training, score selection, or quota inference."""
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from tralo.medmnist_data import load_archive


def run(path, manifest_path, output):
    path, manifest_path, output = Path(path), Path(manifest_path), Path(output)
    manifest = json.loads(manifest_path.read_text())
    dataset = load_archive(path,manifest['md5'],manifest['samples'],
        manifest['resolution'],len(manifest['labels']),manifest['channels'])
    hashes, splits = {}, {}
    for split,(images,labels) in dataset.items():
        values = [hashlib.sha256(im.tobytes()).hexdigest() for im in images]
        hashes[split] = set(values)
        splits[split] = dict(shape=list(images.shape),dtype=str(images.dtype),
            class_support=[int((labels==c).sum()) for c in range(len(manifest['labels']))],
            exact_duplicate_rows=len(values)-len(set(values)),
            ordered_images_sha256=hashlib.sha256(images.tobytes()).hexdigest(),
            ordered_labels_sha256=hashlib.sha256(labels.tobytes()).hexdigest())
    intersections = {a+' / '+b:len(hashes[a]&hashes[b])
        for a in hashes for b in hashes if a < b}
    report = dict(status='archive_integrity_and_schema_passed',
        paper_replication_ready=False,archive_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        manifest_sha256=hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        splits=splits,cross_split_identical_image_hashes=intersections,
        limitation='Image-byte checks do not certify lesion or patient independence. '
        'Paper release/resolution/split and experiment settings remain unconfirmed.')
    output.mkdir(parents=True,exist_ok=False)
    (output/'audit.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report))


if __name__ == '__main__':
    run(*sys.argv[1:])
