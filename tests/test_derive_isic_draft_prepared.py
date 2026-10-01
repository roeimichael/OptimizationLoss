"""A faster decoder version must keep the old cohort and private labels intact."""

import hashlib
import json

import pytest
from PIL import Image

from analysis.derive_isic_draft_prepared import derive
from tralo.tabular_image_data import ISIC_DRAFT_POLICY, load_runner_cohort


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_derived_isic_cohort_preserves_split_and_private_targets(tmp_path):
    images = tmp_path / "images"
    images.mkdir()
    old = tmp_path / "old"
    (old / "runner").mkdir(parents=True)
    (old / "scorer").mkdir()
    records = {
        "train": [{"sample_id": "a", "file": "a.jpg", "group": "female", "label": 1}],
        "stop": [{"sample_id": "b", "file": "b.jpg", "group": "male", "label": 0}],
        "development_pool": [{"sample_id": "c", "file": "c.jpg", "group": "female"}],
    }
    for split, rows in records.items():
        (old / "runner" / f"{split}.jsonl").write_text(json.dumps(rows[0]) + "\n")
        Image.new("RGB", (1600, 1000)).save(images / rows[0]["file"])
    old_manifest = {
        "dataset": "isic2020", "image_dir": str(images), "image_count": 3,
        "files_sha256": {f"runner/{split}.jsonl": _sha(old / "runner" / f"{split}.jsonl")
                         for split in records},
        "supports": {"train/all": {"images": 1, "positive": 1},
                     "train/female": {"images": 1, "positive": 1},
                     "stop/all": {"images": 1, "positive": 0},
                     "stop/male": {"images": 1, "positive": 0},
                     "development/all": {"images": 1},
                     "development/female": {"images": 1}},
    }
    (old / "manifest.json").write_text(json.dumps(old_manifest))
    labels = old / "scorer/development_labels.jsonl"
    labels.write_text('{"sample_id":"c","label":1}\n')
    (old / "scorer/manifest.json").write_text(json.dumps({
        "runner_manifest_sha256": _sha(old / "manifest.json"),
        "development_labels_sha256": _sha(labels)}))
    new = tmp_path / "new"
    receipt = derive(old, new)
    derived, reread = load_runner_cohort(new, "isic2020")
    assert reread == records
    assert derived["decode_policy"] == ISIC_DRAFT_POLICY
    assert receipt["origin_sha256"] == _sha(old / "manifest.json")
    assert (new / "scorer/development_labels.jsonl").read_bytes() == labels.read_bytes()
    assert json.loads((new / "scorer/manifest.json").read_text())[
        "runner_manifest_sha256"] == _sha(new / "manifest.json")
    with pytest.raises(FileExistsError):
        derive(old, new)
