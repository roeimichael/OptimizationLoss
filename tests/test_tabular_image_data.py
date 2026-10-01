"""The training reader authenticates rows and refuses development targets."""

import hashlib
import json

import pytest
from PIL import Image

from tralo.tabular_image_data import (ISIC_DRAFT_POLICY, PreparedImageRows,
                                      load_runner_cohort)


def _fixture(tmp_path):
    images = tmp_path / "images"
    runner = tmp_path / "prepared" / "runner"
    images.mkdir()
    runner.mkdir(parents=True)
    rows = {
        "train": [{"sample_id": "a", "file": "a.jpg", "group": "female", "label": 0},
                  {"sample_id": "b", "file": "b.jpg", "group": "male", "label": 1}],
        "stop": [{"sample_id": "c", "file": "c.jpg", "group": "female", "label": 1}],
        "development_pool": [{"sample_id": "d", "file": "d.jpg", "group": "male"}],
    }
    hashes = {}
    for split, records in rows.items():
        path = runner / f"{split}.jsonl"
        path.write_text("".join(json.dumps(row) + "\n" for row in records))
        hashes[f"runner/{split}.jsonl"] = hashlib.sha256(path.read_bytes()).hexdigest()
        for row in records:
            Image.new("RGB", (4, 4), (255, 0, 0)).save(images / row["file"])
    manifest = {"dataset": "celeba", "image_dir": str(images), "image_count": 4,
                "files_sha256": hashes,
                "supports": {"train/all": {"images": 2, "positive": 1},
                             "train/female": {"images": 1, "positive": 0},
                             "train/male": {"images": 1, "positive": 1},
                             "stop/all": {"images": 1, "positive": 1},
                             "stop/female": {"images": 1, "positive": 1},
                             "development/all": {"images": 1},
                             "development/male": {"images": 1}}}
    (runner.parent / "manifest.json").write_text(json.dumps(manifest))
    return runner.parent, manifest


def test_prepared_reader_seals_dev_labels_and_preserves_external_groups(tmp_path):
    root, manifest = _fixture(tmp_path)
    loaded, rows = load_runner_cohort(root, "celeba")
    assert loaded == manifest
    assert len(rows["train"]) == 2 and "label" not in rows["development_pool"][0]
    data = PreparedImageRows(loaded["image_dir"], rows["development_pool"],
                             lambda image: image.size)
    assert data[0] == ((4, 4), -1, "male", "d")
    with pytest.raises(RuntimeError, match="wrong prepared dataset"):
        load_runner_cohort(root, "isic2020")


def test_public_support_leak_and_private_development_row_rejected(tmp_path):
    root, manifest = _fixture(tmp_path)
    manifest["supports"]["development/male"]["positive"] = 1
    (root / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(RuntimeError, match="exposes development outcomes"):
        load_runner_cohort(root, "celeba")
    del manifest["supports"]["development/male"]["positive"]
    path = root / "runner/development_pool.jsonl"
    path.write_text(json.dumps({"sample_id": "d", "file": "d.jpg",
                                "group": "male", "label": 1}) + "\n")
    manifest["files_sha256"]["runner/development_pool.jsonl"] = hashlib.sha256(
        path.read_bytes()).hexdigest()
    (root / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(RuntimeError, match="development label boundary"):
        load_runner_cohort(root, "celeba")


def test_isic_decoder_draft_is_explicit_and_rejects_wrong_data(tmp_path):
    path = tmp_path / "image.jpg"
    Image.new("RGB", (1600, 1000), (51, 91, 131)).save(path)
    row = [{"sample_id": "one", "file": path.name, "group": "female", "label": 1}]
    full = PreparedImageRows(tmp_path, row, lambda image: image.size)
    draft = PreparedImageRows(tmp_path, row, lambda image: image.size,
                              ISIC_DRAFT_POLICY)
    assert full[0][0] == (1600, 1000)
    assert draft[0][0][0] < 1600 and draft[0][0][1] >= 224
    with pytest.raises(ValueError, match="unfrozen"):
        PreparedImageRows(tmp_path, row, lambda image: image.size, "unknown")
    Image.new("RGB", (1600, 1000)).save(path, format="PNG")
    with pytest.raises(RuntimeError, match="requires JPEG"):
        draft[0]


def test_decoder_policy_cannot_be_applied_to_other_dataset(tmp_path):
    root, manifest = _fixture(tmp_path)
    manifest["decode_policy"] = ISIC_DRAFT_POLICY
    (root / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(RuntimeError, match="decoder policy"):
        load_runner_cohort(root, "celeba")
