"""Synthetic metadata tests for sealed, identity-disjoint dataset preparation."""

import csv
import json

import pytest

from analysis.prepare_tabular_pools import partition, prepare


def _units(dataset):
    found = {}
    for index in range(1000):
        name = f"unit{index}"
        found.setdefault(partition(dataset, name), name)
        if len(found) == 3:
            return found
    raise AssertionError("could not find each deterministic split")


def test_isic_duplicate_and_label_seal(tmp_path):
    raw = tmp_path / "isic"
    images = raw / "images" / "train"
    images.mkdir(parents=True)
    split_units = _units("isic2020")
    rows = []
    for index, split in enumerate(("train", "train", "stop", "development")):
        name = f"ISIC_{index:07d}"
        (images / (name + ".jpg")).write_bytes(b"synthetic")
        rows.append({"image_name": name, "patient_id": split_units[split],
                     "sex": "male" if index % 2 else "female",
                     "target": str(int(index > 1))})
    with (raw / "ISIC_2020_Training_GroundTruth_v2.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with (raw / "ISIC_2020_Training_Duplicates.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["image_name_1", "image_name_2"])
        writer.writerow([rows[0]["image_name"], rows[1]["image_name"]])
    output = tmp_path / "prepared"
    manifest = prepare("isic2020", raw, output)
    assert manifest["listed_duplicate_pairs"] == 1
    assert manifest["split_unit_counts"] == {"train": 1, "stop": 1,
                                              "development": 1}
    pool = (output / "runner/development_pool.jsonl").read_text()
    assert "label" not in pool and "target" not in pool
    assert manifest["supports"]["development/all"] == {"images": 1}
    assert "scorer/development_labels.jsonl" not in manifest["files_sha256"]
    private = json.loads((output / "scorer/manifest.json").read_text())
    assert private["development_supports"]["all"] == {"images": 1, "positive": 1}
    assert json.loads((output / "scorer/development_labels.jsonl").read_text()) == {
        "sample_id": rows[3]["image_name"], "label": 1}
    with pytest.raises(FileExistsError):
        prepare("isic2020", raw, output)
    with (raw / "ISIC_2020_Training_Duplicates.csv").open("a") as stream:
        stream.write(f"{rows[0]['image_name']},{rows[2]['image_name']}\n")
    with pytest.raises(ValueError, match="crosses split"):
        prepare("isic2020", raw, tmp_path / "bad")


def test_celeba_identity_join_and_label_seal(tmp_path):
    base = tmp_path / "celeba" / "celeba"
    images = base / "img_align_celeba"
    images.mkdir(parents=True)
    units = _units("celeba")
    names = ["000001.jpg", "000002.jpg", "000003.jpg"]
    for name in names:
        (images / name).write_bytes(b"synthetic")
    (base / "list_attr_celeba.txt").write_text(
        "3\nSmiling Male\n000001.jpg 1 -1\n000002.jpg -1 1\n000003.jpg 1 1\n")
    (base / "identity_CelebA.txt").write_text(
        "".join(f"{name} {units[split]}\n" for name, split in
                zip(names, ("train", "stop", "development"))))
    (base / "list_eval_partition.txt").write_text(
        "000001.jpg 0\n000002.jpg 1\n000003.jpg 2\n")
    output = tmp_path / "prepared"
    manifest = prepare("celeba", tmp_path / "celeba", output)
    assert manifest["image_count"] == 3
    assert manifest["split_unit_counts"] == {"train": 1, "stop": 1,
                                              "development": 1}
    pool = json.loads((output / "runner/development_pool.jsonl").read_text())
    assert pool == {"file": "000003.jpg", "group": "male",
                    "sample_id": "000003.jpg"}
    assert "label" not in pool
    assert manifest["supports"]["development/all"] == {"images": 1}
    assert json.loads((output / "scorer/manifest.json").read_text())[
        "development_supports"]["all"]["positive"] == 1
    (images / "extra.jpg").write_bytes(b"synthetic")
    with pytest.raises(ValueError, match="sets differ"):
        prepare("celeba", tmp_path / "celeba", tmp_path / "bad")
