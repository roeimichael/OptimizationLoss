"""Synthetic examples for source authentication and private label separation."""

import csv
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import zipfile

import pytest
from PIL import Image

from analysis.prepare_isic2024_labels import prepare_label_candidate
from analysis.prepare_isic2024_metadata import prepare_metadata_candidate
from tralo.tabular_image_data import PreparedImageRows, load_runner_cohort


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fixture(tmp_path, labels=None, header=None):
    # Independent fixed routes: patient1 -> train, patient0 -> stop, patient2 -> dev.
    rows = [("ISIC_0000001", "patient1", "female"),
            ("ISIC_0000002", "patient0", "male"),
            ("ISIC_0000003", "patient2", ""),
            ("ISIC_0000004", "patient1", "female")]
    stream = io.StringIO(newline="")
    writer = csv.writer(stream)
    writer.writerow(["isic_id", "patient_id", "sex"])
    writer.writerows(rows)
    archive = tmp_path / "synthetic.zip"
    with zipfile.ZipFile(archive, "w") as z:
        z.writestr("ISIC_2024_Training_Input/metadata.csv", stream.getvalue())
        for sample, _, _ in rows:
            z.writestr("ISIC_2024_Training_Input/" + sample + ".jpg", b"unread image")
    metadata = tmp_path / "metadata"
    prepare_metadata_candidate(archive, metadata)
    labels = labels if labels is not None else [
        ["ISIC_0000004", "0"], ["ISIC_0000003", "1"],
        ["ISIC_0000002", "0"], ["ISIC_0000001", "1"]]
    target = tmp_path / "synthetic_labels.csv"
    with target.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header or ["isic_id", "malignant"])
        writer.writerows(labels)
    return metadata, target


def _prepare(metadata, target, output, manifest_sha=None, label_sha=None):
    return prepare_label_candidate(
        metadata, target,
        manifest_sha or _sha(metadata / "metadata_candidate_manifest.json"),
        label_sha or _sha(target), output)


def _rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_exact_routing_and_private_public_boundary(tmp_path):
    metadata, target = _fixture(tmp_path)
    output = tmp_path / "labels"
    manifest = _prepare(metadata, target, output)
    assert _rows(output / "public/train.jsonl") == [
        {"sample_id": "ISIC_0000001", "file": "ISIC_0000001.jpg", "group": "female", "label": 1},
        {"sample_id": "ISIC_0000004", "file": "ISIC_0000004.jpg", "group": "female", "label": 0}]
    assert _rows(output / "public/stop.jsonl") == [
        {"sample_id": "ISIC_0000002", "file": "ISIC_0000002.jpg", "group": "male", "label": 0}]
    assert _rows(output / "public/development_pool.jsonl") == [
        {"sample_id": "ISIC_0000003", "file": "ISIC_0000003.jpg", "group": "missing"}]
    assert _rows(output / "private/development_labels.jsonl") == [
        {"sample_id": "ISIC_0000003", "label": 1}]
    assert manifest["status"] == "candidate_only_not_frozen_not_training_ready"
    assert manifest["files_sha256"] == {
        "public/" + name + ".jsonl": _sha(output / "public" / (name + ".jsonl"))
        for name in ["train", "stop", "development_pool"]}
    assert manifest["supports"] == {
        "train": {"images": 2, "groups": {"female": 2}},
        "stop": {"images": 1, "groups": {"male": 1}},
        "development": {"images": 1, "groups": {"missing": 1}}}
    assert not any("private" in key for key in manifest["files_sha256"])
    assert "positive" not in json.dumps(manifest["supports"])
    private = json.loads((output / "private/label_candidate_manifest.json").read_text())
    assert private["public_manifest_sha256"] == _sha(output / "label_candidate_manifest.json")
    assert private["development_labels_sha256"] == _sha(output / "private/development_labels.jsonl")
    assert not (output / "manifest.json").exists()
    with pytest.raises(FileNotFoundError):
        load_runner_cohort(output, "isic2024")


def test_development_label_change_cannot_change_public_rows_or_support(tmp_path):
    metadata, target = _fixture(tmp_path)
    a = _prepare(metadata, target, tmp_path / "a")
    target.write_text(target.read_text().replace("ISIC_0000003,1", "ISIC_0000003,0"))
    b = _prepare(metadata, target, tmp_path / "b")
    for name in ["train", "stop", "development_pool"]:
        assert (tmp_path / "a/public" / (name + ".jsonl")).read_bytes() == (
            tmp_path / "b/public" / (name + ".jsonl")).read_bytes()
    assert a["supports"] == b["supports"]
    assert _rows(tmp_path / "b/private/development_labels.jsonl")[0]["label"] == 0


@pytest.mark.parametrize("change,match", [
    ("manifest_hash", "manifest hash"), ("rows_hash", "row hash"),
    ("wrong_namespace", "namespace"), ("duplicate_sample", "identity"),
    ("wrong_patient_split", "patient split"), ("label_in_metadata", "row schema"),
    ("unknown_group", "group"), ("image_path", "image member"),
])
def test_metadata_refused_before_any_label_file_access(tmp_path, monkeypatch, change, match):
    metadata, target = _fixture(tmp_path)
    manifest_path, rows_path = metadata / "metadata_candidate_manifest.json", metadata / "candidate_rows.jsonl"
    manifest = json.loads(manifest_path.read_text())
    rows = _rows(rows_path)
    manifest_sha = _sha(manifest_path)
    if change == "manifest_hash":
        manifest_sha = "0" * 64
    elif change == "rows_hash":
        rows[0]["group"] = "male"
    elif change == "wrong_namespace":
        manifest["candidate_namespace"] = "different_split"
    elif change == "duplicate_sample":
        rows[1]["sample_id"] = rows[0]["sample_id"]
    elif change == "wrong_patient_split":
        rows[0]["split"] = "development"
    elif change == "label_in_metadata":
        rows[0]["label"] = 1
    elif change == "unknown_group":
        rows[0]["group"] = "guessed"
    elif change == "image_path":
        rows[0]["image_member"] = "../ISIC_0000001.jpg"
    if change not in ("manifest_hash", "wrong_namespace"):
        rows_path.write_text("".join(json.dumps(row) + "\n" for row in rows))
        if change != "rows_hash":
            manifest["candidate_rows_sha256"] = _sha(rows_path)
    if change != "manifest_hash":
        manifest_path.write_text(json.dumps(manifest))
        manifest_sha = _sha(manifest_path)
    label_sha = _sha(target)
    original = Path.open

    def denied_labels(path, *args, **kwargs):
        assert path != target, "labels opened before metadata rejection"
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", denied_labels)
    with pytest.raises(ValueError, match=match):
        _prepare(metadata, target, tmp_path / "out", manifest_sha, label_sha)
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize("labels,header,match", [
    ([["ISIC_0000001", "1"]], None, "ID sets"),
    ([["ISIC_0000001", "1"], ["ISIC_0000001", "0"]], None, "repeated"),
    ([["ISIC_0000001", "1.0"]], None, "binary"),
    ([["ISIC_0000001", "True"]], None, "binary"),
    ([["ISIC_0000001", ""]], None, "binary"),
    ([["ISIC_0000001", "1", "extra"]], None, "CSV row"),
    ([["ISIC_0000001"]], None, "CSV row"),
    ([["ISIC_0000001", "1"]], ["isic_id", "diagnosis"], "label schema"),
    ([["ISIC_0000001", "1"]], ["isic_id", "isic_id"], "label schema"),
])
def test_invalid_labels_fail_without_outputs(tmp_path, labels, header, match):
    metadata, target = _fixture(tmp_path, labels=labels, header=header)
    with pytest.raises(ValueError, match=match):
        _prepare(metadata, target, tmp_path / "out")
    assert not (tmp_path / "out").exists()


def test_label_hash_and_repeat_refusal(tmp_path, monkeypatch):
    metadata, target = _fixture(tmp_path)
    with pytest.raises(ValueError, match="label hash"):
        _prepare(metadata, target, tmp_path / "wrong", label_sha="0" * 64)
    assert not (tmp_path / "wrong").exists()
    output = tmp_path / "out"
    _prepare(metadata, target, output)
    before = {str(p.relative_to(output)): p.read_bytes() for p in output.rglob("*") if p.is_file()}
    with pytest.raises(FileExistsError):
        _prepare(metadata, target, output)
    assert before == {str(p.relative_to(output)): p.read_bytes() for p in output.rglob("*") if p.is_file()}


def test_actual_cli_routes_synthetic_rows_and_refuses_repeat(tmp_path):
    metadata, target = _fixture(tmp_path)
    output = tmp_path / "out"
    command = [sys.executable, "-m", "analysis.prepare_isic2024_labels", str(metadata), str(target),
               _sha(metadata / "metadata_candidate_manifest.json"), _sha(target), str(output)]
    run = subprocess.run(command, capture_output=True, text=True, timeout=30)
    assert run.returncode == 0, run.stderr
    assert json.loads(run.stdout)["status"] == "candidate_only_not_frozen_not_training_ready"
    repeat = subprocess.run(command, capture_output=True, text=True, timeout=30)
    assert repeat.returncode != 0 and "FileExistsError" in repeat.stderr


def test_parsing_uses_the_authenticated_bytes_without_reopening_text(tmp_path, monkeypatch):
    metadata, target = _fixture(tmp_path)
    manifest_sha, label_sha = _sha(metadata / "metadata_candidate_manifest.json"), _sha(target)
    original = Path.read_text

    def no_second_read(path, *args, **kwargs):
        assert path != target and path.parent != metadata, "authenticated source reopened for parsing"
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", no_second_read)
    result = _prepare(metadata, target, tmp_path / "out", manifest_sha, label_sha)
    assert result["supports"]["development"]["images"] == 1


def test_existing_runner_interface_never_opens_private_candidate_labels(tmp_path, monkeypatch):
    metadata, target = _fixture(tmp_path)
    candidate = tmp_path / "candidate"
    _prepare(metadata, target, candidate)
    # This bridge is a test fixture only. Product preparation deliberately emits
    # no runtime manifest, and this does not approve/register a real dataset.
    root, images = tmp_path / "synthetic_runtime", tmp_path / "synthetic_images"
    (root / "runner").mkdir(parents=True)
    images.mkdir()
    hashes, supports = {}, {}
    for name in ["train", "stop", "development_pool"]:
        source = candidate / "public" / (name + ".jsonl")
        destination = root / "runner" / (name + ".jsonl")
        destination.write_bytes(source.read_bytes())
        hashes["runner/" + name + ".jsonl"] = _sha(destination)
        rows = _rows(destination)
        key = "development" if name == "development_pool" else name
        supports[key + "/all"] = {"images": len(rows)}
        for group in {row["group"] for row in rows}:
            current = [row for row in rows if row["group"] == group]
            supports[key + "/" + group] = {"images": len(current)}
            if name != "development_pool":
                supports[key + "/" + group]["positive"] = sum(row["label"] for row in current)
        for row in rows:
            Image.new("RGB", (4, 4), (50, 90, 130)).save(images / row["file"])
    (root / "manifest.json").write_text(json.dumps({
        "dataset": "synthetic_isic2024_interface", "image_dir": str(images), "image_count": 4,
        "files_sha256": hashes, "supports": supports}))
    original = Path.open

    def deny_private(path, *args, **kwargs):
        assert "private" not in path.parts, "runner accessed scorer-only candidate labels"
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", deny_private)
    _, rows = load_runner_cohort(root, "synthetic_isic2024_interface")
    data = PreparedImageRows(images, rows["development_pool"], lambda image: (image.mode, image.size))
    assert data[0] == (("RGB", (4, 4)), -1, "missing", "ISIC_0000003")
