"""Independent synthetic examples for diagnosis-blind ISIC2024 preparation."""

import csv
import hashlib
import io
import json
import subprocess
import sys
import warnings
import zipfile

import pytest

from analysis.prepare_isic2024_metadata import prepare_metadata_candidate
from tralo.tabular_image_data import load_runner_cohort
from tralo.tabular_quota_policy import caps_for_unlabeled_pool


PREFIX = "ISIC_2024_Training_Input/"
METADATA = PREFIX + "metadata.csv"


def _rows():
    # Independently calculated hash buckets: patient1=3553, patient0=8446,
    # patient2=9616. Two images of patient1 must remain in the same partition.
    return [
        ["ISIC_0000001", "patient1", "female", "unused"],
        ["ISIC_0000002", "patient0", "male", "unused"],
        ["ISIC_0000003", "patient2", "", "unused"],
        ["ISIC_0000004", "patient1", "female", "unused"],
    ]


def _archive(path, rows=None, names=None, header=None, reverse=False):
    rows = _rows() if rows is None else rows
    header = header or ["isic_id", "patient_id", "sex", "extra_feature"]
    stream = io.StringIO(newline="")
    writer = csv.writer(stream)
    writer.writerow(header)
    writer.writerows(reversed(rows) if reverse else rows)
    metadata = stream.getvalue().encode("utf-8")
    names = names if names is not None else [PREFIX + row[0] + ".jpg" for row in rows]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        with zipfile.ZipFile(path, "w") as archive:
            archive.writestr(METADATA, metadata)
            for name in reversed(names) if reverse else names:
                # Deliberately not a decodable image: metadata stage never decodes.
                archive.writestr(name, b"image payload must not be read")
            archive.writestr("ISIC_2024_Training_GroundTruth.csv", b"DO NOT OPEN")
            archive.writestr("ISIC_2024_Training_Supplement.csv", b"DO NOT OPEN")
    return metadata


def _read_rows(output):
    return [json.loads(line) for line in (output / "candidate_rows.jsonl").read_text().splitlines()]


def test_patient_routing_and_read_boundary(tmp_path, monkeypatch):
    raw, output = tmp_path / "source.zip", tmp_path / "candidate"
    metadata = _archive(raw)
    opened = []
    original = zipfile.ZipFile.open

    def metadata_only(self, name, *args, **kwargs):
        opened.append(name)
        assert name == METADATA, "image or diagnosis content was accessed"
        return original(self, name, *args, **kwargs)

    monkeypatch.setattr(zipfile.ZipFile, "open", metadata_only)
    manifest = prepare_metadata_candidate(raw, output)
    rows = _read_rows(output)
    assert opened == [METADATA]
    assert [(row["sample_id"], row["patient_id"], row["group"], row["split"])
            for row in rows] == [
        ("ISIC_0000001", "patient1", "female", "train"),
        ("ISIC_0000002", "patient0", "male", "stop"),
        ("ISIC_0000003", "patient2", "missing", "development"),
        ("ISIC_0000004", "patient1", "female", "train"),
    ]
    assert manifest["image_count"] == 4
    assert manifest["patient_count"] == 3
    assert manifest["partitions"] == {
        "train": {"images": 2, "patients": 1, "groups": {"female": 2}},
        "stop": {"images": 1, "patients": 1, "groups": {"male": 1}},
        "development": {"images": 1, "patients": 1, "groups": {"missing": 1}},
    }
    assert manifest["source_metadata_sha256"] == hashlib.sha256(metadata).hexdigest()
    assert manifest["candidate_rows_sha256"] == hashlib.sha256(
        (output / "candidate_rows.jsonl").read_bytes()).hexdigest()
    assert manifest["diagnosis_files_opened"] is False
    assert manifest["image_payloads_opened"] is False
    assert manifest["status"] == "candidate_only_not_frozen_not_training_ready"
    assert all(set(row) == {"sample_id", "image_member", "patient_id", "group", "split"}
               for row in rows)
    # No runtime manifest or labels are emitted, so this cannot become a campaign.
    assert sorted(path.name for path in output.iterdir()) == [
        "candidate_rows.jsonl", "metadata_candidate_manifest.json"]
    with pytest.raises(FileNotFoundError):
        load_runner_cohort(output, "isic2024")
    with pytest.raises(ValueError):
        caps_for_unlabeled_pool("isic2024", ["female", "male"])


def test_order_independence_and_exclusive_outputs(tmp_path):
    first, second = tmp_path / "first.zip", tmp_path / "second.zip"
    _archive(first)
    _archive(second, reverse=True)
    out_a, out_b = tmp_path / "a", tmp_path / "b"
    a = prepare_metadata_candidate(first, out_a)
    b = prepare_metadata_candidate(second, out_b)
    assert _read_rows(out_a) == _read_rows(out_b)
    assert a["candidate_rows_sha256"] == b["candidate_rows_sha256"]
    before = {path.name: path.read_bytes() for path in out_a.iterdir()}
    with pytest.raises(FileExistsError):
        prepare_metadata_candidate(first, out_a)
    assert before == {path.name: path.read_bytes() for path in out_a.iterdir()}


@pytest.mark.parametrize("change,match", [
    ("duplicate_id", "metadata ID"),
    ("empty_patient", "patient"),
    ("ambiguous_patient", "patient"),
    ("unknown_sex", "sex"),
    ("path_id", "metadata ID"),
    ("short_row", "CSV row"),
    ("long_row", "CSV row"),
    ("missing_image", "image/metadata"),
    ("extra_image", "image/metadata"),
    ("duplicate_member", "duplicate ZIP"),
    ("nested_image", "image member"),
])
def test_invalid_source_is_rejected_before_output(tmp_path, change, match):
    rows = _rows()
    names = [PREFIX + row[0] + ".jpg" for row in rows]
    if change == "duplicate_id":
        rows[1][0] = rows[0][0]
    elif change == "empty_patient":
        rows[0][1] = ""
    elif change == "ambiguous_patient":
        rows[0][1] = " patient1 "
    elif change == "unknown_sex":
        rows[0][2] = "not-recorded-category"
    elif change == "path_id":
        rows[0][0] = "../ISIC_0000001"
    elif change == "short_row":
        rows[0].pop()
    elif change == "long_row":
        rows[0].append("extra")
    elif change == "missing_image":
        names.pop()
    elif change == "extra_image":
        names.append(PREFIX + "ISIC_9999999.jpg")
    elif change == "duplicate_member":
        names.append(names[0])
    elif change == "nested_image":
        names[0] = PREFIX + "nested/ISIC_0000001.jpg"
    raw, output = tmp_path / "bad.zip", tmp_path / "output"
    _archive(raw, rows=rows, names=names)
    with pytest.raises(ValueError, match=match):
        prepare_metadata_candidate(raw, output)
    assert not output.exists()


@pytest.mark.parametrize("header", [
    ["isic_id", "patient_id", "sex", "malignant"],
    ["isic_id", "patient_id", "sex", "target"],
    ["isic_id", "patient_id", "sex", "iddx_full"],
    ["isic_id", "patient_id", "sex", "sex"],
    ["isic_id", "other_patient", "sex", "extra_feature"],
])
def test_diagnosis_or_ambiguous_metadata_schema_refused(tmp_path, header):
    raw, output = tmp_path / "bad.zip", tmp_path / "output"
    _archive(raw, header=header)
    with pytest.raises(ValueError, match="metadata schema"):
        prepare_metadata_candidate(raw, output)
    assert not output.exists()


def test_empty_metadata_refused(tmp_path):
    raw, output = tmp_path / "empty.zip", tmp_path / "output"
    _archive(raw, rows=[])
    with pytest.raises(ValueError, match="empty"):
        prepare_metadata_candidate(raw, output)
    assert not output.exists()


def test_real_cli_candidate_and_repeat_refusal(tmp_path):
    raw, output = tmp_path / "source.zip", tmp_path / "candidate"
    _archive(raw)
    command = [sys.executable, "-m", "analysis.prepare_isic2024_metadata",
               str(raw), str(output)]
    result = subprocess.run(command, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["status"] == "candidate_only_not_frozen_not_training_ready"
    repeated = subprocess.run(command, capture_output=True, text=True, timeout=30)
    assert repeated.returncode != 0
    assert "FileExistsError" in repeated.stderr
