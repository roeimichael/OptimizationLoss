"""Authenticate and separate label candidate rows; not a frozen training adapter.

Usage: python -m analysis.prepare_isic2024_labels METADATA_CANDIDATE LABEL_CSV
       METADATA_MANIFEST_SHA256 LABEL_CSV_SHA256 NEW_OUTPUT

Only synthetic inputs have been validated. Real diagnosis access requires the
separate scientific/data-access review. Hash arguments authenticate bytes, not
approval. This emits no runtime manifest, quota registration or launch recipe.
"""

from collections import Counter
import csv
import hashlib
import io
import json
from pathlib import Path
import re
import sys

from analysis.prepare_isic2024_metadata import (
    CANDIDATE_NAMESPACE, IMAGE_ID, PARTITIONS, PREFIX, STATUS, _partition)


def _digest(path):
    if path.is_symlink() or not path.is_file():
        raise ValueError("regular input file required")
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _authenticate(path, expected, name):
    if type(expected) is not str or re.fullmatch(r"[0-9a-f]{64}", expected) is None:
        raise ValueError(name + " hash must be a complete SHA256")
    if path.is_symlink() or not path.is_file():
        raise ValueError("regular input file required")
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected:
        raise ValueError(name + " hash mismatch")
    return raw


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("repeated JSON key")
        result[key] = value
    return result


def _metadata(root, expected_manifest):
    if root.is_symlink() or not root.is_dir():
        raise ValueError("regular metadata candidate directory required")
    path = root / "metadata_candidate_manifest.json"
    manifest = json.loads(_authenticate(path, expected_manifest, "manifest").decode("utf-8"),
                          object_pairs_hook=_unique_object)
    if (manifest.get("dataset") != "isic2024" or manifest.get("status") != STATUS or
            manifest.get("scientific_protocol_approved") is not False):
        raise ValueError("metadata candidate status changed")
    if manifest.get("candidate_namespace") != CANDIDATE_NAMESPACE:
        raise ValueError("candidate namespace changed")
    path = root / "candidate_rows.jsonl"
    raw = _authenticate(path, manifest.get("candidate_rows_sha256"), "row")
    rows, identities = [], set()
    with io.StringIO(raw.decode("utf-8")) as stream:
        for line in stream:
            row = json.loads(line, object_pairs_hook=_unique_object)
            if set(row) != {"sample_id", "image_member", "patient_id", "group", "split"}:
                raise ValueError("metadata row schema changed")
            sample, patient = row["sample_id"], row["patient_id"]
            if (type(sample) is not str or IMAGE_ID.fullmatch(sample) is None or
                    sample in identities):
                raise ValueError("malformed or duplicate sample identity")
            if type(patient) is not str or not patient or patient != patient.strip():
                raise ValueError("unambiguous patient identity required")
            if row["split"] != _partition(patient):
                raise ValueError("candidate patient split changed")
            if row["group"] not in ("female", "male", "missing"):
                raise ValueError("candidate group changed")
            if row["image_member"] != PREFIX + sample + ".jpg":
                raise ValueError("candidate image member changed")
            identities.add(sample)
            rows.append(row)
    if not rows or len(rows) != manifest.get("image_count"):
        raise ValueError("candidate image count changed")
    if len({r["patient_id"] for r in rows}) != manifest.get("patient_count"):
        raise ValueError("candidate patient count changed")
    for split in PARTITIONS:
        current = [r for r in rows if r["split"] == split]
        geometry = dict(images=len(current), patients=len({r["patient_id"] for r in current}),
                        groups=dict(Counter(r["group"] for r in current)))
        if not current or manifest.get("partitions", {}).get(split) != geometry:
            raise ValueError("candidate partition support changed")
    return sorted(rows, key=lambda row: row["sample_id"])


def _labels(path, expected, identities):
    raw = _authenticate(path, expected, "label")
    reader = csv.DictReader(io.StringIO(raw.decode("utf-8-sig"), newline=""), strict=True)
    if len(reader.fieldnames or []) != 2 or set(reader.fieldnames) != {"isic_id", "malignant"}:
        raise ValueError("unexpected label schema")
    labels = {}
    for row in reader:
        if None in row or any(value is None for value in row.values()):
            raise ValueError("malformed label CSV row")
        sample = row["isic_id"]
        if IMAGE_ID.fullmatch(sample) is None or sample in labels:
            raise ValueError("malformed or repeated label ID")
        if row["malignant"] not in ("0", "1"):
            raise ValueError("official binary label must be exactly 0 or 1")
        labels[sample] = int(row["malignant"])
    if set(labels) != identities:
        raise ValueError("label/metadata ID sets differ")
    return labels


def _write_rows(path, rows):
    raw = "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows).encode("utf-8")
    with path.open("xb") as stream:
        stream.write(raw)
    return hashlib.sha256(raw).hexdigest()


def prepare_label_candidate(metadata_root, label_csv, metadata_manifest_sha256,
                            label_csv_sha256, output):
    """Separate authenticated candidate labels without approving data or training.

    Metadata is fully checked before the label file is opened. Development
    targets never enter public row files or support summaries. Input SHA256s
    must come from independent provenance; supplying a hash grants no access.
    """
    metadata_root, label_csv, output = Path(metadata_root), Path(label_csv), Path(output)
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    rows = _metadata(metadata_root, metadata_manifest_sha256)
    labels = _labels(label_csv, label_csv_sha256, {r["sample_id"] for r in rows})
    public = {split: [] for split in PARTITIONS}
    private = []
    for row in rows:
        item = {"sample_id": row["sample_id"], "file": row["sample_id"] + ".jpg", "group": row["group"]}
        if row["split"] == "development":
            private.append({"sample_id": row["sample_id"], "label": labels[row["sample_id"]]})
        else:
            item["label"] = labels[row["sample_id"]]
        public[row["split"]].append(item)
    output.mkdir(parents=True, exist_ok=False)
    (output / "public").mkdir()
    (output / "private").mkdir()
    files = {}
    for split, values in public.items():
        name = "development_pool" if split == "development" else split
        relative = "public/" + name + ".jsonl"
        files[relative] = _write_rows(output / relative, values)
    private_hash = _write_rows(output / "private/development_labels.jsonl", private)
    manifest = dict(dataset="isic2024", status=STATUS,
                    metadata_manifest_sha256=metadata_manifest_sha256,
                    label_source_sha256=label_csv_sha256, files_sha256=files,
                    supports={split: dict(images=len(values), groups=dict(Counter(
                        row["group"] for row in values))) for split, values in public.items()},
                    scientific_protocol_approved=False, source_archive_integrity_verified=False,
                    image_decoding_verified=False, training_adapter_registered=False)
    manifest_path = output / "label_candidate_manifest.json"
    with manifest_path.open("x", encoding="utf-8") as stream:
        json.dump(manifest, stream, sort_keys=True, indent=2); stream.write("\n")
    private_manifest = dict(status=STATUS, public_manifest_sha256=_digest(manifest_path),
                            development_labels_sha256=private_hash,
                            label_source_sha256=label_csv_sha256)
    with (output / "private/label_candidate_manifest.json").open("x", encoding="utf-8") as stream:
        json.dump(private_manifest, stream, sort_keys=True, indent=2); stream.write("\n")
    return manifest


if __name__ == "__main__":
    if len(sys.argv) != 6:
        raise SystemExit("usage: python -m analysis.prepare_isic2024_labels METADATA_CANDIDATE LABEL_CSV METADATA_MANIFEST_SHA256 LABEL_CSV_SHA256 NEW_OUTPUT")
    print(json.dumps(prepare_label_candidate(*sys.argv[1:]), sort_keys=True))
