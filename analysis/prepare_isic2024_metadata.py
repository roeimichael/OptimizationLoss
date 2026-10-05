"""Diagnosis-blind ISIC2024 metadata stage, without a training cohort.

Usage: python -m analysis.prepare_isic2024_metadata ARCHIVE NEW_OUTPUT

Only embedded metadata and ZIP member names are accessed. This reproduces the
one existing prospective patient split; it does not approve or freeze it. No
diagnosis file, image payload, quota, runner manifest or training label is read
or emitted. Source integrity, full decoding and campaign gates remain separate.
"""

import csv
import hashlib
import io
import json
import re
import sys
import zipfile
from collections import Counter
from pathlib import Path


PREFIX = "ISIC_2024_Training_Input/"
METADATA_MEMBER = PREFIX + "metadata.csv"
CANDIDATE_NAMESPACE = "tralo-isic2024-patient-split-20261005"
PARTITIONS = ("train", "stop", "development")
STATUS = "candidate_only_not_frozen_not_training_ready"
IMAGE_ID = re.compile(r"ISIC_[0-9]{7}")


def _partition(patient):
    value = int.from_bytes(hashlib.sha256(
        (CANDIDATE_NAMESPACE + "|" + patient).encode("utf-8")
    ).digest()[:8], "big") % 10000
    return "train" if value < 7000 else "stop" if value < 8500 else "development"


def _source_rows(archive_path):
    if archive_path.is_symlink() or not archive_path.is_file():
        raise ValueError("regular archive file required")
    with zipfile.ZipFile(archive_path) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)):
            raise ValueError("duplicate ZIP member")
        images = set()
        for name in names:
            if name.startswith(PREFIX) and name.lower().endswith((".jpg", ".jpeg", ".png")):
                stem = name[len(PREFIX):-4]
                if not name.endswith(".jpg") or IMAGE_ID.fullmatch(stem) is None:
                    raise ValueError("unexpected image member")
                images.add(stem)
        # Do not extract the archive, open JPEGs, or read diagnosis CSVs.
        metadata = archive.read(METADATA_MEMBER)

    reader = csv.DictReader(io.StringIO(metadata.decode("utf-8-sig"), newline=""),
                            strict=True)
    header = reader.fieldnames or []
    forbidden = {"target", "label", "malignant", "diagnosis"}
    if (len(header) != len(set(header)) or
            not {"isic_id", "patient_id", "sex"}.issubset(header) or
            any(name.strip().lower() in forbidden or
                name.strip().lower().startswith("iddx") for name in header)):
        raise ValueError("unexpected or diagnosis-bearing metadata schema")
    rows, seen = [], set()
    for source in reader:
        if None in source or any(value is None for value in source.values()):
            raise ValueError("malformed CSV row")
        sample, patient = source["isic_id"], source["patient_id"]
        if IMAGE_ID.fullmatch(sample) is None or sample in seen:
            raise ValueError("malformed or repeated metadata ID")
        if not patient or patient != patient.strip():
            raise ValueError("nonempty unambiguous patient ID required")
        sex = source["sex"].strip().lower() or "missing"
        if sex not in ("female", "male", "missing"):
            raise ValueError("unexpected recorded sex")
        seen.add(sample)
        rows.append({"sample_id": sample, "image_member": PREFIX + sample + ".jpg",
                     "patient_id": patient, "group": sex, "split": _partition(patient)})
    if not rows:
        raise ValueError("empty metadata cohort")
    if seen != images:
        raise ValueError("image/metadata ID sets differ")
    return sorted(rows, key=lambda row: row["sample_id"]), metadata


def prepare_metadata_candidate(archive_path, output):
    """Write an exclusive label-free index, explicitly unsuitable for training.

    Extra nondiagnostic metadata is ignored. Recorded sex is not imputed across
    images of one patient; whole patients nevertheless share a split. No cohort
    size, target support or quota is selected by this stage.
    """
    archive_path, output = Path(archive_path), Path(output)
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    rows, metadata = _source_rows(archive_path)
    patients = {split: set() for split in PARTITIONS}
    groups = {split: Counter() for split in PARTITIONS}
    for row in rows:
        patients[row["split"]].add(row["patient_id"])
        groups[row["split"]][row["group"]] += 1
    if any(patients[a] & patients[b] for index, a in enumerate(PARTITIONS)
           for b in PARTITIONS[index + 1:]):
        raise ValueError("patient split overlap")

    row_bytes = "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows).encode("utf-8")
    image_index = "".join(row["image_member"] + "\n" for row in rows).encode("utf-8")
    manifest = {
        "dataset": "isic2024", "status": STATUS,
        "source_archive_path": str(archive_path.resolve()),
        "source_metadata_member": METADATA_MEMBER,
        "source_metadata_sha256": hashlib.sha256(metadata).hexdigest(),
        "source_image_member_index_sha256": hashlib.sha256(image_index).hexdigest(),
        "candidate_namespace": CANDIDATE_NAMESPACE,
        "candidate_split_rule": "sha256(namespace|patient_id)[:8] big-endian modulo10000: train<7000 stop<8500 development",
        "candidate_rows_sha256": hashlib.sha256(row_bytes).hexdigest(),
        "image_count": len(rows),
        "patient_count": sum(len(values) for values in patients.values()),
        "partitions": {split: {
            "images": sum(groups[split].values()), "patients": len(patients[split]),
            "groups": dict(sorted(groups[split].items())),
        } for split in PARTITIONS},
        "diagnosis_files_opened": False, "image_payloads_opened": False,
        "source_archive_integrity_verified": False,
        "image_decoding_verified": False,
        "target_support_verified": False,
        "scientific_protocol_approved": False,
    }
    output.mkdir(parents=True, exist_ok=False)
    with (output / "candidate_rows.jsonl").open("xb") as stream:
        stream.write(row_bytes)
    with (output / "metadata_candidate_manifest.json").open("x", encoding="utf-8") as stream:
        json.dump(manifest, stream, sort_keys=True, indent=2)
        stream.write("\n")
    return manifest


def main():
    if len(sys.argv) != 3:
        raise SystemExit("usage: python -m analysis.prepare_isic2024_metadata ARCHIVE NEW_OUTPUT")
    print(json.dumps(prepare_metadata_candidate(sys.argv[1], sys.argv[2]), sort_keys=True))


if __name__ == "__main__":
    main()
