"""Freeze image/metadata cohorts before a local-constraint training campaign.

Usage: python analysis/prepare_tabular_pools.py isic2020|celeba RAW_ROOT NEW_OUTPUT_ROOT

Only public, already downloaded data are read. The runner receives labels for
train/stop and label-free development IDs; development labels are written to a
separate scorer file. This tool does not train or select a quota or setting.
"""

import csv
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path


SPLIT_RULE = "sha256(namespace|unit)[:8] big-endian modulo10000: train<7000 stop<8500 dev"
NAMESPACES = {
    "isic2020": "tralo-isic-patient-split-20261001",
    "celeba": "tralo-celeba-identity-split-20261001",
}


def digest(path):
    sha = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            sha.update(chunk)
    return sha.hexdigest()


def partition(dataset, unit):
    if dataset not in NAMESPACES or not isinstance(unit, str) or not unit:
        raise ValueError("dataset and nonempty split unit required")
    value = int.from_bytes(hashlib.sha256(
        (NAMESPACES[dataset] + "|" + unit).encode()).digest()[:8], "big") % 10000
    return "train" if value < 7000 else "stop" if value < 8500 else "development"


def _image_names(directory, suffix):
    if not directory.is_dir():
        raise FileNotFoundError(directory)
    files = {path.name for path in directory.glob("*" + suffix)}
    if not files:
        raise ValueError("image directory empty")
    return files


def _isic2020(root):
    meta = root / "ISIC_2020_Training_GroundTruth_v2.csv"
    duplicates = root / "ISIC_2020_Training_Duplicates.csv"
    images = root / "images" / "train"
    records = []
    seen = set()
    lesion_splits = {}
    with meta.open(newline="", encoding="utf-8-sig") as stream:
        for source in csv.DictReader(stream):
            image = source["image_name"]
            patient = source["patient_id"]
            label = source["target"]
            if not image or image in seen or not patient or label not in ("0", "1"):
                raise ValueError("ISIC image, patient or target malformed")
            seen.add(image)
            sex = (source.get("sex") or "missing").strip().lower()
            if sex not in ("male", "female", "missing"):
                raise ValueError("unexpected ISIC sex metadata")
            lesion = (source.get("lesion_id") or "").strip()
            if lesion:
                split = partition("isic2020", patient)
                if lesion in lesion_splits and lesion_splits[lesion] != split:
                    raise ValueError("ISIC lesion crosses patient split")
                lesion_splits[lesion] = split
            records.append(dict(sample_id=image, file=image + ".jpg",
                                unit=patient, group=sex, label=int(label)))
    if {row["file"] for row in records} != _image_names(images, ".jpg"):
        raise ValueError("ISIC image/metadata names differ")
    split_by_id = {row["sample_id"]: partition("isic2020", row["unit"])
                   for row in records}
    label_by_id = {row["sample_id"]: row["label"] for row in records}
    duplicate_count = 0
    with duplicates.open(newline="", encoding="utf-8-sig") as stream:
        for pair in csv.DictReader(stream):
            a, b = pair["image_name_1"], pair["image_name_2"]
            if a not in split_by_id or b not in split_by_id:
                raise ValueError("listed ISIC duplicate not in metadata")
            if split_by_id[a] != split_by_id[b]:
                raise ValueError("listed ISIC duplicate crosses split")
            if label_by_id[a] != label_by_id[b]:
                raise ValueError("listed ISIC duplicate has conflicting targets")
            duplicate_count += 1
    return (records, images, {"metadata_csv": digest(meta),
                              "duplicate_pairs_csv": digest(duplicates)},
            {"listed_duplicate_pairs": duplicate_count,
             "distinct_lesions": len(lesion_splits)})


def _celeba(root):
    base = root / "celeba"
    attrs, identities, partitions = (base / "list_attr_celeba.txt",
                                     base / "identity_CelebA.txt",
                                     base / "list_eval_partition.txt")
    images = base / "img_align_celeba"
    unit = {}
    with identities.open(encoding="utf-8") as stream:
        for line in stream:
            name, identity = line.split()
            if name in unit:
                raise ValueError("CelebA identity repeats image")
            unit[name] = identity
    official = {}
    with partitions.open(encoding="utf-8") as stream:
        for line in stream:
            name, split = line.split()
            if name in official or split not in ("0", "1", "2"):
                raise ValueError("CelebA official partition malformed")
            official[name] = split
    records = []
    with attrs.open(encoding="utf-8") as stream:
        declared = int(stream.readline().strip())
        columns = stream.readline().split()
        smile, male = columns.index("Smiling"), columns.index("Male")
        for line in stream:
            fields = line.split()
            name = fields[0]
            if len(fields) != 1 + len(columns) or name not in unit:
                raise ValueError("CelebA attribute/identity join malformed")
            if fields[1 + smile] not in ("-1", "1") or fields[1 + male] not in ("-1", "1"):
                raise ValueError("CelebA binary attribute malformed")
            records.append(dict(sample_id=name, file=name, unit=unit[name],
                                group="male" if fields[1 + male] == "1" else "female",
                                label=int(fields[1 + smile] == "1")))
    names = {row["file"] for row in records}
    if (len(records) != declared or len(names) != declared or names != set(unit) or
            names != set(official) or names != _image_names(images, ".jpg")):
        raise ValueError("CelebA image/metadata/identity sets differ")
    return (records, images, {"attributes": digest(attrs),
                              "identities": digest(identities),
                              "official_partition": digest(partitions)},
            {"official_partition_present": True})


def prepare(dataset, raw_root, output):
    if dataset not in NAMESPACES:
        raise ValueError("unsupported dataset")
    root, output = Path(raw_root), Path(output)
    records, images, sources, extra = (
        _isic2020(root) if dataset == "isic2020" else _celeba(root))
    counts = defaultdict(Counter)
    units = defaultdict(set)
    files = {name: [] for name in ("train", "stop", "development_pool",
                                  "development_labels")}
    for row in sorted(records, key=lambda value: value["sample_id"]):
        split = partition(dataset, row["unit"])
        units[split].add(row["unit"])
        counts[(split, "all")]["images"] += 1
        counts[(split, "all")]["positive"] += row["label"]
        counts[(split, row["group"])]["images"] += 1
        counts[(split, row["group"])]["positive"] += row["label"]
        public = {key: row[key] for key in ("sample_id", "file", "group")}
        if split == "development":
            files["development_pool"].append(public)
            files["development_labels"].append({"sample_id": row["sample_id"],
                                                 "label": row["label"]})
        else:
            files[split].append({**public, "label": row["label"]})
    if any(units[a] & units[b] for a in units for b in units if a != b):
        raise RuntimeError("split-unit overlap")
    if not all(files.values()):
        raise RuntimeError("empty split")
    output.mkdir(parents=True, exist_ok=False)
    (output / "runner").mkdir()
    (output / "scorer").mkdir()
    hashes = {}
    for name, rows in files.items():
        parent = output / ("scorer" if name == "development_labels" else "runner")
        path = parent / (name + ".jsonl")
        with path.open("x", encoding="utf-8", newline="\n") as stream:
            for row in rows:
                stream.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
        hashes[str(path.relative_to(output))] = digest(path)
    manifest = dict(dataset=dataset, target="malignant" if dataset == "isic2020" else "Smiling",
                    group_attribute="sex" if dataset == "isic2020" else "Male",
                    image_dir=str(images.resolve()), image_count=len(records),
                    split_rule=SPLIT_RULE, split_namespace=NAMESPACES[dataset],
                    source_sha256=sources, files_sha256=hashes,
                    split_unit_counts={key: len(value) for key, value in units.items()},
                    supports={f"{split}/{group}": dict(value)
                              for (split, group), value in sorted(counts.items())}, **extra)
    with (output / "manifest.json").open("x", encoding="utf-8") as stream:
        json.dump(manifest, stream, sort_keys=True, indent=2)
        stream.write("\n")
    return manifest


if __name__ == "__main__":
    if len(sys.argv) != 4:
        raise SystemExit(__doc__)
    print(json.dumps(prepare(sys.argv[1], sys.argv[2], sys.argv[3]), sort_keys=True))
