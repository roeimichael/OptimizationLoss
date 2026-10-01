"""Audited image-only inputs with an external tabular constraint group.

The classifier receives RGB pixels. The group field is returned separately
for quota accounting and the common allocator; development targets are never
loaded by a training process.
"""

import hashlib
import json
from collections import Counter
from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset


RUNNER_FILES = ("train", "stop", "development_pool")
ISIC_DRAFT_POLICY = "isic2020_jpeg_decoder_draft_256_v1"


def _digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load_runner_cohort(prepared_root, expected_dataset):
    """Authenticate a prepared cohort without touching scorer-only labels."""
    root = Path(prepared_root)
    manifest_path = root / "manifest.json"
    if manifest_path.is_symlink():
        raise RuntimeError("linked runner manifest")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("dataset") != expected_dataset:
        raise RuntimeError("wrong prepared dataset")
    policy = manifest.get("decode_policy")
    if policy not in (None, ISIC_DRAFT_POLICY) or (
            policy == ISIC_DRAFT_POLICY and expected_dataset != "isic2020"):
        raise RuntimeError("unknown or mismatched image decoder policy")
    expected_files = {f"runner/{name}.jsonl" for name in RUNNER_FILES}
    if set(manifest.get("files_sha256", {})) != expected_files:
        raise RuntimeError("runner manifest includes missing or private files")
    supports = manifest.get("supports", {})
    if (not isinstance(supports, dict) or
            any("positive" in row for key, row in supports.items()
                if key.startswith("development/"))):
        raise RuntimeError("public manifest exposes development outcomes")
    image_dir = Path(manifest["image_dir"])
    if not image_dir.is_dir() or image_dir.is_symlink():
        raise RuntimeError("prepared image directory missing or linked")
    rows = {}
    identities = set()
    for split in RUNNER_FILES:
        path = root / "runner" / f"{split}.jsonl"
        if path.is_symlink() or _digest(path) != manifest["files_sha256"][
                f"runner/{split}.jsonl"]:
            raise RuntimeError("prepared runner rows changed: " + split)
        current = []
        group_counts = Counter()
        positive_counts = Counter()
        with path.open(encoding="utf-8") as stream:
            for line in stream:
                row = json.loads(line)
                expected = ({"sample_id", "file", "group"} if split == "development_pool"
                            else {"sample_id", "file", "group", "label"})
                if set(row) != expected:
                    raise RuntimeError("development label boundary or row schema changed")
                sample_id, name, group = (row[key] for key in
                                          ("sample_id", "file", "group"))
                if (any(type(value) is not str or not value for value in
                        (sample_id, name, group)) or
                        Path(name).name != name or "/" in name or "\\" in name or
                        name in (".", "..") or
                        sample_id in identities):
                    raise RuntimeError("duplicate identity or invalid image/group path")
                if split != "development_pool" and row["label"] not in (0, 1):
                    raise RuntimeError("training or stop target is not binary")
                if not (image_dir / name).is_file():
                    raise RuntimeError("prepared image is missing: " + name)
                identities.add(sample_id)
                group_counts[group] += 1
                positive_counts[group] += row.get("label", 0)
                current.append(row)
        if not current:
            raise RuntimeError("empty prepared split")
        expected_support = {key.removeprefix(
            "development/" if split == "development_pool" else split + "/"):
            value for key, value in supports.items() if key.startswith(
                "development/" if split == "development_pool" else split + "/")}
        if (set(expected_support) != {"all", *group_counts} or
                sum(group_counts.values()) != expected_support.get("all", {}).get("images")):
            raise RuntimeError("split size differs from frozen manifest")
        for group, count in group_counts.items():
            recorded = expected_support.get(group)
            if (recorded is None or recorded.get("images") != count or
                    (split != "development_pool" and
                     recorded.get("positive") != positive_counts[group])):
                raise RuntimeError("group support differs from frozen manifest")
        rows[split] = current
    if len(identities) != manifest.get("image_count"):
        raise RuntimeError("prepared cohort image count differs")
    return manifest, rows


class PreparedImageRows(Dataset):
    """Use a frozen row list; development rows have no target field."""

    def __init__(self, image_dir, rows, transform, decode_policy=None):
        if decode_policy not in (None, ISIC_DRAFT_POLICY):
            raise ValueError("unfrozen image decoder policy")
        self.image_dir = Path(image_dir)
        self.rows = rows
        self.transform = transform
        self.decode_policy = decode_policy

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        with Image.open(self.image_dir / row["file"]) as image:
            if self.decode_policy == ISIC_DRAFT_POLICY:
                if image.format != "JPEG":
                    raise RuntimeError("ISIC decoder draft requires JPEG input")
                image.draft("RGB", (256, 256))
            pixels = self.transform(image.convert("RGB"))
        return pixels, row.get("label", -1), row["group"], row["sample_id"]
