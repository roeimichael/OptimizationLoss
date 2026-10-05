"""Neutral-path inputs for the approved knee snapshot pilot; no test-tree access.

Preparation is a trusted label-handling role. Loading the public pack has no
private-path argument and never opens the original Chen folder hierarchy.
"""

from collections import Counter
import hashlib
import json
from pathlib import Path
import re

from .local_policy import size_share_caps

NAMESPACE = "tralo-knee-artificial-local-20261005:"
FORMAT = "knee_snapshot_local_20261005"
EXPECTED_COUNTS = {"train": 5778, "development": 826}
ROW_KEYS = {"sample_id", "subject", "path", "sha256", "pixel_sha256", "size", "split", "group"}


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def encode(value):
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()


def group_for(subject):
    if type(subject) is not str or not re.fullmatch(r"\d{7}", subject):
        raise ValueError("expected seven-digit subject ID")
    return "H" + str(int(sha256((NAMESPACE + subject).encode()), 16) % 2)


def stopping_subject(subject):
    return int(sha256(subject.encode()), 16) % 10 == 0


def _exclusive(path, data):
    with path.open("xb") as stream:
        stream.write(data)


def prepare(source, public, private, expected_counts=None):
    """Copy only train/val; development labels leave through a separate file.

    A nonstandard expected_counts is only for fictitious fixture validation and
    is marked in the output. Neither mode claims campaign readiness.
    """
    from PIL import Image
    import io

    source, public, private = Path(source).resolve(), Path(public).resolve(), Path(private).resolve()
    counts = EXPECTED_COUNTS if expected_counts is None else expected_counts
    if (public == source or source in public.parents or public in private.parents
            or private == public or source in private.parents):
        raise ValueError("outputs must be outside original data and private labels outside public pack")
    if public.exists() or private.exists():
        raise FileExistsError("refusing existing preparation outputs")
    # Discover only the two authorized source roles; do not list the root/test.
    discovered = {}
    for split, role in (("train", "train"), ("val", "development")):
        if (source / split).is_symlink():
            raise ValueError("source role symlink refused")
        # Source grade directory order would itself disclose development labels.
        paths = sorted((source / split).glob("*/*.png"), key=lambda path: path.stem)
        if len(paths) != counts[role]:
            raise ValueError("unexpected source count: " + role)
        discovered[split] = paths
    public.mkdir(parents=True, exist_ok=False)
    rows = {"train": [], "stop": [], "development": []}
    private_rows, identities, source_hashes = [], set(), {}
    subjects = {"train": set(), "development": set()}
    pixels = {"train": set(), "development": set()}
    for split, role in (("train", "train"), ("val", "development")):
        image_dir = public / "images" / role
        image_dir.mkdir(parents=True)
        for path in discovered[split]:
            match = re.fullmatch(r"(\d{7})([LR])", path.stem)
            if (path.is_symlink() or path.parent.is_symlink() or not match
                    or path.parent.name not in {"0", "1", "2", "3", "4"}):
                raise ValueError("unexpected source image identity/path")
            if path.stem in identities:
                raise ValueError("repeated image identity across authorized roles")
            identities.add(path.stem)
            data = path.read_bytes()
            # Original bytes may fingerprint grade-bearing PNG metadata. Keep
            # that lineage only in the separate trusted/private artifact.
            source_hashes[path.stem] = sha256(data)
            with Image.open(io.BytesIO(data)) as image:
                image.load()
                rgb = image.convert("RGB")
                if min(rgb.size) < 2:
                    raise ValueError("invalid image size")
                pixel_hash = sha256(str(rgb.size).encode() + rgb.tobytes())
                size = list(rgb.size)
                neutral = io.BytesIO()
                # Drop arbitrary PNG text/metadata; preserve native RGB pixels.
                Image.frombytes("RGB", rgb.size, rgb.tobytes()).save(neutral, format="PNG")
                neutral_bytes = neutral.getvalue()
            subject = match[1]
            subjects[role].add(subject)
            pixels[role].add(pixel_hash)
            destination = image_dir / (path.stem + ".png")
            _exclusive(destination, neutral_bytes)
            row = dict(sample_id=path.stem, subject=subject,
                       path=destination.relative_to(public).as_posix(), sha256=sha256(neutral_bytes),
                       pixel_sha256=pixel_hash, size=size, split=split, group=group_for(subject))
            if role == "development":
                rows[role].append(row)
                private_rows.append(dict(sample_id=path.stem, label=int(path.parent.name)))
            else:
                row["label"] = int(path.parent.name)
                rows["stop" if stopping_subject(subject) else "train"].append(row)
    if subjects["train"] & subjects["development"] or pixels["train"] & pixels["development"]:
        raise ValueError("train/development subject or exact RGB overlap")
    if not all(rows.values()):
        raise ValueError("empty train/stop/development role")
    groups = [row["group"] for row in rows["development"]]
    if set(groups) != {"H0", "H1"}:
        raise ValueError("fixed hash did not produce two observed groups")
    row_bytes = encode(rows)
    _exclusive(public / "rows.json", row_bytes)
    manifest = dict(format=FORMAT, status="prepared_not_campaign_ready",
                    synthetic_or_nonstandard_counts=counts != EXPECTED_COUNTS,
                    namespace=NAMESPACE, capped_class=3, global_cap=76, local_total=95,
                    local_caps=size_share_caps(groups, 95), group_sizes=dict(sorted(Counter(groups).items())),
                    counts={role: len(items) for role, items in rows.items()},
                    rows_sha256=sha256(row_bytes),
                    overlap=dict(subject_pairs=0, exact_rgb_pairs=0),
                    limitation="Filename subject identity; no semantic/near-duplicate or OS isolation certification.")
    manifest_bytes = encode(manifest)
    _exclusive(public / "manifest.json", manifest_bytes)
    private.parent.mkdir(parents=True, exist_ok=True)
    _exclusive(private, encode(dict(format=FORMAT, public_manifest_sha256=sha256(manifest_bytes),
                                    rows=private_rows, source_sha256_by_id=source_hashes)))
    # Public return/hash tables deliberately contain no private label path/hash.
    return dict(public_manifest_sha256=sha256(manifest_bytes), **manifest)


def load_public(public, manifest_sha256, *, allow_synthetic=False):
    """Authenticate and validate the exact bytes used; no private label access."""
    public = Path(public).resolve()
    manifest_path, rows_path = public / "manifest.json", public / "rows.json"
    if manifest_path.is_symlink() or rows_path.is_symlink():
        raise ValueError("public metadata symlink refused")
    data = manifest_path.read_bytes()
    if sha256(data) != manifest_sha256:
        raise ValueError("public manifest hash mismatch")
    manifest = json.loads(data)
    expected = {"format", "status", "synthetic_or_nonstandard_counts", "namespace", "capped_class",
                "global_cap", "local_total", "local_caps", "group_sizes", "counts", "rows_sha256",
                "overlap", "limitation"}
    if (set(manifest) != expected or manifest["format"] != FORMAT or manifest["namespace"] != NAMESPACE
            or manifest["capped_class"] != 3 or manifest["global_cap"] != 76
            or manifest["local_total"] != 95 or manifest["status"] != "prepared_not_campaign_ready"):
        raise ValueError("public manifest differs from approved input contract")
    if type(manifest["synthetic_or_nonstandard_counts"]) is not bool:
        raise ValueError("invalid synthetic marker")
    if manifest["synthetic_or_nonstandard_counts"] and not allow_synthetic:
        raise ValueError("synthetic/nonstandard pack cannot enter campaign runtime")
    row_bytes = rows_path.read_bytes()
    if sha256(row_bytes) != manifest["rows_sha256"]:
        raise ValueError("public row hash mismatch")
    rows = json.loads(row_bytes)
    if set(rows) != {"train", "stop", "development"}:
        raise ValueError("unexpected public roles")
    subjects, pixels, ids = {}, {}, set()
    for role, items in rows.items():
        if not isinstance(items, list) or not items:
            raise ValueError("empty or invalid public role")
        subjects[role], pixels[role] = set(), set()
        for row in items:
            if set(row) != (ROW_KEYS if role == "development" else ROW_KEYS | {"label"}):
                raise ValueError("unexpected row fields or development target")
            if any(not isinstance(row[key], str) or not re.fullmatch(r"[0-9a-f]{64}", row[key])
                   for key in ("sha256", "pixel_sha256")):
                raise ValueError("invalid image hash")
            sid, subject = row["sample_id"], row["subject"]
            if not re.fullmatch(r"\d{7}[LR]", sid) or sid[:7] != subject or sid in ids:
                raise ValueError("invalid/repeated sample identity")
            ids.add(sid)
            image_role = "development" if role == "development" else "train"
            if (row["path"] != f"images/{image_role}/{sid}.png"
                    or row["split"] != ("val" if role == "development" else "train")
                    or row["group"] != group_for(subject)):
                raise ValueError("nonneutral image path, split or fixed group mismatch")
            if role != "development":
                if type(row["label"]) is not int or not 0 <= row["label"] <= 4:
                    raise ValueError("invalid training label")
                if stopping_subject(subject) != (role == "stop"):
                    raise ValueError("training stopping carve mismatch")
            path = public / row["path"]
            if (any(p.is_symlink() for p in (path, path.parent, path.parent.parent))
                    or public not in path.resolve().parents or sha256(path.read_bytes()) != row["sha256"]):
                raise ValueError("public image changed or escapes pack")
            subjects[role].add(subject)
            pixels[role].add(row["pixel_sha256"])
        if [r["sample_id"] for r in items] != sorted(r["sample_id"] for r in items):
            raise ValueError("public rows must use label-independent ID ordering")
    for a, b in (("train", "stop"), ("train", "development"), ("stop", "development")):
        if subjects[a] & subjects[b] or (b == "development" and pixels[a] & pixels[b]):
            raise ValueError("public role overlap")
    counts = {role: len(items) for role, items in rows.items()}
    if counts != manifest["counts"] or (not allow_synthetic and
            (counts["train"] + counts["stop"] != 5778 or counts["development"] != 826)):
        raise ValueError("public count mismatch")
    groups = [row["group"] for row in rows["development"]]
    if (set(groups) != {"H0", "H1"} or manifest["local_caps"] != size_share_caps(groups, 95)
            or manifest["group_sizes"] != dict(sorted(Counter(groups).items()))
            or manifest["overlap"] != dict(subject_pairs=0, exact_rgb_pairs=0)):
        raise ValueError("public quota/overlap mismatch")
    return manifest, rows
