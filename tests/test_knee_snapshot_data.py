import builtins
import json
from pathlib import Path

from PIL import Image, PngImagePlugin
import pytest

from tralo.knee_snapshot_data import (encode, group_for, load_public, prepare,
                                      sha256, stopping_subject)


def source_fixture(tmp_path, reverse_labels=False):
    source = tmp_path / "source"
    for role, count, offset in (("train", 40, 1000000), ("val", 100, 2000000)):
        for index in range(count):
            label = (4 - index % 5) if reverse_labels else index % 5
            path = source / role / str(label) / f"{offset + index:07d}L.png"
            path.parent.mkdir(parents=True, exist_ok=True)
            info = PngImagePlugin.PngInfo()
            info.add_text("grade", str(label))
            Image.new("RGB", (3, 3), (index, offset // 1000000, index // 2)).save(path, pnginfo=info)
    (source / "test").mkdir()
    (source / "test" / "SEALED_DO_NOT_READ").write_text("private")
    return source


def prepared(tmp_path):
    source = source_fixture(tmp_path)
    public, private = tmp_path / "public", tmp_path / "private" / "labels.json"
    record = prepare(source, public, private, {"train": 40, "development": 100})
    return source, public, private, record


def repin(public, change):
    manifest = json.loads((public / "manifest.json").read_bytes())
    rows = json.loads((public / "rows.json").read_bytes())
    change(manifest, rows)
    row_bytes = encode(rows)
    (public / "rows.json").write_bytes(row_bytes)
    manifest["rows_sha256"] = sha256(row_bytes)
    data = encode(manifest)
    (public / "manifest.json").write_bytes(data)
    return sha256(data)


def test_fixed_subject_hash_is_independent_of_knee_and_label():
    assert group_for("1234567") in {"H0", "H1"}
    assert group_for("1234567") == group_for("1234567")
    for bad in ("123", "1234567L", "../1234567", 1234567):
        with pytest.raises(ValueError):
            group_for(bad)


def test_prepare_and_public_loader_never_touch_test_or_private(tmp_path, monkeypatch):
    source = source_fixture(tmp_path)
    public, private = tmp_path / "public", tmp_path / "private_labels.json"
    original_glob, original_read = Path.glob, Path.read_bytes
    opened = []

    def guarded_glob(path, pattern):
        assert path != source and "test" not in path.parts
        return original_glob(path, pattern)

    def guarded_read(path):
        assert "test" not in path.parts
        opened.append(path)
        return original_read(path)

    monkeypatch.setattr(Path, "glob", guarded_glob)
    monkeypatch.setattr(Path, "read_bytes", guarded_read)
    record = prepare(source, public, private, {"train": 40, "development": 100})
    opened.clear()

    def no_private_read(path):
        assert path != private and source not in path.parents
        return guarded_read(path)

    monkeypatch.setattr(Path, "read_bytes", no_private_read)
    manifest, rows = load_public(public, record["public_manifest_sha256"], allow_synthetic=True)
    assert sum(manifest["local_caps"].values()) == 95
    assert len(rows["development"]) == 100
    assert all("label" not in row for row in rows["development"])
    assert all(row["path"].startswith("images/development/") for row in rows["development"])
    assert all(stopping_subject(row["subject"]) for row in rows["stop"])
    assert not any(stopping_subject(row["subject"]) for row in rows["train"])
    assert "private" not in (public / "manifest.json").read_text()
    assert all(source not in path.parents and path != private for path in opened)


def test_neutral_development_pack_does_not_encode_grade_in_bytes_or_order(tmp_path):
    packs = []
    for reverse in (False, True):
        directory = tmp_path / str(reverse)
        directory.mkdir()
        source = source_fixture(directory, reverse)
        public = directory / "public"
        result = prepare(source, public, directory / "private.json", {"train": 40, "development": 100})
        _, rows = load_public(public, result["public_manifest_sha256"], allow_synthetic=True)
        packs.append((public, rows))
        for row in rows["development"]:
            with Image.open(public / row["path"]) as image:
                assert "grade" not in image.info
    a, b = packs
    # Even original PNG hashes could fingerprint grade-bearing metadata: only
    # the private artifact receives them. Every public development field agrees.
    assert a[1]["development"] == b[1]["development"]
    assert [row["sample_id"] for row in a[1]["development"]] == sorted(
        row["sample_id"] for row in a[1]["development"])


@pytest.mark.parametrize("mutation", [
    lambda m, r: r["development"][0].update(label=3),
    lambda m, r: r["development"][0].update(path="val/3/2000000L.png"),
    lambda m, r: r["development"][0].update(path="../private.json"),
    lambda m, r: r["development"][0].update(group="other"),
    lambda m, r: r["development"].reverse(),
    lambda m, r: m.update(local_caps={"H0": 50, "H1": 45}),
    lambda m, r: m.update(global_cap=77),
    lambda m, r: r["train"][0].update(label=True),
])
def test_loader_rejects_rehashed_boundary_or_policy_changes(tmp_path, mutation):
    _, public, _, _ = prepared(tmp_path)
    pin = repin(public, mutation)
    with pytest.raises(ValueError):
        load_public(public, pin, allow_synthetic=True)


def test_loader_authenticates_exact_metadata_and_image_bytes(tmp_path):
    _, public, _, record = prepared(tmp_path)
    pin = record["public_manifest_sha256"]
    with pytest.raises(ValueError, match="manifest hash"):
        load_public(public, "0" * 64, allow_synthetic=True)
    rows = json.loads((public / "rows.json").read_bytes())
    path = public / rows["development"][0]["path"]
    path.write_bytes(b"changed")
    with pytest.raises(ValueError, match="image changed"):
        load_public(public, pin, allow_synthetic=True)


def test_synthetic_pack_and_existing_outputs_are_refused_for_campaign(tmp_path):
    source, public, private, record = prepared(tmp_path)
    with pytest.raises(ValueError, match="synthetic"):
        load_public(public, record["public_manifest_sha256"])
    with pytest.raises(FileExistsError):
        prepare(source, public, private, {"train": 40, "development": 100})
    with pytest.raises(ValueError, match="outside"):
        prepare(source, tmp_path / "new", tmp_path / "new" / "labels.json")


def test_duplicate_subject_and_exact_rgb_split_overlap_fail_without_cleanup(tmp_path):
    source = source_fixture(tmp_path)
    val = next((source / "val").glob("*/*.png"))
    train = next((source / "train").glob("*/*.png"))
    val.write_bytes(train.read_bytes())
    with pytest.raises(ValueError, match="overlap"):
        prepare(source, tmp_path / "public", tmp_path / "private.json", {"train": 40, "development": 100})
    assert (tmp_path / "public" / "images" / "development").exists()
    assert not (tmp_path / "private.json").exists()
