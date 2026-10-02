"""Version a decoded-RGB memory cache without changing ISIC source images or rows.

Usage: python -m analysis.derive_isic_cached_prepared PREPARED_DRAFT NEW_PREPARED
"""

import json
import shutil
import sys
from pathlib import Path

from tralo.knee_experiment import digest
from tralo.tabular_image_data import (ISIC_CACHED_DRAFT_POLICY,
                                      ISIC_DRAFT_POLICY, RUNNER_FILES,
                                      load_runner_cohort)


def derive(original_root, output_root):
    original, output = Path(original_root), Path(output_root)
    manifest, rows = load_runner_cohort(original, "isic2020")
    if manifest.get("decode_policy") != ISIC_DRAFT_POLICY:
        raise RuntimeError("cache derivation requires the fixed draft decoder")
    private_path = original / "scorer/manifest.json"
    labels_path = original / "scorer/development_labels.jsonl"
    if private_path.is_symlink() or labels_path.is_symlink():
        raise RuntimeError("linked private artifact")
    private = json.loads(private_path.read_text(encoding="utf-8"))
    if (private["runner_manifest_sha256"] != digest(original / "manifest.json") or
            private["development_labels_sha256"] != digest(labels_path)):
        raise RuntimeError("private artifact provenance changed")
    output.mkdir(parents=True, exist_ok=False)
    (output / "runner").mkdir()
    (output / "scorer").mkdir()
    for name in RUNNER_FILES:
        source = original / "runner" / f"{name}.jsonl"
        target = output / "runner" / source.name
        shutil.copyfile(source, target)
        if digest(source) != digest(target):
            raise RuntimeError("runner row changed during cache derivation")
    target_labels = output / "scorer/development_labels.jsonl"
    shutil.copyfile(labels_path, target_labels)
    if digest(labels_path) != digest(target_labels):
        raise RuntimeError("private row changed during cache derivation")
    derived = dict(manifest)
    derived["decode_policy"] = ISIC_CACHED_DRAFT_POLICY
    derived["derived_from_manifest_sha256"] = digest(original / "manifest.json")
    with (output / "manifest.json").open("x", encoding="utf-8") as stream:
        json.dump(derived, stream, sort_keys=True, indent=2)
        stream.write("\n")
    private["runner_manifest_sha256"] = digest(output / "manifest.json")
    with (output / "scorer/manifest.json").open("x", encoding="utf-8") as stream:
        json.dump(private, stream, sort_keys=True, indent=2)
        stream.write("\n")
    verified, reread = load_runner_cohort(output, "isic2020")
    if verified != derived or reread != rows:
        raise RuntimeError("cache cohort changed identities, groups or labels")
    return {"output": str(output), "image_count": derived["image_count"],
            "decoder_policy": ISIC_CACHED_DRAFT_POLICY,
            "origin_sha256": derived["derived_from_manifest_sha256"],
            "manifest_sha256": digest(output / "manifest.json")}


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    print(json.dumps(derive(sys.argv[1], sys.argv[2]), sort_keys=True))
