"""Version the ISIC JPEG decoder policy without changing any split or label.

Usage: python -m analysis.derive_isic_draft_prepared PREPARED_V2 NEW_PREPARED_V3
The original JPEG directory stays untouched. The new runner decodes those
same files through Pillow's JPEG draft path before fixed image transforms.
"""

import json
import shutil
import sys
from pathlib import Path

from tralo.knee_experiment import digest
from tralo.tabular_image_data import (ISIC_DRAFT_POLICY, RUNNER_FILES,
                                      load_runner_cohort)


def derive(original_root, output_root):
    original, output = Path(original_root), Path(output_root)
    manifest, rows = load_runner_cohort(original, "isic2020")
    if manifest.get("decode_policy") is not None:
        raise RuntimeError("original cohort already has a decoder policy")
    old_private = original / "scorer/manifest.json"
    old_labels = original / "scorer/development_labels.jsonl"
    private = json.loads(old_private.read_text(encoding="utf-8"))
    if (private["runner_manifest_sha256"] != digest(original / "manifest.json") or
            private["development_labels_sha256"] != digest(old_labels)):
        raise RuntimeError("original private manifest does not match labels")
    output.mkdir(parents=True, exist_ok=False)
    (output / "runner").mkdir()
    (output / "scorer").mkdir()
    for name in RUNNER_FILES:
        src = original / "runner" / f"{name}.jsonl"
        dst = output / "runner" / src.name
        shutil.copyfile(src, dst)
        if digest(src) != digest(dst):
            raise RuntimeError("runner split changed during derivation")
    shutil.copyfile(old_labels, output / "scorer/development_labels.jsonl")
    if digest(old_labels) != digest(output / "scorer/development_labels.jsonl"):
        raise RuntimeError("private targets changed during derivation")
    derived = dict(manifest)
    derived["decode_policy"] = ISIC_DRAFT_POLICY
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
        raise RuntimeError("derived cohort changed identities or labels")
    return {"output": str(output), "image_count": derived["image_count"],
            "decoder_policy": ISIC_DRAFT_POLICY,
            "origin_sha256": derived["derived_from_manifest_sha256"],
            "manifest_sha256": digest(output / "manifest.json")}


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    print(json.dumps(derive(sys.argv[1], sys.argv[2]), sort_keys=True))
