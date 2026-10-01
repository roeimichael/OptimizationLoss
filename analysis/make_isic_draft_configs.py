"""Add frozen, fresh ISIC JPEG-draft seed blocks to an existing config directory."""

import json
from pathlib import Path

from tralo.tabular_persistent_train import STUDY, validate_config


def make(root):
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(root)
    for backbone, base in (("mobilenet_v3_large", 6810), ("vit_b_16", 6820)):
        for seed in range(base, base + 5):
            config = {"study": STUDY, "dataset": "isic2020",
                      "backbone": backbone, "seed": seed,
                      "pilot": seed == base}
            validate_config(config)
            with (root / f"isic2020_{backbone}_{seed}.json").open(
                    "x", encoding="utf-8", newline="\n") as stream:
                json.dump(config, stream, sort_keys=True, indent=2)
                stream.write("\n")


if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    make(sys.argv[1])
