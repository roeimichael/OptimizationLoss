"""Freeze the prospective six-cell pilot and four-seed screening matrix."""

import json
from pathlib import Path

from tralo.tabular_persistent_train import BACKBONES, STUDY


def main(output):
    root = Path(output)
    root.mkdir(parents=True, exist_ok=False)
    for dataset in ("isic2020", "celeba"):
        for backbone in BACKBONES:
            for seed in range(6800, 6805):
                config = {"study": STUDY, "dataset": dataset,
                          "backbone": backbone, "seed": seed,
                          "pilot": seed == 6800}
                path = root / f"{dataset}_{backbone}_{seed}.json"
                with path.open("x", encoding="utf-8", newline="\n") as stream:
                    json.dump(config, stream, sort_keys=True, indent=2)
                    stream.write("\n")


if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    main(sys.argv[1])
