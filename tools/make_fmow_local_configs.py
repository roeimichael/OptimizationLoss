"""Write the fixed fmow2 local-constraint configs for an immutable release.

Usage: python -m tools.make_fmow_local_configs OUTPUT_DIRECTORY
The target directory must not exist; no existing config is ever overwritten.
"""

import json
from pathlib import Path
import sys

from tralo.fmow_local import PILOT, RECIPE, SEEDS, validate


def write_configs(directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    configs = [(PILOT, True), (PILOT, False)] + [(seed, True) for seed in SEEDS]
    for seed, steps in configs:
        config = dict(RECIPE, seed=seed, snapshot_steps=steps)
        validate(config)
        suffix = "step" if steps else "ref"
        path = directory / f"fmow_local_{seed}_{suffix}.json"
        path.write_text(json.dumps(config, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return len(configs)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    print(write_configs(sys.argv[1]))
