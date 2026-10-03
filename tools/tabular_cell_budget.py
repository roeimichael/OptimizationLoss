"""Account for completed GPU jobs in one calibrated tabular cell."""

import json
from pathlib import Path
import sys


def remaining_cell_seconds(root: Path, base_seed: int, ceiling_seconds: int) -> int:
    spent = 0
    for seed in range(base_seed, base_seed + 5):
        phase = "pilot" if seed == base_seed else "full"
        path = root / f"{phase}_{seed}.complete.json"
        if not path.exists():
            if seed == base_seed:
                raise ValueError("pilot completion receipt missing")
            continue
        row = json.loads(path.read_text(encoding="utf-8"))
        if row.get("seed") != seed:
            raise ValueError(f"completion receipt seed mismatch: {path}")
        if row.get("exit_code") != 0:
            raise ValueError(f"completion receipt exit is not zero: {path}")
        elapsed = row.get("elapsed_seconds")
        if type(elapsed) is not int or elapsed < 0:
            raise ValueError(f"invalid elapsed seconds: {path}")
        spent += elapsed
    return ceiling_seconds - spent


if __name__ == "__main__":
    if len(sys.argv) != 4:
        raise SystemExit("expected ROOT BASE_SEED CEILING_SECONDS")
    print(remaining_cell_seconds(Path(sys.argv[1]), int(sys.argv[2]),
                                 int(sys.argv[3])))
