"""Trusted train/development preparation; never opens the Chen test tree.

Usage: python -m analysis.prepare_knee_snapshot_local SOURCE PUBLIC_DIR PRIVATE_LABEL_FILE
Outputs are exclusive and remain preserved if preparation fails.
"""

import json
import sys

from tralo.knee_snapshot_data import prepare


if __name__ == "__main__":
    if len(sys.argv) != 4:
        raise SystemExit(__doc__)
    print(json.dumps(prepare(*sys.argv[1:])))
