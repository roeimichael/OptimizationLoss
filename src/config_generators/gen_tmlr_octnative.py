"""TMLR Track B / B2 (REDIRECTED): native-resolution OCT replication.

Advisor-approved redirect of B2 from "native HAM10000" (redundant -- DermMNIST
is already native 224) to native-resolution OCT, because OctMNIST -- the paper's
HEADLINE tight-cap dataset -- is the one that is genuinely 28px-upsampled.

Clones the frozen paper_final OctMNIST configs and swaps ONLY the dataset:
dataset_mode octmnist -> octnative, data_dir -> data/octnative/slice_1 (built by
scripts/prep_oct_native.py to the SAME 12k/1k shape, so only image RESOLUTION
differs). Everything else -- constrained_class=2 (DRUSEN), synth_group, caps,
HPs, backbones, methods -- is identical, making this a clean native-vs-28px
replication of the headline result.

  methods : tralo, fioretto_ldf, hounie_rcl, heuristic   (handoff B2's 4)
  caps    : L30_G30, L40_G40
  backbones: MobileNetV3, RegNetY400MF, ViTB16
  seeds   : 1-4
  total   : 4 x 2 x 3 x 4 = 96 runs

base_model_id is recomputed (new dataset_mode/data_dir) so warmups are fresh
(12 = 3 bb x 4 seeds, shared across methods+caps of a cell -> one lane per cell).

Root: results/tmlr_track_b/octnative_2026-07/. Idempotent.

Run ON THE SERVER from repo root (after prep_oct_native.py has built the slice):
    python -m src.config_generators.gen_tmlr_octnative
"""

import glob
import json
from pathlib import Path

from src.config_generators.generate_configs import compute_base_model_id

PF = "results/pending_runs/paper_final"
DST_ROOT = Path("results/tmlr_track_b/octnative_2026-07")
N_LANES = 3

METHODS = ["tralo", "fioretto_ldf", "hounie_rcl", "heuristic"]
CAPS = ["L30_G30", "L40_G40"]
BACKBONES = ["MobileNetV3", "RegNetY400MF", "ViTB16"]
SEEDS = [1, 2, 3, 4]

NEW_MODE = "octnative"
NEW_DIR = "data/octnative/slice_1"

_cell_lane = {}


def _lane_for(cell_key):
    # all methods+caps of a (model, seed) cell share one lane -> the fresh
    # native-OCT warmup trains once and is reused (no cross-GPU cache race).
    if cell_key not in _cell_lane:
        _cell_lane[cell_key] = len(_cell_lane) % N_LANES
    return _cell_lane[cell_key]


def _src(model, cap, meth, seed):
    hits = glob.glob(f"{PF}/lane*/{model}/octmnist/{cap}/{meth}/seed_{seed}/config.json")
    return hits[0] if hits else None


def main():
    assert "pending_runs" not in str(DST_ROOT), "failsafe: must not write into the corpus root"
    emit = skip = 0
    lane_counts = [0] * N_LANES
    for model in BACKBONES:
        for seed in SEEDS:
            lane = _lane_for((model, seed))
            for cap in CAPS:
                for meth in METHODS:
                    src = _src(model, cap, meth, seed)
                    assert src, f"missing frozen {model}/octmnist/{cap}/{meth}/seed_{seed}"
                    dst_dir = (DST_ROOT / f"lane_gpu{lane}" / model / NEW_MODE
                               / cap / meth / f"seed_{seed}")
                    dst = dst_dir / "config.json"
                    if dst.exists():
                        skip += 1
                        continue
                    c = json.loads(json.dumps(json.load(open(src))))
                    c.pop("results", None)
                    c["status"] = "pending"
                    c["dataset_mode"] = NEW_MODE
                    c["dataset_config"]["data_dir"] = NEW_DIR
                    # native-OCT is a distinct dataset -> fresh warmup identity
                    c["base_model_id"] = compute_base_model_id(
                        model, c["hyperparams"], NEW_MODE, NEW_DIR, c["dataset_config"])
                    c["sweep_tag"] = "tmlr_octnative_2026-07"
                    c["cloned_from"] = src
                    c["exp_name"] = f"tmlroctnat_{model}_{meth}_{cap}_seed{seed}"
                    c["experiment_path"] = str(dst_dir)
                    dst_dir.mkdir(parents=True, exist_ok=True)
                    with open(dst, "w") as f:
                        json.dump(c, f, indent=1)
                    emit += 1
                    lane_counts[lane] += 1
    print(f"emitted={emit} skipped(existing)={skip} per_lane={lane_counts}")
    print(f"root={DST_ROOT}")


if __name__ == "__main__":
    main()
