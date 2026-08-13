"""Expand the ALM baseline from the B3 tight-cap probe to the FULL paper grid.

B3 ran ALM on OctMNIST x {L30,L40} x 3 backbones x 4 seeds (24 runs). The paper
now needs ALM as a full column in the headline tables, i.e. everywhere the other
six methods already are:

    3 datasets x 3 backbones x 9 symmetric caps x 4 seeds = 324 runs

Configs are cloned from the frozen `paper_final` fioretto_ldf configs and only
the dual rule is swapped (methodology -> fioretto_alm, plus alm_eta / alm_mu0 /
alm_mu_step). Everything else -- warmup keys, constraint_epochs, step size,
early stop -- is byte-identical, so ALM reuses the exact frozen CE warmup cache
and pairs apples-to-apples against TraLO and Fioretto-LDF at the same cells.

The 24 B3 runs already live under results/track_b/b3 and are NOT regenerated;
they are merged at analysis time. This generator emits only what is missing.

FAILSAFE: writes to results/track_b/b3_full, outside results/pending_runs, so
the frozen corpus can never be touched. Idempotent -- existing targets skipped.

Run ON THE SERVER from repo root:
    python -m src.config_generators.gen_alm_full
"""

import glob
import json
from pathlib import Path

PF = "results/pending_runs/paper_final"
B3 = "results/track_b/b3"
DST_ROOT = Path("results/track_b/b3_full")
N_LANES = 4

DATASETS = ["octmnist", "dermmnist", "tissuemnist"]
BACKBONES = ["MobileNetV3", "RegNetY400MF", "ViTB16"]
CAPS = ["L%d_G%d" % (p, p) for p in range(10, 100, 10)]
SEEDS = [1, 2, 3, 4]

# Identical to gen_tmlr_alm.py so the new cells are comparable to the B3 ones.
ALM_HP = {"alm_eta": 0.005, "alm_mu0": 0.01, "alm_mu_step": 0.01}

_cell_lane = {}


def _lane_for(cell_key):
    # One lane per (model, dataset, seed): all caps of a cell share a warmup
    # cache entry, so keeping them on one GPU avoids a cross-GPU cache race.
    if cell_key not in _cell_lane:
        _cell_lane[cell_key] = len(_cell_lane) % N_LANES
    return _cell_lane[cell_key]


def _src(model, dataset, cap, seed):
    hits = glob.glob(
        "%s/lane*/%s/%s/%s/fioretto_ldf/seed_%d/config.json" % (PF, model, dataset, cap, seed))
    return hits[0] if hits else None


def _already_in_b3(model, dataset, cap, seed):
    hits = glob.glob(
        "%s/lane*/%s/%s/%s/fioretto_alm/seed_%d/config.json" % (B3, model, dataset, cap, seed))
    return bool(hits)


def main():
    assert "pending_runs" not in str(DST_ROOT), "failsafe: must not write into the corpus root"
    emit = skip_b3 = skip_exists = 0
    missing_src = []
    lane_counts = [0] * N_LANES
    for dataset in DATASETS:
        for model in BACKBONES:
            for seed in SEEDS:
                lane = _lane_for((model, dataset, seed))
                for cap in CAPS:
                    if _already_in_b3(model, dataset, cap, seed):
                        skip_b3 += 1
                        continue
                    src = _src(model, dataset, cap, seed)
                    if not src:
                        missing_src.append((dataset, model, cap, seed))
                        continue
                    dst_dir = (DST_ROOT / ("lane_gpu%d" % lane) / model / dataset
                               / cap / "fioretto_alm" / ("seed_%d" % seed))
                    dst = dst_dir / "config.json"
                    if dst.exists():
                        skip_exists += 1
                        continue
                    c = json.loads(json.dumps(json.load(open(src))))
                    c.pop("results", None)
                    c["status"] = "pending"
                    c["methodology"] = "fioretto_alm"
                    c["hyperparams"].update(ALM_HP)
                    # base_model_id untouched: warmup keys are unchanged, so the
                    # frozen CE warmup cache is reused (fair + fast).
                    c["sweep_tag"] = "alm_full_2026-08"
                    c["cloned_from"] = src
                    c["exp_name"] = "almfull_%s_%s_%s_seed%d" % (model, dataset, cap, seed)
                    c["experiment_path"] = str(dst_dir)
                    dst_dir.mkdir(parents=True, exist_ok=True)
                    with open(dst, "w") as f:
                        json.dump(c, f, indent=1)
                    emit += 1
                    lane_counts[lane] += 1

    print("emitted=%d  skipped(already in b3)=%d  skipped(existing)=%d"
          % (emit, skip_b3, skip_exists))
    print("per_lane=%s" % lane_counts)
    print("root=%s" % DST_ROOT)
    if missing_src:
        print("WARNING: %d cells had no frozen fioretto_ldf source:" % len(missing_src))
        for m in missing_src[:10]:
            print("   ", m)


if __name__ == "__main__":
    main()
