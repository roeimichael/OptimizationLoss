"""TMLR Track B / B3: ALM (augmented-Lagrangian) baseline on OctMNIST tight caps.

Clones frozen paper_final `fioretto_ldf` OctMNIST configs
(L30_G30 + L40_G40 x 3 backbones x seeds 1-4 = 24 runs) and swaps ONLY the dual
rule: methodology -> fioretto_alm, adding alm_eta / alm_mu0 / alm_mu_step. The
warmup-cache identity (base_model_id) depends solely on warmup keys, which are
unchanged, so ALM reuses the exact frozen CE warmup and pairs apples-to-apples
against frozen Fioretto-LDF / TraLO at the same cells (constraint_epochs=300,
step=0.005, same early stop -- only the multiplier update differs).

FAILSAFE: output root results/tmlr_track_b/alm_2026-07 is OUTSIDE
results/pending_runs, so the frozen corpus can never be touched. Idempotent:
existing target configs are never overwritten.

Run ON THE SERVER from repo root:
    python -m src.config_generators.gen_tmlr_alm
"""

import glob
import json
from pathlib import Path

PF = "results/pending_runs/paper_final"
DST_ROOT = Path("results/tmlr_track_b/alm_2026-07")
N_LANES = 3

BACKBONES = ["MobileNetV3", "RegNetY400MF", "ViTB16"]
CAPS = ["L30_G30", "L40_G40"]
SEEDS = [1, 2, 3, 4]
# ALM update HPs. eta = projected dual-ascent step on the raw residual (matches
# the frozen Fioretto step so ALM and Fioretto differ ONLY in the augmentation
# term); mu0 / mu_step = augmentation penalty coefficient and its linear growth.
ALM_HP = {"alm_eta": 0.005, "alm_mu0": 0.01, "alm_mu_step": 0.01}

_cell_lane = {}


def _lane_for(cell_key):
    # Both caps of a (model, seed) cell share one lane so a missing warmup would
    # train once and be reused by the second cap (no cross-GPU cache race).
    if cell_key not in _cell_lane:
        _cell_lane[cell_key] = len(_cell_lane) % N_LANES
    return _cell_lane[cell_key]


def _src(model, cap, seed):
    hits = glob.glob(
        f"{PF}/lane*/{model}/octmnist/{cap}/fioretto_ldf/seed_{seed}/config.json")
    return hits[0] if hits else None


def main():
    assert "pending_runs" not in str(DST_ROOT), "failsafe: must not write into the corpus root"
    emit = skip = 0
    lane_counts = [0] * N_LANES
    for model in BACKBONES:
        for seed in SEEDS:
            lane = _lane_for((model, seed))
            for cap in CAPS:
                src = _src(model, cap, seed)
                assert src, f"missing frozen fioretto_ldf {model}/octmnist/{cap}/seed_{seed}"
                dst_dir = (DST_ROOT / f"lane_gpu{lane}" / model / "octmnist"
                           / cap / "fioretto_alm" / f"seed_{seed}")
                dst = dst_dir / "config.json"
                if dst.exists():
                    skip += 1
                    continue
                c = json.loads(json.dumps(json.load(open(src))))
                c.pop("results", None)
                c["status"] = "pending"
                c["methodology"] = "fioretto_alm"
                c["hyperparams"].update(ALM_HP)
                # base_model_id kept as-is: warmup keys unchanged -> frozen CE
                # warmup cache is reused (fair + fast).
                c["sweep_tag"] = "tmlr_alm_2026-07"
                c["cloned_from"] = src
                c["exp_name"] = f"tmlralm_{model}_octmnist_{cap}_seed{seed}"
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
