"""TMLR Track B / B8: MobileNetV2 corroboration of the OctMNIST tight-cap result.

The headline tight-cap OctMNIST cc-F1 comparison (TraLO vs clip / duals) was shown on
MobileNetV3 / RegNet / ViT. B8 adds MobileNetV2 as an independent 2nd backbone to confirm
the result is not backbone-specific.

Clones the frozen paper_final MobileNetV3 octmnist configs (core comparison methods x
L30/L40 x seeds 1-4) and swaps ONLY model_name -> MobileNetV2, recomputing base_model_id
so a FRESH MNV2 CE warmup is trained (the MNV3 cache is never reused -- model_name is in
the hash). Everything else (method HPs, caps, dataset_config, seeds) is identical, so the
comparison is apples-to-apples at the same cells.

4 methods x 2 caps x 4 seeds = 32 runs. Root: results/track_b/b8/
FAILSAFE: writes OUTSIDE results/pending_runs. Idempotent. Run on server from repo root:
    python -m src.config_generators.gen_b8_mnv2_oct
"""
import glob
import json
from pathlib import Path

from src.config_generators.generate_configs import compute_base_model_id

PF = "results/pending_runs/paper_final"
DST_ROOT = Path("results/track_b/b8")
NEW_MODEL = "MobileNetV2"
SRC_MODEL = "MobileNetV3"
DS = "octmnist"
METHODS = ["tralo", "danits_lp", "fioretto_ldf", "hounie_rcl"]
CAPS = ["L30_G30", "L40_G40"]
SEEDS = [1, 2, 3, 4]


def _src(method, cap, seed):
    hits = glob.glob(f"{PF}/lane*/{SRC_MODEL}/{DS}/{cap}/{method}/seed_{seed}/config.json")
    return hits[0] if hits else None


def main():
    assert "pending_runs" not in str(DST_ROOT), "failsafe: must not write into the corpus root"
    emit = skip = miss = 0
    for method in METHODS:
        for cap in CAPS:
            for seed in SEEDS:
                src = _src(method, cap, seed)
                if not src:
                    miss += 1
                    print(f"  MISS frozen src {SRC_MODEL}/{DS}/{cap}/{method}/seed_{seed}")
                    continue
                dst_dir = DST_ROOT / NEW_MODEL / DS / cap / method / f"seed_{seed}"
                dst = dst_dir / "config.json"
                if dst.exists():
                    skip += 1
                    continue
                c = json.loads(json.dumps(json.load(open(src))))
                c.pop("results", None)
                c["status"] = "pending"
                c["model_name"] = NEW_MODEL
                c["base_model_id"] = compute_base_model_id(
                    NEW_MODEL, c["hyperparams"], c["dataset_mode"],
                    c["dataset_config"]["data_dir"], c["dataset_config"])
                c["sweep_tag"] = "track_b_b8_2026-07-28"
                c["cloned_from"] = src
                c["exp_name"] = f"b8_{NEW_MODEL}_{DS}_{cap}_{method}_seed{seed}"
                c["experiment_path"] = str(dst_dir)
                dst_dir.mkdir(parents=True, exist_ok=True)
                with open(dst, "w") as f:
                    json.dump(c, f, indent=1)
                emit += 1
    print(f"emitted={emit} skipped(existing)={skip} missing_src={miss} root={DST_ROOT}")


if __name__ == "__main__":
    main()
