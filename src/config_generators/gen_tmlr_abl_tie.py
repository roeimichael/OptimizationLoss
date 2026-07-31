"""TMLR Track B / B4: component ablation at a TIE-REGION cell (DermMNIST L50).

The paper's leave-one-out component ablation (reset / hinge / rho-schedule /
lambda-freeze) was run only on tight-cap OctMNIST -- the win region. R1 M4 +
Devil's Advocate asked: are those components load-bearing in a TIE region too?
B4 repeats the ablation at DermMNIST L50 symmetric with MobileNetV3.

Clones the frozen dermmnist L50 `tralo` config (= the 'full' variant) and
toggles ONE component per variant. base_model_id is unchanged (warmup keys
untouched), so all 5 variants reuse the frozen CE warmup for that seed.

Mirrors gen_g5_component_ablation's toggle set (the 4 handoff components +
full). Does NOT touch `disable_lambda_toggle` (the lambda toggle is sacrosanct
per project rule -- only `disable_freeze_on_satisfy` = post-satisfaction freeze
is ablated as the 'lambda freeze' component).

5 variants x 4 seeds = 20 runs. Root: results/tmlr_track_b/abl_tie_2026-07/
"""

import glob
import json
from pathlib import Path

PF = "results/pending_runs/paper_final"
DST_ROOT = Path("results/tmlr_track_b/abl_tie_2026-07")
N_LANES = 3
MODEL = "MobileNetV3"
DS = "dermmnist"
CAP = "L50_G50"
SEEDS = [1, 2, 3, 4]

# full + leave-one-out on the 4 handoff components. `no_rho_sched` is special
# (flatten the schedule: rho_target <- initial_rho), handled in code.
VARIANTS = {
    "full": {},
    "no_reset": {"reset_optimizer_at_sat": False},
    "no_hinge": {"hybrid_mode": "bounded_only"},
    "no_rho_sched": "FLATTEN_RHO",
    "no_freeze": {"disable_freeze_on_satisfy": True},
}

_cell_lane = {}


def _lane(seed):
    if seed not in _cell_lane:
        _cell_lane[seed] = len(_cell_lane) % N_LANES
    return _cell_lane[seed]


def _src(seed):
    hits = glob.glob(f"{PF}/lane*/{MODEL}/{DS}/{CAP}/tralo/seed_{seed}/config.json")
    return hits[0] if hits else None


def main():
    assert "pending_runs" not in str(DST_ROOT), "failsafe: must not write into the corpus root"
    emit = skip = 0
    lc = [0] * N_LANES
    for seed in SEEDS:
        src = _src(seed)
        assert src, f"missing frozen {MODEL}/{DS}/{CAP}/tralo/seed_{seed}"
        lane = _lane(seed)
        for vname, ov in VARIANTS.items():
            dst_dir = DST_ROOT / f"lane_gpu{lane}" / MODEL / DS / CAP / vname / f"seed_{seed}"
            dst = dst_dir / "config.json"
            if dst.exists():
                skip += 1
                continue
            c = json.loads(json.dumps(json.load(open(src))))
            c.pop("results", None)
            c["status"] = "pending"
            c["methodology"] = "tralo"
            hp = c["hyperparams"]
            if ov == "FLATTEN_RHO":
                hp["rho_target"] = hp.get("initial_rho", hp.get("rho_target"))
            elif ov:
                hp.update(ov)
            # base_model_id kept as-is: warmup keys unchanged -> frozen CE warmup reused.
            c["sweep_tag"] = "tmlr_abl_tie_2026-07"
            c["cloned_from"] = src
            c["exp_name"] = f"tmlrabl_{vname}_{DS}_{CAP}_seed{seed}"
            c["experiment_path"] = str(dst_dir)
            dst_dir.mkdir(parents=True, exist_ok=True)
            with open(dst, "w") as f:
                json.dump(c, f, indent=1)
            emit += 1
            lc[lane] += 1
    print(f"emitted={emit} skipped(existing)={skip} per_lane={lc}")
    print(f"root={DST_ROOT}")


if __name__ == "__main__":
    main()
