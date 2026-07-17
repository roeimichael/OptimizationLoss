"""Round-3 review control: TraLO-side hyperparameter sensitivity sweep.

Answers the methodology reviewer's one real fairness gap (MAJOR-3): the dual
baselines' step size is swept x10 to confirm fairness, but TraLO's own load-bearing
knobs (the hinge weight beta and the ratchet step lambda_step) are only ever
ablated on/off, never swept. In a transductive setting there is no held-out split,
so "fixed a priori" does not by itself rule out test-informed selection. This sweep
is symmetric to the baseline step-fairness sweep: it perturbs TraLO's two continuous
knobs around the frozen recipe and checks the OctMNIST tight-cap win survives.

The rank-1 mechanism (Sec. 3) predicts lambda_step is ~neutral (it rescales the
multiplier ratchet, a positive scalar on the fixed soft-count direction, absorbed by
Adam's normalization and the clip rail) while beta -- the hinge, a genuine direction
change active only when S<K -- should matter but the win should be robust across
reasonable values.

Clones the frozen paper_final OctMNIST L30/L40 tralo configs VERBATIM (base_model_id
-> read-only warmup-cache reuse, dataset_config, everything), changing only the one
probed knob and the output path. Frozen defaults confirmed on server:
fior_beta=0.5, lambda_step=0.002.

5 arms (beta 0.25/0.75/1.0, lambda_step 0.001/0.004)
 x 2 caps x 3 backbones x 4 seeds = 120 runs.

FAILSAFE: output root is results/review_controls3_2026-07 -- OUTSIDE
results/pending_runs, so this campaign and the corpus dispatchers cannot see each
other's runs. Existing configs are never overwritten (re-runs are idempotent).

Run from the repo root ON THE SERVER:
    python -m src.config_generators.gen_review_controls3
"""

import glob
import json
from pathlib import Path

SRC_PATTERN = ("results/pending_runs/paper_final/lane*/{model}/octmnist/"
               "{tag}/tralo/seed_{seed}/config.json")
DST_ROOT = Path("results/review_controls3_2026-07")
MODELS = ["MobileNetV3", "RegNetY400MF", "ViTB16"]
TAGS = ["L30_G30", "L40_G40"]
SEEDS = [1, 2, 3, 4]

# arm label -> hyperparam override (all arms stay methodology=tralo)
ARMS = {
    "beta025":  {"fior_beta": 0.25},
    "beta075":  {"fior_beta": 0.75},
    "beta100":  {"fior_beta": 1.0},
    "lstep001": {"lambda_step": 0.001},
    "lstep004": {"lambda_step": 0.004},
}
# frozen-recipe values the source MUST already carry, so an override is a real change
FROZEN = {"fior_beta": 0.5, "lambda_step": 0.002}
N_LANES = 2  # GPU-sharing discipline: max 2 GPUs


def main():
    assert "pending_runs" not in str(DST_ROOT), "failsafe: must not write into the corpus root"
    emitted, skipped = 0, 0
    per_lane = {f"gpu{i+1}": 0 for i in range(N_LANES)}
    lane_idx = 0
    for model in MODELS:
        for tag in TAGS:
            for seed in SEEDS:
                pattern = SRC_PATTERN.format(model=model, tag=tag, seed=seed)
                hits = glob.glob(pattern)
                assert len(hits) == 1, f"expected exactly 1 source, got {hits} for {pattern}"
                with open(hits[0]) as f:
                    src = json.load(f)
                assert src.get("status") == "completed", f"source not completed: {hits[0]}"
                assert src["methodology"] == "tralo", hits[0]
                for k, v in FROZEN.items():
                    assert src["hyperparams"].get(k) == v, \
                        f"source {k}={src['hyperparams'].get(k)} != frozen {v}: {hits[0]}"
                for arm, over in ARMS.items():
                    lane = f"gpu{(lane_idx % N_LANES) + 1}"
                    lane_idx += 1
                    dst_dir = DST_ROOT / lane / model / "octmnist" / tag / arm / f"seed_{seed}"
                    dst = dst_dir / "config.json"
                    if dst.exists():
                        skipped += 1
                        continue
                    c = json.loads(json.dumps(src))  # deep copy
                    c.pop("results", None)
                    c["status"] = "pending"
                    c["sweep_tag"] = "review_controls3_2026-07"
                    c["cloned_from"] = hits[0]
                    c["arm"] = arm
                    c["exp_name"] = f"rc3_{model}_octmnist_{arm}_{tag}_seed{seed}"
                    c["experiment_path"] = str(dst_dir)
                    c["hyperparams"].update(over)
                    dst_dir.mkdir(parents=True, exist_ok=True)
                    with open(dst, "w") as f:
                        json.dump(c, f, indent=1)
                    emitted += 1
                    per_lane[lane] += 1
    total = emitted + skipped
    print(f"emitted={emitted} skipped(existing)={skipped} total={total} (expect 120)")
    print(f"per lane: {per_lane}")
    assert total == 120, f"expected 120 configs, got {total}"


if __name__ == "__main__":
    main()
