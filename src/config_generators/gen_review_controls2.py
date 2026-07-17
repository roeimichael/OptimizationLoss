"""Round-2 review controls on the OctMNIST win region (the six tight-cap cells).

Answers the two attribution challenges the round-2 board raised against the
paper's remaining TraLO-specific claims:

  C2 (penalty shape). The ablation reports the rho schedule and lambda freeze as
  cc-F1-neutral and the graft transfers most of the margin, so the bounded penalty
  is never tested directly. 'tralo_linear' swaps Eq.(1)'s bounded shape for E/K
  (unbounded, constant 1/K gradient) with the ratchet, freeze, optimizer reset and
  undershoot hinge held fixed. If it matches TraLO, the bounded penalty is inert
  and the paper must re-center; if it does not, this run IS the vindication.

  M2 (convergence confound). TraLO starts at lambda=0.05 while both duals start at
  lambda=0; the reported ordering (TraLO 12 < Fioretto 21 < Hounie 38 epochs) is
  what that initialization alone predicts. 'fioretto_init' / 'hounie_init' give the
  hosts TraLO's own starting multiplier, changing nothing else.

Clones the frozen paper_final OctMNIST L30/L40 configs VERBATIM (hyperparams,
base_model_id -> read-only warmup-cache reuse, dataset_config), changing only the
methodology, the probed knob, and the output path.

4 arms (tralo control, tralo_linear, fioretto_init, hounie_init)
 x 2 caps x 3 backbones x 4 seeds = 96 runs.

FAILSAFE: output root is results/review_controls2_2026-07 -- OUTSIDE
results/pending_runs, so this campaign and the corpus dispatchers cannot see each
other's runs. Existing configs are never overwritten (re-runs are idempotent).

Run from the repo root ON THE SERVER:
    python -m src.config_generators.gen_review_controls2
"""

import glob
import json
from pathlib import Path

SRC_PATTERN = ("results/pending_runs/paper_final/lane*/{model}/octmnist/"
               "{tag}/{meth}/seed_{seed}/config.json")
DST_ROOT = Path("results/review_controls2_2026-07")
MODELS = ["MobileNetV3", "RegNetY400MF", "ViTB16"]
TAGS = ["L30_G30", "L40_G40"]
SEEDS = [1, 2, 3, 4]

# host method -> [(emitted methodology, hyperparam overrides, arm label)]
CLONES = {
    "tralo": [
        ("tralo", {}, "control re-run (same hardware/warmup as the probes)"),
        ("tralo", {"penalty_mode": "linear"}, "C2: bounded shape -> E/K"),
    ],
    "fioretto_ldf": [
        ("fioretto_ldf", {"fioretto_lambda_init": 0.05}, "M2: TraLO's starting multiplier"),
    ],
    "hounie_rcl": [
        ("hounie_rcl", {"hounie_lambda_init": 0.05}, "M2: TraLO's starting multiplier"),
    ],
}
# arm name suffix keyed by the override that defines it, so paths stay distinct
ARM_NAME = {
    ("tralo", frozenset()): "tralo",
    ("tralo", frozenset({("penalty_mode", "linear")})): "tralo_linear",
    ("fioretto_ldf", frozenset({("fioretto_lambda_init", 0.05)})): "fioretto_init",
    ("hounie_rcl", frozenset({("hounie_lambda_init", 0.05)})): "hounie_init",
}
N_LANES = 2  # GPU-sharing discipline: max 2 GPUs


def main():
    assert "pending_runs" not in str(DST_ROOT), "failsafe: must not write into the corpus root"
    emitted, skipped = 0, 0
    per_lane = {f"gpu{i+1}": 0 for i in range(N_LANES)}
    for model in MODELS:
        lane_idx = 0  # per-model round robin so each lane gets half the ViT runs
        for tag in TAGS:
            for seed in SEEDS:
                for host, arms in CLONES.items():
                    pattern = SRC_PATTERN.format(model=model, tag=tag, meth=host, seed=seed)
                    hits = glob.glob(pattern)
                    assert len(hits) == 1, f"expected exactly 1 source, got {hits} for {pattern}"
                    with open(hits[0]) as f:
                        src = json.load(f)
                    assert src.get("status") == "completed", f"source not completed: {hits[0]}"
                    assert src["methodology"] == host, hits[0]
                    for meth, over, _why in arms:
                        arm = ARM_NAME[(meth, frozenset(over.items()))]
                        lane = f"gpu{(lane_idx % N_LANES) + 1}"
                        lane_idx += 1
                        dst_dir = DST_ROOT / lane / model / "octmnist" / tag / arm / f"seed_{seed}"
                        dst = dst_dir / "config.json"
                        if dst.exists():
                            skipped += 1
                            continue
                        c = json.loads(json.dumps(src))  # deep copy
                        c.pop("results", None)
                        c["methodology"] = meth
                        c["status"] = "pending"
                        c["sweep_tag"] = "review_controls2_2026-07"
                        c["cloned_from"] = hits[0]
                        c["arm"] = arm
                        c["exp_name"] = f"rc2_{model}_octmnist_{arm}_{tag}_seed{seed}"
                        c["experiment_path"] = str(dst_dir)
                        if over:
                            c["hyperparams"].update(over)
                        dst_dir.mkdir(parents=True, exist_ok=True)
                        with open(dst, "w") as f:
                            json.dump(c, f, indent=1)
                        emitted += 1
                        per_lane[lane] += 1
    total = emitted + skipped
    print(f"emitted={emitted} skipped(existing)={skipped} total={total} (expect 96)")
    print(f"per lane: {per_lane}")
    assert total == 96, f"expected 96 configs, got {total}"


if __name__ == "__main__":
    main()
