"""Review-response campaign: graft (B1) + anti-windup (B2) on the OctMNIST win region.

Answers the peer-review attribution question (DA-CRITICAL-2 / P0-2): is TraLO's
tight-cap quality edge carried by the bounded penalty, or by the two portable
components (optimizer reset at satisfaction + undershoot hinge)? And does an
anti-windup dual baseline (dual restart) erase the edge (R1-W2 / R2-W1)?

Clones the frozen paper_final OctMNIST L30/L40 configs VERBATIM (hyperparams,
base_model_id -> read-only warmup-cache reuse, dataset_config), changing only
the methodology and the output path:

  host tralo        -> 'tralo'            (same-hardware control re-run)
  host fioretto_ldf -> 'fioretto_ldf'     (control re-run)
                    -> 'fioretto_rh'      (B1 graft: + reset + hinge)
                    -> 'fioretto_restart' (B2 anti-windup: dual restart)
  host hounie_rcl   -> 'hounie_rcl'       (control re-run)
                    -> 'hounie_rh'        (B1 graft: + reset + hinge)

6 methods x 2 caps x 3 backbones x 4 seeds = 144 runs.

FAILSAFE: output root is results/review_graft_2026-07 -- OUTSIDE
results/pending_runs, so neither this campaign's dispatchers nor the corpus
dispatchers can see each other's runs. Existing target configs are never
overwritten (idempotent re-runs of this script are safe). Lane subdirs gpu1/
gpu2 partition the campaign for two per-GPU dispatchers, round-robin per
model so ViT load is split evenly.

Run from the repo root ON THE SERVER:
    python -m src.config_generators.gen_review_graft
"""

import glob
import json
from pathlib import Path

SRC_PATTERN = ("results/pending_runs/paper_final/lane*/{model}/octmnist/"
               "{tag}/{meth}/seed_{seed}/config.json")
DST_ROOT = Path("results/review_graft_2026-07")
MODELS = ["MobileNetV3", "RegNetY400MF", "ViTB16"]
TAGS = ["L30_G30", "L40_G40"]
SEEDS = [1, 2, 3, 4]
CLONES = {  # host method -> methodologies to emit from that host's config
    "tralo": ["tralo"],
    "fioretto_ldf": ["fioretto_ldf", "fioretto_rh", "fioretto_restart"],
    "hounie_rcl": ["hounie_rcl", "hounie_rh"],
}
GRAFT_HP = {"fior_beta": 0.5, "reset_optimizer_at_sat": True}  # TraLO's shipped values
N_LANES = 2  # dispatch on 2 GPUs (GPU-sharing discipline: max 2)


def main():
    assert "pending_runs" not in str(DST_ROOT), "failsafe: must not write into the corpus root"
    emitted, skipped = 0, 0
    per_lane = {f"gpu{i+1}": 0 for i in range(N_LANES)}
    for model in MODELS:
        lane_idx = 0  # per-model round robin so each lane gets half the ViT runs
        for tag in TAGS:
            for seed in SEEDS:
                for host, outs in CLONES.items():
                    pattern = SRC_PATTERN.format(model=model, tag=tag, meth=host, seed=seed)
                    hits = glob.glob(pattern)
                    assert len(hits) == 1, f"expected exactly 1 source, got {hits} for {pattern}"
                    with open(hits[0]) as f:
                        src = json.load(f)
                    assert src.get("status") == "completed", f"source not completed: {hits[0]}"
                    assert src["methodology"] == host, hits[0]
                    for new_meth in outs:
                        lane = f"gpu{(lane_idx % N_LANES) + 1}"
                        lane_idx += 1
                        dst_dir = DST_ROOT / lane / model / "octmnist" / tag / new_meth / f"seed_{seed}"
                        dst = dst_dir / "config.json"
                        if dst.exists():
                            skipped += 1
                            continue
                        c = json.loads(json.dumps(src))  # deep copy
                        c.pop("results", None)
                        c["methodology"] = new_meth
                        c["status"] = "pending"
                        c["sweep_tag"] = "review_graft_2026-07"
                        c["cloned_from"] = hits[0]
                        c["exp_name"] = f"reviewgraft_{model}_octmnist_{new_meth}_{tag}_seed{seed}"
                        c["experiment_path"] = str(dst_dir)
                        if new_meth in ("fioretto_rh", "hounie_rh"):
                            c["hyperparams"].update(GRAFT_HP)
                        dst_dir.mkdir(parents=True, exist_ok=True)
                        with open(dst, "w") as f:
                            json.dump(c, f, indent=1)
                        emitted += 1
                        per_lane[lane] += 1
    total = emitted + skipped
    print(f"emitted={emitted} skipped(existing)={skipped} total={total} (expect 144)")
    print(f"per lane: {per_lane}")
    assert total == 144, f"expected 144 configs, got {total}"


if __name__ == "__main__":
    main()
