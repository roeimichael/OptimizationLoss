"""ALM baseline on MobileNetV2 -- the fourth backbone.

gen_alm_full.py clones the frozen `paper_final` tree, which contains only the
three main-grid backbones. MobileNetV2 lives in older sweeps, so its ALM arm
needs its own generator.

Clone source: any completed MobileNetV2 fioretto_ldf config at warmup_epochs=50.
Those configs carry hyperparameters byte-identical to paper_final (lr 1e-4,
lr_constraint 5e-6, dropout 0.3, batch 64, warmup 50, constraint_epochs 300,
fioretto_step 0.005), so swapping only the dual rule gives the same
apples-to-apples pairing as the other three backbones, and the existing
MobileNetV2 warmup cache is reused.

Preference order when several sources exist for one (dataset, cap, seed): the
sweep with the most complete coverage wins, then lexical order, so the choice is
deterministic across reruns.

FAILSAFE: writes to results/track_b/b3_mnv2, outside results/pending_runs.
Idempotent -- existing targets are skipped.

Run ON THE SERVER from repo root:
    python -m src.config_generators.gen_alm_mnv2
"""

import glob
import json
from pathlib import Path

SEARCH_ROOTS = ["results/pending_runs", "results/track_b", "results/baselines"]
DST_ROOT = Path("results/track_b/b3_mnv2")
N_LANES = 4

DATASETS = ["octmnist", "dermmnist", "tissuemnist"]
# Caps the headline tables use come first so they finish first if the run is cut
# short; the rest complete the nine-cap sweep for the figures.
PRIORITY_CAPS = ["L30_G30", "L50_G50", "L70_G70"]
OTHER_CAPS = ["L20_G20", "L40_G40", "L60_G60", "L80_G80"]
SEEDS = [1, 2, 3, 4]

ALM_HP = {"alm_eta": 0.005, "alm_mu0": 0.01, "alm_mu_step": 0.01}

_cell_lane = {}


def _lane_for(cell_key):
    # All caps of a (dataset, seed) cell share one warmup cache entry, so they
    # must stay on one GPU or two lanes would train the same warmup twice.
    if cell_key not in _cell_lane:
        _cell_lane[cell_key] = len(_cell_lane) % N_LANES
    return _cell_lane[cell_key]


def _find_sources():
    """Map (dataset, cap, seed) -> path of a usable MobileNetV2 fioretto_ldf config."""
    best = {}
    for root in SEARCH_ROOTS:
        pat = "%s/**/MobileNetV2/**/fioretto_ldf/**/config.json" % root
        for f in glob.glob(pat, recursive=True):
            try:
                c = json.load(open(f))
            except Exception:
                continue
            hp = c.get("hyperparams", {})
            if c.get("model_name") != "MobileNetV2" or hp.get("warmup_epochs") != 50:
                continue
            key = (c.get("dataset_mode"), c.get("constraint_tag"), hp.get("seed"))
            if None in key:
                continue
            # deterministic pick: shortest path, then lexical
            if key not in best or (len(f), f) < (len(best[key]), best[key]):
                best[key] = f
    return best


def main():
    assert "pending_runs" not in str(DST_ROOT), "failsafe: must not write into the corpus root"
    sources = _find_sources()
    print("found %d MobileNetV2 fioretto_ldf sources at warmup 50" % len(sources))

    emit = skip_exists = 0
    no_source = []
    lane_counts = [0] * N_LANES
    for cap in PRIORITY_CAPS + OTHER_CAPS:
        for dataset in DATASETS:
            for seed in SEEDS:
                src = sources.get((dataset, cap, seed))
                if not src:
                    no_source.append((dataset, cap, seed))
                    continue
                lane = _lane_for((dataset, seed))
                dst_dir = (DST_ROOT / ("lane_gpu%d" % lane) / "MobileNetV2" / dataset
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
                # base_model_id untouched: warmup keys unchanged -> reuse the
                # existing MobileNetV2 warmup cache.
                c["sweep_tag"] = "alm_mnv2_2026-08"
                c["cloned_from"] = src
                c["exp_name"] = "almmnv2_%s_%s_seed%d" % (dataset, cap, seed)
                c["experiment_path"] = str(dst_dir)
                dst_dir.mkdir(parents=True, exist_ok=True)
                with open(dst, "w") as f:
                    json.dump(c, f, indent=1)
                emit += 1
                lane_counts[lane] += 1

    print("emitted=%d  skipped(existing)=%d" % (emit, skip_exists))
    print("per_lane=%s" % lane_counts)
    print("root=%s" % DST_ROOT)
    prio = [k for k in no_source if k[1] in PRIORITY_CAPS]
    print("no source: %d cells (%d of them at the headline caps L30/L50/L70)"
          % (len(no_source), len(prio)))
    for k in prio[:12]:
        print("    MISSING HEADLINE CELL:", k)


if __name__ == "__main__":
    main()
