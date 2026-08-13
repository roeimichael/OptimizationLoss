"""Config generator for the three campaigns answering blind review round 1.

The review (docs/review/round1_2026-08-13.md) scored 58/100. Most deductions are
text or regeneration; these three are the ones that genuinely need GPU time.

  R1  alm_rh graft      24 runs   ALM + reset + hinge on the six tight-cap cells.
                                  The reviewer's structural point: the paper's
                                  thesis is that the two components carry the
                                  effect, and ALM -- the strongest comparator --
                                  is the only baseline that never received them.

  R2  seeds 5-10       252 runs   Six more seeds on the six tight-cap cells, all
                                  seven methods. n=4 floors a paired Wilcoxon at
                                  p=0.125, so no headline cell is individually
                                  testable today. Every method needs the same
                                  seeds or the comparison stops being paired.

  R3  rerun variance    10 runs   One config, ten repeats, same seed, FRESH
                                  warmup each time. The reviewer measured
                                  0.013 macro-F1 / 0.025 cc-F1 drift between
                                  campaigns re-running identical configurations,
                                  against a +/-0.005 tie band. This measures the
                                  pipeline's own noise floor directly.

All three clone frozen `paper_final` configs and change one axis each, so they
pair against the existing corpus. Nothing is written inside `results/pending_runs`.

Run ON THE SERVER from repo root:
    python -m src.config_generators.gen_review_r1
"""

import glob
import json
from pathlib import Path

from src.config_generators.generate_configs import compute_base_model_id

PF = "results/pending_runs/paper_final"
N_LANES = 4

# The six tight-cap cells the headline rests on.
TIGHT_DATASET = "octmnist"
BACKBONES = ["MobileNetV3", "RegNetY400MF", "ViTB16"]
TIGHT_CAPS = ["L30_G30", "L40_G40"]

# Every method that appears in a headline row, so seeds 5-10 keep the grid square.
METHODS = ["tralo", "tralo_bounded", "fioretto_ldf", "hounie_rcl",
           "fioretto_alm", "heuristic", "danits_lp"]
NEW_SEEDS = [5, 6, 7, 8, 9, 10]

# Matches gen_alm_full.py ALM_HP and src/methodologies/alm_rh/hp_defaults.py.
ALM_HP = {"alm_eta": 0.005, "alm_mu0": 0.01, "alm_mu_step": 0.01}
GRAFT_HP = {"fior_beta": 0.5, "reset_optimizer_at_sat": True}

R3_CELL = ("MobileNetV3", "octmnist", "L30_G30", "tralo", 1)
R3_REPEATS = 10

_cell_lane = {}


def _lane_for(cell_key):
    # One lane per (model, dataset, seed): all caps of a cell share a warmup
    # cache entry, so keeping them on one GPU avoids a cross-GPU cache race.
    if cell_key not in _cell_lane:
        _cell_lane[cell_key] = len(_cell_lane) % N_LANES
    return _cell_lane[cell_key]


def _src(model, dataset, cap, method, seed):
    hits = glob.glob("%s/lane*/%s/%s/%s/%s/seed_%d/config.json"
                     % (PF, model, dataset, cap, method, seed))
    return hits[0] if hits else None


def _clone(src):
    c = json.load(open(src))
    c.pop("results", None)
    c["status"] = "pending"
    c["cloned_from"] = src
    return c


def _write(c, dst_dir):
    dst = dst_dir / "config.json"
    if dst.exists():
        return False
    c["experiment_path"] = str(dst_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)
    with open(dst, "w") as f:
        json.dump(c, f, indent=1)
    return True


def gen_alm_rh(root):
    """R1: ALM + reset + hinge, the six tight-cap cells x 4 seeds."""
    emit, missing = 0, []
    for model in BACKBONES:
        for seed in [1, 2, 3, 4]:
            lane = _lane_for(("r1", model, seed))
            for cap in TIGHT_CAPS:
                # Cloned from fioretto_ldf, exactly as gen_alm_full.py does, so
                # alm_rh and fioretto_alm inherit identical host hyperparameters
                # and the only difference between the arms is the two components.
                src = _src(model, TIGHT_DATASET, cap, "fioretto_ldf", seed)
                if not src:
                    missing.append((model, cap, seed))
                    continue
                c = _clone(src)
                c["methodology"] = "alm_rh"
                c["hyperparams"].update(ALM_HP)
                c["hyperparams"].update(GRAFT_HP)
                # base_model_id untouched: warmup keys unchanged, frozen cache reused.
                c["sweep_tag"] = "review_almrh_2026-08"
                c["exp_name"] = "almrh_%s_%s_%s_seed%d" % (model, TIGHT_DATASET, cap, seed)
                if _write(c, root / ("lane_gpu%d" % lane) / model / TIGHT_DATASET
                          / cap / "alm_rh" / ("seed_%d" % seed)):
                    emit += 1
    return emit, missing


def gen_seeds(root):
    """R2: seeds 5-10 on the six tight-cap cells, every method."""
    emit, missing = 0, []
    for model in BACKBONES:
        for seed in NEW_SEEDS:
            # New seed => new warmup hash => a fresh warmup per (model, seed),
            # shared across that seed's caps and methods. Lane by (model, seed)
            # so the 18 fresh warmups are each trained exactly once.
            lane = _lane_for(("r2", model, seed))
            for cap in TIGHT_CAPS:
                for method in METHODS:
                    src = _src(model, TIGHT_DATASET, cap, method, 1)
                    if not src:
                        missing.append((model, cap, method))
                        continue
                    c = _clone(src)
                    c["hyperparams"]["seed"] = seed
                    # Recompute: the cache key includes the seed, and reusing
                    # seed 1's warmup would make the "new seeds" identical runs.
                    c["base_model_id"] = compute_base_model_id(
                        c["model_name"], c["hyperparams"], c["dataset_mode"],
                        c["dataset_config"]["data_dir"], c["dataset_config"])
                    c["sweep_tag"] = "review_seeds10_2026-08"
                    c["exp_name"] = "seeds10_%s_%s_%s_%s_seed%d" % (
                        model, TIGHT_DATASET, cap, method, seed)
                    if _write(c, root / ("lane_gpu%d" % lane) / model / TIGHT_DATASET
                              / cap / method / ("seed_%d" % seed)):
                        emit += 1
    return emit, missing


def gen_rerun_var(root):
    """R3: one configuration, ten repeats, same seed, fresh warmup each."""
    model, dataset, cap, method, seed = R3_CELL
    src = _src(model, dataset, cap, method, seed)
    if not src:
        return 0, [R3_CELL]
    emit = 0
    for rep in range(1, R3_REPEATS + 1):
        c = _clone(src)
        # Same seed on purpose. The question is how much the pipeline moves when
        # NOTHING is varied -- nondeterministic CUDA kernels, cuDNN autotuning,
        # atomics. Suffixing base_model_id forces a fresh warmup per repeat;
        # sharing one cached warmup would measure only the constraint phase.
        c["base_model_id"] = "%s_rerunvar%02d" % (c["base_model_id"], rep)
        c["sweep_tag"] = "review_rerunvar_2026-08"
        c["exp_name"] = "rerunvar_%s_%s_%s_rep%02d" % (model, dataset, cap, rep)
        if _write(c, root / ("lane_gpu%d" % ((rep - 1) % N_LANES)) / model / dataset
                  / cap / method / ("rep_%02d" % rep)):
            emit += 1
    return emit, []


def main():
    total = 0
    for name, fn in [("r1_almrh", gen_alm_rh),
                     ("r2_seeds10", gen_seeds),
                     ("r3_rerunvar", gen_rerun_var)]:
        root = Path("results/track_b") / name
        assert "pending_runs" not in str(root), "failsafe: must not write into the corpus root"
        emit, missing = fn(root)
        total += emit
        print("%-14s emitted=%-4d root=%s" % (name, emit, root))
        if missing:
            print("   WARNING: %d cells had no frozen source: %s" % (len(missing), missing[:5]))
    print("total emitted = %d" % total)


if __name__ == "__main__":
    main()
