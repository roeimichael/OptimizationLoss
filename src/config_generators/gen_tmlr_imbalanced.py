"""TMLR Track B / B1: imbalanced-learning baselines campaign (expanded, rigorous).

Emits focal / class_balanced / logit_adjust runs (fine-tune the shared CE warmup
with the imbalanced loss, then Shifman-LP clip) across a grid that EXCEEDS the
handoff's minimal 108-run scope:

  methods : focal, class_balanced, logit_adjust  (+ tralo, danits_lp reference
            arms ONLY where paper_final has no frozen result to pair against)
  datasets: octmnist, dermmnist, tissuemnist
  backbones: MobileNetV3, RegNetY400MF, ViTB16 (frozen refs reused) + MobileNetV2 (B8)
  caps    : L30_G30, L40_G40
  seeds   : 1-8  (paper_final has 1-4; 5-8 are new -> fresh per-seed warmups)

Cloned VERBATIM from frozen paper_final configs (same dataset_config, HPs,
warmup-cache identity) so the comparison is apples-to-apples by construction --
exactly the review-graft pattern. Only methodology / model_name / seed / cap /
output-path change; base_model_id is recomputed so new-model/new-seed cells get
their own warmup while reused-warmup cells share the frozen one.

Priority via tier prefix (t1 core -> t2 seed-expansion -> t3 MobileNetV2) and
round-robin lane assignment so all 3 GPUs clear the core tier first.

FAILSAFE: output root results/tmlr_track_b/imbalanced_2026-07 is OUTSIDE
results/pending_runs, so this campaign and the frozen corpus can never see each
other. Idempotent: existing target configs are never overwritten.

Run ON THE SERVER from repo root:
    python -m src.config_generators.gen_tmlr_imbalanced
"""

import glob
import json
from pathlib import Path

from src.config_generators.generate_configs import compute_base_model_id

PF = "results/pending_runs/paper_final"
DST_ROOT = Path("results/tmlr_track_b/imbalanced_2026-07")
N_LANES = 3

DATASETS = ["octmnist", "dermmnist", "tissuemnist"]
EXISTING_BB = ["MobileNetV3", "RegNetY400MF", "ViTB16"]
NEW_BB = "MobileNetV2"
CAPS = ["L30_G30", "L40_G40"]
IMB = ["focal", "class_balanced", "logit_adjust"]
REFS = ["tralo", "danits_lp"]  # paired reference arms for NEW cells only

IMB_HP = {
    "focal": {"focal_alpha": 0.25, "focal_gamma": 2.0},
    "class_balanced": {"cb_beta": 0.9999},
    "logit_adjust": {"logit_adjust_tau": 1.0},
}
# Imbalanced baselines TRAIN the backbone with the imbalanced loss in the shared
# warmup phase (hp['warmup_loss']), reusing the CE warmup's epochs/lr for
# fairness -- only the loss differs. The trained model caches under a
# loss-suffixed base_model_id so both caps (L30/L40) reuse a single training.

# constraint-method-only HPs to drop when repurposing a tralo config as an
# imbalanced-baseline config (harmless if kept; dropped for cleanliness).
TRALO_ONLY = ["lambda_global", "lambda_local", "lambda_step", "initial_rho",
              "rho_target", "alpha_kl", "penalty_mode", "hybrid_mode",
              "fior_beta", "reset_optimizer_at_sat", "enable_ce_skip"]

_stats = {"emit": 0, "skip": 0, "lane": [0] * N_LANES, "warmups_new": set()}
# Both caps of a cell share ONE lane so they run sequentially on the same GPU:
# the first trains the warmup, the second reuses the cache (no duplicate train,
# no cache write-race across GPUs). Cells are round-robined across lanes.
_cell_lane = {}


def _lane_for(cell_key):
    if cell_key not in _cell_lane:
        _cell_lane[cell_key] = len(_cell_lane) % N_LANES
    return _cell_lane[cell_key]


def _pf_config(model, ds, cap, meth, seed):
    hits = glob.glob(f"{PF}/lane*/{model}/{ds}/{cap}/{meth}/seed_{seed}/config.json")
    return hits[0] if hits else None


def _cap_pair(cap):
    local = int(cap[1:3]) / 100.0
    glob_ = int(cap.split("_G")[1][:2]) / 100.0
    return [local, glob_]


def _emit(src_path, tier, model, ds, cap, meth, seed, is_imb):
    # Group ALL methods+caps of a (tier,model,ds,seed) cell onto one lane so
    # warmups (incl. the CE warmup shared by tralo+danits_lp refs) train once
    # and are reused sequentially -- no duplicate training, no cache write-race.
    lane = _lane_for((tier, model, ds, seed))
    dst_dir = DST_ROOT / f"lane_gpu{lane}" / f"t{tier}" / model / ds / cap / meth / f"seed_{seed}"
    dst = dst_dir / "config.json"
    if dst.exists():
        _stats["skip"] += 1
        return
    c = json.loads(json.dumps(json.load(open(src_path))))
    c.pop("results", None)
    c["status"] = "pending"
    c["methodology"] = meth
    c["model_name"] = model
    c["constraint"] = _cap_pair(cap)
    c["constraint_tag"] = cap
    c["hyperparams"]["seed"] = seed
    if is_imb:
        for k in TRALO_ONLY + ["constraint_epochs", "lr_constraint"]:
            c["hyperparams"].pop(k, None)
        c["hyperparams"]["warmup_loss"] = meth
        c["hyperparams"].update(IMB_HP[meth])
    c["base_model_id"] = compute_base_model_id(
        model, c["hyperparams"], c["dataset_mode"],
        c["dataset_config"]["data_dir"], c["dataset_config"])
    if is_imb:
        c["base_model_id"] += f"_{meth}"  # distinct cache per loss; both caps share one training
    c["sweep_tag"] = "tmlr_imbalanced_2026-07"
    c["cloned_from"] = src_path
    c["exp_name"] = f"tmlrimb_{model}_{ds}_{meth}_{cap}_seed{seed}"
    c["experiment_path"] = str(dst_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)
    with open(dst, "w") as f:
        json.dump(c, f, indent=1)
    _stats["emit"] += 1
    _stats["lane"][lane] += 1
    _stats["warmups_new"].add(c["base_model_id"])


def main():
    assert "pending_runs" not in str(DST_ROOT), "failsafe: must not write into the corpus root"

    # Tier 1 -- core: existing backbones, seeds 1-4, imbalanced only.
    # Pairs against FROZEN tralo/danits_lp already in paper_final (no re-runs).
    for ds in DATASETS:
        for model in EXISTING_BB:
            for cap in CAPS:
                for seed in (1, 2, 3, 4):
                    src = _pf_config(model, ds, cap, "tralo", seed)
                    assert src, f"missing frozen source {model}/{ds}/{cap}/tralo/seed_{seed}"
                    for meth in IMB:
                        _emit(src, 1, model, ds, cap, meth, seed, True)

    # Tier 2 -- seed expansion: existing backbones, seeds 5-8.
    # New per-seed warmups; run imbalanced + refs so the paired test spans 8 seeds.
    for ds in DATASETS:
        for model in EXISTING_BB:
            for cap in CAPS:
                for seed in (5, 6, 7, 8):
                    tralo_src = _pf_config(model, ds, cap, "tralo", 4)
                    assert tralo_src
                    for meth in IMB:
                        _emit(tralo_src, 2, model, ds, cap, meth, seed, True)
                    for meth in REFS:
                        ref = _pf_config(model, ds, cap, meth, 4)
                        assert ref, f"missing ref {meth} {model}/{ds}/{cap}"
                        _emit(ref, 2, model, ds, cap, meth, seed, False)

    # Tier 3 -- MobileNetV2 (B8): new backbone, seeds 1-8, imbalanced + refs.
    for ds in DATASETS:
        for cap in CAPS:
            for seed in (1, 2, 3, 4, 5, 6, 7, 8):
                mnv3 = _pf_config("MobileNetV3", ds, cap, "tralo", min(seed, 4))
                assert mnv3
                for meth in IMB:
                    _emit(mnv3, 3, NEW_BB, ds, cap, meth, seed, True)
                for meth in REFS:
                    ref = _pf_config("MobileNetV3", ds, cap, meth, min(seed, 4))
                    assert ref
                    _emit(ref, 3, NEW_BB, ds, cap, meth, seed, False)

    print(f"emitted={_stats['emit']} skipped(existing)={_stats['skip']} "
          f"per_lane={_stats['lane']} distinct_new_warmups={len(_stats['warmups_new'])}")
    print(f"root={DST_ROOT}")


if __name__ == "__main__":
    main()
