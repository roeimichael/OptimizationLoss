"""Does the no-restore win survive outside its four cells?

The claim, from `norestore` (48 runs, one campaign, equal compute):

    at a fixed budget K, TraLO with the checkpoint restore disabled puts more
    true positives inside the budget than plain CE + post-hoc clipping
    (+0.0146 cc-F1, p=0.0075, 13/16 pairs, positive in all four cells), with
    macro-F1 unchanged and roughly half the post-hoc flips.

Every one of those four cells is L30_G30. That is the single most likely way
for this to be an artifact, and this project has watched exactly that failure
before: focal looked like a result at 11/16 cells on L30 and collapsed to 4/8
at L20 and L40. So the cap axis is tested first and hardest.

Three slices, each a set of WHOLE cells (both arms, all seeds) so that every
paired comparison stays inside one lane on one card -- the cross-campaign drift
here is 0.027, which is twice the effect being measured.

    caps_derm   dermmnist   x {MobileNetV3, RegNetY400MF} x {L20, L40, L50}
    caps_oct    octmnist    x {MobileNetV3, RegNetY400MF} x {L20, L40, L50}
    wide        tissuemnist x {MobileNetV3, RegNetY400MF} x {L20, L30, L40}
                + {dermmnist, octmnist} x {MobileNetV2, ShuffleNetV2} x L30

The cap tag is a FRACTION OF THE TRUE POSITIVE COUNT, not an absolute count
(dermmnist L30 -> K=67 against n_pos=223), so these levels really do move the
budget: L20 is a tight cap, L50 a loose one.

`constraint` is not part of base_model_id, so the two caps slices reuse the
warm-ups already cached by `norestore`. The wide slice trains new ones.

    usage: gen_replicate.py --slice caps_derm --root results/repcaps_derm
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.abspath("."))
sys.path.insert(0, os.path.abspath("paper/scripts"))
from src.config_generators.generate_configs import compute_base_model_id  # noqa: E402
from gen_headroom import DS_CFG  # noqa: E402

SHARED = {"lr": 1e-4, "dropout": 0.3, "batch_size": 64, "pretrained": True,
          "class_weighted_ce": False, "constraint_chunk_size": 256,
          "enable_ce_skip": False, "stable_count_threshold": 31}
TRALO = {"lambda_step": 0.05, "rho_step": 0.05, "initial_rho": 0.5,
         "alpha_kl": 0.0, "penalty_mode": "both"}

# Two arms only. The restore question is closed -- an arm-level campaign and a
# within-run probe independently measured the same -0.0351 AP -- so spending a
# third of the budget re-confirming it would buy nothing. What is open is
# whether restore-off still beats the clipper away from L30.
#
# Equal compute: 30 optimizer epochs on both sides.
ARMS = {
    "tralo_norestore": dict(methodology="tralo",
                            hp={"warmup_epochs": 1, "constraint_epochs": 29,
                                "lr_constraint": 1e-4,
                                "enable_checkpoint_restore": False}),
    "clip": dict(methodology="heuristic",
                 hp={"warmup_epochs": 30, "constraint_epochs": 0,
                     "lr_constraint": 1e-4}),
}

SEEDS = [1, 2, 3, 4]

SLICES = {
    "caps_derm": [("dermmnist", m, c)
                  for m in ("MobileNetV3", "RegNetY400MF")
                  for c in ("L20_G20", "L40_G40", "L50_G50")],
    "caps_oct": [("octmnist", m, c)
                 for m in ("MobileNetV3", "RegNetY400MF")
                 for c in ("L20_G20", "L40_G40", "L50_G50")],
    "wide": ([("tissuemnist", m, c)
              for m in ("MobileNetV3", "RegNetY400MF")
              for c in ("L20_G20", "L30_G30", "L40_G40")]
             + [(d, m, "L30_G30")
                for d in ("dermmnist", "octmnist")
                for m in ("MobileNetV2", "ShuffleNetV2")]),
}


def pair(tag):
    p = tag.split("_")
    return [int(p[0][1:]) / 100, int(p[1][1:]) / 100]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slice", required=True, choices=sorted(SLICES))
    ap.add_argument("--root", required=True)
    args = ap.parse_args()

    cells = SLICES[args.slice]
    todo = [(ds, mo, tag, arm, spec, seed)
            for (ds, mo, tag) in cells
            for arm, spec in ARMS.items() for seed in SEEDS]
    # Seed-major. If the night runs short this leaves EVERY cell at a lower seed
    # count rather than some cells complete and others absent -- the paired test
    # is across cells, so cell coverage is what the claim actually needs.
    todo.sort(key=lambda c: (c[5], c[0], c[1], c[2], c[3]))

    for ds, model, tag, arm, spec, seed in todo:
        dc = DS_CFG[ds]
        hp = {**SHARED, **TRALO, **spec["hp"], "seed": seed}
        bmid = compute_base_model_id(model, hp, ds, dc["data_dir"], dc)
        path = "%s/%s/%s/%s/%s/seed_%d" % (args.root, model, ds, tag, arm, seed)
        cfg = {"methodology": spec["methodology"], "model_name": model,
               "constraint": pair(tag), "constraint_tag": tag,
               "dataset_mode": ds, "dataset_config": dc, "hyperparams": hp,
               "base_model_id": bmid, "sweep_tag": args.slice, "arm": arm,
               "exp_name": "%s_%s_%s_%s_%s_seed%d" % (args.slice, model, ds, arm, tag, seed),
               "experiment_path": path, "status": "pending"}
        os.makedirs(path, exist_ok=True)
        json.dump(cfg, open(os.path.join(path, "config.json"), "w"), indent=2)

    print("slice %s: %d cells x %d seeds x %d arms = %d runs -> %s"
          % (args.slice, len(cells), len(SEEDS), len(ARMS), len(todo), args.root))
    for c in cells:
        print("   %-12s %-14s %s" % c)
    return 0


if __name__ == "__main__":
    sys.exit(main())
