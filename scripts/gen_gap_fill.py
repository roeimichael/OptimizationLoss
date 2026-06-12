"""Gap-fill generator: create heuristic+danits_lp configs for TraLO push-pull cells.

Run on dsisco02 directly (configs need to land in results/pending_runs/).

For each TraLO config at a push-pull cell (warmup<=3, fsat<0.85) that lacks
heuristic+danits_lp baselines: clone the config, swap methodology and
experiment_path, save to disk. The main dispatcher picks them up.
"""
import glob
import json
import os
import shutil
import sys
from pathlib import Path

ROOT = "results/pending_runs"
NEW_METHODS = ["heuristic", "danits_lp"]

# Push-pull TraLO cells with missing heuristic+danits_lp baselines
# (sweep, ds, model, tag, cls) tuples — from scripts/find_gaps.py output.
# These are the 21 cells where we want to add the 2 missing post-hoc baselines.
TARGET_CELLS = [
    # blackwell_validation - main wins from warmup=1/3
    ("blackwell_validation", "aider", "MobileNetV2", "L50_G50", 0),
    ("blackwell_validation", "aider", "ShuffleNetV2", "L50_G50", 0),
    ("blackwell_validation", "dermmnist", "MobileNetV2", "L50_G50", 4),
    ("blackwell_validation", "dermmnist", "RegNetY400MF", "L50_G50", 4),
    ("blackwell_validation", "dermmnist", "ShuffleNetV2", "L50_G50", 4),
    # blackwell_new_backbones - MobileViTS
    ("blackwell_new_backbones", "aider", "MobileViTS", "L50_G50", 0),
    ("blackwell_new_backbones", "dermmnist", "MobileViTS", "L50_G50", 4),
    # model_search
    ("model_search", "aider", "MobileNetV2", "L50_G50", 0),
    ("model_search", "aider", "RegNetY400MF", "L50_G50", 0),
    ("model_search", "aider", "ShuffleNetV2", "L50_G50", 0),
    ("model_search", "aider", "SqueezeNet11", "L50_G50", 0),
    ("model_search", "dermmnist", "MobileNetV2", "L50_G50", 4),
    ("model_search", "dermmnist", "RegNetY400MF", "L50_G50", 4),
    ("model_search", "dermmnist", "ShuffleNetV2", "L50_G50", 4),
    ("model_search", "dermmnist", "SqueezeNet11", "L50_G50", 4),
    ("model_search", "tissuemnist", "MobileNetV2", "L50_G50", 4),
    ("model_search", "tissuemnist", "ShuffleNetV2", "L50_G50", 4),
    # turing_new_datasets - bloodmnist/retinamnist
    ("turing_new_datasets", "bloodmnist", "MobileNetV2", "L50_G50", 0),
    ("turing_new_datasets", "retinamnist", "MobileNetV2", "L50_G50", 2),
    # warmup_confirm
    ("warmup_confirm", "dermmnist", "EfficientNetB0", "L50_G50", 4),
]


def find_tralo_configs(sweep, ds, model, tag, cls):
    """Find TraLO config.jsons under results/pending_runs/<sweep>/ matching cell."""
    pattern = f"{ROOT}/{sweep}/**/tralo/seed_*/config.json"
    matches = []
    for p in glob.glob(pattern, recursive=True):
        try:
            with open(p) as f:
                cfg = json.load(f)
        except Exception:
            continue
        if cfg.get("methodology") != "tralo":
            continue
        if cfg.get("model_name") != model:
            continue
        if cfg.get("dataset_mode") != ds:
            continue
        if cfg.get("constraint_tag") != tag:
            continue
        dsc = cfg.get("dataset_config", {})
        if dsc.get("constrained_class") != cls:
            continue
        matches.append((p, cfg))
    return matches


def make_baseline_config(tralo_cfg, new_method):
    """Clone tralo config, swap methodology + path for new_method."""
    cfg = json.loads(json.dumps(tralo_cfg))
    cfg["methodology"] = new_method
    # Replace /tralo/ in experiment_path with /new_method/
    old_path = cfg.get("experiment_path", "")
    new_path = old_path.replace("/tralo/", f"/{new_method}/")
    cfg["experiment_path"] = new_path
    # exp_name swap if present
    if "exp_name" in cfg:
        cfg["exp_name"] = cfg["exp_name"].replace("_tralo_", f"_{new_method}_")
    # Strip TraLO-specific hyperparams that baselines don't need
    hp = cfg.get("hyperparams", {})
    for k in ["lambda_global", "lambda_local", "lambda_step",
              "initial_rho", "rho_target", "alpha_kl", "penalty_mode",
              "enable_ce_skip", "hybrid_mode", "fior_beta",
              "reset_optimizer_at_sat", "disable_freeze_on_satisfy"]:
        hp.pop(k, None)
    # Strip prior results + status so dispatcher picks it up
    cfg.pop("results", None)
    cfg.pop("status", None)
    cfg.pop("code_version", None)
    return cfg


def main():
    n_made = 0
    n_skipped_existing = 0
    n_no_tralo = 0
    for target in TARGET_CELLS:
        sweep, ds, model, tag, cls = target
        tralos = find_tralo_configs(sweep, ds, model, tag, cls)
        if not tralos:
            print(f"NO TRALO: {target}", file=sys.stderr)
            n_no_tralo += 1
            continue
        for tralo_path, tralo_cfg in tralos:
            for new_method in NEW_METHODS:
                new_cfg = make_baseline_config(tralo_cfg, new_method)
                new_dir = new_cfg["experiment_path"]
                new_config_path = os.path.join(new_dir, "config.json")
                if os.path.exists(new_config_path):
                    # Check if it's completed
                    try:
                        with open(new_config_path) as f:
                            existing = json.load(f)
                        if existing.get("status") == "completed":
                            n_skipped_existing += 1
                            continue
                    except Exception:
                        pass
                os.makedirs(new_dir, exist_ok=True)
                with open(new_config_path, "w") as f:
                    json.dump(new_cfg, f, indent=2)
                n_made += 1
                print(f"  -> {new_config_path}")
    print(f"\nTotal: made={n_made} skipped_existing={n_skipped_existing} "
          f"no_tralo_in_sweep={n_no_tralo}", file=sys.stderr)


if __name__ == "__main__":
    main()
