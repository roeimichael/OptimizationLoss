"""B1 imbalanced-baseline WARMUP SWEEP (Track B redo, 2026-07-27).

Reviewers asked: does an imbalanced-learning-aware clipper (focal / class-balanced /
logit-adjust + LP-clip) close TraLO's macro-F1 gap over vanilla clipping?

Key point (per the student): the imbalanced baselines are HEURISTIC -- they train the
backbone with the imbalanced loss in the WARMUP phase (hp['warmup_loss']) then Shifman-LP
clip. So their warmup length IS their entire training budget. TraLO instead trains warmup
(CE) + up to 300 constraint epochs. This generator sweeps warmup in {1, 5, 50} so we can
see, as a function of how much the heuristic classifier trains, whether it ever matches
TraLO -- and whether short warmup (which gives TraLO constraint-phase headroom) changes
the verdict.

Clones VERBATIM from frozen paper_final configs (same dataset_config/HPs); only
warmup/method/seed/cap/path change. base_model_id recomputed (includes warmup_epochs);
imbalanced caches get a per-loss suffix so both caps of a cell share one training.

Root: results/track_b/b1/<ds>/<model>/warmup<W>/<cap>/<method>/seed_<s>/
Idempotent. Run ON THE SERVER from repo root:
    python -m src.config_generators.gen_b1_warmup_sweep
Smoke a subset by monkeypatching the module globals via `python -c`.
"""
import glob
import json
from pathlib import Path

from src.config_generators.generate_configs import compute_base_model_id

PF = "results/pending_runs/paper_final"
DST = Path("results/track_b/b1")

IMB = ["focal", "class_balanced", "logit_adjust"]
IMB_HP = {
    "focal": {"focal_alpha": 0.25, "focal_gamma": 2.0},
    "class_balanced": {"cb_beta": 0.9999},
    "logit_adjust": {"logit_adjust_tau": 1.0},
}
# constraint-method-only HPs dropped when repurposing a tralo config as imbalanced
TRALO_ONLY = ["lambda_global", "lambda_local", "lambda_step", "initial_rho",
              "rho_target", "alpha_kl", "penalty_mode", "hybrid_mode",
              "fior_beta", "reset_optimizer_at_sat", "enable_ce_skip"]

DATASETS = ["octmnist", "dermmnist", "tissuemnist"]
BACKBONES = ["MobileNetV3", "RegNetY400MF"]
METHODS = ["tralo", "danits_lp"] + IMB   # tralo + clip references + 3 imbalanced
CAPS = {"L30_G30": [0.3, 0.3]}
WARMUPS = [1, 5, 50]
SEEDS = [1, 2, 3, 4]


def _src(model, ds, meth):
    """A frozen paper_final template. Imbalanced baselines repurpose a tralo config
    (only the loss differs); tralo/danits_lp clone their own frozen config."""
    base = meth if meth in ("tralo", "danits_lp") else "tralo"
    for cap in ("L30_G30", "L40_G40", "L20_G20"):
        h = glob.glob(f"{PF}/lane*/{model}/{ds}/{cap}/{base}/seed_1/config.json")
        if h:
            return h[0]
    return None


def main():
    assert "pending_runs" not in str(DST), "failsafe"
    emit = skip = 0
    for ds in DATASETS:
        for model in BACKBONES:
            for meth in METHODS:
                src = _src(model, ds, meth)
                assert src, f"no paper_final template for {model}/{ds}/{meth}"
                base = json.load(open(src))
                is_imb = meth in IMB
                for W in WARMUPS:
                    for captag, cvec in CAPS.items():
                        for seed in SEEDS:
                            dst_dir = DST / ds / model / f"warmup{W}" / captag / meth / f"seed_{seed}"
                            dst = dst_dir / "config.json"
                            if dst.exists():
                                skip += 1
                                continue
                            c = json.loads(json.dumps(base))
                            c.pop("results", None)
                            c["status"] = "pending"
                            c["model_name"] = model
                            c["methodology"] = meth
                            c["constraint"] = list(cvec)
                            c["constraint_tag"] = captag
                            c["hyperparams"]["seed"] = seed
                            c["hyperparams"]["warmup_epochs"] = W
                            if is_imb:
                                for k in TRALO_ONLY + ["constraint_epochs", "lr_constraint"]:
                                    c["hyperparams"].pop(k, None)
                                c["hyperparams"]["warmup_loss"] = meth
                                c["hyperparams"].update(IMB_HP[meth])
                            c["base_model_id"] = compute_base_model_id(
                                model, c["hyperparams"], c["dataset_mode"],
                                c["dataset_config"]["data_dir"], c["dataset_config"])
                            if is_imb:
                                c["base_model_id"] += f"_{meth}"
                            c["sweep_tag"] = "track_b_b1_2026-07-27"
                            c["cloned_from"] = src
                            c["exp_name"] = f"b1_{ds}_{model}_{meth}_w{W}_{captag}_seed{seed}"
                            c["experiment_path"] = str(dst_dir)
                            dst_dir.mkdir(parents=True, exist_ok=True)
                            with open(dst, "w") as f:
                                json.dump(c, f, indent=1)
                            emit += 1
    print(f"emitted={emit} skipped(existing)={skip} root={DST}")
    for ds in DATASETS:
        n = len(glob.glob(str(DST / ds / "**" / "config.json"), recursive=True))
        if n:
            print(f"  {ds}: {n}")


if __name__ == "__main__":
    main()
