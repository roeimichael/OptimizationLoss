"""200-run thesis experiment suite.

Blocks A-N addressing audit findings + exploring correlations.
Run count: exactly 200 our_approach + companion baselines (heuristic + danits_lp).
All our_approach runs use diagnostic_level=2.

Usage:
    python -m danits_research.gen_200run_thesis [--phase 1|2|3|4|5|all]
"""

from __future__ import annotations
import argparse
from pathlib import Path

from src.config_generators.generate_configs import (
    compute_base_model_id, constraint_tag, save_configs,
)

MODEL = "MobileNetV3"
DATA_DIR = "data/tissuemnist/slice_1"
ROOT = "results/pending_runs/thesis_200"

BASELINE_HP = {
    "lr": 0.0001, "lr_constraint": 5e-06, "dropout": 0.3, "batch_size": 64,
    "warmup_epochs": 50, "constraint_epochs": 300,
    "lambda_global": 0.01, "lambda_local": 0.01, "lambda_step": 0.002,
    "use_sum_loss": True, "initial_rho": 5.0, "rho_target": 100.0,
    "alpha_kl": 0.1, "kl_temperature": 1.0, "pretrained": True,
    "class_weighted_ce": False, "constraint_chunk_size": 64,
    "lambda_mode": "ratchet", "diagnostic_level": 2,
}

SCENARIOS = {
    "single_GE":          {"cc": 4,         "con": (0.3, 0.5)},
    "single_STR":         {"cc": 6,         "con": (0.3, 0.5)},
    "dual_GE_CST":        {"cc": [4, 2],    "con": (0.3, 0.5)},
    "dual_GE_STR":        {"cc": [4, 6],    "con": (0.3, 0.5)},
    "triple_GE_CST_PTC":  {"cc": [4, 2, 5], "con": (0.3, 0.5)},
    "quad_rare":          {"cc": [4, 2, 5, 1], "con": (0.3, 0.5)},
}


def _ds(cc):
    return {
        "target_column": "label", "group_column": "synth_group",
        "num_classes": 8, "image_size": 224, "data_dir": DATA_DIR,
        "constrained_class": cc,
    }


def _build(block, methodology, scenario_name, con, model, seed, hp_overrides=None, tag_extra=""):
    hp = dict(BASELINE_HP)
    if methodology != "our_approach":
        hp.pop("diagnostic_level", None)
        hp.pop("lambda_mode", None)
    hp["seed"] = seed
    if hp_overrides:
        hp.update(hp_overrides)
    sc = SCENARIOS.get(scenario_name)
    cc = sc["cc"] if sc else hp_overrides.get("_cc", 4)
    if not sc:
        cc = hp_overrides.pop("_cc", 4) if hp_overrides and "_cc" in hp_overrides else 4
    ctag = constraint_tag(con)
    variant = f"s{seed}{tag_extra}"
    path = Path(ROOT) / block / scenario_name / ctag / model / methodology / variant
    ds_config = _ds(cc)
    return {
        "methodology": methodology, "model_name": model,
        "constraint": list(con), "constraint_tag": ctag,
        "dataset_mode": "tissuemnist", "dataset_config": ds_config,
        "hyperparams": hp,
        "base_model_id": compute_base_model_id(
            model, hp, dataset_mode="tissuemnist", data_dir=DATA_DIR,
            dataset_config=ds_config),
        "exp_name": f"t200_{block}_{scenario_name}_{ctag}_{methodology}_s{seed}{tag_extra}",
        "status": "pending", "experiment_path": str(path),
    }


def _with_baselines(block, scenario_name, con, model, seed, hp_overrides=None, tag_extra=""):
    cfgs = [_build(block, "our_approach", scenario_name, con, model, seed, hp_overrides, tag_extra)]
    bl_hp = dict(hp_overrides) if hp_overrides else {}
    for k in ["diagnostic_level", "lambda_mode", "lambda_max", "lambda_k",
              "lambda_ema_alpha", "_cc"]:
        bl_hp.pop(k, None)
    cfgs.append(_build(block, "heuristic", scenario_name, con, model, seed, bl_hp, tag_extra))
    cfgs.append(_build(block, "danits_lp", scenario_name, con, model, seed, bl_hp, tag_extra))
    return cfgs


# ── Block A: CE-only ablation (15 runs) ─────────────────────────────
def block_a():
    cfgs = []
    ablation_hp = {"warmup_epochs": 350, "constraint_epochs": 0}
    for sc in ["single_GE", "dual_GE_CST", "triple_GE_CST_PTC"]:
        con = SCENARIOS[sc]["con"]
        for s in range(1, 6):
            cfgs.extend(_with_baselines("A_ceonly", sc, con, MODEL, s, ablation_hp))
    return cfgs


# ── Block B: Multiclass replication n=5 (30 runs) ───────────────────
def block_b():
    cfgs = []
    for sc in SCENARIOS:
        con = SCENARIOS[sc]["con"]
        for s in range(1, 6):
            cfgs.extend(_with_baselines("B_multiclass", sc, con, MODEL, s))
    return cfgs


# ── Block C: Constraint tightness sweep (18 runs) ───────────────────
def block_c():
    cfgs = []
    tiers_single = [(0.8, 0.8), (0.5, 0.8), (0.5, 0.5), (0.2, 0.3)]
    for con in tiers_single:
        for s in [1, 2, 3]:
            cfgs.extend(_with_baselines("C_tightness", "single_GE", con, MODEL, s))
    tiers_multi = [(0.8, 0.8), (0.2, 0.3)]
    for con in tiers_multi:
        for s in [1, 2, 3]:
            cfgs.extend(_with_baselines("C_tightness", "dual_GE_CST", con, MODEL, s))
    return cfgs


# ── Block D: Chunk size effect (12 runs) ────────────────────────────
def block_d():
    cfgs = []
    for cs in [256, 2400]:
        for sc in ["single_GE", "dual_GE_CST"]:
            con = SCENARIOS[sc]["con"]
            for s in [1, 2, 3]:
                cfgs.extend(_with_baselines("D_chunk", sc, con, MODEL, s,
                    {"constraint_chunk_size": cs}, f"_cs{cs}"))
    return cfgs


# ── Block E: Weight drift exploration (15 runs) ─────────────────────
def block_e():
    cfgs = []
    combos = [
        (1e-5, 0.0), (2e-5, 0.0), (5e-6, 0.0),
        (5e-6, 0.5), (1e-5, 0.5),
    ]
    con = SCENARIOS["single_GE"]["con"]
    for lrc, akl in combos:
        for s in [1, 2, 3]:
            cfgs.extend(_with_baselines("E_drift", "single_GE", con, MODEL, s,
                {"lr_constraint": lrc, "alpha_kl": akl},
                f"_lrc{lrc}_akl{akl}"))
    return cfgs


# ── Block F: Lambda schedule variants (18 runs) ─────────────────────
def block_f():
    cfgs = []
    modes = [
        ("ratchet_frozen", {}),
        ("proportional", {"lambda_max": 0.5, "lambda_k": 30.0, "lambda_ema_alpha": 0.3}),
        ("cosine", {"lambda_max": 0.2}),
    ]
    for sc in ["single_GE", "dual_GE_CST"]:
        con = SCENARIOS[sc]["con"]
        for mode_name, extras in modes:
            hp_ov = {"lambda_mode": mode_name, **extras}
            for s in [1, 2, 3]:
                cfgs.extend(_with_baselines("F_lambda", sc, con, MODEL, s,
                    hp_ov, f"_{mode_name}"))
    return cfgs


# ── Block G: Warmup epoch count (15 runs) ───────────────────────────
def block_g():
    cfgs = []
    for we in [30, 100, 150]:
        for s in [1, 2, 3]:
            cfgs.extend(_with_baselines("G_warmup", "single_GE",
                SCENARIOS["single_GE"]["con"], MODEL, s,
                {"warmup_epochs": we}, f"_we{we}"))
    for we in [30, 150]:
        for s in [1, 2, 3]:
            cfgs.extend(_with_baselines("G_warmup", "dual_GE_CST",
                SCENARIOS["dual_GE_CST"]["con"], MODEL, s,
                {"warmup_epochs": we}, f"_we{we}"))
    return cfgs


# ── Block H: EfficientNetB0 (24 runs) ───────────────────────────────
def block_h():
    cfgs = []
    for sc in ["single_GE", "dual_GE_CST", "triple_GE_CST_PTC"]:
        con = SCENARIOS[sc]["con"]
        for s in range(1, 6):
            cfgs.extend(_with_baselines("H_effnet", sc, con, "EfficientNetB0", s))
    ablation_hp = {"warmup_epochs": 350, "constraint_epochs": 0}
    for sc in ["single_GE", "dual_GE_CST", "triple_GE_CST_PTC"]:
        con = SCENARIOS[sc]["con"]
        for s in [1, 2, 3]:
            cfgs.extend(_with_baselines("H_effnet_ceonly", sc, con, "EfficientNetB0", s, ablation_hp))
    return cfgs


# ── Block J: KL temp / alpha_kl sweep (15 runs) ─────────────────────
def block_j():
    cfgs = []
    combos = [
        (0.5, 0.1), (2.0, 0.1), (1.0, 0.3), (0.5, 0.3), (2.0, 0.3),
    ]
    con = SCENARIOS["single_GE"]["con"]
    for klt, akl in combos:
        for s in [1, 2, 3]:
            cfgs.extend(_with_baselines("J_kl", "single_GE", con, MODEL, s,
                {"kl_temperature": klt, "alpha_kl": akl},
                f"_klt{klt}_akl{akl}"))
    return cfgs


# ── Block K: Rho schedule exploration (15 runs) ─────────────────────
def block_k():
    cfgs = []
    combos = [(1.0, 50.0), (25.0, 200.0), (50.0, 50.0), (0.5, 500.0)]
    con = SCENARIOS["single_GE"]["con"]
    for ir, rt in combos:
        for s in [1, 2, 3]:
            cfgs.extend(_with_baselines("K_rho", "single_GE", con, MODEL, s,
                {"initial_rho": ir, "rho_target": rt},
                f"_rho{ir}-{rt}"))
    con2 = SCENARIOS["dual_GE_CST"]["con"]
    for s in [1, 2, 3]:
        cfgs.extend(_with_baselines("K_rho", "dual_GE_CST", con2, MODEL, s,
            {"initial_rho": 25.0, "rho_target": 200.0}, "_rho25-200"))
    return cfgs


# ── Block L: Extended constraint epochs (6 runs) ────────────────────
def block_l():
    cfgs = []
    for sc in ["single_GE", "dual_GE_CST"]:
        con = SCENARIOS[sc]["con"]
        for s in [1, 2, 3]:
            cfgs.extend(_with_baselines("L_extended", sc, con, MODEL, s,
                {"constraint_epochs": 600}, "_ep600"))
    return cfgs


# ── Block M: Interaction placeholder (6 runs) ───────────────────────
def block_m():
    cfgs = []
    for sc in ["single_GE", "dual_GE_CST"]:
        con = SCENARIOS[sc]["con"]
        for s in [1, 2, 3]:
            cfgs.extend(_with_baselines("M_interaction", sc, con, MODEL, s,
                {"constraint_chunk_size": 256, "lambda_mode": "cosine",
                 "lambda_max": 0.2}, "_cs256_cosine"))
    return cfgs


# ── Block N: Seed diversity (6 runs) ────────────────────────────────
def block_n():
    cfgs = []
    for sc in ["single_GE", "dual_GE_CST"]:
        con = SCENARIOS[sc]["con"]
        for s in [6, 7, 8]:
            cfgs.extend(_with_baselines("N_seeds", sc, con, MODEL, s))
    return cfgs


PHASES = {
    1: [("A", block_a), ("B", block_b), ("N", block_n)],
    2: [("C", block_c), ("D", block_d), ("E", block_e)],
    3: [("F", block_f), ("G", block_g), ("H", block_h), ("L", block_l)],
    4: [("J", block_j), ("K", block_k)],
    5: [("M", block_m)],
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", default="all", help="1-5 or 'all'")
    args = ap.parse_args()

    if args.phase == "all":
        phases = [1, 2, 3, 4, 5]
    else:
        phases = [int(args.phase)]

    all_cfgs = []
    for p in phases:
        for label, fn in PHASES[p]:
            block_cfgs = fn()
            all_cfgs.extend(block_cfgs)

    oa = [c for c in all_cfgs if c["methodology"] == "our_approach"]
    bl = [c for c in all_cfgs if c["methodology"] != "our_approach"]
    blocks = {}
    for c in oa:
        b = c["exp_name"].split("_")[1]
        blocks[b] = blocks.get(b, 0) + 1

    print("=" * 70)
    print(f"THESIS 200-RUN SUITE — phases {phases}")
    print("=" * 70)
    for b, n in sorted(blocks.items()):
        print(f"  Block {b}: {n} our_approach")
    print(f"  TOTAL our_approach: {len(oa)}")
    print(f"  TOTAL baselines:    {len(bl)}")
    print(f"  TOTAL configs:      {len(all_cfgs)}")

    hashes = sorted({c["base_model_id"] for c in all_cfgs})
    print(f"  Warmup hashes:      {len(hashes)}")

    save_configs(all_cfgs, output_dir=ROOT)


if __name__ == "__main__":
    main()
