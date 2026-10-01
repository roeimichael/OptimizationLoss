"""Frozen persistent fmow2 study configs; no development metric may select them."""

import json
from pathlib import Path

from tralo.fmow_persistent_local import CONFIG, PREPROCESSING, validate_config


ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ROOT / "experiments" / "configs" / "fmow_persistent_local_20261001"
WEIGHT_SHA = "5c1a416349c4cf298f2a6a5e2600ed0ee55e604713578f5e74e6bc8bcaef7997"


def test_exact_fixed_pilot_reference_and_full_seed_family():
    wanted = {"fmow_persistent_6700_step.json", "fmow_persistent_6700_ref.json"}
    wanted |= {f"fmow_persistent_{seed}_step.json" for seed in range(6701, 6713)}
    assert {p.name for p in CONFIGS.glob("*.json")} == wanted
    assert len(wanted) == 14
    for path in CONFIGS.glob("*.json"):
        raw = path.read_bytes()
        assert raw.endswith(b"\n") and not raw.endswith(b"\n\n")
        config = json.loads(raw)
        validate_config(config)
        seed, role = path.stem.removeprefix("fmow_persistent_").split("_")
        assert config["seed"] == int(seed)
        assert config["pilot"] is (seed == "6700")
        assert config["reference"] is (role == "ref")
        assert config["pretrained_sha256"] == WEIGHT_SHA
        assert {key: config[key] for key in CONFIG} == CONFIG


def test_scientific_recipe_identical_across_arms_and_scorer():
    from analysis.score_fmow_persistent_local import (
        PREPROCESSING as SCORER_PREPROCESSING,
        RECIPE, WEIGHT_SHA as SCORER_SHA,
    )

    assert CONFIG == RECIPE
    assert PREPROCESSING == SCORER_PREPROCESSING
    assert WEIGHT_SHA == SCORER_SHA
    values = [json.loads(path.read_bytes()) for path in CONFIGS.glob("*.json")]
    fixed = [{key: value for key, value in row.items()
              if key not in {"seed", "pilot", "reference"}} for row in values]
    assert all(row == fixed[0] for row in fixed)
