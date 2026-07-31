"""Native-Resolution Campaign generator (docs/NATIVE_RES_CAMPAIGN.md).

Emits the full method x warmup x cap x seed x backbone grid for each native-224
dataset, cloning per-METHOD from the frozen paper_final OctMNIST configs (so each
method keeps its own tuned HPs: TraLO's undershoot_hinge+reset, Fioretto's eta,
Hounie's etas, LP-clip). Swaps ONLY: dataset fields, warmup length, cap, seed.

Warmup {10, 50}: 10 = headroom-preserving (stop before the constrained class
saturates), 50 = saturated control. base_model_id includes warmup_epochs+seed, so
each (model, dataset, warmup, seed) trains ONE fresh native warmup, reused across
methods+caps of that cell.

Skips any dataset whose data_dir/test_meta.csv is not staged yet, so this can be
run early for the ready best-shots (derm/retina/blood) and re-run once
tissue-native/organa finish staging.

Root: results/native_res_campaign/<dataset>/<model>/warmup<W>/<cap>/<method>/seed_<s>/
Idempotent. Run ON THE SERVER from repo root:
    python -m src.config_generators.gen_native_res
"""
import glob
import json
import os
from pathlib import Path

from src.config_generators.generate_configs import compute_base_model_id

PF = "results/pending_runs/paper_final"
DST_ROOT = Path("results/native_res_campaign")

# key -> (dataset_mode, data_dir, num_classes, constrained_class, group_column)
# tissuenative/organa constrained_class locked from prep report before their run.
DATASETS = {
    "derm":         ("dermmnist",    "data/dermmnist/slice_1",             7,  4, "loc_group"),
    "retina":       ("retinamnist",  "data/native224/retinamnist/slice_1", 5,  2, "synth_group"),
    "blood":        ("bloodmnist",   "data/native224/bloodmnist/slice_1",  8,  4, "synth_group"),
    "tissuenative": ("tissuenative", "data/native224/tissuemnist/slice_1", 8,  2, "synth_group"),
    "organa":       ("organamnist",  "data/native224/organamnist/slice_1", 11, 1, "synth_group"),
}
METHODS = ["tralo", "fioretto_ldf", "hounie_rcl", "danits_lp"]
BACKBONES = ["MobileNetV3", "RegNetY400MF"]
CAPS = {"L20_G20": [0.2, 0.2], "L30_G30": [0.3, 0.3], "L40_G40": [0.4, 0.4]}
WARMUPS = [10, 50]
SEEDS = [1, 2, 3, 4]


def _src(model, meth):
    """Any frozen paper_final OctMNIST config of this method+model, for its HPs."""
    for cap in ("L30_G30", "L40_G40", "L20_G20"):
        h = glob.glob(f"{PF}/lane*/{model}/octmnist/{cap}/{meth}/seed_1/config.json")
        if h:
            return h[0]
    return None


def main():
    assert "pending_runs" not in str(DST_ROOT), "failsafe"
    emit = skip = nodata = 0
    for dskey, (mode, ddir, ncls, ccls, gcol) in DATASETS.items():
        if not os.path.exists(os.path.join(ddir, "test_meta.csv")):
            print(f"[skip-no-data] {dskey}: {ddir} not staged yet")
            nodata += 1
            continue
        for model in BACKBONES:
            for meth in METHODS:
                src = _src(model, meth)
                assert src, f"no paper_final template for {model}/{meth}"
                base = json.load(open(src))
                for W in WARMUPS:
                    for captag, cvec in CAPS.items():
                        for seed in SEEDS:
                            dst_dir = (DST_ROOT / dskey / model / f"warmup{W}"
                                       / captag / meth / f"seed_{seed}")
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
                            c["dataset_mode"] = mode
                            c["dataset_config"] = {
                                "data_dir": ddir, "num_classes": ncls,
                                "image_size": 224, "target_column": "label",
                                "group_column": gcol, "constrained_class": ccls,
                            }
                            c["hyperparams"]["warmup_epochs"] = W
                            c["hyperparams"]["seed"] = seed
                            c["base_model_id"] = compute_base_model_id(
                                model, c["hyperparams"], mode, ddir, c["dataset_config"])
                            c["sweep_tag"] = "native_res_2026-07"
                            c["cloned_from"] = src
                            c["exp_name"] = f"nativeres_{dskey}_{model}_{meth}_w{W}_{captag}_seed{seed}"
                            c["experiment_path"] = str(dst_dir)
                            dst_dir.mkdir(parents=True, exist_ok=True)
                            with open(dst, "w") as f:
                                json.dump(c, f, indent=1)
                            emit += 1
    print(f"emitted={emit} skipped(existing)={skip} datasets_without_data={nodata}")
    print(f"root={DST_ROOT}")
    # per-dataset counts
    for dskey in DATASETS:
        n = len(glob.glob(str(DST_ROOT / dskey / "**" / "config.json"), recursive=True))
        if n:
            print(f"  {dskey}: {n} configs")


if __name__ == "__main__":
    main()
