"""Smoke test: which (dataset, model, class) combos have models naturally
OVER-predicting → good targets for TraLO (constraint actually bites).

For each cached warmup checkpoint we can find, runs inference on the test
set once and outputs per-class:
  true_count   (ground truth)
  pred_count   (model raw argmax)
  pred/true    (over-prediction ratio; >1 means model over-predicts)

A "good" constrained class for TraLO is one where pred_count > true_count
(model over-predicts). At tightness α=50%, K = ceil(true_count × α). The
constraint bites when pred_count > K, i.e. pred_count > 0.5 × true_count.

Output: paper_results/dataset_smoke.csv + readable table.
"""
import csv
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.models.model_factory import get_model
from src.utils.data_loader import load_experiment_data

OUT = Path("paper_results/dataset_smoke.csv")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Datasets to probe. Match num_classes & group_column to existing configs.
DATASETS = [
    ("tissuemnist", "data/tissuemnist/slice_1", 8, "synth_group", "label"),
    ("dermmnist", "data/dermmnist/slice_1", 7, "sex", "label"),
    ("so2sat", "data/so2sat", 17, "city_id", "label"),
]
MODELS = ["MobileNetV3", "ResNet18", "EfficientNetB0"]


def make_config(dataset_mode, data_dir, n_classes, group_col, target_col,
                model_name):
    return {
        "methodology": "warmup",
        "model_name": model_name,
        "dataset_mode": dataset_mode,
        "constraint": [1.0, 1.0],  # dummy
        "constraint_tag": "smoke",
        "dataset_config": {
            "data_dir": data_dir,
            "num_classes": n_classes,
            "image_size": 224,
            "target_column": target_col,
            "group_column": group_col,
            "constrained_class": 0,  # dummy
        },
        "hyperparams": {"warmup_epochs": 50, "dropout": 0.3, "pretrained": True,
                        "batch_size": 64},
    }


def find_warmup_cache(model_name, dataset_mode):
    """Find any cached warmup checkpoint for (model, dataset). Picks first."""
    cache_dir = Path("model_cache")
    pattern = f"{model_name}_{dataset_mode}_*.pt"
    found = sorted(cache_dir.glob(pattern))
    return found[0] if found else None


def chunked_forward(model, X, chunk=128):
    """Return logits from model(X) in chunks to fit GPU."""
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(X), chunk):
            x = torch.tensor(X[i:i+chunk]).to(DEVICE).float()
            out.append(model(x).cpu())
    return torch.cat(out, dim=0)


def probe(dataset_mode, data_dir, n_classes, group_col, target_col, model_name):
    cache = find_warmup_cache(model_name, dataset_mode)
    if cache is None:
        return None, f"no cached warmup for {model_name}_{dataset_mode}"
    cfg = make_config(dataset_mode, data_dir, n_classes, group_col, target_col,
                      model_name)
    try:
        (X_train, X_test, y_train, y_test, groups, gcon, lcon, nc) = \
            load_experiment_data(cfg)
    except Exception as e:
        return None, f"data_loader error: {e}"

    model = get_model(model_name, n_classes, pretrained=True, dropout=0.3).to(DEVICE)
    try:
        state = torch.load(cache, map_location=DEVICE, weights_only=True)
    except TypeError:
        state = torch.load(cache, map_location=DEVICE)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    try:
        model.load_state_dict(state, strict=False)
    except Exception as e:
        return None, f"state_dict load failed: {e}"

    logits = chunked_forward(model, X_test)
    pred = logits.argmax(dim=1).numpy()
    probs = F.softmax(logits.float(), dim=1).numpy()

    rows = []
    for c in range(n_classes):
        true_n = int((y_test == c).sum())
        pred_n = int((pred == c).sum())
        mean_prob_c = float(probs[:, c].mean())
        ratio = pred_n / true_n if true_n > 0 else float("nan")
        # K at common tightness levels
        k_30 = int(np.floor(true_n * 0.3))
        k_50 = int(np.floor(true_n * 0.5))
        k_70 = int(np.floor(true_n * 0.7))
        # constraint bites at α if pred_n > K_α
        bites_30 = pred_n > k_30
        bites_50 = pred_n > k_50
        bites_70 = pred_n > k_70
        rows.append({
            "dataset": dataset_mode, "model": model_name, "cls": c,
            "true_count": true_n, "pred_count": pred_n,
            "pred_over_true": round(ratio, 3) if ratio == ratio else None,
            "mean_prob_c": round(mean_prob_c, 4),
            "K_at_L30": k_30, "K_at_L50": k_50, "K_at_L70": k_70,
            "bites_L30": int(bites_30), "bites_L50": int(bites_50),
            "bites_L70": int(bites_70),
            "n_classes": n_classes,
        })
    return rows, None


def main():
    all_rows = []
    for ds, ddir, nc, gcol, tcol in DATASETS:
        for m in MODELS:
            print(f"=== probe {m} / {ds}", file=sys.stderr)
            rows, err = probe(ds, ddir, nc, gcol, tcol, m)
            if err:
                print(f"   skip: {err}", file=sys.stderr)
                continue
            all_rows.extend(rows)

    if not all_rows:
        print("No probes succeeded — no cached warmups.")
        return

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, all_rows[0].keys())
        w.writeheader()
        w.writerows(all_rows)
    print(f"\nWrote {OUT} ({len(all_rows)} rows)")

    print(f"\n=== Over-prediction summary (good TraLO targets where pred/true > 1) ===")
    print(f"{'dataset':<12} {'model':<14} {'cls':>3} "
          f"{'true':>5} {'pred':>5} {'ratio':>6} "
          f"{'K30':>4} {'K50':>4} {'K70':>4} "
          f"{'bite30':>6} {'bite50':>6} {'bite70':>6} {'verdict':<20}")
    print("-" * 130)
    for r in sorted(all_rows, key=lambda x: (x["dataset"], x["model"],
                                              -x["pred_over_true"] if x["pred_over_true"] else 0)):
        good = r["pred_over_true"] and r["pred_over_true"] > 1.05
        weak = r["pred_over_true"] and 0.95 <= r["pred_over_true"] <= 1.05
        verdict = "🟢 over-predicts" if good else ("🟡 ~natural" if weak else "🔴 under-predicts")
        print(f"{r['dataset']:<12} {r['model']:<14} {r['cls']:>3} "
              f"{r['true_count']:>5} {r['pred_count']:>5} "
              f"{r['pred_over_true'] or 0:>6.3f} "
              f"{r['K_at_L30']:>4} {r['K_at_L50']:>4} {r['K_at_L70']:>4} "
              f"{r['bites_L30']:>6} {r['bites_L50']:>6} {r['bites_L70']:>6} "
              f"{verdict}")


if __name__ == "__main__":
    main()
