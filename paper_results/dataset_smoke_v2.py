"""v2: probe the EXACT warmup cache that paper_rerun configs reference.

For each (dataset, model) we find a representative config.json in paper_rerun,
extract its base_model_id, load that specific warmup checkpoint, and run
inference on the test set. Then report per-class natural prediction count.
"""
import csv
import glob
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.models.model_factory import get_model
from src.utils.data_loader import load_experiment_data

OUT = Path("paper_results/dataset_smoke_v2.csv")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def chunked_logits(model, X, chunk=128):
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(X), chunk):
            x = torch.tensor(X[i:i+chunk]).to(DEVICE).float()
            out.append(model(x).cpu())
    return torch.cat(out, dim=0)


def find_representative_config(dataset, model):
    """Find any tralo config for (dataset, model) and return parsed JSON."""
    pat = f"results/pending_runs/paper_rerun/{dataset}/{model}/cls_*/L*/tralo/seed_*/config.json"
    candidates = sorted(glob.glob(pat))
    if not candidates:
        return None
    return json.load(open(candidates[0]))


def main():
    targets = []
    for dataset, _, _, _, _ in [
        ("tissuemnist", None, None, None, None),
        ("dermmnist", None, None, None, None),
        ("so2sat", None, None, None, None),
    ]:
        for model in ["MobileNetV3", "ResNet18", "EfficientNetB0"]:
            targets.append((dataset, model))

    all_rows = []
    for dataset, model_name in targets:
        cfg = find_representative_config(dataset, model_name)
        if cfg is None:
            print(f"=== skip {model_name}/{dataset}: no paper_rerun config", file=sys.stderr)
            continue
        bmid = cfg.get("base_model_id")
        if not bmid:
            print(f"=== skip {model_name}/{dataset}: no base_model_id", file=sys.stderr)
            continue
        cache_path = Path("model_cache") / f"{bmid}.pt"
        if not cache_path.exists():
            print(f"=== skip {model_name}/{dataset}: missing {cache_path}", file=sys.stderr)
            continue

        n_classes = cfg["dataset_config"]["num_classes"]
        print(f"=== probe {model_name}/{dataset} via {cache_path.name}", file=sys.stderr)
        try:
            (X_train, X_test, y_train, y_test, groups, gcon, lcon, nc) = \
                load_experiment_data(cfg)
        except Exception as e:
            print(f"   data_loader: {e}", file=sys.stderr)
            continue
        dropout = cfg.get("hyperparams", {}).get("dropout", 0.3)
        model = get_model(model_name, n_classes, pretrained=True, dropout=dropout).to(DEVICE)
        try:
            state = torch.load(cache_path, map_location=DEVICE, weights_only=True)
        except TypeError:
            state = torch.load(cache_path, map_location=DEVICE)
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        model.load_state_dict(state, strict=False)

        logits = chunked_logits(model, X_test)
        pred = logits.argmax(dim=1).numpy()
        probs = F.softmax(logits.float(), dim=1).numpy()

        for c in range(n_classes):
            true_n = int((y_test == c).sum())
            pred_n = int((pred == c).sum())
            ratio = pred_n / true_n if true_n else float("nan")
            k50 = int(np.floor(true_n * 0.5))
            k30 = int(np.floor(true_n * 0.3))
            k70 = int(np.floor(true_n * 0.7))
            all_rows.append({
                "dataset": dataset, "model": model_name, "cls": c,
                "true_count": true_n, "pred_count": pred_n,
                "ratio": round(ratio, 3) if ratio == ratio else None,
                "mean_prob_c": round(float(probs[:, c].mean()), 4),
                "K_L30": k30, "K_L50": k50, "K_L70": k70,
                "bites_L30": int(pred_n > k30),
                "bites_L50": int(pred_n > k50),
                "bites_L70": int(pred_n > k70),
                "base_model_id": bmid,
            })

    if not all_rows:
        print("No probes succeeded")
        return

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, all_rows[0].keys())
        w.writeheader()
        w.writerows(all_rows)
    print(f"\nWrote {OUT} ({len(all_rows)} rows)\n")

    print(f"\n=== Strong TraLO targets (natural over-prediction, constraint bites at α=50%) ===")
    print(f"{'dataset':<12} {'model':<14} {'cls':>3} {'true':>5} {'pred':>5} "
          f"{'ratio':>6} {'K50':>5} {'L30bite':>7} {'L50bite':>7} {'L70bite':>7} {'verdict':<22}")
    print("-" * 130)
    for r in sorted(all_rows, key=lambda x: (x["dataset"], x["model"],
                                              -(x["ratio"] or 0))):
        good = (r["ratio"] or 0) > 1.05
        weak = 0.85 <= (r["ratio"] or 0) <= 1.05
        verdict = ("STRONG: TraLO suppresses" if good
                   else "weak/natural" if weak
                   else "BAD: model under-predicts")
        print(f"{r['dataset']:<12} {r['model']:<14} {r['cls']:>3} "
              f"{r['true_count']:>5} {r['pred_count']:>5} "
              f"{r['ratio'] or 0:>6.3f} {r['K_L50']:>5} "
              f"{r['bites_L30']:>7} {r['bites_L50']:>7} {r['bites_L70']:>7} "
              f"{verdict}")


if __name__ == "__main__":
    main()
