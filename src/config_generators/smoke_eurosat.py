"""Quick smoke for EuroSAT: train 50-epoch warmup on MobileNetV3 + ResNet18,
then report per-class natural prediction counts on the test set.

A class is a "strong TraLO target" if pred_count > true_count (over-prediction
ratio > 1) — at α=0.5 the constraint will bite and TraLO has actual work to do.
"""
import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.models.model_factory import get_model
from src.utils.data_loader import load_experiment_data
from src.pipeline.warmup import run_warmup

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CLASS_NAMES = {
    0: "AnnCrop", 1: "Forest", 2: "HerbVeg", 3: "Highway", 4: "Industrial",
    5: "Pasture", 6: "PermCrop", 7: "Resid", 8: "River", 9: "SeaLake",
}


def make_config(model_name):
    return {
        "methodology": "smoke",
        "model_name": model_name,
        "dataset_mode": "eurosat",
        "constraint": [0.5, 0.5],
        "constraint_tag": "smoke_L50",
        "dataset_config": {
            "data_dir": "data/eurosat",
            "num_classes": 10,
            "image_size": 224,
            "target_column": "label",
            "group_column": "synth_group",
            "constrained_class": 0,
        },
        "hyperparams": {
            "lr": 1e-4, "lr_constraint": 5e-6, "dropout": 0.3, "batch_size": 64,
            "warmup_epochs": 50, "constraint_epochs": 0,
            "use_sum_loss": True, "kl_temperature": 1.0, "pretrained": True,
            "class_weighted_ce": False, "constraint_chunk_size": 256,
            "seed": 1,
        },
    }


def chunked_logits(model, X, chunk=128):
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(X), chunk):
            x = torch.tensor(X[i:i+chunk]).to(DEVICE).float()
            out.append(model(x).cpu())
    return torch.cat(out, dim=0)


def smoke_one(model_name):
    print(f"\n=== {model_name} on EuroSAT (warmup 50ep) ===")
    cfg = make_config(model_name)
    from src.config_generators.generate_configs import compute_base_model_id
    cfg["base_model_id"] = compute_base_model_id(
        model_name, cfg["hyperparams"], cfg["dataset_mode"],
        cfg["dataset_config"]["data_dir"], cfg["dataset_config"])
    (X_train, X_test, y_train, y_test, groups, gcon, lcon, nc) = \
        load_experiment_data(cfg)
    X_train_t = torch.from_numpy(X_train).float()
    y_train_t = torch.from_numpy(y_train).long()
    model, from_cache = run_warmup(cfg, nc, X_train_t, y_train_t, DEVICE)
    print(f"  (warmup cache used: {from_cache}, base_model_id={cfg['base_model_id']})")
    logits = chunked_logits(model.to(DEVICE), X_test)
    pred = logits.argmax(dim=1).numpy()
    print(f"  {'cls':>3} {'name':<10} {'true':>5} {'pred':>5} {'ratio':>7} "
          f"{'K_L30':>5} {'K_L50':>5} {'K_L70':>5} {'verdict':<22}")
    for c in range(nc):
        true_n = int((y_test == c).sum())
        pred_n = int((pred == c).sum())
        ratio = pred_n / true_n if true_n else float("nan")
        k30 = int(np.floor(true_n * 0.3))
        k50 = int(np.floor(true_n * 0.5))
        k70 = int(np.floor(true_n * 0.7))
        if ratio > 1.05: verdict = "STRONG over-predict"
        elif 0.85 <= ratio <= 1.05: verdict = "natural"
        else: verdict = "UNDER-predict"
        print(f"  {c:>3} {CLASS_NAMES[c]:<10} {true_n:>5} {pred_n:>5} "
              f"{ratio:>7.3f} {k30:>5} {k50:>5} {k70:>5} {verdict}")


if __name__ == "__main__":
    import traceback
    for m in ["MobileNetV3", "ResNet18", "EfficientNetB0"]:
        try:
            smoke_one(m)
        except Exception as e:
            print(f"=== {m} FAILED: {e}")
            traceback.print_exc()
