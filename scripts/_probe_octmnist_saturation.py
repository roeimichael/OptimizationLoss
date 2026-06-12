"""OCTMNIST warmup-saturation probe (the model-search 'probe before sweep' step).

Trains each backbone for N CE epochs and logs train/test accuracy per epoch.
Verdict per REJECTED.md headroom band:
  train-acc stays ~[0.70, 0.82]  -> HEADROOM (good TraLO candidate)
  train-acc -> ~1.0 by ep1-2     -> SATURATED (like aider; reject)
Mirrors src/pipeline/warmup.py: ImageNet-normed data, Adam lr=1e-4, bs=64, bf16.
"""
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from src.utils.data_loader import load_experiment_data
from src.models.model_factory import get_model

EPOCHS = 8
BACKBONES = ["MobileNetV3", "MobileNetV2", "RegNetY400MF", "ResNet18"]
CFG = {
    "dataset_mode": "octmnist", "constraint": [0.5, 0.5],
    "dataset_config": {
        "num_classes": 4, "image_size": 224, "target_column": "label",
        "group_column": "synth_group", "constrained_class": 0,
        "data_dir": "data/octmnist/slice_1",
    },
}


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    Xtr, Xte, ytr, yte, _, _, _, ncls = load_experiment_data(CFG)
    Xtr = torch.tensor(Xtr, dtype=torch.float32)
    ytr = torch.tensor(ytr, dtype=torch.long)
    Xte = torch.tensor(Xte, dtype=torch.float32)
    yte = torch.tensor(yte, dtype=torch.long)
    dl = DataLoader(TensorDataset(Xtr, ytr), batch_size=64, shuffle=True,
                    num_workers=2, drop_last=False)
    print(f"device={dev} classes={ncls} train={len(ytr)} test={len(yte)}",
          flush=True)

    summary = {}
    for bb in BACKBONES:
        torch.manual_seed(1)
        model = get_model(bb, ncls, dropout=0.3, pretrained=True).to(dev)
        opt = torch.optim.Adam(model.parameters(), lr=1e-4)
        crit = nn.CrossEntropyLoss()
        print(f"\n=== {bb} ===", flush=True)
        accs = []
        for ep in range(1, EPOCHS + 1):
            t0 = time.time()
            model.train()
            cor = tot = 0
            for xb, yb in dl:
                xb, yb = xb.to(dev), yb.to(dev)
                opt.zero_grad(set_to_none=True)
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev == "cuda"):
                    out = model(xb)
                    loss = crit(out, yb)
                loss.backward()
                opt.step()
                cor += (out.argmax(1) == yb).sum().item()
                tot += len(yb)
            tr_acc = cor / tot
            # test acc
            model.eval()
            tc = tt = 0
            with torch.no_grad():
                for i in range(0, len(Xte), 128):
                    xb = Xte[i:i + 128].to(dev)
                    yb = yte[i:i + 128].to(dev)
                    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev == "cuda"):
                        out = model(xb)
                    tc += (out.argmax(1) == yb).sum().item()
                    tt += len(yb)
            te_acc = tc / tt
            accs.append(tr_acc)
            print(f"  ep{ep}: train_acc={tr_acc:.3f}  test_acc={te_acc:.3f}  "
                  f"[{time.time()-t0:.0f}s]", flush=True)
        summary[bb] = accs

    print("\n" + "=" * 64)
    print("SATURATION VERDICT (headroom band ~[0.70, 0.82])")
    print("=" * 64)
    for bb, accs in summary.items():
        e1, e3, ef = accs[0], accs[min(2, len(accs)-1)], accs[-1]
        if ef >= 0.95:
            verd = "SATURATED (reject, like aider)"
        elif 0.68 <= ef <= 0.86 or 0.68 <= e3 <= 0.86:
            verd = "HEADROOM (good candidate!)"
        else:
            verd = f"other (final train-acc {ef:.2f})"
        print(f"  {bb:16s} ep1={e1:.3f} ep3={e3:.3f} ep{len(accs)}={ef:.3f}  -> {verd}")


if __name__ == "__main__":
    main()
