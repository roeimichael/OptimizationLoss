"""WHERE inside run_warmup does a rerun stop matching?

The noise floor is 0.0358 macro-F1 over three identical runs, and the
training logs show them apart ALREADY AT EPOCH 1 (L_CE 0.762393 /
0.762403 / 0.762397). A different batch order would move epoch-1 loss by
~1e-2, not 1e-5, so the batch order is probably fine and something
numerical drifts inside the epoch.

Stage A separately proved a bare ViT-B/16 forward+backward is bit-identical
on this GPU in fp16, with and without warn_only. So the divergence needs the
rest of the real path: the project model, the DataLoader, fused Adam,
GradScaler.

This replays exactly that path using the project own functions and prints a
hash at each stage. Run it twice and diff -- the first differing line is the
culprit.

    python -m scripts.bisect_determinism <a config.json> --steps 126
"""
import argparse
import hashlib
import json
from pathlib import Path

import torch

from src.pipeline.setup import seed_all, setup_runtime
from src.pipeline.data import load_data
from src.pipeline.warmup import make_ce_criterion, make_optimizer, make_dataloader
from src.models import get_model


def h_params(model):
    m = hashlib.md5()
    for k in sorted(model.state_dict()):
        m.update(model.state_dict()[k].detach().float().cpu().numpy().tobytes())
    return m.hexdigest()[:12]


def h_grads(model):
    m = hashlib.md5()
    for n, p in sorted(model.named_parameters()):
        if p.grad is not None:
            m.update(p.grad.detach().float().cpu().numpy().tobytes())
    return m.hexdigest()[:12]


def main():
    a = argparse.ArgumentParser()
    a.add_argument("config")
    a.add_argument("--steps", type=int, default=126)
    args = a.parse_args()

    config = json.loads(Path(args.config).read_text())
    device = torch.device("cuda")
    seed = config.get("hyperparams", {}).get("seed")
    seed_all(seed)
    import os
    if os.environ.get("OPTLOSS_STRICT_DET") == "1":
        torch.use_deterministic_algorithms(True, warn_only=False)
    if os.environ.get("OPTLOSS_NOFUSED_SDPA") == "1":
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        try:
            torch.backends.cuda.enable_cudnn_sdp(False)
        except AttributeError:
            pass
    print("seed=%s strict=%s nofused=%s" % (seed,
          os.environ.get("OPTLOSS_STRICT_DET"), os.environ.get("OPTLOSS_NOFUSED_SDPA")))

    data = load_data(config)
    hp = config["hyperparams"]
    use_amp, amp_dtype, scaler = setup_runtime(device)
    print("amp=%s dtype=%s scaler=%s" % (use_amp, amp_dtype, scaler is not None))

    model = get_model(config["model_name"],
                      n_classes=data.num_classes, dropout=hp["dropout"],
                      pretrained=hp.get("pretrained", False)).to(device)
    print("STAGE init_model      %s" % h_params(model))

    criterion = make_ce_criterion(config, data.y_train, data.num_classes, device)
    optimizer = make_optimizer(model.parameters(), hp["lr"], device)
    loader = make_dataloader(data.X_train, data.y_train, hp["batch_size"])
    print("STAGE loader_len      %d  workers=%s" % (len(loader), loader.num_workers))

    model.train()
    order = hashlib.md5()
    for i, (bx, by) in enumerate(loader):
        order.update(by.numpy().tobytes())
        if i >= args.steps:
            break
        bx, by = bx.to(device), by.to(device)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
            loss = criterion(model(bx), by)
        if scaler:
            scaler.scale(loss).backward()
        else:
            loss.backward()
        if i in (0, 1, 5):
            print("STAGE grad_step%-3d    %s   loss=%.10f  scale=%s"
                  % (i, h_grads(model), loss.item(),
                     scaler.get_scale() if scaler else "-"))
        if scaler:
            scaler.step(optimizer); scaler.update()
        else:
            optimizer.step()
        if i in (0, 1, 5, 20, 60, 125):
            print("STAGE param_step%-3d   %s   scale=%s"
                  % (i, h_params(model), scaler.get_scale() if scaler else "-"))
    print("STAGE batch_order     %s" % order.hexdigest()[:12])


if __name__ == "__main__":
    main()
