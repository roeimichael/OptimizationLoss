"""CIFAR100 frozen ImageNet-ResNet18 feature baseline (CUDA only)."""

import hashlib
import json
import math
import random
import sys
from pathlib import Path
from .events import EventLog


CONFIG_KEYS = {
    "data_root", "seed", "train_samples", "development_samples", "image_size",
    "feature_batch_size", "head_batch_size", "head_epochs", "head_lr", "precision",
}
PRECISIONS = {"fp32", "fp16", "bf16"}


def _plain_int(value):
    return type(value) is int


def validate_config(config):
    if not isinstance(config, dict) or set(config) != CONFIG_KEYS:
        raise ValueError("config must contain exactly the declared baseline keys")
    if not isinstance(config["data_root"], str) or not config["data_root"]:
        raise ValueError("data_root must be a nonempty string")
    for key in ("seed", "train_samples", "development_samples", "image_size",
                "feature_batch_size", "head_batch_size", "head_epochs"):
        if not _plain_int(config[key]) or config[key] <= 0:
            raise ValueError("%s must be a positive integer" % key)
    if (isinstance(config["head_lr"], bool) or
            not isinstance(config["head_lr"], (int, float)) or
            not math.isfinite(config["head_lr"]) or config["head_lr"] <= 0):
        raise ValueError("head_lr must be a positive finite number")
    if config["precision"] not in PRECISIONS:
        raise ValueError("precision must be fp32, fp16, or bf16")
    if config["train_samples"] + config["development_samples"] > 50000:
        raise ValueError("requested split exceeds CIFAR100 train split")
    return dict(config)


def split_indices(n_items, train_samples, development_samples, seed):
    if not all(_plain_int(x) for x in (n_items, train_samples, development_samples, seed)):
        raise ValueError("split sizes and seed must be integers")
    if n_items <= 0 or train_samples <= 0 or development_samples <= 0 or seed < 0:
        raise ValueError("split sizes must be positive")
    if train_samples + development_samples > n_items:
        raise ValueError("requested split exceeds dataset size")
    order = list(range(n_items))
    random.Random(seed).shuffle(order)
    return order[:train_samples], order[train_samples:train_samples + development_samples]


def config_digest(config):
    payload = json.dumps(config, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode()).hexdigest()


def sample_ids(indices):
    return ["cifar100:train:%06d" % int(index) for index in indices]


def _file_hashes(root):
    result = {}
    for path in sorted(Path(root).rglob("*")):
        if path.is_file():
            digest = hashlib.sha256()
            with path.open("rb") as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(block)
            result[str(path.relative_to(root))] = digest.hexdigest()
    return result


def _source_hash():
    digest = hashlib.sha256()
    for path in sorted(Path(__file__).resolve().parent.glob("*.py")):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _event(log, event, **fields):
    log.emit(event, **fields)


def _image_hashes(raw, indices):
    return {hashlib.sha256(bytes(raw[i])).hexdigest() for i in indices}


def run(config, output):
    config = validate_config(config)
    output = Path(output)
    if output.exists():
        raise FileExistsError("output already exists: %s" % output)
    output.mkdir(parents=True)
    events = output / "events.jsonl"
    event_log = EventLog(events)
    try:
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required; refusing CPU fallback")
        from torchvision import datasets, models, transforms

        device = torch.device("cuda")
        _event(event_log, "started", config_sha256=config_digest(config), source_sha256=_source_hash(),
               torch_version=str(torch.__version__), device_name=str(torch.cuda.get_device_name(device)),
               precision=config["precision"])
        dataset = datasets.CIFAR100(root=config["data_root"], train=True, download=True)
        raw = dataset.data
        labels = list(dataset.targets)
        if len(labels) != len(raw) or any(type(y) is not int or not 0 <= y < 100 for y in labels):
            raise ValueError("CIFAR100 train labels are malformed")
        train_idx, dev_idx = split_indices(len(raw), config["train_samples"],
                                            config["development_samples"], config["seed"])
        train_hashes, dev_hashes = _image_hashes(raw, train_idx), _image_hashes(raw, dev_idx)
        duplicate_hashes = sorted(train_hashes & dev_hashes)
        if duplicate_hashes:
            raise ValueError("cross-split image-byte duplicates detected")
        dataset_hashes = _file_hashes(Path(config["data_root"]))
        (output / "config.json").write_text(json.dumps(config, sort_keys=True, indent=2), encoding="utf-8")
        (output / "split_indices.json").write_text(json.dumps({"train": train_idx, "development": dev_idx},
                                                                 sort_keys=True), encoding="utf-8")
        (output / "dataset_hashes.json").write_text(json.dumps(dataset_hashes, sort_keys=True), encoding="utf-8")
        support = {str(c): labels.count(c) for c in range(100)}
        selected_support = {"train": {str(c): sum(labels[i] == c for i in train_idx) for c in range(100)},
                            "development": {str(c): sum(labels[i] == c for i in dev_idx) for c in range(100)}}
        (output / "class_support.json").write_text(json.dumps({"full": support, "selected": selected_support},
                                                                sort_keys=True), encoding="utf-8")

        weights = models.ResNet18_Weights.IMAGENET1K_V1
        model = models.resnet18(weights=weights).to(device)
        weight_path = Path(torch.hub.get_dir()) / 'checkpoints' / weights.url.rsplit('/', 1)[-1]
        _event(event_log, 'model_loaded', weights=str(weights), weights_url=weights.url,
               weights_sha256=hashlib.sha256(weight_path.read_bytes()).hexdigest(),
               cuda_capability=list(torch.cuda.get_device_capability(device)),
               feature_precision=config['precision'], head_precision='fp32')
        feature_dim = model.fc.in_features
        model.fc = torch.nn.Identity()
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        preprocess = transforms.Compose([
            transforms.Resize((config["image_size"], config["image_size"])),
            transforms.ToTensor(),
            transforms.Normalize(weights.meta["mean"] if "mean" in weights.meta else (0.485, 0.456, 0.406),
                                 weights.meta["std"] if "std" in weights.meta else (0.229, 0.224, 0.225)),
        ])
        dataset.transform = preprocess
        amp_dtype = {"fp32": None, "fp16": torch.float16, "bf16": torch.bfloat16}[config["precision"]]
        if (amp_dtype is torch.bfloat16 and
                (torch.cuda.get_device_capability(device)[0] < 8 or
                 not torch.cuda.is_bf16_supported())):
            raise RuntimeError("bf16 requested but CUDA device does not support bf16")

        def extract(indices, name):
            chunks = []
            with torch.no_grad():
                for start in range(0, len(indices), config["feature_batch_size"]):
                    batch = torch.stack([dataset[i][0] for i in indices[start:start + config["feature_batch_size"]]]).to(device)
                    with torch.autocast("cuda", dtype=amp_dtype, enabled=amp_dtype is not None):
                        features = model(batch)
                    if not torch.isfinite(features).all():
                        raise RuntimeError("non-finite feature output")
                    chunks.append(features.float().cpu())
            result = torch.cat(chunks)
            torch.save(result, output / name)
            return result

        train_features = extract(train_idx, "train_features.pt")
        dev_features = extract(dev_idx, "development_features.pt")
        with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
            torch.manual_seed(config["seed"] + 2)
            head = torch.nn.Linear(feature_dim, 100, device=device)
        optimizer = torch.optim.SGD(head.parameters(), lr=float(config["head_lr"]))
        generator = torch.Generator(device="cpu").manual_seed(config["seed"] + 1)
        train_y = torch.tensor([labels[i] for i in train_idx], dtype=torch.long)
        head.train()
        for epoch in range(config["head_epochs"]):
            permutation = torch.randperm(len(train_idx), generator=generator)
            losses = []
            planned = (len(train_idx) + config["head_batch_size"] - 1) // config["head_batch_size"]
            attempted = applied = 0
            for start in range(0, len(train_idx), config["head_batch_size"]):
                batch_idx = permutation[start:start + config["head_batch_size"]]
                logits = head(train_features[batch_idx].to(device))
                loss = torch.nn.functional.cross_entropy(logits, train_y[batch_idx].to(device))
                if not torch.isfinite(loss):
                    raise RuntimeError("non-finite head loss")
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                attempted += 1
                if not all(parameter.grad is None or torch.isfinite(parameter.grad).all()
                           for parameter in head.parameters()):
                    raise RuntimeError("non-finite head gradient")
                optimizer.step()
                applied += 1
                losses.append((float(loss.detach().cpu()), len(batch_idx)))
            _event(event_log, "head_epoch", epoch=epoch + 1, samples=len(train_idx),
                   batches=planned, updates_planned=planned, updates_attempted=attempted,
                   updates_applied=applied, updates_skipped=attempted - applied,
                   loss=sum(value * count for value, count in losses) / len(train_idx),
                   loss_finite=all(math.isfinite(value) for value, _ in losses))
        torch.save({"state_dict": head.state_dict(), "feature_dim": feature_dim}, output / "head_checkpoint.pt")
        head.eval()
        with torch.no_grad():
            dev_probs = torch.softmax(head(dev_features.to(device)), dim=1).float().cpu().tolist()
        if not all(math.isfinite(value) for row in dev_probs for value in row):
            raise RuntimeError("non-finite development probabilities")
        (output / "development_probabilities.json").write_text(json.dumps({
            "sample_ids": sample_ids(dev_idx),
            "labels": [labels[i] for i in dev_idx],
            "probabilities": dev_probs,
        }, allow_nan=False), encoding="utf-8")
        artifacts = {str(path.name): hashlib.sha256(path.read_bytes()).hexdigest()
                     for path in output.iterdir() if path.is_file() and path.name != "events.jsonl"}
        _event(event_log, "completed", artifacts=artifacts, train_samples=len(train_idx), development_samples=len(dev_idx))
    except Exception as error:
        try:
            _event(event_log, "failed", error_type=type(error).__name__, error=str(error))
        finally:
            event_log.__exit__(type(error), error, error.__traceback__)
            raise
    else:
        event_log.__exit__(None, None, None)


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 2:
        raise SystemExit("usage: python -m tralo.image_baseline CONFIG OUTPUT")
    config = json.loads(Path(argv[0]).read_text(encoding="utf-8"))
    run(config, argv[1])


if __name__ == "__main__":
    main()
