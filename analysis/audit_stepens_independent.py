"""Independent, read-only audit of the four preregistered knee snapshot blocks.

This scorer intentionally imports neither score_stepens nor tralo allocation or
metric code. It operates on the archived epoch probability tensors and the
already-viewed development manifest; it never trains, changes a run, or opens
the sealed Chen test. Its fixed study identities come from the four knee
step-ensemble preregistrations. Usage:

  python analysis/audit_stepens_independent.py BLOCK RUN_ROOT NEW_OUTPUT.json
"""

import argparse
import hashlib
import json
import math
from pathlib import Path
from statistics import mean, stdev

from scipy import stats
import torch


STUDIES = {
    "resnet18": dict(seeds=range(4500, 4572), model_class="ResNet", cap=76, images=826, positives=106),
    "regnet_y_400mf": dict(seeds=range(4600, 4672), model_class="RegNet", cap=76, images=826, positives=106),
    "mobilenet_v3_large": dict(seeds=range(4700, 4772), model_class="MobileNetV3", cap=76, images=826, positives=106),
    "efficientnet_b5": dict(seeds=range(4800, 4848), model_class="EfficientNet", cap=76, images=826, positives=106),
}
ARMS = {"pto": "", "tralo": "_tralo", "sham": "_sham"}
METRICS = ("cc_f1", "accuracy", "macro_f1", "weighted_f1")


def capped_first_single(probabilities, sample_ids, capped, cap):
    """Fill one capped class by score, then assign every other item to an uncapped class."""
    if (len(probabilities) != len(sample_ids) or len(set(sample_ids)) != len(sample_ids)
            or any(not isinstance(sid, str) or not sid for sid in sample_ids)):
        raise ValueError("sample IDs must align and be unique")
    if (not probabilities or len(probabilities[0]) < 2 or not 0 <= capped < len(probabilities[0])
            or not 0 <= cap <= len(probabilities)):
        raise ValueError("invalid cap or probability shape")
    classes = range(len(probabilities[0]))
    for row in probabilities:
        if len(row) != len(classes) or any(not math.isfinite(p) or p < 0 or p > 1 for p in row):
            raise ValueError("invalid probability row")
        if not math.isclose(sum(row), 1.0, rel_tol=0, abs_tol=1e-5):
            raise ValueError("probability row does not sum to one")
    selected = set(sorted(range(len(sample_ids)), key=lambda i: (-probabilities[i][capped], sample_ids[i]))[:cap])
    other = [c for c in classes if c != capped]
    return [capped if i in selected else max(other, key=probabilities[i].__getitem__)
            for i in range(len(sample_ids))]


def classification_metrics(labels, predictions, capped, classes):
    if len(labels) != len(predictions) or not labels or any(y not in range(classes) for y in labels + predictions):
        raise ValueError("labels and predictions must be aligned class indices")
    f1, support = [], []
    for c in range(classes):
        actual = sum(y == c for y in labels)
        predicted = sum(y == c for y in predictions)
        tp = sum(y == c and p == c for y, p in zip(labels, predictions))
        f1.append(2 * tp / (actual + predicted) if actual + predicted else 0.0)
        support.append(actual)
    return dict(cc_f1=f1[capped], accuracy=sum(y == p for y, p in zip(labels, predictions)) / len(labels),
                macro_f1=sum(f1) / classes, weighted_f1=sum(f * n for f, n in zip(f1, support)) / len(labels))


def paired_interval(differences):
    n = len(differences)
    if n < 2:
        raise ValueError("a paired interval requires at least two seeds")
    center = mean(differences)
    spread = stdev(differences)
    if spread == 0:
        return dict(n=n, mean=center, sd=0.0, lo=center, hi=center, p=0.0 if center else 1.0)
    se = spread / math.sqrt(n)
    half = stats.t.ppf(0.975, n - 1) * se
    return dict(n=n, mean=center, sd=spread, lo=center - half, hi=center + half,
                p=float(2 * stats.t.sf(abs(center / se), n - 1)))


def holm_adjust(p_values):
    adjusted = [0.0] * len(p_values)
    running = 0.0
    for rank, i in enumerate(sorted(range(len(p_values)), key=p_values.__getitem__)):
        running = max(running, min(1.0, (len(p_values) - rank) * p_values[i]))
        adjusted[i] = running
    return adjusted


def preflight(root, architecture, study):
    expected = {f"seed{seed}" for seed in study["seeds"]}
    found = {d.name for d in root.glob("seed*") if d.is_dir()}
    if found != expected:
        raise RuntimeError(f"fixed block incomplete or mixed: missing {sorted(expected - found)}, extra {sorted(found - expected)}")
    plan = []
    for seed in study["seeds"]:
        directory = root / f"seed{seed}"
        summary_path = directory / "summary.json"
        if not summary_path.is_file():
            raise RuntimeError(f"seed{seed}: no completed summary")
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if summary.get("seed") != seed or len(summary.get("retrains", [])) != 1:
            raise RuntimeError(f"seed{seed}: wrong seed or retrain count")
        events = [json.loads(line) for line in (directory / "events.jsonl").read_text(encoding="utf-8").splitlines()]
        event = next((row for row in events if row.get("event") == "model_initialized"), None)
        if event is None or (event.get("architecture"), event.get("model_class")) != (
                architecture, study["model_class"]):
            raise RuntimeError(f"seed{seed}: model identity mismatch")
        retrain = summary["retrains"][0]
        best, last = retrain["best_epoch"], retrain["epochs_run"]
        if not 1 <= best <= last:
            raise RuntimeError(f"seed{seed}: invalid snapshot window")
        steps = retrain.get("snapshot_steps") or {}
        if set(steps) != {str(e) for e in range(1, last + 1)}:
            raise RuntimeError(f"seed{seed}: incomplete snapshot steps")
        for e in range(1, last + 1):
            target, sham = steps[str(e)]["tralo"], steps[str(e)]["sham"]
            if (target.get("radius") != sham.get("radius") or target["applied"] != sham["applied"]
                    or (target["applied"] and target["hard_after"] > study["cap"])):
                raise RuntimeError(f"seed{seed} epoch{e}: unmatched or invalid step")
        window = range(max(1, best - 2), last + 1)
        required = [directory / "manifest.json"]
        required += [directory / arm / "report.json" for arm in ("pto", "tralo_final", "sham_final")]
        required += [directory / "retrain1" / f"epoch{e:02d}{suffix}.pt"
                     for e in window for suffix in ARMS.values()]
        if any(not path.is_file() for path in required):
            raise RuntimeError(f"seed{seed}: missing manifest, report or snapshot")
        plan.append((seed, directory, window))
    return plan


def audit_block(root, architecture, study):
    plan = preflight(root, architecture, study)  # No development labels read before the fixed-set inventory.
    common_ids = common_labels = None
    seeds = []
    pto_hashes = set()
    for seed, directory, window in plan:
        rows = [r for r in json.loads((directory / "manifest.json").read_text(encoding="utf-8"))["rows"]
                if r["split"] == "val"]
        ids, labels = [r["sample_id"] for r in rows], [r["label"] for r in rows]
        if len(ids) != study["images"] or len(set(ids)) != len(ids) or sum(y == 3 for y in labels) != study["positives"]:
            raise RuntimeError(f"seed{seed}: development pool identity or support mismatch")
        if common_ids is None:
            common_ids, common_labels = ids, labels
        elif ids != common_ids or labels != common_labels:
            raise RuntimeError(f"seed{seed}: development pool differs from the first seed")
        arms = {}
        for arm, suffix in ARMS.items():
            tensors = [torch.load(directory / "retrain1" / f"epoch{e:02d}{suffix}.pt",
                                  map_location="cpu", weights_only=True) for e in window]
            if any(tuple(t.shape) != (study["images"], 5) or not bool(torch.isfinite(t).all()) for t in tensors):
                raise RuntimeError(f"seed{seed} {arm}: invalid saved tensor")
            probabilities = torch.stack(tensors).mean(0).tolist()
            predictions = capped_first_single(probabilities, ids, capped=3, cap=study["cap"])
            arms[arm] = dict(classification_metrics(labels, predictions, capped=3, classes=5),
                             prediction_sha256=hashlib.sha256(json.dumps(predictions).encode()).hexdigest())
        if arms["pto"]["prediction_sha256"] in pto_hashes:
            raise RuntimeError(f"seed{seed}: repeated PTO prediction vector")
        pto_hashes.add(arms["pto"]["prediction_sha256"])
        seeds.append(dict(seed=seed, arms=arms))
    contrasts = {}
    for name, control in (("E1_tralo_minus_sham", "sham"), ("E2_tralo_minus_pto", "pto")):
        contrasts[name] = {metric: paired_interval([row["arms"]["tralo"][metric] - row["arms"][control][metric]
                                                    for row in seeds]) for metric in METRICS}
    for metric in METRICS:
        ps = [contrasts[name][metric]["p"] for name in contrasts]
        for name, adjusted in zip(contrasts, holm_adjust(ps)):
            contrasts[name][metric]["holm_p"] = adjusted
    return dict(architecture=architecture, seed_count=len(seeds), cap=study["cap"],
                development_images=study["images"], development_grade3=study["positives"],
                contrasts=contrasts, per_seed=seeds)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("block", choices=STUDIES)
    parser.add_argument("root", type=Path)
    parser.add_argument("output", type=Path, help="new exclusive JSON file")
    args = parser.parse_args()
    result = audit_block(args.root, args.block, STUDIES[args.block])
    result["source_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    result["run_root"] = str(args.root.resolve())
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({name: row["cc_f1"] for name, row in result["contrasts"].items()}, sort_keys=True))


if __name__ == "__main__":
    main()
