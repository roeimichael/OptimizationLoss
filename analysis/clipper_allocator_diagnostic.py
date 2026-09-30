"""Label-free allocator comparison on the completed fmow2 PHR study.

Run from an immutable study release with this file supplied on stdin, or from a
checkout with the same analysis dependencies:

    python analysis/clipper_allocator_diagnostic.py RUN_ROOT RELEASE_ROOT

Only PTO probability snapshots, sample IDs, country IDs, and run receipts are
opened. No label array, label-bearing manifest rows, or reserved images are read.
"""

from collections import Counter
import hashlib
import json
from pathlib import Path
import sys


def _hash_json(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def compare_allocators(probabilities, ids, groups, quota, capped_class):
    """Compare two fixed policies on identical, label-free model probabilities."""
    from tralo.global_clipper import allocate, allocate_local_capped_first

    n = len(probabilities)
    if n == 0 or not (n == len(ids) == len(groups)) or len(set(ids)) != n:
        raise ValueError("probabilities, unique IDs, and countries must align")
    if set(quota["local_caps"]) != set(groups):
        raise ValueError("country quota scope differs from pool")
    classes = len(probabilities[0])
    caps = [None] * classes
    caps[capped_class] = quota["global_cap"]
    global_predictions = allocate(probabilities, caps, ids, "capped_first")
    local_predictions = allocate_local_capped_first(
        probabilities, caps, ids, groups, quota["local_caps"])

    def selected(predictions):
        return {sid for sid, value in zip(ids, predictions) if value == capped_class}

    global_ids, local_ids = selected(global_predictions), selected(local_predictions)
    if len(global_ids) != quota["global_cap"] or len(local_ids) != quota["global_cap"]:
        raise RuntimeError("one allocator did not fill the declared pooled cap")
    # Independent one-capped-class set oracle; no labels or allocator internals.
    order = lambda index: (-probabilities[index][capped_class], ids[index])
    global_expected = {ids[i] for i in sorted(range(n), key=order)[:quota["global_cap"]]}
    eligible = [i for country in sorted(quota["local_caps"])
                for i in sorted((j for j, g in enumerate(groups) if g == country), key=order)
                [:quota["local_caps"][country]]]
    local_expected = {ids[i] for i in sorted(eligible, key=order)[:quota["global_cap"]]}
    if global_ids != global_expected or local_ids != local_expected:
        raise RuntimeError("allocator output disagrees with independent selected-set oracle")
    global_counts = Counter(group for sid, group in zip(ids, groups) if sid in global_ids)
    local_counts = Counter(group for sid, group in zip(ids, groups) if sid in local_ids)
    by_country = {}
    for group in sorted(quota["local_caps"]):
        cap = quota["local_caps"][group]
        by_country[group] = {"cap": cap,
                             "global_clipper_count": global_counts[group],
                             "global_clipper_excess": max(0, global_counts[group] - cap),
                             "local_clipper_count": local_counts[group]}
        if local_counts[group] > cap:
            raise RuntimeError("local allocator violates a country cap")
    overlap = len(global_ids & local_ids)
    return {"global_selected": len(global_ids), "local_selected": len(local_ids),
            "country_counts": by_country,
            "global_country_cap_excess_total": sum(row["global_clipper_excess"]
                                                    for row in by_country.values()),
            "global_violated_countries": [g for g, row in by_country.items()
                                          if row["global_clipper_excess"] > 0],
            "selected_overlap": overlap,
            "global_only_selected": len(global_ids - local_ids),
            "local_only_selected": len(local_ids - global_ids),
            "selected_jaccard": overlap / len(global_ids | local_ids),
            "global_selected_id_sha256": _hash_json(sorted(global_ids)),
            "local_selected_id_sha256": _hash_json(sorted(local_ids))}


def main(run_root, release_root):
    """Require all 12 completed immutable runs and audit every PTO artifact."""
    root, release = Path(run_root).resolve(), Path(release_root).resolve()
    if not root.is_dir() or not release.is_dir():
        raise FileNotFoundError("run or release root is absent")
    sys.path.insert(0, str(release))
    import torch
    from analysis import score_fmow_local_alm as alm
    from analysis import score_fmow_local as prior

    names = sorted(p.name for p in root.glob("seed*") if p.is_dir())
    expected = sorted(f"seed{seed}" for seed in alm.SEEDS)
    if names != expected:
        raise RuntimeError(f"expected the complete fixed 12-seed block; found {names}")
    rows = []
    reference_pool_hash = None
    for seed in alm.SEEDS:
        directory = root / f"seed{seed}"
        config, summary, started, _, ids, groups = alm._receipt(directory)
        if config["seed"] != seed or not config["snapshot_steps"]:
            raise RuntimeError("seed/config differs from completed study")
        if alm._launch_receipt(directory, config, started)["release_commit"] != release.name:
            raise RuntimeError("run launch release differs from supplied immutable release")
        pool_hash = started["pool_identity_sha256"]
        if reference_pool_hash is None:
            reference_pool_hash = pool_hash
        elif pool_hash != reference_pool_hash:
            raise RuntimeError("development pool identity differs across seeds")
        epoch_count, best_epoch = (summary["retrain"][key]
                                   for key in ("epochs_run", "best_epoch"))
        window = list(range(max(1, best_epoch - 2), epoch_count + 1))
        snapshots = []
        hashes = {}
        for epoch in window:
            path = directory / "retrain1" / f"epoch{epoch:02d}.pt"
            digest = prior.sha256(path)
            if digest != summary["pto_snapshot_sha256"][str(epoch)]:
                raise RuntimeError("PTO snapshot differs from creation-time hash")
            hashes[str(epoch)] = digest
            snapshots.append(prior._probabilities(path, len(ids)))
        ensemble = torch.stack(snapshots).mean(0)
        values = ensemble.tolist()
        comparisons = {}
        for divisor in prior.DIVISORS:
            quota = started["quotas"][str(divisor)]
            if quota != prior._budget(groups, divisor):
                raise RuntimeError("country quotas differ from label-free size policy")
            comparisons[str(divisor)] = compare_allocators(
                values, ids, groups, quota, prior.CAPPED)
        rows.append({"seed": seed, "best_epoch": best_epoch,
                     "epochs_run": epoch_count, "ensemble_window": window,
                     "config_sha256": started["config_sha256"],
                     "pto_snapshot_sha256": hashes,
                     "ensemble_probabilities_sha256": hashlib.sha256(
                         ensemble.contiguous().numpy().tobytes()).hexdigest(),
                     "caps": comparisons})

    aggregate = {}
    for divisor in prior.DIVISORS:
        cells = [row["caps"][str(divisor)] for row in rows]
        aggregate[str(divisor)] = {
            "seeds": len(cells),
            "global_clipper_country_cap_violation_seeds": sum(
                cell["global_country_cap_excess_total"] > 0 for cell in cells),
            "global_clipper_country_cap_excess_total": sum(
                cell["global_country_cap_excess_total"] for cell in cells),
            "selected_overlap_mean": sum(cell["selected_overlap"] for cell in cells) / len(cells),
            "selected_overlap_range": [min(cell["selected_overlap"] for cell in cells),
                                       max(cell["selected_overlap"] for cell in cells)]}
    return {"status": "complete_label_free_allocator_diagnostic",
            "study_release_commit": release.name,
            "run_root": str(root), "release_root": str(release),
            "runner_source_sha256": prior.source(),
            "pool_identity_sha256": reference_pool_hash,
            "data_file_sha256_receipts": prior.FILES,
            "labels_accessed": False, "reserved_countries_scored": False,
            "methods": ["global_capped_first", "local_capped_first"],
            "aggregate": aggregate, "per_seed": rows}


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    print(json.dumps(main(sys.argv[1], sys.argv[2]), indent=2, allow_nan=False))
