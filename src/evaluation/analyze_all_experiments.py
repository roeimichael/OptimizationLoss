#!/usr/bin/env python3
"""Comprehensive analysis of ALL completed experiments in pending_runs."""

import json
import os
import csv
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd
import numpy as np

BASE_DIR = Path(__file__).resolve().parents[2] / "results" / "pending_runs"

pd.set_option("display.max_columns", 30)
pd.set_option("display.width", 200)
pd.set_option("display.max_rows", 200)
pd.set_option("display.float_format", "{:.4f}".format)
pd.set_option("display.max_colwidth", 40)


def load_all_configs():
    """Walk all config.json files, filter completed, build records."""
    records = []
    total = 0
    statuses = defaultdict(int)

    for root, dirs, files in os.walk(BASE_DIR):
        if "config.json" not in files:
            continue
        total += 1
        config_path = os.path.join(root, "config.json")
        try:
            with open(config_path, "r") as f:
                cfg = json.load(f)
        except Exception as e:
            print(f"  ERROR reading {config_path}: {e}")
            continue

        status = cfg.get("status", "unknown")
        statuses[status] += 1

        if status != "completed":
            continue

        # Extract path components
        rel = os.path.relpath(root, BASE_DIR)
        parts = rel.replace("\\", "/").split("/")
        # Expected: dataset/scenario/constraint_tag/model/methodology/slice_N
        if len(parts) < 6:
            print(f"  WARNING: unexpected path depth: {rel}")
            continue

        dataset = parts[0]
        scenario = parts[1]
        constraint_tag = parts[2]
        model_name = parts[3]
        methodology = parts[4]
        slice_str = parts[5]
        slice_num = int(slice_str.replace("slice_", "")) if "slice_" in slice_str else -1

        results = cfg.get("results", {})
        results_comparison = cfg.get("results_comparison", {})

        rec = {
            "dataset": dataset,
            "scenario": scenario,
            "constraint_tag": constraint_tag,
            "model_name": model_name,
            "methodology": methodology,
            "slice": slice_num,
            "accuracy": results.get("accuracy"),
            "f1_macro": results.get("f1_macro"),
            "precision_macro": results.get("precision_macro"),
            "recall_macro": results.get("recall_macro"),
            "training_time": results.get("training_time"),
            "samples_adjusted": results.get("samples_adjusted"),
            "checkpoint_source": results.get("checkpoint_source", "N/A"),
            "lp_fallback_used": results.get("lp_fallback_used", None),
            "constraint": cfg.get("constraint"),
            "constrained_class": cfg.get("dataset_config", {}).get("constrained_class"),
            "exp_dir": root,
            "has_final_predictions": os.path.exists(os.path.join(root, "final_predictions.csv")),
        }
        # For our_approach, also store comparison data
        if results_comparison:
            for key in ["final", "bracket_best", "bracket_previous"]:
                sub = results_comparison.get(key, {})
                rec[f"cmp_{key}_f1"] = sub.get("f1_macro")
                rec[f"cmp_{key}_acc"] = sub.get("accuracy")
                rec[f"cmp_{key}_adjusted"] = sub.get("adjusted")
                rec[f"cmp_{key}_lp_fallback"] = sub.get("lp_fallback_used")

        records.append(rec)

    return records, total, statuses


def check_constraint_violations(df, n_samples=30):
    """
    For a sample of completed experiments, read final_predictions.csv and
    check if constraints are violated.
    """
    print("\n" + "=" * 80)
    print("SECTION 4: CONSTRAINT VIOLATION CHECK")
    print("=" * 80)

    # Sample from experiments that have final_predictions
    candidates = df[df["has_final_predictions"]].copy()
    if len(candidates) == 0:
        print("No experiments with final_predictions.csv found!")
        return

    # Sample across different scenarios/methodologies
    np.random.seed(42)
    sample_indices = []
    for (dataset, scenario, methodology), grp in candidates.groupby(["dataset", "scenario", "methodology"]):
        n = min(3, len(grp))
        sample_indices.extend(grp.sample(n=n, random_state=42).index.tolist())

    sample_indices = list(set(sample_indices))
    if len(sample_indices) > n_samples:
        sample_indices = sorted(np.random.choice(sample_indices, n_samples, replace=False))

    sampled = candidates.loc[sample_indices]
    print(f"\nSampling {len(sampled)} experiments for constraint violation checks...\n")

    violations_found = []
    for idx, row in sampled.iterrows():
        exp_dir = row["exp_dir"]
        pred_path = os.path.join(exp_dir, "final_predictions.csv")
        config_path = os.path.join(exp_dir, "config.json")

        try:
            with open(config_path, "r") as f:
                cfg = json.load(f)
        except:
            continue

        try:
            preds = pd.read_csv(pred_path)
        except:
            continue

        constraint = cfg.get("constraint", [])
        constrained_classes = cfg.get("dataset_config", {}).get("constrained_class", [])
        if not isinstance(constrained_classes, list):
            constrained_classes = [constrained_classes]
        if not isinstance(constraint, list):
            constraint = [constraint]

        n_test = len(preds)
        groups = preds["Group_ID"].unique()

        issues = []
        for ci, cls_idx in enumerate(constrained_classes):
            local_frac = constraint[0] if len(constraint) > 0 else None
            global_frac = constraint[1] if len(constraint) > 1 else constraint[0] if len(constraint) > 0 else None

            # Global constraint check
            global_count = (preds["Predicted_Label"] == cls_idx).sum()
            global_limit = int(np.ceil(global_frac * n_test)) if global_frac and global_frac < 1 else None

            if global_limit is not None and global_count > global_limit:
                issues.append(
                    f"  GLOBAL VIOLATION class {cls_idx}: {global_count} predicted > limit {global_limit} "
                    f"({global_frac*100:.0f}% of {n_test})"
                )

            # Local (per-group) constraint check
            if local_frac and local_frac < 1:
                for gid in groups:
                    grp_mask = preds["Group_ID"] == gid
                    grp_size = grp_mask.sum()
                    local_count = ((preds["Predicted_Label"] == cls_idx) & grp_mask).sum()
                    local_limit = int(np.ceil(local_frac * grp_size))
                    if local_count > local_limit:
                        issues.append(
                            f"  LOCAL VIOLATION class {cls_idx} group {gid}: {local_count} predicted > "
                            f"limit {local_limit} ({local_frac*100:.0f}% of {grp_size})"
                        )

        short_path = os.path.relpath(exp_dir, BASE_DIR).replace("\\", "/")
        if issues:
            violations_found.append((short_path, issues))
            print(f"VIOLATION in {short_path}:")
            for iss in issues:
                print(iss)
        else:
            print(f"  OK: {short_path}")

    print(f"\n--- Summary: {len(violations_found)} / {len(sampled)} sampled experiments have constraint violations ---")
    if violations_found:
        print("\nExperiments with violations:")
        for path, issues in violations_found:
            print(f"  {path}: {len(issues)} violation(s)")


def main():
    print("=" * 80)
    print("COMPREHENSIVE EXPERIMENT ANALYSIS")
    print("=" * 80)
    print(f"Base directory: {BASE_DIR}\n")

    records, total_configs, statuses = load_all_configs()

    print(f"Total config.json files found: {total_configs}")
    print(f"Status distribution:")
    for s, c in sorted(statuses.items()):
        print(f"  {s}: {c}")
    print(f"\nCompleted experiments loaded: {len(records)}")

    if not records:
        print("No completed experiments found!")
        sys.exit(1)

    df = pd.DataFrame(records)

    # =========================================================================
    # SECTION 3a: Count of completed experiments by dataset/scenario/methodology
    # =========================================================================
    print("\n" + "=" * 80)
    print("SECTION 3a: COUNT OF COMPLETED EXPERIMENTS")
    print("=" * 80)

    pivot_count = df.groupby(["dataset", "scenario", "methodology"]).size().reset_index(name="count")
    print("\nBy dataset / scenario / methodology:")
    print(pivot_count.to_string(index=False))

    pivot_full = df.groupby(["dataset", "scenario", "constraint_tag", "methodology"]).size().reset_index(name="count")
    print("\n\nFull breakdown (dataset / scenario / constraint_tag / methodology):")
    print(pivot_full.to_string(index=False))

    # =========================================================================
    # SECTION 3b: Mean accuracy and f1_macro by methodology, grouped by constraint_tag
    # =========================================================================
    print("\n" + "=" * 80)
    print("SECTION 3b: MEAN ACCURACY & F1 BY METHODOLOGY, GROUPED BY CONSTRAINT_TAG")
    print("=" * 80)

    agg_method = df.groupby(["constraint_tag", "methodology"]).agg(
        accuracy_mean=("accuracy", "mean"),
        accuracy_std=("accuracy", "std"),
        f1_mean=("f1_macro", "mean"),
        f1_std=("f1_macro", "std"),
        n=("accuracy", "count"),
    ).reset_index()
    print("\n" + agg_method.to_string(index=False))

    # Also show overall methodology means
    print("\n\nOverall by methodology:")
    overall_method = df.groupby("methodology").agg(
        accuracy_mean=("accuracy", "mean"),
        accuracy_std=("accuracy", "std"),
        f1_mean=("f1_macro", "mean"),
        f1_std=("f1_macro", "std"),
        n=("accuracy", "count"),
    ).reset_index()
    print(overall_method.to_string(index=False))

    # =========================================================================
    # SECTION 3c: Mean accuracy and f1_macro by model
    # =========================================================================
    print("\n" + "=" * 80)
    print("SECTION 3c: MEAN ACCURACY & F1 BY MODEL")
    print("=" * 80)

    agg_model = df.groupby(["model_name", "methodology"]).agg(
        accuracy_mean=("accuracy", "mean"),
        accuracy_std=("accuracy", "std"),
        f1_mean=("f1_macro", "mean"),
        f1_std=("f1_macro", "std"),
        n=("accuracy", "count"),
    ).reset_index()
    print("\n" + agg_model.to_string(index=False))

    agg_model_overall = df.groupby("model_name").agg(
        accuracy_mean=("accuracy", "mean"),
        f1_mean=("f1_macro", "mean"),
        n=("accuracy", "count"),
    ).reset_index()
    print("\n\nOverall by model (both methodologies combined):")
    print(agg_model_overall.to_string(index=False))

    # =========================================================================
    # SECTION 3d: Number of experiments where lp_fallback_used == True
    # =========================================================================
    print("\n" + "=" * 80)
    print("SECTION 3d: LP FALLBACK USAGE")
    print("=" * 80)

    lp_data = df[df["lp_fallback_used"].notna()]
    lp_true = lp_data[lp_data["lp_fallback_used"] == True]
    print(f"\nExperiments with lp_fallback_used field: {len(lp_data)}")
    print(f"Experiments where lp_fallback_used == True: {len(lp_true)}")

    if len(lp_true) > 0:
        print("\nLP fallback experiments breakdown:")
        lp_breakdown = lp_true.groupby(["dataset", "scenario", "constraint_tag", "model_name"]).size().reset_index(name="count")
        print(lp_breakdown.to_string(index=False))

    # =========================================================================
    # SECTION 3e: Distribution of checkpoint_source
    # =========================================================================
    print("\n" + "=" * 80)
    print("SECTION 3e: CHECKPOINT SOURCE DISTRIBUTION")
    print("=" * 80)

    # Only for our_approach (heuristic doesn't have this field typically)
    oa_df = df[df["methodology"] == "our_approach"]
    ckpt_dist = oa_df["checkpoint_source"].value_counts()
    print(f"\nCheckpoint source distribution (our_approach only, n={len(oa_df)}):")
    for src, cnt in ckpt_dist.items():
        print(f"  {src}: {cnt} ({cnt/len(oa_df)*100:.1f}%)")

    # Also show F1 by checkpoint source
    if len(oa_df) > 0:
        print("\nMean F1 by checkpoint source:")
        ckpt_f1 = oa_df.groupby("checkpoint_source").agg(
            f1_mean=("f1_macro", "mean"),
            f1_std=("f1_macro", "std"),
            acc_mean=("accuracy", "mean"),
            samples_adj_mean=("samples_adjusted", "mean"),
            n=("f1_macro", "count"),
        ).reset_index()
        print(ckpt_f1.to_string(index=False))

    # =========================================================================
    # SECTION 3f: Mean samples_adjusted by methodology
    # =========================================================================
    print("\n" + "=" * 80)
    print("SECTION 3f: MEAN SAMPLES ADJUSTED BY METHODOLOGY")
    print("=" * 80)

    adj_data = df[df["samples_adjusted"].notna()]
    adj_by_method = adj_data.groupby("methodology").agg(
        samples_adj_mean=("samples_adjusted", "mean"),
        samples_adj_std=("samples_adjusted", "std"),
        samples_adj_median=("samples_adjusted", "median"),
        samples_adj_max=("samples_adjusted", "max"),
        n=("samples_adjusted", "count"),
    ).reset_index()
    print("\n" + adj_by_method.to_string(index=False))

    # Also break down by constraint_tag
    print("\n\nSamples adjusted by methodology & constraint_tag:")
    adj_by_ct = adj_data.groupby(["constraint_tag", "methodology"]).agg(
        adj_mean=("samples_adjusted", "mean"),
        adj_median=("samples_adjusted", "median"),
        adj_max=("samples_adjusted", "max"),
        n=("samples_adjusted", "count"),
    ).reset_index()
    print(adj_by_ct.to_string(index=False))

    # =========================================================================
    # SECTION 3g: Suspiciously low accuracy or F1
    # =========================================================================
    print("\n" + "=" * 80)
    print("SECTION 3g: SUSPICIOUS LOW-PERFORMANCE EXPERIMENTS")
    print("=" * 80)

    low_acc = df[df["accuracy"] < 0.3]
    low_f1 = df[df["f1_macro"] < 0.15]
    suspicious = df[(df["accuracy"] < 0.3) | (df["f1_macro"] < 0.15)]

    print(f"\nExperiments with accuracy < 0.3: {len(low_acc)}")
    print(f"Experiments with f1_macro < 0.15: {len(low_f1)}")
    print(f"Union (either criterion): {len(suspicious)}")

    if len(suspicious) > 0:
        print("\nSuspicious experiments:")
        cols = ["dataset", "scenario", "constraint_tag", "model_name", "methodology", "slice", "accuracy", "f1_macro", "training_time", "checkpoint_source"]
        print(suspicious[cols].to_string(index=False))

    # Also check for moderately low F1 (< 0.3) which might still be concerning
    moderate_low = df[(df["f1_macro"] < 0.30) & (df["f1_macro"] >= 0.15)]
    print(f"\nExperiments with 0.15 <= f1_macro < 0.30: {len(moderate_low)}")
    if len(moderate_low) > 0:
        print(moderate_low[["dataset", "scenario", "constraint_tag", "model_name", "methodology", "slice", "accuracy", "f1_macro"]].to_string(index=False))

    # =========================================================================
    # SECTION 3h: Head-to-head comparison: our_approach vs heuristic
    # =========================================================================
    print("\n" + "=" * 80)
    print("SECTION 3h: HEAD-TO-HEAD COMPARISON (our_approach vs heuristic)")
    print("=" * 80)

    # Pivot to get our_approach and heuristic side by side
    merge_cols = ["dataset", "scenario", "constraint_tag", "model_name", "slice"]

    oa = df[df["methodology"] == "our_approach"][merge_cols + ["accuracy", "f1_macro", "samples_adjusted"]].copy()
    oa.columns = merge_cols + ["acc_oa", "f1_oa", "adj_oa"]

    hr = df[df["methodology"] == "heuristic"][merge_cols + ["accuracy", "f1_macro", "samples_adjusted"]].copy()
    hr.columns = merge_cols + ["acc_hr", "f1_hr", "adj_hr"]

    h2h = pd.merge(oa, hr, on=merge_cols, how="inner")
    h2h["f1_diff"] = h2h["f1_oa"] - h2h["f1_hr"]
    h2h["acc_diff"] = h2h["acc_oa"] - h2h["acc_hr"]

    print(f"\nMatched pairs (same scenario/constraint/model/slice): {len(h2h)}")

    if len(h2h) > 0:
        print(f"\nOverall head-to-head statistics:")
        print(f"  Mean F1 diff (oa - hr): {h2h['f1_diff'].mean():.4f} (std={h2h['f1_diff'].std():.4f})")
        print(f"  Median F1 diff: {h2h['f1_diff'].median():.4f}")
        print(f"  our_approach wins: {(h2h['f1_diff'] > 0).sum()} / {len(h2h)} ({(h2h['f1_diff'] > 0).mean()*100:.1f}%)")
        print(f"  heuristic wins:   {(h2h['f1_diff'] < 0).sum()} / {len(h2h)} ({(h2h['f1_diff'] < 0).mean()*100:.1f}%)")
        print(f"  ties (exact):     {(h2h['f1_diff'] == 0).sum()}")
        print(f"\n  Mean Acc diff (oa - hr): {h2h['acc_diff'].mean():.4f} (std={h2h['acc_diff'].std():.4f})")
        print(f"  Median Acc diff: {h2h['acc_diff'].median():.4f}")

        # By constraint_tag
        print("\n\nHead-to-head by constraint_tag:")
        h2h_by_ct = h2h.groupby("constraint_tag").agg(
            f1_diff_mean=("f1_diff", "mean"),
            f1_diff_std=("f1_diff", "std"),
            acc_diff_mean=("acc_diff", "mean"),
            oa_wins=("f1_diff", lambda x: (x > 0).sum()),
            hr_wins=("f1_diff", lambda x: (x < 0).sum()),
            n=("f1_diff", "count"),
        ).reset_index()
        h2h_by_ct["oa_win_rate"] = h2h_by_ct["oa_wins"] / h2h_by_ct["n"]
        print(h2h_by_ct.to_string(index=False))

        # By dataset
        print("\n\nHead-to-head by dataset:")
        h2h_by_ds = h2h.groupby("dataset").agg(
            f1_diff_mean=("f1_diff", "mean"),
            acc_diff_mean=("acc_diff", "mean"),
            oa_wins=("f1_diff", lambda x: (x > 0).sum()),
            hr_wins=("f1_diff", lambda x: (x < 0).sum()),
            n=("f1_diff", "count"),
        ).reset_index()
        h2h_by_ds["oa_win_rate"] = h2h_by_ds["oa_wins"] / h2h_by_ds["n"]
        print(h2h_by_ds.to_string(index=False))

        # By model
        print("\n\nHead-to-head by model:")
        h2h_by_model = h2h.groupby("model_name").agg(
            f1_diff_mean=("f1_diff", "mean"),
            acc_diff_mean=("acc_diff", "mean"),
            oa_wins=("f1_diff", lambda x: (x > 0).sum()),
            hr_wins=("f1_diff", lambda x: (x < 0).sum()),
            n=("f1_diff", "count"),
        ).reset_index()
        h2h_by_model["oa_win_rate"] = h2h_by_model["oa_wins"] / h2h_by_model["n"]
        print(h2h_by_model.to_string(index=False))

        # By scenario
        print("\n\nHead-to-head by scenario:")
        h2h_by_scenario = h2h.groupby(["dataset", "scenario"]).agg(
            f1_diff_mean=("f1_diff", "mean"),
            acc_diff_mean=("acc_diff", "mean"),
            oa_wins=("f1_diff", lambda x: (x > 0).sum()),
            hr_wins=("f1_diff", lambda x: (x < 0).sum()),
            n=("f1_diff", "count"),
        ).reset_index()
        h2h_by_scenario["oa_win_rate"] = h2h_by_scenario["oa_wins"] / h2h_by_scenario["n"]
        print(h2h_by_scenario.to_string(index=False))

        # Worst and best head-to-head comparisons
        print("\n\nTop 10 LARGEST F1 advantages for our_approach:")
        top_oa = h2h.nlargest(10, "f1_diff")[merge_cols + ["f1_oa", "f1_hr", "f1_diff"]]
        print(top_oa.to_string(index=False))

        print("\n\nTop 10 LARGEST F1 advantages for heuristic:")
        top_hr = h2h.nsmallest(10, "f1_diff")[merge_cols + ["f1_oa", "f1_hr", "f1_diff"]]
        print(top_hr.to_string(index=False))

    # =========================================================================
    # SECTION 3 (bonus): Training time comparison
    # =========================================================================
    print("\n" + "=" * 80)
    print("BONUS: TRAINING TIME COMPARISON")
    print("=" * 80)

    time_by_method = df.groupby("methodology").agg(
        time_mean=("training_time", "mean"),
        time_std=("training_time", "std"),
        time_median=("training_time", "median"),
        time_min=("training_time", "min"),
        time_max=("training_time", "max"),
        n=("training_time", "count"),
    ).reset_index()
    print("\nTraining time (seconds) by methodology:")
    print(time_by_method.to_string(index=False))

    # =========================================================================
    # SECTION 4: Constraint violations
    # =========================================================================
    check_constraint_violations(df, n_samples=30)

    # =========================================================================
    # FINAL SUMMARY
    # =========================================================================
    print("\n" + "=" * 80)
    print("FINAL SUMMARY")
    print("=" * 80)
    print(f"Total experiments found: {total_configs}")
    print(f"Completed experiments: {len(df)}")
    print(f"Datasets: {df['dataset'].unique().tolist()}")
    print(f"Scenarios: {df['scenario'].unique().tolist()}")
    print(f"Constraint tags: {sorted(df['constraint_tag'].unique().tolist())}")
    print(f"Models: {sorted(df['model_name'].unique().tolist())}")
    print(f"Methodologies: {df['methodology'].unique().tolist()}")
    print(f"Slices per experiment: {sorted(df['slice'].unique().tolist())}")
    n_oa = len(df[df['methodology'] == 'our_approach'])
    n_hr = len(df[df['methodology'] == 'heuristic'])
    print(f"our_approach experiments: {n_oa}")
    print(f"heuristic experiments: {n_hr}")
    if len(h2h) > 0:
        print(f"\nKey finding: our_approach vs heuristic (matched pairs: {len(h2h)})")
        print(f"  our_approach mean F1: {h2h['f1_oa'].mean():.4f}")
        print(f"  heuristic mean F1:    {h2h['f1_hr'].mean():.4f}")
        print(f"  F1 advantage (oa):    {h2h['f1_diff'].mean():+.4f}")
        print(f"  our_approach win rate: {(h2h['f1_diff'] > 0).mean()*100:.1f}%")


if __name__ == "__main__":
    main()
