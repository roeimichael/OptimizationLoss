#!/usr/bin/env python3
"""
Run heuristic evaluation on all experiments in the heuristic folder.

This script will:
1. Find all experiment config files in results/heuristic/
2. Run the heuristic evaluation for each one
3. Save results in the heuristic folder structure (preserving optimization results)

NOTE: Run setup_heuristic_structure.py first to create the folder structure!
"""

import json
import subprocess
import sys
from pathlib import Path
from collections import defaultdict

def main():
    print("=" * 80)
    print("BATCH HEURISTIC EVALUATION")
    print("=" * 80)

    # Find all experiment configs in HEURISTIC folder
    results_dir = Path('results/heuristic')

    if not results_dir.exists():
        print(f"\nERROR: {results_dir} does not exist!")
        print("Please run setup_heuristic_structure.py first to create the folder structure.")
        return False

    all_configs = sorted(list(results_dir.rglob('config.json')))

    print(f"\nFound {len(all_configs)} experiments to evaluate")

    # Group by model and constraint for organized execution
    experiments_by_group = defaultdict(list)
    for config_path in all_configs:
        with open(config_path) as f:
            config = json.load(f)

        model_name = config.get('model_name', 'unknown')
        constraint = config.get('constraint', [])
        constraint_str = f"{constraint[0]}_{constraint[1]}" if len(constraint) == 2 else "unknown"

        experiments_by_group[f"{model_name}_{constraint_str}"].append(config_path)

    # Print summary
    print("\nExperiments grouped by model and constraint:")
    for group_name in sorted(experiments_by_group.keys()):
        count = len(experiments_by_group[group_name])
        print(f"  {group_name}: {count} experiments")

    print("\n" + "=" * 80)
    print("Starting heuristic evaluations...")
    print("=" * 80)

    success_count = 0
    error_count = 0
    errors = []

    for i, config_path in enumerate(all_configs, 1):
        # Read config to get metadata
        with open(config_path) as f:
            config = json.load(f)

        model_name = config.get('model_name', 'unknown')
        constraint = config.get('constraint', [])
        constraint_str = f"{constraint[0]}_{constraint[1]}" if len(constraint) == 2 else "unknown"
        variation = config.get('variation_name', 'unknown')

        print(f"\n[{i}/{len(all_configs)}] {model_name} | Constraint {constraint_str} | {variation}")

        # Run heuristic
        try:
            result = subprocess.run(
                ['python', 'run_heuristic.py', str(config_path)],
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout per experiment
            )

            if result.returncode == 0:
                success_count += 1
                # Extract accuracy from output
                for line in result.stdout.split('\n'):
                    if 'Acc:' in line:
                        print(f"  ✓ {line.strip()}")
                        break
            else:
                error_count += 1
                error_msg = f"{model_name}/{constraint_str}/{variation}: {result.stderr[:100]}"
                errors.append(error_msg)
                print(f"  ✗ ERROR: {result.stderr[:100]}")

        except subprocess.TimeoutExpired:
            error_count += 1
            error_msg = f"{model_name}/{constraint_str}/{variation}: Timeout (>5min)"
            errors.append(error_msg)
            print(f"  ✗ TIMEOUT")
        except Exception as e:
            error_count += 1
            error_msg = f"{model_name}/{constraint_str}/{variation}: {str(e)[:100]}"
            errors.append(error_msg)
            print(f"  ✗ EXCEPTION: {e}")

    # Final summary
    print("\n" + "=" * 80)
    print("HEURISTIC EVALUATION COMPLETE")
    print("=" * 80)
    print(f"Total experiments: {len(all_configs)}")
    print(f"Successful: {success_count}")
    print(f"Errors: {error_count}")

    if errors:
        print("\n⚠️  Errors encountered:")
        for error in errors[:10]:  # Show first 10 errors
            print(f"  - {error}")
        if len(errors) > 10:
            print(f"  ... and {len(errors) - 10} more")
    else:
        print("\n✓ ALL HEURISTIC EVALUATIONS COMPLETED SUCCESSFULLY!")

    print("\nResults are saved in results/heuristic/ folder:")
    print("  - evaluation_metrics.csv (heuristic results)")
    print("  - final_predictions.csv (heuristic predictions)")
    print("  - config.json (with heuristic results + original optimization_results)")
    print("\nOriginal optimization results preserved in results/our_approach/")
    print("=" * 80)

    return success_count == len(all_configs)

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
