import subprocess
import sys
from pathlib import Path
from typing import List


def find_all_configs(base_dir: Path) -> List[Path]:
    configs = sorted(base_dir.glob('**/config.json'))
    return configs


def run_experiment(config_path: Path) -> bool:
    try:
        result = subprocess.run(
            [sys.executable, 'src/experiments/run_experiment.py', str(config_path)],
            capture_output=False,
            text=True,
            timeout=3600  # 1 hour timeout per experiment
        )
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        print(f"[TIMEOUT] Experiment timed out after 1 hour: {config_path.parent}")
        return False
    except Exception as e:
        print(f"[ERROR] Failed to run experiment: {e}")
        return False


def main():
    base_dir = Path('results/longer_saturation')
    if not base_dir.exists():
        print(f"[ERROR] Directory not found: {base_dir}")
        sys.exit(1)

    configs = find_all_configs(base_dir)

    if len(configs) == 0:
        print(f"[ERROR] No config.json files found in {base_dir}")
        sys.exit(1)

    print(f"Found {len(configs)} experiments to run")

    successful = 0
    failed = 0
    skipped = 0

    for i, config_path in enumerate(configs, 1):
        parts = config_path.parts
        model_idx = parts.index('longer_saturation') + 1
        model_name = parts[model_idx] if model_idx < len(parts) else 'unknown'
        constraint_idx = model_idx + 1
        constraint_name = parts[constraint_idx] if constraint_idx < len(parts) else 'unknown'
        conv_idx = constraint_idx + 2  # Skip 'convergence_test'
        conv_name = parts[conv_idx] if conv_idx < len(parts) else 'unknown'

        print(f"\n{'='*80}")
        print(f"[{i}/{len(configs)}] Running: {model_name}/{constraint_name}/{conv_name}")
        print(f"{'='*80}")

        # Check if already completed
        exp_dir = config_path.parent
        status_marker = exp_dir / '.completed'
        if status_marker.exists():
            print(f"[SKIP] Already completed")
            skipped += 1
            continue

        # Run experiment
        success = run_experiment(config_path)

        if success:
            successful += 1
            # Mark as completed
            status_marker.touch()
            print(f"[SUCCESS] Completed {i}/{len(configs)}")
        else:
            failed += 1
            print(f"[FAILED] Failed {i}/{len(configs)}")

        # Print progress summary
        print(f"\nProgress: {i}/{len(configs)} | ✓ {successful} | ✗ {failed} | ⊘ {skipped}")

    # Final summary
    print(f"\n{'='*80}")
    print(f"FINAL SUMMARY")
    print(f"{'='*80}")
    print(f"Total experiments: {len(configs)}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Skipped (already done): {skipped}")
    print()

    if failed > 0:
        print("[WARNING] Some experiments failed. Check logs above for details.")
        sys.exit(1)
    else:
        print("[SUCCESS] All experiments completed!")
        sys.exit(0)


if __name__ == '__main__':
    main()
