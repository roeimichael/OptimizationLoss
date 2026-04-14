"""Statistical significance analysis for our_approach vs heuristic.

Runs paired comparisons on matched experiments (same scenario, constraint,
model, slice) using Wilcoxon signed-rank tests and paired t-tests.

Produces:
  - Overall significance across all matched pairs
  - Per-dataset breakdown
  - Per-constraint-tag breakdown
  - Per-model breakdown
  - Per-scenario breakdown
  - Effect size (Cohen's d)
  - Summary CSV and LaTeX table

Usage:
    python -m analysis.statistical_significance [--results-dir results/pending_runs]
"""

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats


def load_matched_pairs(results_dir):
    """Load all completed experiments and match our_approach vs heuristic pairs.

    Returns list of dicts with keys: dataset, scenario, constraint_tag, model,
    slice, oa_accuracy, h_accuracy, oa_f1, h_f1, oa_adj, h_adj.
    """
    # Collect all completed experiments indexed by matching key
    by_key = defaultdict(dict)

    for config_path in Path(results_dir).rglob('config.json'):
        try:
            with open(config_path) as f:
                cfg = json.load(f)
        except (json.JSONDecodeError, ValueError):
            continue
        if cfg.get('status') != 'completed':
            continue

        results = cfg.get('results', {})
        if not results:
            continue

        dataset = cfg.get('dataset_mode', 'unknown')
        parts = config_path.relative_to(results_dir).parts
        scenario = parts[1] if len(parts) > 1 else 'unknown'
        tag = cfg.get('constraint_tag', '')
        model = cfg.get('model_name', '')
        method = cfg.get('methodology', '')
        slice_name = config_path.parent.name

        key = (dataset, scenario, tag, model, slice_name)
        by_key[key][method] = {
            'accuracy': results.get('accuracy', 0),
            'f1_macro': results.get('f1_macro', 0),
            'precision_macro': results.get('precision_macro', 0),
            'recall_macro': results.get('recall_macro', 0),
            'samples_adjusted': results.get('samples_adjusted', 0),
        }

    # Build matched pairs
    pairs = []
    for (dataset, scenario, tag, model, slice_name), methods in by_key.items():
        if 'our_approach' in methods and 'heuristic' in methods:
            oa = methods['our_approach']
            h = methods['heuristic']
            pairs.append({
                'dataset': dataset,
                'scenario': scenario,
                'constraint_tag': tag,
                'model': model,
                'slice': slice_name,
                'oa_accuracy': oa['accuracy'],
                'h_accuracy': h['accuracy'],
                'oa_f1': oa['f1_macro'],
                'h_f1': h['f1_macro'],
                'oa_precision': oa['precision_macro'],
                'h_precision': h['precision_macro'],
                'oa_recall': oa['recall_macro'],
                'h_recall': h['recall_macro'],
                'oa_adj': oa['samples_adjusted'],
                'h_adj': h['samples_adjusted'],
            })

    return pairs


def cohens_d(a, b):
    """Compute Cohen's d for paired samples."""
    diff = np.array(a) - np.array(b)
    if diff.std() == 0:
        return 0.0
    return diff.mean() / diff.std()


def effect_size_label(d):
    """Interpret Cohen's d magnitude."""
    d = abs(d)
    if d < 0.2:
        return 'negligible'
    elif d < 0.5:
        return 'small'
    elif d < 0.8:
        return 'medium'
    else:
        return 'large'


def run_significance_tests(oa_values, h_values, label=""):
    """Run Wilcoxon signed-rank and paired t-test on matched values.

    Returns dict with test statistics.
    """
    n = len(oa_values)
    if n < 3:
        return {
            'label': label,
            'n': n,
            'note': 'Too few pairs for significance testing',
        }

    oa = np.array(oa_values)
    h = np.array(h_values)
    diff = oa - h

    mean_diff = diff.mean()
    std_diff = diff.std(ddof=1) if n > 1 else 0
    oa_wins = int((diff > 1e-6).sum())
    h_wins = int((diff < -1e-6).sum())
    ties = n - oa_wins - h_wins

    # Paired t-test
    t_stat, t_pval = stats.ttest_rel(oa, h)

    # Wilcoxon signed-rank test (non-parametric)
    # Remove zero differences for Wilcoxon
    nonzero_diff = diff[np.abs(diff) > 1e-10]
    if len(nonzero_diff) >= 10:
        w_stat, w_pval = stats.wilcoxon(nonzero_diff)
    else:
        w_stat, w_pval = float('nan'), float('nan')

    d = cohens_d(oa, h)

    return {
        'label': label,
        'n': n,
        'oa_mean': float(oa.mean()),
        'h_mean': float(h.mean()),
        'mean_diff': float(mean_diff),
        'std_diff': float(std_diff),
        'oa_wins': oa_wins,
        'h_wins': h_wins,
        'ties': ties,
        'win_rate': oa_wins / n if n > 0 else 0,
        't_stat': float(t_stat),
        't_pval': float(t_pval),
        'w_stat': float(w_stat) if not np.isnan(w_stat) else None,
        'w_pval': float(w_pval) if not np.isnan(w_pval) else None,
        'cohens_d': float(d),
        'effect_size': effect_size_label(d),
    }


def print_result(r, metric_name="F1"):
    """Pretty-print a single significance test result."""
    if 'note' in r:
        print(f"  {r['label']}: n={r['n']} — {r['note']}")
        return

    sig_t = "***" if r['t_pval'] < 0.001 else (
        "**" if r['t_pval'] < 0.01 else (
            "*" if r['t_pval'] < 0.05 else "n.s."))

    w_str = ""
    if r['w_pval'] is not None:
        sig_w = "***" if r['w_pval'] < 0.001 else (
            "**" if r['w_pval'] < 0.01 else (
                "*" if r['w_pval'] < 0.05 else "n.s."))
        w_str = f"  Wilcoxon: W={r['w_stat']:.1f}, p={r['w_pval']:.4f} {sig_w}"

    print(f"\n  {r['label']} (n={r['n']})")
    print(f"    OA mean {metric_name}={r['oa_mean']:.4f}  "
          f"H mean {metric_name}={r['h_mean']:.4f}  "
          f"diff={r['mean_diff']:+.4f} +/- {r['std_diff']:.4f}")
    print(f"    OA wins: {r['oa_wins']}/{r['n']} ({r['win_rate']:.0%})  "
          f"H wins: {r['h_wins']}  ties: {r['ties']}")
    print(f"    Paired t: t={r['t_stat']:.3f}, p={r['t_pval']:.4f} {sig_t}")
    if w_str:
        print(f"  {w_str}")
    print(f"    Cohen's d={r['cohens_d']:.3f} ({r['effect_size']})")


def generate_latex_table(all_results, output_path):
    """Generate a LaTeX significance table."""
    lines = [
        "% Auto-generated statistical significance table",
        r"\begin{table}[htbp]",
        r"\centering",
        r"\small",
        r"\caption{Statistical significance of our approach vs.\ heuristic "
        r"baseline (F1 Macro). Paired t-test and Wilcoxon signed-rank test "
        r"on matched experiment pairs. "
        r"$^{***}p<0.001$, $^{**}p<0.01$, $^{*}p<0.05$.}",
        r"\label{tab:significance}",
        r"\begin{tabular}{l r r r r r r}",
        r"\toprule",
        r"Group & $n$ & $\Delta$F1 & Win\% & $t$ & $p$ (t-test)"
        r" & Cohen's $d$ \\",
        r"\midrule",
    ]

    for r in all_results:
        if 'note' in r:
            lines.append(f"{r['label']} & {r['n']} & "
                         f"\\multicolumn{{5}}{{c}}{{insufficient data}} \\\\")
            continue

        sig = "^{***}" if r['t_pval'] < 0.001 else (
            "^{**}" if r['t_pval'] < 0.01 else (
                "^{*}" if r['t_pval'] < 0.05 else ""))

        pval_str = f"${r['t_pval']:.4f}{sig}$"
        if r['t_pval'] < 0.0001:
            pval_str = f"$<0.0001{sig}$"

        lines.append(
            f"{r['label']} & {r['n']} & "
            f"${r['mean_diff']:+.4f}$ & "
            f"{r['win_rate']:.0%} & "
            f"${r['t_stat']:.2f}$ & "
            f"{pval_str} & "
            f"${r['cohens_d']:.3f}$ \\\\")

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])

    Path(output_path).write_text("\n".join(lines), encoding='utf-8')
    print(f"\nWrote LaTeX table: {output_path}")


def save_csv(all_results, pairs, output_dir):
    """Save detailed results to CSV."""
    import csv

    # Summary CSV
    summary_path = Path(output_dir) / 'significance_summary.csv'
    with open(summary_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'label', 'n', 'oa_mean', 'h_mean', 'mean_diff', 'std_diff',
            'oa_wins', 'h_wins', 'ties', 'win_rate',
            't_stat', 't_pval', 'w_stat', 'w_pval',
            'cohens_d', 'effect_size',
        ])
        writer.writeheader()
        for r in all_results:
            if 'note' not in r:
                writer.writerow(r)
    print(f"Wrote: {summary_path}")

    # Paired data CSV
    pairs_path = Path(output_dir) / 'matched_pairs.csv'
    with open(pairs_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'dataset', 'scenario', 'constraint_tag', 'model', 'slice',
            'oa_f1', 'h_f1', 'f1_diff',
            'oa_accuracy', 'h_accuracy', 'acc_diff',
            'oa_adj', 'h_adj',
        ])
        writer.writeheader()
        for p in pairs:
            writer.writerow({
                'dataset': p['dataset'],
                'scenario': p['scenario'],
                'constraint_tag': p['constraint_tag'],
                'model': p['model'],
                'slice': p['slice'],
                'oa_f1': p['oa_f1'],
                'h_f1': p['h_f1'],
                'f1_diff': p['oa_f1'] - p['h_f1'],
                'oa_accuracy': p['oa_accuracy'],
                'h_accuracy': p['h_accuracy'],
                'acc_diff': p['oa_accuracy'] - p['h_accuracy'],
                'oa_adj': p['oa_adj'],
                'h_adj': p['h_adj'],
            })
    print(f"Wrote: {pairs_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Statistical significance analysis')
    parser.add_argument('--results-dir', default='results/pending_runs')
    parser.add_argument('--output-dir', default='analysis/output')
    args = parser.parse_args()

    pairs = load_matched_pairs(args.results_dir)
    print(f"Loaded {len(pairs)} matched pairs")

    if not pairs:
        print("ERROR: No matched pairs found.")
        return

    all_results = []

    # ── Overall ──
    print("\n" + "=" * 70)
    print("OVERALL (all datasets, scenarios, constraints, models)")
    print("=" * 70)
    r = run_significance_tests(
        [p['oa_f1'] for p in pairs],
        [p['h_f1'] for p in pairs],
        label="Overall")
    print_result(r)
    all_results.append(r)

    # ── By dataset ──
    print("\n" + "=" * 70)
    print("BY DATASET")
    print("=" * 70)
    datasets = sorted(set(p['dataset'] for p in pairs))
    for ds in datasets:
        sub = [p for p in pairs if p['dataset'] == ds]
        r = run_significance_tests(
            [p['oa_f1'] for p in sub],
            [p['h_f1'] for p in sub],
            label=ds)
        print_result(r)
        all_results.append(r)

    # ── By scenario ──
    print("\n" + "=" * 70)
    print("BY SCENARIO")
    print("=" * 70)
    scenarios = sorted(set((p['dataset'], p['scenario']) for p in pairs))
    for ds, sc in scenarios:
        sub = [p for p in pairs
               if p['dataset'] == ds and p['scenario'] == sc]
        r = run_significance_tests(
            [p['oa_f1'] for p in sub],
            [p['h_f1'] for p in sub],
            label=f"{ds}/{sc}")
        print_result(r)
        all_results.append(r)

    # ── By constraint tag ──
    print("\n" + "=" * 70)
    print("BY CONSTRAINT TAG")
    print("=" * 70)
    tags = sorted(set(p['constraint_tag'] for p in pairs))
    for tag in tags:
        sub = [p for p in pairs if p['constraint_tag'] == tag]
        r = run_significance_tests(
            [p['oa_f1'] for p in sub],
            [p['h_f1'] for p in sub],
            label=tag)
        print_result(r)
        all_results.append(r)

    # ── By model ──
    print("\n" + "=" * 70)
    print("BY MODEL")
    print("=" * 70)
    models = sorted(set(p['model'] for p in pairs))
    for model in models:
        sub = [p for p in pairs if p['model'] == model]
        r = run_significance_tests(
            [p['oa_f1'] for p in sub],
            [p['h_f1'] for p in sub],
            label=model)
        print_result(r)
        all_results.append(r)

    # ── By constraint tag × model (detailed) ──
    print("\n" + "=" * 70)
    print("BY CONSTRAINT TAG x MODEL")
    print("=" * 70)
    for tag in tags:
        for model in models:
            sub = [p for p in pairs
                   if p['constraint_tag'] == tag and p['model'] == model]
            if len(sub) < 3:
                continue
            r = run_significance_tests(
                [p['oa_f1'] for p in sub],
                [p['h_f1'] for p in sub],
                label=f"{tag}/{model}")
            print_result(r)
            all_results.append(r)

    # ── Accuracy analysis (secondary metric) ──
    print("\n" + "=" * 70)
    print("ACCURACY (overall)")
    print("=" * 70)
    r_acc = run_significance_tests(
        [p['oa_accuracy'] for p in pairs],
        [p['h_accuracy'] for p in pairs],
        label="Overall (Accuracy)")
    print_result(r_acc, metric_name="Acc")

    # ── Save outputs ──
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_csv(all_results, pairs, output_dir)
    generate_latex_table(
        [r for r in all_results if r.get('n', 0) >= 3
         and 'note' not in r and '/' not in r.get('label', '')
         or r.get('label') == 'Overall'],
        output_dir / 'significance_table.tex')


if __name__ == '__main__':
    main()
