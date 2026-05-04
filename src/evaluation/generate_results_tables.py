"""Generate LaTeX results tables from completed experiments.

Reads all completed config.json files, aggregates over slices (mean ± std),
and produces publication-ready LaTeX tables for the thesis paper.

Output: paper/results_tables.tex

Usage:
    python -m analysis.generate_results_tables [--results-dir results/pending_runs]
"""

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

# Display names
SCENARIO_DISPLAY = {
    'single_MEL': 'Single-class: MEL',
    'multi_MEL_BCC': 'Multi-class: MEL + BCC',
    'single_GE': 'Single-class: GE',
    'multi_GE_PTC': 'Multi-class: GE + PTC',
}

MODEL_SHORT = {
    'MobileNetV3': 'MV3',
    'EfficientNetB0': 'EffB0',
    'ConvNeXtTiny': 'CNxT',
}

MODEL_ORDER = ['MobileNetV3', 'EfficientNetB0', 'ConvNeXtTiny']

CONSTRAINT_SORT_KEY = {
    'L20_G50': (0.2, 0.5),
    'L20_G80': (0.2, 0.8),
    'L30_G30': (0.3, 0.3),
    'L50_G50': (0.5, 0.5),
    'L80_G30': (0.8, 0.3),
    'L80_G80': (0.8, 0.8),
}


def constraint_display(tag):
    """L20_G50 -> (0.2, 0.5)"""
    parts = tag.split('_')
    local_pct = int(parts[0][1:]) / 100
    global_pct = int(parts[1][1:]) / 100
    return f"({local_pct}, {global_pct})"


def load_all_completed(results_dir):
    """Load all completed experiments into a nested dict structure."""
    experiments = defaultdict(lambda: defaultdict(lambda: defaultdict(
        lambda: defaultdict(list))))

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

        key = f"{dataset}/{scenario}"
        experiments[key][tag][(model, method)]['accuracy'].append(
            results.get('accuracy', 0))
        experiments[key][tag][(model, method)]['f1_macro'].append(
            results.get('f1_macro', 0))
        experiments[key][tag][(model, method)]['precision_macro'].append(
            results.get('precision_macro', 0))
        experiments[key][tag][(model, method)]['recall_macro'].append(
            results.get('recall_macro', 0))
        experiments[key][tag][(model, method)]['samples_adjusted'].append(
            results.get('samples_adjusted', 0))
        experiments[key][tag][(model, method)]['training_time'].append(
            results.get('training_time', 0))

    return experiments


def fmt_mean_std(values):
    """Format as 0.XXX±0.XXX."""
    if not values:
        return '---'
    m = statistics.mean(values)
    s = statistics.stdev(values) if len(values) > 1 else 0
    return f"{m:.3f}" + r"{\scriptsize$\pm$" + f"{s:.3f}" + "}"


def make_table(scenario_key, experiments, metric, metric_display, label_prefix):
    """Generate one LaTeX table for a given scenario and metric."""
    scenario_display = SCENARIO_DISPLAY.get(
        scenario_key.split('/')[-1], scenario_key)
    dataset = scenario_key.split('/')[0]

    tags_data = experiments[scenario_key]
    if not tags_data:
        return ''

    # Sort constraint tags
    sorted_tags = sorted(tags_data.keys(),
                         key=lambda t: CONSTRAINT_SORT_KEY.get(t, (0, 0)))

    # Check which models have data
    all_models = []
    for tag in sorted_tags:
        for (model, method) in tags_data[tag]:
            if model not in all_models:
                all_models.append(model)
    models = [m for m in MODEL_ORDER if m in all_models]

    n_models = len(models)
    col_spec = f"l *{{{n_models}}}{{cc}}"

    lines = []
    safe_scenario = scenario_key.replace('/', '_').replace(' ', '_')
    safe_metric = metric.replace('_', '')
    table_label = f"tab:{label_prefix}_{safe_scenario}"

    lines.append(r"\begin{table}[htbp]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(
        f"\\caption{{{metric_display} for {scenario_display} — {dataset} "
        f"(mean{{\\scriptsize$\\pm$}}std over 5 slices, "
        f"\\textbf{{bold}} = winner per pair).}}")
    lines.append(f"\\label{{{table_label}}}")
    lines.append(f"\\begin{{tabular}}{{{col_spec}}}")
    lines.append(r"\toprule")

    # Header row
    header = r"Constraint $(\alpha_l, \alpha_g)$"
    for i, m in enumerate(models):
        header += f" & \\multicolumn{{2}}{{c}}{{{MODEL_SHORT.get(m, m)}}}"
    header += r" \\"
    lines.append(header)

    # Cmidrules
    cmidrules = ""
    for i, m in enumerate(models):
        col_start = 2 + i * 2
        col_end = col_start + 1
        cmidrules += f"\\cmidrule(lr){{{col_start}-{col_end}}} "
    lines.append(cmidrules)

    # Sub-header
    sub = ""
    for m in models:
        sub += " & Ours & Heur."
    sub += r" \\"
    lines.append(sub)
    lines.append(r"\midrule")

    # Data rows
    for tag in sorted_tags:
        row = constraint_display(tag)
        for model in models:
            ours_vals = tags_data[tag].get((model, 'tralo'), {}).get(
                metric, [])
            heur_vals = tags_data[tag].get((model, 'heuristic'), {}).get(
                metric, [])

            ours_mean = statistics.mean(ours_vals) if ours_vals else None
            heur_mean = statistics.mean(heur_vals) if heur_vals else None

            ours_str = fmt_mean_std(ours_vals) if ours_vals else '---'
            heur_str = fmt_mean_std(heur_vals) if heur_vals else '---'

            # Bold the winner
            if ours_mean is not None and heur_mean is not None:
                if ours_mean > heur_mean + 1e-6:
                    ours_str = r"\textbf{" + ours_str + "}"
                elif heur_mean > ours_mean + 1e-6:
                    heur_str = r"\textbf{" + heur_str + "}"
                else:
                    # Tie — bold both
                    ours_str = r"\textbf{" + ours_str + "}"
                    heur_str = r"\textbf{" + heur_str + "}"

            row += f" & {ours_str} & {heur_str}"
        row += r" \\"
        lines.append(row)

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    lines.append("")

    return "\n".join(lines)


def make_adjustment_table(experiments):
    """Generate a table showing mean post-hoc adjustments by constraint and model."""
    lines = []
    lines.append(r"% === Post-hoc Adjustment Summary ===")
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(
        r"\caption{Mean post-hoc adjustments (our approach) by constraint "
        r"tightness and model, averaged across scenarios and slices.}")
    lines.append(r"\label{tab:posthoc_adjustments}")

    models = MODEL_ORDER
    lines.append(r"\begin{tabular}{l" + " c" * len(models) + "}")
    lines.append(r"\toprule")

    header = r"Constraint $(\alpha_l, \alpha_g)$"
    for m in models:
        header += f" & {MODEL_SHORT.get(m, m)}"
    header += r" \\"
    lines.append(header)
    lines.append(r"\midrule")

    # Aggregate across all scenarios
    adj_by_tag_model = defaultdict(lambda: defaultdict(list))
    for scenario_key, tags_data in experiments.items():
        for tag, model_data in tags_data.items():
            for (model, method), metrics in model_data.items():
                if method == 'tralo':
                    adj_by_tag_model[tag][model].extend(
                        metrics.get('samples_adjusted', []))

    all_tags = sorted(adj_by_tag_model.keys(),
                      key=lambda t: CONSTRAINT_SORT_KEY.get(t, (0, 0)))

    for tag in all_tags:
        row = constraint_display(tag)
        for model in models:
            vals = adj_by_tag_model[tag].get(model, [])
            if vals:
                row += f" & {statistics.mean(vals):.1f}"
            else:
                row += " & ---"
        row += r" \\"
        lines.append(row)

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    lines.append("")
    return "\n".join(lines)


def make_summary_table(experiments):
    """Generate an overall summary table: mean F1 by methodology across all experiments."""
    lines = []
    lines.append(r"% === Overall Summary ===")
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(
        r"\caption{Overall performance summary by methodology and dataset "
        r"(mean over all constraint configurations, models, and slices).}")
    lines.append(r"\label{tab:overall_summary}")
    lines.append(r"\begin{tabular}{l l c c c c}")
    lines.append(r"\toprule")
    lines.append(
        r"Dataset & Method & Accuracy & F1 Macro & Precision & Recall \\")
    lines.append(r"\midrule")

    by_dataset_method = defaultdict(lambda: defaultdict(
        lambda: defaultdict(list)))
    for scenario_key, tags_data in experiments.items():
        dataset = scenario_key.split('/')[0]
        for tag, model_data in tags_data.items():
            for (model, method), metrics in model_data.items():
                for metric_name in ['accuracy', 'f1_macro',
                                    'precision_macro', 'recall_macro']:
                    by_dataset_method[dataset][method][metric_name].extend(
                        metrics.get(metric_name, []))

    for dataset in sorted(by_dataset_method.keys()):
        for method in ['tralo', 'heuristic']:
            m = by_dataset_method[dataset][method]
            if not m['accuracy']:
                continue
            display_method = 'Ours' if method == 'tralo' else 'Heuristic'
            acc = statistics.mean(m['accuracy'])
            f1 = statistics.mean(m['f1_macro'])
            prec = statistics.mean(m['precision_macro'])
            rec = statistics.mean(m['recall_macro'])

            best_f1 = max(
                statistics.mean(
                    by_dataset_method[dataset][mm].get('f1_macro', [0]))
                for mm in ['tralo', 'heuristic']
                if by_dataset_method[dataset][mm].get('f1_macro'))

            f1_str = f"{f1:.3f}"
            if abs(f1 - best_f1) < 1e-6:
                f1_str = r"\textbf{" + f1_str + "}"

            lines.append(
                f"{dataset} & {display_method} & {acc:.3f} & {f1_str} "
                f"& {prec:.3f} & {rec:.3f} \\\\")
        lines.append(r"\midrule")

    # Remove last midrule, replace with bottomrule
    lines[-1] = r"\bottomrule"
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    lines.append("")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description='Generate LaTeX results tables from completed experiments')
    parser.add_argument('--results-dir', default='results/pending_runs',
                        help='Root results directory')
    parser.add_argument('--output', default='paper/results_tables.tex',
                        help='Output LaTeX file')
    args = parser.parse_args()

    experiments = load_all_completed(args.results_dir)

    if not experiments:
        print("ERROR: No completed experiments found.")
        return

    # Report what we found
    total = 0
    for sk, tags in experiments.items():
        n = sum(len(v) for mm in tags.values() for v in mm.values())
        print(f"  {sk}: {n} metric lists")
        total += n
    print(f"  Total experiment entries: {total}")

    output_lines = [
        "% Auto-generated results tables",
        f"% Generated from {args.results_dir}",
        "% Bold = winner (our approach vs heuristic) per constraint-model pair",
        "",
    ]

    # Sort scenarios: dermmnist first, then tissuemnist
    scenario_keys = sorted(experiments.keys(),
                           key=lambda k: (0 if 'derm' in k else 1, k))

    # Per-scenario tables for accuracy and F1
    for metric, display, prefix in [
        ('accuracy', 'Accuracy', 'acc'),
        ('f1_macro', 'F1 Macro', 'f1'),
    ]:
        for sk in scenario_keys:
            table = make_table(sk, experiments, metric, display, prefix)
            if table:
                output_lines.append(
                    f"% === {sk} - {display} ===")
                output_lines.append(table)

    # Summary table
    output_lines.append(make_summary_table(experiments))

    # Post-hoc adjustment table
    output_lines.append(make_adjustment_table(experiments))

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(output_lines), encoding='utf-8')
    print(f"\nWrote {output_path} ({len(output_lines)} lines)")


if __name__ == '__main__':
    main()
