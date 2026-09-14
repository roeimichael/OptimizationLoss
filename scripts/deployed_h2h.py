"""Receipt-backed exploratory comparisons of complete fresh campaigns."""

import argparse
import json
import os
from pathlib import Path
import sys
import collections
import numpy as np
from scipy.stats import t
from src.pipeline.campaign import validate_receipts, safe_path
from src.training.metrics import compute_metrics
from src.methodologies.heuristic.train import verify_allocation

CURRENT_RECIPE = {"constraint_fp32": True, "constraint_grad_mode": "normalize"}


def on_recipe(cfg):
    hp = cfg.get("hyperparams") or {}
    if int(hp.get("constraint_epochs") or 0) <= 0:
        return True
    return (
        hp.get("constraint_fp32") is CURRENT_RECIPE["constraint_fp32"]
        and hp.get("constraint_grad_mode") == CURRENT_RECIPE["constraint_grad_mode"]
    )


def capped_classes(cfg):
    cls = (cfg.get("dataset_config") or {}).get("constrained_class")
    if isinstance(cls, int):
        cls = [cls]
    return tuple(sorted(cls or []))


def read_run(run_dir):
    fin = os.path.join(run_dir, "final_predictions.csv")
    cj = os.path.join(run_dir, "config.json")
    if not (os.path.exists(fin) and os.path.exists(cj)):
        return None
    try:
        cfg = json.load(open(cj))
    except Exception:
        return None
    if not on_recipe(cfg):
        return None
    classes = capped_classes(cfg)
    if not classes:
        return None
    import pandas as pd

    df = pd.read_csv(fin)
    if "Predicted_Label" not in df.columns or "True_Label" not in df.columns:
        return None
    (p, y) = (df["Predicted_Label"], df["True_Label"])

    def counts(c):
        return dict(
            TP=int(((p == c) & (y == c)).sum()),
            K=int((p == c).sum()),
            n=int((y == c).sum()),
        )

    per = dict(((c, counts(c)) for c in classes))
    allc = range(cfg['dataset_config']['num_classes'])
    all_per = dict(((int(c), counts(c)) for c in allc))
    return dict(
        cfg=cfg,
        classes=classes,
        per=per,
        all_per=all_per,
        TP=sum((per[c]["TP"] for c in classes)),
    )


def ccf1(per, classes):
    vals = []
    for c in classes:
        d = per[c]
        den = d["K"] + d["n"]
        vals.append(2.0 * d["TP"] / den if den else 0.0)
    return sum(vals) / len(vals) if vals else float("nan")


def macrof1(all_per):
    vals = []
    for c in sorted(all_per):
        d = all_per[c]
        den = d["K"] + d["n"]
        vals.append(2.0 * d["TP"] / den if den else 0.0)
    return sum(vals) / len(vals) if vals else float("nan")


def paired_difference(tralo, other):
    seeds = sorted(set(tralo) & set(other))
    deltas = np.array([tralo[s] - other[s] for s in seeds])
    n = len(deltas)
    mean = float(deltas.mean()) if n else None
    sd = float(deltas.std(ddof=1)) if n > 1 else None
    ci = None
    if sd is not None and sd > 0:
        half = float(t.ppf(.975, n-1) * sd / np.sqrt(n))
        ci = [mean-half, mean+half]
    return dict(seeds=seeds, deltas=deltas.tolist(), n=n, mean=mean, ci95=ci,
                pilot=n < 4, missing_pairs=sorted(set(tralo) ^ set(other)),
                interval_limitation=('n<2' if n < 2 else 'zero empirical variance') if ci is None else None)


def reject_duplicate_observations(records):
    seen = set()
    for record in records:
        key = tuple(record[k] for k in ('dataset', 'backbone', 'cap', 'arm', 'seed'))
        if key in seen:
            raise ValueError('duplicate observation: %s' % (key,))
        seen.add(key)


def markdown_report(records):
    reject_duplicate_observations(records)
    cells = collections.defaultdict(list)
    for record in records:
        cells[record['dataset'], record['backbone'], record['cap']].append(record)
    lines = ['Best observed quality means are bolded (all exact ties), not significance.',
             'Exploratory seed-paired effects conditional on the fixed inspected dataset and recipe.',
             'Small-n approximate normality is unverified; marginal 95% Student-t intervals are not multiplicity-adjusted.',
             'Compute/event-dose evidence remains unknown until the shared runtime logging gates pass.']
    lines.append('tralo_null is the phase-matched zero-constraint control. The reference budget is 30 task epochs; clip/focal_clip use continuous warm-up, while trained arms reseed and create a fresh Adam optimizer after warm-up. Wall-clock compute and optimizer/RNG trajectories are not equal by this convention.')
    quality = ['cc_f1', 'macro_f1', 'constrained_precision', 'constrained_recall', 'collateral_f1']
    for cell, runs in sorted(cells.items()):
        arms = collections.defaultdict(list)
        for row in runs:
            arms[row['arm']].append(row)
        means = {arm: {k: float(np.mean([r[k] for r in rows])) if all(r[k] is not None for r in rows) else None
                        for k in quality} for arm, rows in arms.items()}
        best = {k: max((m[k] for m in means.values() if m[k] is not None), default=None) for k in quality}
        lines.extend(['', ' / '.join(cell), '',
                      '| Arm | n | cc-F1 | Macro-F1 | Constrained P | Constrained R | Collateral F1 | Collateral support | Feasible |',
                      '|---|---:|---:|---:|---:|---:|---:|---:|---:|'])
        for arm, rows in sorted(arms.items()):
            values = []
            for key in quality:
                value = means[arm][key]
                printed = 'NA' if value is None else '%.4f' % value
                if value is not None and value == best[key]:
                    printed = '**%s**' % printed
                if len(rows) > 1 and value is not None:
                    printed += ' ± %.4f' % np.std([r[key] for r in rows], ddof=1)
                values.append(printed)
            lines.append('| %s | %d | %s | %s | %d/%d |' %
                         (arm, len(rows), ' | '.join(values),
                          ','.join(str(r['collateral_support']) for r in sorted(rows, key=lambda r:r['seed'])),
                          sum(r['feasible'] for r in rows), len(rows)))
        if 'tralo' in arms:
            tralo = {r['seed']: r['cc_f1'] for r in arms['tralo']}
            for arm in sorted(set(arms)-{'tralo'}):
                effect = paired_difference(tralo, {r['seed']: r['cc_f1'] for r in arms[arm]})
                lines.append('TraLO - %s (primary cc-F1): %s' % (arm, json.dumps(effect)))
            other = [means[a]['cc_f1'] for a in arms if a != 'tralo']
            delta = [means['tralo']['cc_f1'] - v for v in other]
            lines.append('Observed TraLO cc-F1 mean is higher than %d, tied with %d, and lower than %d declared comparators; no superiority verdict.' %
                         (sum(d > 0 for d in delta), sum(d == 0 for d in delta), sum(d < 0 for d in delta)))
    return '\n'.join(lines) + '\n'


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--campaign", nargs="+", required=True)
    ap.add_argument("--json", dest="output")
    ap.add_argument('--markdown', dest='markdown_output')
    args = ap.parse_args()
    records = []
    try:
        outputs = [safe_path(p) for p in (args.output, args.markdown_output) if p]
        if len(args.campaign) != 1:
            raise ValueError('exactly one complete campaign root is required; report disjoint roots separately')
        admitted = [validate_receipts(root) for root in args.campaign]
        observations = []
        for root, (_, inventory) in zip(args.campaign, admitted):
            for rel in inventory['completed']:
                cfg = json.loads((safe_path(root)/rel).read_text(encoding='utf-8'))
                observations.append(dict(dataset=cfg['dataset_mode'], backbone=cfg['model_name'],
                                         cap=cfg['constraint_tag'], arm=cfg['arm'],
                                         seed=cfg['hyperparams']['seed']))
        reject_duplicate_observations(observations)  # Before reading/scoring any prediction CSV.
        for root, (manifest, inventory) in zip(args.campaign, admitted):
            for rel in inventory['completed']:
                p = (safe_path(root)/rel).with_name('final_predictions.csv')
                cfg = json.loads((p.parent / "config.json").read_text(encoding="utf-8"))
                if cfg.get("status") != "completed":
                    raise ValueError("run is not completed: " + str(p.parent))
                rec = read_run(str(p.parent))
                if rec is None:
                    raise ValueError("invalid or off-recipe run: " + str(p.parent))
                import pandas as pd
                frame = pd.read_csv(p, float_precision='round_trip')
                data = manifest['data'][manifest['runs'][rel]['data_id']]
                if len(frame) != data['test_rows']:
                    raise ValueError('prediction rows differ from frozen data')
                n_classes = cfg['dataset_config']['num_classes']
                probs = frame[['Prob_Class_%d' % c for c in range(n_classes)]].to_numpy()
                if not np.isfinite(probs).all():
                    raise ValueError('non-finite probabilities')
                metrics = compute_metrics(frame.True_Label.to_numpy(), frame.Predicted_Label.to_numpy(),
                                          probs, constrained_classes=rec['classes'])
                quotas = data['quotas']
                violations = verify_allocation(frame.Predicted_Label.to_numpy(), frame.Group_ID.to_numpy(),
                                               quotas['global'], {int(k):v for k,v in quotas['local'].items()}, n_classes)
                records.append(
                    dict(
                        root=str(Path(root).resolve()),
                        dataset=cfg["dataset_mode"],
                        backbone=cfg["model_name"],
                        cap=cfg["constraint_tag"],
                        arm=cfg["arm"],
                        seed=cfg["hyperparams"]["seed"],
                        cc_f1=ccf1(rec["per"], rec["classes"]),
                        macro_f1=macrof1(rec["all_per"]),
                        constrained_precision=metrics['constrained_precision'],
                        constrained_recall=metrics['constrained_recall'],
                        collateral_f1=metrics['collateral_f1'],
                        collateral_support=metrics['collateral_support'],
                        feasible=not violations, violations=violations,
                        per_class=rec["all_per"],
                    )
                )
        if not records:
            raise ValueError("no completed runs")
        protected = {'config.json', 'campaign_plan.json', 'campaign_manifest.json',
                     'completion_receipt.json', 'final_predictions.csv',
                     'final_predictions_raw.csv', 'evaluation_metrics.csv'}
        if any(p.name in protected for p in outputs):
            raise ValueError('report output would overwrite campaign evidence')
    except (OSError, ValueError, TypeError, KeyError) as exc:
        print("FAIL deployed_h2h: %s" % exc)
        return 1
    text = json.dumps(records, indent=2)
    if args.output:
        Path(args.output).write_text(text + "\n", encoding="utf-8")
    markdown = markdown_report(records)
    if args.markdown_output:
        Path(args.markdown_output).write_text(markdown, encoding='utf-8')
    print(markdown)
    return 0


if __name__ == "__main__":
    sys.exit(main())
