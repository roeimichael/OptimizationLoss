"""Partial-correlation universal test.

Within each dataset, residualize d_macro by dataset MEAN (i.e., subtract
the per-dataset mean d_macro), residualize each feature the same way,
then correlate the residuals. If the correlation survives this control,
the feature predicts TraLO winning ABOVE AND BEYOND what dataset identity
alone tells you. That's a true universal signal.

We also test partial-correlation controlling for (dataset, tightness)
jointly — a stricter test.

Inputs: paper/HANDOFF/tables/deep_paired_vs_<baseline>.csv
"""
import csv
from pathlib import Path
import numpy as np

TBL = Path("paper/HANDOFF/tables")
FEATS = [
    "borderline_010","borderline_005","high_conf_090","high_conf_099",
    "entropy_mean","entropy_std","soft_count_cstr","hard_count_cstr",
    "soft_minus_hard_cstr","binding_ratio","soft_binding_ratio",
    "cstr_prob_mean","cstr_prob_std","cstr_prob_q90","uncstr_prob_mean",
    "pred_balance_entropy",
]
TARGETS = ["d_macro","d_f1c","d_f1u"]


def load(baseline):
    out = []
    with open(TBL / f"deep_paired_vs_{baseline}.csv") as f:
        for r in csv.DictReader(f):
            d = dict(r)
            for k in FEATS + TARGETS + ["tight_pct"]:
                try: d[k] = float(d[k])
                except (ValueError, KeyError): d[k] = np.nan
            out.append(d)
    return out


def residualize_by_group(rows, key_field, value_field):
    """Subtract group mean from value. Returns list aligned with rows."""
    groups = {}
    for r in rows:
        groups.setdefault(r[key_field], []).append(r[value_field])
    means = {g: np.nanmean(vs) for g, vs in groups.items()}
    return np.array([r[value_field] - means[r[key_field]] for r in rows])


def residualize_by_two(rows, k1, k2, vf):
    groups = {}
    for r in rows:
        key = (r[k1], r[k2])
        groups.setdefault(key, []).append(r[vf])
    means = {k: np.nanmean(vs) for k, vs in groups.items()}
    return np.array([r[vf] - means[(r[k1], r[k2])] for r in rows])


def corr(x, y):
    m = ~(np.isnan(x) | np.isnan(y))
    x, y = x[m], y[m]
    if len(x) < 5 or np.std(x) == 0 or np.std(y) == 0: return np.nan, len(x)
    return float(np.corrcoef(x, y)[0,1]), len(x)


def report(baseline):
    rows = load(baseline)
    print(f"\n{'='*78}")
    print(f"PARTIAL-CORRELATION TEST  TraLO vs {baseline}  (n={len(rows)})")
    print(f"{'='*78}")
    print("Each |r| below is the within-dataset signal AFTER removing dataset baseline.")
    print("A non-trivial |r| here = feature predicts winning ABOVE dataset identity.")
    print()
    print(f"  {'feature':<25}{'r|ds':>9}{'r|ds+tight':>14}  for d_macro")
    print(f"  {'-'*60}")
    y = residualize_by_group(rows, "dataset", "d_macro")
    y2 = residualize_by_two(rows, "dataset", "tight_pct", "d_macro")
    ranked = []
    for f in FEATS:
        x  = residualize_by_group(rows, "dataset", f)
        x2 = residualize_by_two(rows, "dataset", "tight_pct", f)
        r1, n1 = corr(x, y)
        r2, n2 = corr(x2, y2)
        ranked.append((f, r1, r2, n1))
    # sort by |r|ds+tight|
    ranked.sort(key=lambda t: abs(t[2]) if not np.isnan(t[2]) else 0,
                reverse=True)
    for f, r1, r2, n in ranked:
        print(f"  {f:<25}{r1:>+9.3f}{r2:>+14.3f}    (n={n})")

    # Per-dataset directional consistency: does sign of within-dataset r match
    # the sign of residual r?
    print(f"\n  Sign consistency across 3 datasets vs d_macro:")
    print(f"  {'feature':<25}{'tissue':>8}{'derm':>8}{'aider':>8}{'all3same':>10}")
    for f, r1, r2, n in ranked[:8]:
        signs = []
        for ds in ("tissuemnist","dermmnist","aider"):
            sub = [r for r in rows if r["dataset"] == ds]
            if len(sub) < 20: signs.append(np.nan); continue
            xs = np.array([r[f] for r in sub])
            ys = np.array([r["d_macro"] for r in sub])
            rc, _ = corr(xs, ys)
            signs.append(rc)
        same = (not any(np.isnan(s) for s in signs)
                and len({np.sign(s) for s in signs}) == 1)
        cells = "  ".join(f"{s:>+6.2f}" if not np.isnan(s) else "  nan  "
                          for s in signs)
        print(f"  {f:<25}  {cells}    {str(same):>8}")


def main():
    for b in ("fioretto_ldf","hounie_rcl","danits_lp"):
        report(b)


if __name__ == "__main__":
    main()
