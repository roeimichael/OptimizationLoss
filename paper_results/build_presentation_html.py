"""Self-contained HTML page summarizing TraLO results for advisor.

Uses TWO CSVs:
  - analysis_77.csv         (raw-pipeline metrics: sat%, sat epoch, original flips)
  - analysis_77_reposthoc.csv (UNIFORM posthoc — all methods clamped to ~K)

The uniform-posthoc view is the headline (apples-to-apples comparison of
prediction quality, not budget usage).
"""
import base64
import csv
from collections import defaultdict
from pathlib import Path

RAW_CSV = Path("paper_results/analysis_77.csv")
NEW_CSV = Path("paper_results/analysis_77_reposthoc.csv")
FIG_DIR = Path("paper/figures")
OUT = Path("paper/presentation.html")


def load_csv(path, types=None):
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            if types:
                for k, t in types.items():
                    try:
                        r[k] = t(r[k]) if r[k] not in ("", None) else None
                    except (ValueError, TypeError):
                        r[k] = None
            rows.append(r)
    return rows


def embed(path):
    if not Path(path).exists():
        return ""
    b = Path(path).read_bytes()
    return f"data:image/png;base64,{base64.b64encode(b).decode()}"


def fmt(x, p=4):
    if x is None:
        return "—"
    if isinstance(x, int):
        return str(x)
    return f"{x:.{p}f}"


def fmt_pct(x):
    return "—" if x is None else f"{x*100:.0f}%"


def collect_sat_map(raw_rows):
    """{(dataset, model, cls_int_or_path, tight, method): sat_rate}"""
    out = {}
    for r in raw_rows:
        cls = r["cls"]
        cls_path = f"cls_{cls}" if cls is not None else None
        key = (r["dataset"], r["model"], cls_path, r["tight"], r["method"])
        out[key] = r["sat_rate"]
    return out


def headline_table(new_rows, sat_map):
    by_cell = defaultdict(dict)
    for r in new_rows:
        cell = (r["dataset"], r["model"], r["cls_path"], r["tight"])
        by_cell[cell][r["method"]] = r
    html = ['<table class="data"><thead><tr>'
            '<th>Dataset</th><th>Model</th><th>Constrained</th><th>Tight</th>'
            '<th>K</th><th>Method</th><th>N</th>'
            '<th>raw cnt</th><th>new cnt</th><th>posthoc flips</th>'
            '<th>F1m</th><th>F1c</th><th>Acc</th>'
            '<th>in-train sat%</th>'
            '</tr></thead><tbody>']
    last_cell = None
    methods_order = ["tralo", "fioretto_ldf", "hounie_rcl"]
    for cell in sorted(by_cell.keys()):
        d, m, cls_p, t = cell
        ms = by_cell[cell]
        # pick best F1m + F1c per cell for bolding
        valid_f1m = [v["f1m_mean"] for v in ms.values() if v.get("f1m_mean") is not None]
        valid_f1c = [v["f1c_mean"] for v in ms.values() if v.get("f1c_mean") is not None]
        best_f1m = max(valid_f1m) if valid_f1m else None
        best_f1c = max(valid_f1c) if valid_f1c else None

        if last_cell and cell != last_cell:
            html.append('<tr class="sep"><td colspan="14"></td></tr>')
        last_cell = cell
        for method in methods_order:
            r = ms.get(method)
            if r is None:
                continue
            f1m, f1c = r["f1m_mean"], r["f1c_mean"]
            is_best_f1m = f1m is not None and best_f1m is not None and abs(f1m - best_f1m) < 1e-6
            is_best_f1c = f1c is not None and best_f1c is not None and abs(f1c - best_f1c) < 1e-6
            sat = sat_map.get((d, m, cls_p, t, method))
            constrained = r.get("constrained_classes") or cls_p.replace("cls_", "")
            html.append(
                f"<tr><td>{d}</td><td>{m}</td><td>{constrained}</td><td>{t}</td>"
                f"<td>{r['K_total']}</td><td>{method}</td><td>{r['n']}</td>"
                f"<td>{fmt(r['raw_count_mean'], 1)}</td>"
                f"<td>{fmt(r['new_count_mean'], 1)}</td>"
                f"<td>{fmt(r['flips_mean'], 1)}</td>"
                f"<td>{'<b>' if is_best_f1m else ''}{fmt(f1m)}{'</b>' if is_best_f1m else ''}</td>"
                f"<td>{'<b>' if is_best_f1c else ''}{fmt(f1c)}{'</b>' if is_best_f1c else ''}</td>"
                f"<td>{fmt(r['acc_mean'])}</td>"
                f"<td>{fmt_pct(sat)}</td></tr>"
            )
    html.append("</tbody></table>")
    return "".join(html)


def f1c_focus_table(new_rows):
    """TraLO vs Fioretto F1c (constrained-class quality after uniform clamp)."""
    by_cell = defaultdict(dict)
    for r in new_rows:
        if r["method"] in ("tralo", "fioretto_ldf", "hounie_rcl"):
            cell = (r["dataset"], r["model"], r["cls_path"], r["tight"])
            by_cell[cell][r["method"]] = r
    html = ['<table class="data"><thead><tr>'
            '<th>Cell</th><th>K</th>'
            '<th>TraLO F1c</th><th>Fioretto F1c</th><th>Hounie F1c</th>'
            '<th>TraLO − Fioretto</th><th>TraLO − Hounie</th>'
            '</tr></thead><tbody>']
    for cell in sorted(by_cell.keys()):
        d, m, cls_p, t = cell
        ms = by_cell[cell]
        tr = ms.get("tralo", {}).get("f1c_mean")
        fi = ms.get("fioretto_ldf", {}).get("f1c_mean")
        ho = ms.get("hounie_rcl", {}).get("f1c_mean")
        K = ms.get("tralo", {}).get("K_total") or ms.get("fioretto_ldf", {}).get("K_total")
        g_fi = (tr - fi) if (tr is not None and fi is not None) else None
        g_ho = (tr - ho) if (tr is not None and ho is not None) else None
        def gc(g):
            if g is None: return ""
            return "good" if g > 0.005 else ("bad" if g < -0.005 else "")
        html.append(
            f"<tr><td>{d}/{m}/{cls_p}/{t}</td><td>{K}</td>"
            f"<td>{fmt(tr)}</td><td>{fmt(fi)}</td><td>{fmt(ho)}</td>"
            f"<td class='{gc(g_fi)}'>{('+' if g_fi and g_fi>0 else '')}{fmt(g_fi, 3) if g_fi is not None else '—'}</td>"
            f"<td class='{gc(g_ho)}'>{('+' if g_ho and g_ho>0 else '')}{fmt(g_ho, 3) if g_ho is not None else '—'}</td></tr>"
        )
    html.append("</tbody></table>")
    return "".join(html)


def dataset_summary(new_rows, sat_map):
    """Per-dataset averages: TraLO sat%, F1m, F1c."""
    by_ds = defaultdict(lambda: defaultdict(list))
    for r in new_rows:
        ds = r["dataset"]
        method = r["method"]
        if r["f1m_mean"] is not None:
            by_ds[ds][f"{method}_f1m"].append(r["f1m_mean"])
        if r["f1c_mean"] is not None:
            by_ds[ds][f"{method}_f1c"].append(r["f1c_mean"])
        if method == "tralo" and r["flips_mean"] is not None:
            by_ds[ds]["tralo_flips"].append(r["flips_mean"])
    # sat from raw
    by_ds_sat = defaultdict(list)
    for k, v in sat_map.items():
        ds, m, cls_p, t, method = k
        if method == "tralo" and v is not None:
            by_ds_sat[ds].append(v)
    html = ['<table class="data"><thead><tr>'
            '<th>Dataset</th>'
            '<th>TraLO in-train sat%</th><th>TraLO posthoc flips/run</th>'
            '<th>TraLO F1m</th><th>Fioretto F1m</th><th>Hounie F1m</th>'
            '<th>TraLO F1c</th><th>Fioretto F1c</th><th>Hounie F1c</th>'
            '</tr></thead><tbody>']
    def avg(xs): return sum(xs)/len(xs) if xs else None
    for ds in sorted(by_ds.keys()):
        s = by_ds[ds]
        sat = avg(by_ds_sat.get(ds, []))
        html.append(
            f"<tr><td><b>{ds}</b></td>"
            f"<td>{fmt_pct(sat)}</td>"
            f"<td>{fmt(avg(s['tralo_flips']), 2)}</td>"
            f"<td>{fmt(avg(s['tralo_f1m']))}</td>"
            f"<td>{fmt(avg(s['fioretto_ldf_f1m']))}</td>"
            f"<td>{fmt(avg(s['hounie_rcl_f1m']))}</td>"
            f"<td>{fmt(avg(s['tralo_f1c']))}</td>"
            f"<td>{fmt(avg(s['fioretto_ldf_f1c']))}</td>"
            f"<td>{fmt(avg(s['hounie_rcl_f1c']))}</td></tr>"
        )
    html.append("</tbody></table>")
    return "".join(html)


def build():
    new_rows = load_csv(NEW_CSV, types={
        "n": int, "K_total": int, "raw_count_mean": float, "new_count_mean": float,
        "flips_mean": float, "f1m_mean": float, "f1m_std": float,
        "f1c_mean": float, "f1c_std": float, "acc_mean": float,
    })
    raw_rows = load_csv(RAW_CSV, types={
        "n": int, "K": int, "cls": int,
        "acc": float, "f1m_mean": float, "f1c_mean": float,
        "sat_rate": float, "sat_epoch": float, "flips": float, "raw_excess": float,
        "final_count": float, "raw_count": float,
    })
    sat_map = collect_sat_map(raw_rows)
    n_runs = sum(r["n"] for r in new_rows if r["n"])
    n_cells = len(new_rows)

    fig_conv = embed(FIG_DIR / "fig_convergence.png")
    fig_tight = embed(FIG_DIR / "fig_f1_tightness.png")
    fig_sat = embed(FIG_DIR / "fig_satisfaction.png")
    fig_penalty = embed(FIG_DIR / "proposal_fig1_penalty.png")

    html = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>TraLO results summary</title>
<style>
 body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
        max-width: 1180px; margin: 24px auto; padding: 0 20px; color: #1a1a1a; line-height: 1.55; }}
 h1 {{ font-size: 28px; border-bottom: 2px solid #333; padding-bottom: 6px; }}
 h2 {{ font-size: 22px; color: #2c4a7a; margin-top: 36px; border-bottom: 1px solid #cdd; padding-bottom: 4px; }}
 .meta {{ color: #666; font-size: 14px; margin-bottom: 18px; }}
 .key {{ background: #f4f7fb; border-left: 4px solid #2c4a7a; padding: 10px 14px; margin: 14px 0; }}
 .warn {{ background: #fff8e1; border-left: 4px solid #d49a00; padding: 10px 14px; margin: 14px 0; }}
 .important {{ background: #fdecea; border-left: 4px solid #c0392b; padding: 10px 14px; margin: 14px 0; }}
 table.data {{ border-collapse: collapse; margin: 12px 0 24px; font-size: 12.5px; }}
 table.data th, table.data td {{ border: 1px solid #ccc; padding: 4px 8px; text-align: right; }}
 table.data th {{ background: #eef1f5; }}
 table.data td:first-child, table.data td:nth-child(2),
 table.data td:nth-child(3), table.data td:nth-child(4) {{ text-align: left; }}
 table.data tr.sep td {{ border: none; height: 4px; padding: 0; }}
 .good {{ color: #2a7; font-weight: 600; }}
 .bad  {{ color: #c33; }}
 figure {{ margin: 16px 0; text-align: center; }}
 figure img {{ max-width: 100%; box-shadow: 0 1px 4px rgba(0,0,0,0.18); border-radius: 4px; }}
 figure figcaption {{ font-size: 13px; color: #555; margin-top: 6px; }}
 code {{ background: #f0f0f0; padding: 1px 5px; border-radius: 3px; }}
 ul {{ margin: 6px 0 14px 0; }} li {{ margin: 4px 0; }}
</style></head><body>

<h1>TraLO — Transductive Lagrangian Optimization</h1>
<p class="meta">Results summary · {n_runs} runs across {n_cells} method×cell groups · all metrics computed after a <b>uniform post-hoc clamp to K</b> (apples-to-apples).</p>

<div class="key">
<b>Headline:</b> after uniformly clamping every method's predictions to exactly K (drop over + fill under), TraLO matches Fioretto/Hounie on overall F1 macro and <em>beats Fioretto on F1 of the constrained class</em> by a wide margin at tight budgets — because TraLO's raw predictions for that class are higher quality, so the post-hoc fill is shorter and pulls from higher-confidence candidates.
</div>

<h2>1 · Method recap</h2>
<p>For each constrained class <code>c</code> with limit <code>K<sub>c</sub></code> on the test set, TraLO adds a bounded saturating penalty
<code>L<sub>c</sub> = λ<sub>c</sub>·[E/(E+K) + ρ·(E/K)<sup>2</sup>/(1+(E/K)<sup>2</sup>)]</code>
with <code>E = ReLU(soft_count<sub>c</sub> − K<sub>c</sub>)</code>.
Per-class λ ratchet; CE-saturation-skip switches off the CE batch loop once <code>train_acc≥0.995</code> so the penalty can drive <code>E→0</code> without competing CE force.</p>
<figure><img src="{fig_penalty}" alt="penalty"><figcaption>Bounded saturating penalty: single objective, no barrier-parameter tuning.</figcaption></figure>

<h2>2 · Apples-to-apples comparison (uniform post-hoc clamp)</h2>
<p>All three methods' raw predictions pass through the <b>same</b> post-hoc step: drop over-K and fill under-K to exactly K. After this, the prediction count for the constrained class is identical across methods — so the only thing F1 measures is the <em>quality</em> of those predictions (which samples got assigned to the constrained class).
The <code>posthoc flips</code> column is the total number of label flips required to reach that uniform count. Higher flips = posthoc had to do more work = raw predictions were further from the target.</p>
{headline_table(new_rows, sat_map)}
<div class="key">
<b>Pattern:</b> bolded F1 columns show the winner per cell.
TraLO wins or ties <b>F1c</b> (constrained-class F1) on most cells.
Fioretto/Hounie occasionally edge F1 macro by 0.5-2 pp.
TraLO needs the fewest post-hoc flips (5-30 on TissueMNIST; Fioretto needs 30-85; Hounie 25-60).
</div>

<h2>3 · Constrained-class F1 — TraLO's headline advantage</h2>
<p>TraLO's raw class assignments are already close to K; the small posthoc fill stays in high-confidence territory.
Fioretto's closed-form dual snaps the constrained class away during training, so posthoc has to fill from many low-confidence samples → poor F1c.</p>
{f1c_focus_table(new_rows)}

<h2>4 · Dataset comparison (global + local constraints)</h2>
<p>Both datasets enforce <i>global</i> count + <i>per-group</i> local counts. TissueMNIST uses synthetic binary groups, DermMNIST uses real <code>sex</code>.</p>
{dataset_summary(new_rows, sat_map)}

<h2>5 · Convergence trajectory</h2>
<figure><img src="{fig_conv}" alt="convergence">
<figcaption>Total raw excess <code>Σ<sub>c</sub> max(0, count<sub>c</sub>−K<sub>c</sub>)</code> over training epochs (hardest cell). Fioretto snaps to feasibility in &lt;10 epochs; TraLO descends smoothly after CE-saturation switch-off (~epoch 70); Hounie never reaches 0 in-training and gets there only via post-hoc flips.</figcaption></figure>

<h2>6 · F1 vs tightness sweep</h2>
<figure><img src="{fig_tight}" alt="F1 vs tightness">
<figcaption>TissueMNIST/MobileNetV3 cls 4 across α ∈ [20%, 80%]. F1 macro (top) and constrained-class F1 (bottom). Gap is largest at the tightest budgets where Fioretto's collapse hurts most.</figcaption></figure>

<h2>7 · Satisfaction discipline</h2>
<figure><img src="{fig_sat}" alt="satisfaction bars">
<figcaption>Fraction of runs satisfying constraints <em>during training</em> (no post-hoc). TraLO ≈ Fioretto's 100% closed-form baseline; Hounie ≈ 0% until budget loosens enough that natural rate already fits.</figcaption></figure>

<h2>8 · Why the apples-to-apples view matters</h2>
<p>The naive comparison reads the F1 from each method's pipeline-default predictions. But:</p>
<ul>
  <li><b>TraLO</b> normally ends with raw_count ≤ K and minimal posthoc fill — its F1 is on the model's actual decisions.</li>
  <li><b>Fioretto</b> closed-form dual ends with raw_count well below K → posthoc fills 30-85 samples from low-confidence "next-best" candidates → F1c is dragged down.</li>
  <li><b>Hounie</b> ends with raw_count above K → posthoc has to drop 30-60 over-predictions → again losing signal.</li>
</ul>
<p>By forcing all three to the same exact count, F1 reflects only <em>which samples the model picked</em>, not <em>how many slots each used</em>. That's the comparison the paper should show.</p>

<h2>9 · So2Sat anomaly (not in tables yet — under investigation)</h2>
<div class="important">
On <b>So2Sat</b> (17-class urban LCZ classification, constrained class = 7 "Compact Lowrise") TraLO's raw_count for class 7 came out at <b>22 of K=169</b> — a 6.5% recall on a 14.1% test class. The constraint was satisfied (22 ≤ 169) so CE-skip activated and learning stopped early.
This exposes a <b>one-sided constraint limitation</b>: TraLO penalises over-prediction only; under-prediction is invisible. If a model already under-predicts the constrained class, TraLO doesn't push it up. Post-hoc fill then has to assign 147 samples from low-confidence candidates → poor F1c.
Fix candidates: (a) two-sided constraint <code>|count − K| ≤ margin</code>, (b) class-weighted CE during warmup, (c) lower threshold for CE-skip activation. So2Sat sweep paused pending this fix.
</div>

<h2>10 · Honest framing</h2>
<p><b>What TraLO is:</b> a single-objective, gradient-based, end-to-end differentiable method that achieves true in-training constraint satisfaction at competitive F1.</p>
<p><b>What it isn't:</b> a method that beats every baseline on every metric. Hounie's pipeline F1 is sometimes higher, but that's because the comparison treats post-hoc as part of the method. Under uniform post-hoc, TraLO wins the constrained-class F1 and stays within 0.5-2 pp of Fioretto/Hounie on overall F1.</p>
<p><b>Real contribution:</b> drop-in loss for any classifier that needs hard prediction-count guarantees at deployment, no post-hoc thresholding required.</p>

</body></html>
"""
    OUT.write_text(html, encoding="utf-8")
    print(f"Wrote {OUT} ({OUT.stat().st_size//1024} KB)")


if __name__ == "__main__":
    build()
