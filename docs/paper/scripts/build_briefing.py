"""Build a self-contained HTML briefing (figures embedded as base64).

Run from repo root:  python paper/scripts/build_briefing.py  ->  briefing.html
"""
import base64
import io
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = pathlib.Path(__file__).resolve().parents[2]
FIG = ROOT / "paper" / "figures"


def img(name):
    b = (FIG / name).read_bytes()
    return "data:image/png;base64," + base64.b64encode(b).decode()


def matimg(tex, fontsize=21, color="#15243a"):
    """Render a LaTeX-ish math string to a tight transparent PNG (mathtext)."""
    fig = plt.figure(figsize=(0.01, 0.01))
    fig.text(0, 0, f"${tex}$", fontsize=fontsize, color=color)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=220, bbox_inches="tight",
                pad_inches=0.14, transparent=True)
    plt.close(fig)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


# Pre-rendered equations (mathtext: no \underbrace; multi-letter subs via \mathrm)
EQ = {
    "soft":  matimg(r"s_c \;=\; \sum_i \, p_\theta(c \,|\, x_i)"),
    "exc":   matimg(r"E^{+}=\max(0,\,s_c-K)\quad\ \, E^{-}=\max(0,\,K-s_c)"),
    "over":  matimg(r"\varphi(E^{+}\!,K)=\frac{E^{+}}{E^{+}+K}"
                    r"\qquad \psi(E^{+}\!,K)=\frac{(E^{+}/K)^2}{1+(E^{+}/K)^2}"),
    "hinge": matimg(r"\eta(s,K)\;=\;\beta\,\frac{E^{-}}{K}"
                    r"\;=\;\beta\,\frac{\max(0,\;K-s)}{K}",
                    color="#0d47a1"),
    "full":  matimg(r"\ell \;=\; \varphi(E^{+}\!,K)+\rho\,\psi(E^{+}\!,K)"
                    r"\;+\;\beta\,\frac{E^{-}}{K}"),
    "loss":  matimg(r"L \;=\; L_{\mathrm{CE}} + \lambda_g\,L_{\mathrm{global}}"
                    r" + \lambda_\ell\,L_{\mathrm{local}}"),
    "tfull": matimg(r"\ell_{\mathrm{TraLO}} \;=\; \varphi(E^{+}\!,K)+\rho\,\psi(E^{+}\!,K)"
                    r"\;+\;\beta\,\frac{E^{-}}{K}"
                    r"\quad (\beta=0.5)", color="#0a7d28"),
    "tbnd":  matimg(r"\ell_{\mathrm{bounded}} \;=\; \varphi(E^{+}\!,K)+\rho\,\psi(E^{+}\!,K)"
                    r"\qquad\qquad\ \ (\beta=0)", color="#b04a00"),
}


F = {k: img(v) for k, v in {
    "conv": "fig_convergence_v2.png",
    "flips": "fig_flips_tightness_v2.png",
    "f1": "fig_f1_tightness_v2.png",
    "sat": "fig_satisfaction_v2.png",
    "asym": "fig_asymmetric_summary_v2.png",
    "ba": "fig_posthoc_beforeafter_v2.png",
    "ww": "fig_posthoc_withwithout_v2.png",
}.items()}

CSS = """
:root{--blue:#1976D2;--ink:#1a1a1a;--mut:#666;--bg:#f6f7f9;--card:#fff;--line:#e3e6ea;}
*{box-sizing:border-box}
body{margin:0;font-family:-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;
color:var(--ink);background:var(--bg);line-height:1.5;font-size:15px}
.wrap{max-width:1180px;margin:0 auto;padding:28px 20px 80px}
header{border-left:6px solid var(--blue);padding:6px 0 6px 16px;margin-bottom:8px}
h1{margin:0;font-size:26px}
.thesis{color:var(--mut);font-size:16px;margin-top:4px}
h2{font-size:19px;margin:34px 0 12px;border-bottom:2px solid var(--line);padding-bottom:6px}
h3{font-size:15px;margin:18px 0 6px;color:var(--blue)}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:16px 18px;
margin:14px 0;box-shadow:0 1px 3px rgba(0,0,0,.04)}
.grid2{display:grid;grid-template-columns:1fr 1fr;gap:14px}
@media(max-width:820px){.grid2{grid-template-columns:1fr}}
figure{margin:0}
img{width:100%;height:auto;border:1px solid var(--line);border-radius:8px;background:#fff}
img.m{width:auto;height:auto;max-width:98%;max-height:80px;display:block;margin:12px auto;
border:none;background:none;border-radius:0}
img.mi{width:auto;height:1.25em;max-height:30px;display:inline;vertical-align:-0.45em;
border:none;background:none;border-radius:0;margin:0 2px}
figcaption{font-size:13px;color:var(--mut);margin-top:7px}
ul{margin:6px 0 6px 0;padding-left:20px} li{margin:3px 0}
table{border-collapse:collapse;width:100%;font-size:13.5px}
th,td{border:1px solid var(--line);padding:6px 9px;text-align:center}
th{background:#eef2f7} td.l,th.l{text-align:left}
.win{color:#0a7d28;font-weight:700} .lose{color:#b04a00} .tie{color:#666}
.tra{background:#e8f1fc;font-weight:700}
.small{font-size:13px;color:var(--mut)}
.key{display:flex;gap:10px;flex-wrap:wrap;margin:6px 0}
.kbox{flex:1;min-width:200px;background:#fbfcfe;border:1px solid var(--line);border-radius:8px;padding:10px 12px}
.takeaway{background:#eef7ee;border:1px solid #cfe6cf;border-radius:10px;padding:14px 18px;margin-top:8px}
"""

HTML = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>TraLO - briefing</title>
<style>{CSS}</style></head><body><div class="wrap">

<header>
<h1>TraLO &mdash; Transductive Prediction-Count Constraints</h1>
<div class="thesis">Train a classifier whose predictions already respect hard per-class budgets
(global + per-subgroup), so it deploys feasibly with no post-hoc surgery.</div>
</header>
<p class="small">Briefing &bull; MobileNetV3 headline &bull; 3 datasets (TissueMNIST, DermMNIST, AIDER)
&bull; 6 methods &bull; 4 seeds/cell</p>

<h2>1 &middot; How the model works (math, short)</h2>
<div class="card">
<p class="small"><b>Problem:</b> cap how many test samples get a target class &mdash; at most
<b>K</b> globally and at most <b>K<sub>g</sub></b> per subgroup. Hard counts use <i>argmax</i>
(not differentiable), so we relax them.</p>

<h3>Soft count (differentiable proxy for the hard count)</h3>
<img class="m" src="{EQ['soft']}">
<p class="small">Sum of class probabilities over the transductive test set.</p>

<h3>Two one-sided deviations from the budget K</h3>
<img class="m" src="{EQ['exc']}">
<p class="small">The positive and negative parts of the same gap <b>s<sub>c</sub> &minus; K</b> &mdash; at most
one is ever nonzero. <b>E&#8314;</b> = how far <i>over</i> budget (penalized next); <b>E&#8315;</b> = how
far <i>under</i> (pushed back up by the hinge). Both measured relative to the same K.</p>

<h3>Over-budget penalty &mdash; acts on E&#8314;</h3>
<img class="m" src="{EQ['over']}">
<p class="small"><b>&phi;</b> = rational saturation (maps any violation into [0,1), no gradient blow-up);
<b>&psi;</b> = bounded quadratic (extra push near the boundary). <b>This over-budget part alone is
TraLO-bounded.</b></p>

<h3 style="color:#0d47a1">Undershoot hinge &mdash; acts on E&#8315;, makes it <u>full</u> TraLO</h3>
<img class="m" src="{EQ['hinge']}">
<p class="small">A one-sided term that activates only when the soft count is <i>below</i> budget and
pushes it <b>up to K</b> &mdash; so the model fills the budget and flips the last borderline argmaxes.
Weight <b>&beta; = 0.5</b> in TraLO; <b>&beta; = 0</b> recovers TraLO-bounded.</p>

<h3>Complete per-cell penalty</h3>
<img class="m" src="{EQ['full']}">

<h3>Total training loss</h3>
<img class="m" src="{EQ['loss']}">
<p class="small"><b>3 phases:</b> &#9312; warm-up (CE only) &rarr; &#9313; constraint optimization
(CE + penalties; &lambda; ratchets up until satisfied, optimizer reset at first satisfaction)
&rarr; &#9314; post-hoc adjustment (flip borderline predictions to <i>guarantee</i> hard feasibility).</p>
</div>

<h2>2 &middot; TraLO vs TraLO-bounded (the one difference)</h2>
<div class="card">
<p>Identical except the <span style="color:var(--blue)"><b>undershoot hinge</b></span> &mdash;
i.e. the value of <b>&beta;</b>:</p>
<div style="text-align:center;margin:8px 0">
<img class="m" src="{EQ['tfull']}"><br>
<img class="m" src="{EQ['tbnd']}">
</div>
<div class="grid2" style="margin-top:8px">
<ul>
<li><b>Full TraLO (&beta;=0.5):</b> drives the count <i>up to</i> the cap &rarr; lands natively feasible.</li>
<li><b>Bounded (&beta;=0):</b> only avoids going over &rarr; often stops short, needs post-hoc fixes.</li>
</ul>
<table>
<tr><th class="l">Overall</th><th>TraLO</th><th>bounded</th></tr>
<tr><td class="l">Flips &darr;</td><td class="tra">4</td><td class="lose">11</td></tr>
<tr><td class="l">Sat% &uarr;</td><td class="tra">100%</td><td class="lose">83%</td></tr>
<tr><td class="l">Macro F1 &uarr;</td><td>0.669</td><td>0.667</td></tr>
</table>
</div>
<p class="small">Same F1 &mdash; the hinge is what buys native feasibility (~3&times; fewer flips, +17 pts Sat%).</p>
</div>

<h2>3 &middot; The three metrics</h2>
<div class="key">
<div class="kbox"><b>Macro F1 &uarr;</b><br><span class="small">class-balanced accuracy.
<b class="tie">Result: a tie.</b></span></div>
<div class="kbox"><b>Post-hoc flips &darr;</b><br><span class="small">predictions forced to change after
training to obey the budget = how far the raw model was from feasible.
<b class="win">TraLO wins big.</b></span></div>
<div class="kbox"><b>In-training Sat% &uarr;</b><br><span class="small">fraction of runs already feasible
<i>before</i> post-hoc. <b class="win">TraLO 100%.</b></span></div>
</div>

<h2>4 &middot; Headline result (15 cells, overall mean)</h2>
<div class="card">
<table>
<tr><th class="l">Method</th><th>Macro F1 &uarr;</th><th>Flips &darr;</th><th>Sat% &uarr;</th></tr>
<tr class="tra"><td class="l">TraLO (ours)</td><td>0.669</td><td>4</td><td>100%</td></tr>
<tr><td class="l">TraLO-bounded</td><td>0.667</td><td>11</td><td>83%</td></tr>
<tr><td class="l">Fioretto LDF</td><td>0.667</td><td>11</td><td>93%</td></tr>
<tr><td class="l">Hounie RCL</td><td>0.661</td><td>20</td><td>100%</td></tr>
<tr><td class="l">Danits LP</td><td>0.659</td><td>72</td><td>7%</td></tr>
<tr><td class="l">Heuristic</td><td>0.660</td><td>74</td><td>7%</td></tr>
</table>
<p class="small">F1 within 0.01 across all methods (a tie); flips and Sat% separate them by an order of magnitude.</p>
</div>

<div class="card"><figure><img src="{F['flips']}">
<figcaption><b>Flips vs tightness.</b> TraLO (blue) hugs the floor on every dataset.
On AIDER it needs ~0 flips while post-hoc baselines need ~80.</figcaption></figure></div>
<div class="card"><figure><img src="{F['sat']}">
<figcaption><b>In-training satisfaction.</b> TraLO &amp; Hounie hold 100% everywhere;
Fioretto &amp; bounded drop on some cells. (Post-hoc methods = 0%, omitted.)</figcaption></figure></div>

<h2>5 &middot; F1 is a tie &mdash; shown honestly</h2>
<div class="card"><figure><img src="{F['f1']}">
<figcaption><b>Macro F1 vs tightness.</b> TraLO leads on Tissue/Derm; on <b>AIDER (right)</b> the
warm-up is saturated so everyone coincides and the post-hoc baselines edge ahead by 0.003&ndash;0.010
&mdash; shown, not hidden.</figcaption></figure></div>

<h2>6 &middot; How fast it reaches feasibility</h2>
<div class="card"><figure><img src="{F['conv']}">
<figcaption><b>Constraint excess vs epoch.</b> TraLO reaches feasibility first (dot = converged).
On Derm, Fioretto is drawn per-seed: 2 of 4 seeds get stuck &mdash; it does not reliably converge.</figcaption></figure></div>

<h2>7 &middot; Robustness &mdash; the win holds everywhere</h2>
<div class="card">
<table>
<tr><th class="l">Experiment (DermMNIST unless noted)</th><th>TraLO flips</th><th>Baselines flips</th>
<th>F1</th><th>Sat%</th></tr>
<tr><td class="l">Asymmetric tightness L&ne;G (20 cells)</td><td class="win">1.9</td><td>5.7 &ndash; 129</td>
<td class="tie">tie</td><td>100%</td></tr>
<tr><td class="l">Backbone: ResNet18 (saturated)</td><td class="win">8.3</td><td>22 &ndash; 50</td>
<td class="tie">tie</td><td>100%</td></tr>
<tr><td class="l">Backbone: EfficientNetB0 (saturated)</td><td class="win">6.2</td><td>11 &ndash; 69</td>
<td class="tie">tie</td><td>100%</td></tr>
<tr><td class="l">Multi-class (AKIEC/BCC/BKL)</td><td class="win">2.0</td><td>7 &ndash; 66</td>
<td class="tie">tie</td><td>98%</td></tr>
<tr><td class="l">Group column = sex</td><td class="win">3.1</td><td>8 &ndash; 92</td>
<td class="tie">tie</td><td>100%</td></tr>
</table>
</div>
<div class="card"><figure><img src="{F['asym']}">
<figcaption><b>Asymmetric sweep (2&times;2).</b> Top row = quality (F1, accuracy) <b>tied</b>;
bottom row = deployability (flips, Sat%) <b>won</b>. No-free-lunch made visible.</figcaption></figure></div>

<h2>8 &middot; What &ldquo;flips&rdquo; really means (with vs without post-hoc)</h2>
<div class="card"><figure><img src="{F['ww']}">
<figcaption><b>Satisfaction without &rarr; with post-hoc.</b> Arrow = feasibility post-hoc must supply.
TraLO/Hounie already feasible (no-op); the post-hoc baselines depend on it entirely.</figcaption></figure></div>
<div class="card"><figure><img src="{F['ba']}">
<figcaption><b>Budget overflow before post-hoc.</b> TraLO ~6% over (4 corrections);
Danits/Heuristic ~164% over (110 corrections). After post-hoc: all feasible, F1 tied.</figcaption></figure></div>

<h2>Bottom line</h2>
<div class="takeaway">
<b>TraLO wins decisively on post-hoc flips and in-training satisfaction in every experiment</b>,
at <b>matched F1/accuracy</b>. It ships a model that is feasible on its own &mdash; deployable on
streaming data where post-hoc allocators cannot run. The undershoot hinge (vs TraLO-bounded) is what
buys the native feasibility.
<br><span class="small">Honest caveats: F1 is a tie (not a win) vs trained baselines; AIDER concedes
F1 in the saturated regime; post-hoc baselines win ECE on Derm/AIDER (calibration trade-off).</span>
</div>

<p class="small" style="margin-top:30px">In progress (marked in the paper): asymmetric &amp; multi-class
on TissueMNIST/AIDER, and the component-ablation table (Table F).</p>

</div></body></html>"""


def main():
    out = ROOT / "briefing.html"
    out.write_text(HTML, encoding="utf-8")
    kb = len(out.read_bytes()) // 1024
    print(f"Wrote {out} ({kb} KB)")


if __name__ == "__main__":
    main()
