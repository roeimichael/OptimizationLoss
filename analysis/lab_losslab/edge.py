"""S6: edge cases of the count penalty itself (bounded_count_penalty, count_logit_gradient, targeted_step)."""
import json, math, sys
import numpy as np
import torch
from tralo.global_constraint import bounded_count_penalty, advance_controller
from tralo.streamed_constraint import count_logit_gradient
from tralo.targeted_step import targeted_step
import lab

torch.set_default_dtype(torch.float32)
out = {}
K, C3 = 5, 3


def pen_and_grad(logits, cap, lam=1.0, rho=0.0):
    z = logits.clone().requires_grad_(True)
    caps = [None] * K; caps[C3] = cap
    m = torch.zeros(K); m[C3] = lam
    v = bounded_count_penalty(z, caps, m, rho)
    v.backward()
    an = count_logit_gradient(z.detach().softmax(1).double(), caps, m.double(), rho)
    return float(v), z.grad.clone(), an


g = torch.Generator().manual_seed(0)
Z = torch.randn(800, K, generator=g) * 2
P = Z.softmax(1)
soft, hard = float(P[:, C3].sum()), int((P.argmax(1) == C3).sum())
# E1 cap = 0
v, gr, an = pen_and_grad(Z, 0, rho=1.0)
out['E1_cap0'] = dict(soft=soft, hard=hard, penalty=v, grad_norm=float(gr.norm()),
                      coef_note='scale=max(cap,1)=1 so e=soft=%.1f; lambda coefficient 1/(1+e)^2=%.2e' % (soft, 1 / (1 + soft) ** 2),
                      autograd_vs_analytic_maxabs=float((gr.double() - an).abs().max()))
# E2 cap >= soft and cap >= hard -> exactly inactive
for cap in (int(math.ceil(soft)), int(math.ceil(soft)) + 50, 800):
    v, gr, an = pen_and_grad(Z, cap, rho=1.0)
    out['E2_cap%d' % cap] = dict(penalty=v, grad_exact_zero=bool((gr == 0).all()), analytic_zero=bool((an == 0).all()))
# E2b soft > cap >= hard: penalty active although the hard count already meets the cap
Pb = torch.tensor([[0.45, 0.05, 0.05, 0.40, 0.05]]).repeat(800, 1)  # p3=0.40 but argmax is class 0
Zb = Pb.log(); sb, hb = float(Pb[:, C3].sum()), int((Pb.argmax(1) == C3).sum())
capb = 100
v, gr, an = pen_and_grad(Zb, capb, rho=1.0)
out['E2b_soft_above_hard_below'] = dict(soft=sb, hard=hb, cap=capb, penalty=v, grad_norm=float(gr.norm()),
                                        note='joint penalty keeps pushing although hard<=cap; targeted_step is a no-op here')
# E3 D2: soft < cap < hard -> penalty 0, gradient 0, controller ratchets forever
Zd = torch.zeros(800, K); Zd[:, C3] = 0.6  # p3 = e^.6/(e^.6+4) = 0.313, argmax = 3 for all
Pd = Zd.softmax(1); sd, hd = float(Pd[:, C3].sum()), int((Pd.argmax(1) == C3).sum())
capd = int((sd + hd) / 2)
v, gr, an = pen_and_grad(Zd, capd, rho=1.0)
lam, rho, frozen = [1.0] * K, 1.0, False
caps = [None] * K; caps[C3] = capd
hardc = [0] * K; hardc[C3] = hd
for t in range(50):
    lam, rho, frozen = advance_controller(hardc, caps, lam, rho, 0.5, 0.5, frozen)
out['E3_D2_soft_below_hard_above'] = dict(soft=sd, hard=hd, cap=capd, penalty=v, grad_exact_zero=bool((gr == 0).all()),
                                           after_50_controller_steps=dict(lam=lam[C3], rho=rho, frozen=frozen))
# E4 extreme confidence: per-item logit-gradient norm vs p3, float32 saturation
rows = []
for m in [0, 1, 2, 4, 6, 8, 10, 12, 14, 16, 17, 18, 20, 30]:
    z = torch.zeros(1, K); z[0, C3] = m
    p = z.softmax(1)
    caps = [None] * K; caps[C3] = 0
    gl = count_logit_gradient(p.double(), caps, torch.tensor([0, 0, 0, 1.0, 0], dtype=torch.float64), 0.0)
    rows.append(dict(logit_margin=m, p3_f32=float(p[0, C3]), one_minus_p3_f32=float(1 - p[0, C3]),
                     grad_norm=float(gl.norm()), grad_exact_zero=bool((gl == 0).all())))
out['E4_confidence'] = rows
# E4b bounded-penalty coefficient vs relative excess (the penalty is bounded => gradient vanishes for big violations)
out['E4b_coef_vs_excess'] = [dict(e=e, lam_coef=1 / (1 + e) ** 2, rho_coef_per_rho=2 * e / (1 + e * e) ** 2)
                             for e in (0.01, 0.1, 0.5, 1, 2, 5, 10, 100)]

# ------------- model-based edges on S1 data (seed 5000)
cfg = json.load(open('configs.json'))[sys.argv[1] if len(sys.argv) > 1 else 'S1']
torch.manual_seed(5000)
d = lab.make(cfg, 5000)
X = torch.tensor(d['xtr'], dtype=torch.float32); Y = torch.tensor(d['ytr'])
XP = torch.tensor(d['xp'], dtype=torch.float32); yp = d['yp']
model = lab.MLP(cfg['D'], cfg['H'], cfg.get('depth', 2))
opt = torch.optim.Adam(model.parameters(), lr=cfg.get('lr', 3e-3), weight_decay=cfg.get('wd', 1e-4))
lab.ce_steps(model, opt, X, Y, cfg['E'])
base = {k: v.clone() for k, v in model.state_dict().items()}
p0 = lab.probs(model, XP); p3 = p0[:, C3].numpy().astype(np.float64)
hard0 = int((p0.argmax(1) == C3).sum()); n3 = int((yp == C3).sum()); cap = int(round(0.7 * n3))
sel0 = lab.topcap(p3, cap)
out['model'] = dict(hard0=hard0, soft0=float(p3.sum()), n_true3=n3, cap=cap, clip=lab.prec(sel0, yp))

# E4c share of the soft-count parameter gradient carried by each p3 decile (who drives the step)
model.zero_grad(); model.eval()
pr = model(XP).softmax(1)
per = []
params = [p for p in model.parameters()]
full = torch.autograd.grad(pr[:, C3].sum(), params, retain_graph=True)
fulln = math.sqrt(sum(float(x.square().sum()) for x in full))
bins = [0, 0.01, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99, 0.999, 1.0001]
for lo, hi in zip(bins[:-1], bins[1:]):
    m = (p3 >= lo) & (p3 < hi)
    if not m.any():
        continue
    gsub = torch.autograd.grad(pr[torch.tensor(m), C3].sum(), params, retain_graph=True)
    proj = sum(float((a * b).sum()) for a, b in zip(gsub, full)) / fulln
    per.append(dict(p3_bin='[%g,%g)' % (lo, hi), n=int(m.sum()), frac_true3=float((yp[m] == C3).mean()),
                    grad_norm=math.sqrt(sum(float(x.square().sum()) for x in gsub)), share_of_full_along_full=proj / fulln))
out['E4c_gradient_share_by_p3'] = per

# E7 the per-item (M1) channel alone: free logits, gradient flow on the count to the cap, no network
z = torch.log(p0.double().clamp_min(1e-300))
caps0 = [None] * K; caps0[C3] = 0
steps = 0
while int((z.softmax(1).argmax(1) == C3).sum()) > cap and steps < 200000:
    gl = count_logit_gradient(z.softmax(1), caps0, torch.tensor([0, 0, 0, 1.0, 0], dtype=torch.float64), 0.0)
    z = z - 0.05 * gl / gl.norm() * math.sqrt(len(z))
    steps += 1
pf = z.softmax(1)[:, C3].numpy()
self_ = lab.topcap(pf, cap)
out['E7_free_logit_channel'] = dict(steps=steps, hard_after=int((z.softmax(1).argmax(1) == C3).sum()),
                                    overlap_with_clip=len(set(self_) & set(sel0)) / cap, prec=lab.prec(self_, yp),
                                    clip=lab.prec(sel0, yp),
                                    spearman_p3=float(np.corrcoef(np.argsort(np.argsort(pf)), np.argsort(np.argsort(p3)))[0, 1]))

# E5 hard count along the targeted direction: monotone?  then ties at the cut
caps1 = [None] * K; caps1[C3] = cap
info = targeted_step(model, [XP], caps1)
rad = info['radius']
model.load_state_dict(base)
params = [p for p in model.parameters() if p.requires_grad]
origin = [p.detach().clone() for p in params]
# recompute direction exactly as targeted_step does
model.eval(); model.zero_grad()
model(XP).softmax(1)[:, C3].sum().backward()
gvec = [p.grad.detach().clone() for p in params]
nrm = math.sqrt(sum(float(x.square().sum()) for x in gvec))
counts = []
for r in np.linspace(0, 3 * rad, 301):
    with torch.no_grad():
        for p, o, gg in zip(params, origin, gvec):
            p.copy_(o - r * gg / nrm)
    counts.append(int((lab.probs(model, XP).argmax(1) == C3).sum()))
counts = np.array(counts)
out['E5_monotone'] = dict(radius=rad, hard_after=info['hard_after'], evaluations=info['evaluations'],
                          upticks=int((np.diff(counts) > 0).sum()), max_uptick=int(np.diff(counts).max()),
                          count_at_3x_radius=int(counts[-1]))
model.load_state_dict(base)
# ties: duplicate the LAST item to leave the argmax-3 set along the direction (the one that decides the bisection)
def at(r):
    with torch.no_grad():
        for p, o, gg in zip(params, origin, gvec):
            p.copy_(o - r * gg / nrm)
    return lab.probs(model, XP).argmax(1).numpy() == C3
a_lo, a_hi = at(info['radius_violating']), at(rad)
flip = np.where(a_lo & ~a_hi)[0]
model.load_state_dict(base)
out['E5_last_flippers'] = int(len(flip))
if len(flip):
    k = int(flip[0])
    for dup in (1, 5):
        model.load_state_dict(base)
        XT = torch.cat([XP, XP[k:k + 1].repeat(dup, 1)])
        ht = int((lab.probs(model, XT).argmax(1) == C3).sum())
        for capt in (cap, cap + 2):
            model.load_state_dict(base)
            ct = [None] * K; ct[C3] = capt
            it = targeted_step(model, [XT], ct)
            out['E5_ties_dup%d_cap%d' % (dup, capt)] = dict(hard_before=ht, cap=capt, hard_after=it.get('hard_after'),
                                                         undershoot=capt - it.get('hard_after', capt))
model.load_state_dict(base)
# E1 on a real model: cap = 0 with the targeted step
c0 = [None] * K; c0[C3] = 0
try:
    i0 = targeted_step(model, [XP], c0)
    p = lab.probs(model, XP)
    out['E1_cap0_targeted'] = dict(hard_after=i0['hard_after'], radius=i0['radius'], radius_ratio_vs_cap=i0['radius'] / rad,
                                   pool_acc_after=float((p.argmax(1).numpy() == yp).mean()),
                                   pool_acc_before=float((p0.argmax(1).numpy() == yp).mean()))
except RuntimeError as e:
    out['E1_cap0_targeted'] = dict(error=str(e))
model.load_state_dict(base)
# E2 on a real model: cap = hard0 -> targeted_step must be a no-op
c2 = [None] * K; c2[C3] = hard0
i2 = targeted_step(model, [XP], c2)
same = all(torch.equal(a, b) for a, b in zip(model.state_dict().values(), base.values()))
out['E2_cap_eq_hard_targeted'] = dict(applied=i2['applied'], weights_unchanged=same)

json.dump(out, open('edge_%s.json' % (sys.argv[1] if len(sys.argv) > 1 else 'S1'), 'w'), indent=1)
print(json.dumps(out, indent=1))
