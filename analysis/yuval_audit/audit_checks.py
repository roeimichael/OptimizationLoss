"""Audit checks: Yuval's allocator/loss/metrics vs ours, on synthetic and stored knee probabilities."""
import json, math, sys, random
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score

sys.path.insert(0, r'C:/Users/roeym/Desktop/projects/OL-bandcons')
sys.path.insert(0, r'C:/Users/roeym/Desktop/projects/yuval-ConstrainedClassification')
from tralo.global_clipper import allocate
from tralo.metrics import classification_metrics

HERE = Path(__file__).parent


def yuval_alloc(probs, k, n_k):
    """Verbatim port of optimization.constrained_classification's assignment loop (model-free)."""
    all_p = probs.clone()
    cp = all_p[:, k].view(-1)          # a VIEW, as in his code
    pred = torch.zeros(len(all_p), dtype=torch.long)
    order = torch.argsort(cp, descending=True)
    count = 0
    for idx in order:
        if count < n_k and cp[idx] > 0:
            pred[idx] = k
            count += 1
        else:
            row = all_p[idx]
            row[k] = 0                  # mutates all_p (and cp) in place, as in his code
            pred[idx] = torch.argmax(row)
    return pred.tolist(), count


def ours(probs, k, cap, ids):
    caps = [None] * probs.shape[1]
    caps[k] = cap
    return allocate([[float(x) for x in r] for r in probs.double()], caps, ids, 'capped_first')


def normalize_rows(p):
    p = p.double()
    return p / p.sum(1, keepdim=True)


out = {}
rng = torch.Generator().manual_seed(0)

# ---- 1. allocator agreement, random matrices, integer caps, no ties
agree = total = 0
diff_items = []
for trial in range(300):
    n = int(torch.randint(50, 400, (1,), generator=rng))
    alpha = float(torch.empty(1).uniform_(0.05, 2.0, generator=rng))
    p = torch.distributions.Dirichlet(torch.full((5,), alpha)).sample((n,))
    p = normalize_rows(p)
    cap = int(torch.randint(1, n // 3 + 2, (1,), generator=rng))
    ids = ['s%05d' % i for i in range(n)]
    y, cnt = yuval_alloc(p, 3, cap)
    o = ours(p, 3, cap, ids)
    total += 1
    agree += int(y == o)
    diff_items.append(sum(a != b for a, b in zip(y, o)))
out['alloc_random_integer_caps'] = dict(trials=total, identical=agree, max_items_differing=max(diff_items))

# ---- 2. non-integer N_K: his loop fills ceil(N_K)
fills = []
for n_k in (66.9, 111.5, 156.1, 200.7, 76.0, 75.5):
    p = normalize_rows(torch.distributions.Dirichlet(torch.full((5,), 0.5)).sample((826,)))
    _, cnt = yuval_alloc(p, 3, n_k)
    fills.append((n_k, cnt))
out['noninteger_fill'] = fills

# ---- 3. exact ties: duplicate rows (identical probabilities, different labels)
tie_disagree = 0
tie_trials = 0
for trial in range(200):
    n_unique = 120
    base = normalize_rows(torch.distributions.Dirichlet(torch.full((5,), 0.3)).sample((n_unique,)))
    reps = torch.randint(1, 4, (n_unique,), generator=rng)
    p = torch.cat([base[i:i + 1].repeat(int(r), 1) for i, r in enumerate(reps)])
    perm = torch.randperm(len(p), generator=rng)
    p = p[perm]
    cap = 40
    ids = ['s%05d' % i for i in range(len(p))]
    y, _ = yuval_alloc(p, 3, cap)
    o = ours(p, 3, cap, ids)
    tie_trials += 1
    tie_disagree += int(sorted(i for i, c in enumerate(y) if c == 3) != sorted(i for i, c in enumerate(o) if c == 3))
out['alloc_exact_ties'] = dict(trials=tie_trials, selected_set_differs=tie_disagree,
                               note='only possible when tied p3 values straddle the cut rank')

# ---- 4. stored knee probabilities (ResNet18 v3 study, seeds 1801-1824, cap 76)
man = json.loads((HERE / 'seed1801' / 'manifest.json').read_text())
val = [r for r in man['rows'] if r['split'] == 'val']
labels = [r['label'] for r in val]
ids = [r['sample_id'] for r in val]
knee = dict(n=len(labels), true_grade3=labels.count(3), class_counts=[labels.count(c) for c in range(5)],
            val_order_is_class_sorted=labels == sorted(labels))
rows = []
for s in range(1801, 1825):
    for arm in ('clipper', 'tralo_null', 'tralo_target'):
        f = HERE / f'seed{s}' / arm / 'final_probabilities.pt'
        if not f.exists():
            continue
        p = torch.load(f, map_location='cpu', weights_only=True).double()
        p3 = p[:, 3]
        srt = torch.sort(p3, descending=True).values
        # ties straddling the cut: value at rank 76 equals value at rank 77 (0-based 75/76)
        straddle = bool(srt[75] == srt[76])
        n_exact_one = int((p[:, 3] == 1.0).sum())
        n_dup_p3 = len(p3) - len(torch.unique(p3))
        y, cnt = yuval_alloc(p.float(), 3, 76)       # his code runs on float32 softmax output
        o = ours(p, 3, 76, ids)
        m_y = classification_metrics(labels, y, 5, [3])
        m_o = classification_metrics(labels, o, 5, [3])
        sk_macro = f1_score(labels, o, average='macro')
        sk_weighted = f1_score(labels, o, average='weighted')
        rows.append(dict(seed=s, arm=arm, straddle_tie=straddle, p3_eq_1=n_exact_one, dup_p3=n_dup_p3,
                         identical=y == o, items_differ=sum(a != b for a, b in zip(y, o)),
                         acc_ours=m_o['accuracy'], acc_his=m_y['accuracy'],
                         ccf1_ours=m_o['cc_f1'], ccf1_his=m_y['cc_f1'],
                         macro_ours=m_o['macro_f1'], macro_sklearn=sk_macro, weighted_sklearn=sk_weighted,
                         raw_grade3=int((p.argmax(1) == 3).sum())))
knee['rows'] = len(rows)
knee['identical_all'] = all(r['identical'] for r in rows)
knee['max_items_differ'] = max(r['items_differ'] for r in rows)
knee['any_straddle_tie'] = any(r['straddle_tie'] for r in rows)
knee['max_p3_eq_1'] = max(r['p3_eq_1'] for r in rows)
knee['max_dup_p3'] = max(r['dup_p3'] for r in rows)
knee['macro_ours_eq_sklearn'] = all(abs(r['macro_ours'] - r['macro_sklearn']) < 1e-12 for r in rows)
knee['raw_grade3_counts_clipper'] = [r['raw_grade3'] for r in rows if r['arm'] == 'clipper']
# cross-check one stored report
rep = json.loads((HERE / 'seed1801' / 'clipper' / 'report.json').read_text())
r1801 = [r for r in rows if r['seed'] == 1801 and r['arm'] == 'clipper'][0]
knee['report_crosscheck_1801_clipper'] = dict(stored_ccf1=rep['capped_first']['metrics']['cc_f1'],
                                              recomputed=r1801['ccf1_ours'],
                                              stored_acc=rep['capped_first']['metrics']['accuracy'],
                                              recomputed_acc=r1801['acc_ours'])
out['knee_stored'] = knee
# mean metrics per arm on the three Yuval-style metrics (capped_first)
per_arm = {}
for arm in ('clipper', 'tralo_null', 'tralo_target'):
    rs = [r for r in rows if r['arm'] == arm]
    per_arm[arm] = {k: float(np.mean([r[k] for r in rs])) for k in ('acc_ours', 'ccf1_ours', 'macro_ours', 'weighted_sklearn')}
    per_arm[arm]['n'] = len(rs)
out['knee_per_arm_means'] = per_arm
paired = {}
for a, b in (('tralo_target', 'clipper'), ('tralo_target', 'tralo_null')):
    for k in ('acc_ours', 'ccf1_ours', 'macro_ours', 'weighted_sklearn'):
        d = []
        for s in range(1801, 1825):
            ra = [r for r in rows if r['seed'] == s and r['arm'] == a]
            rb = [r for r in rows if r['seed'] == s and r['arm'] == b]
            if ra and rb:
                d.append(ra[0][k] - rb[0][k])
        d = np.array(d)
        from scipy import stats
        t = stats.t.ppf(0.975, len(d) - 1)
        paired[f'{a}-{b}:{k}'] = dict(n=len(d), mean=float(d.mean()), lo=float(d.mean() - t * d.std(ddof=1) / math.sqrt(len(d))),
                                      hi=float(d.mean() + t * d.std(ddof=1) / math.sqrt(len(d))))
out['knee_paired'] = paired

# ---- 5. CustomLoss == CE at C = 1; == C_y * CE for argmax==k rows
from losses import CustomLoss
torch.manual_seed(1)
logits = torch.randn(4096, 5) * 3
labels_t = torch.randint(0, 5, (4096,))
ce = F.cross_entropy(logits, labels_t, reduction='none')
crit1 = CustomLoss(3, torch.ones(5))
l1 = crit1(logits, labels_t)
C = torch.tensor([4., 4., 4., 1., 4.])
critC = CustomLoss(3, C)
lC = critC(logits, labels_t)
pred_k = logits.argmax(1) == 3
expected = torch.where(pred_k, C[labels_t] * ce, ce).mean()
lg = logits.clone().requires_grad_(True)
critC(lg, labels_t).backward()
g_custom = lg.grad.clone()
lg2 = logits.clone().requires_grad_(True)
(torch.where(pred_k, C[labels_t], torch.ones(4096)) * F.cross_entropy(lg2, labels_t, reduction='none')).mean().backward()
out['customloss'] = dict(ce_mean=float(ce.mean()), custom_C1=float(l1), abs_diff_C1=float(abs(l1 - ce.mean())),
                         custom_C=float(lC), expected_weighted_ce=float(expected),
                         grad_max_abs_diff_vs_weighted_ce=float((g_custom - lg2.grad).abs().max()),
                         frac_pred_k=float(pred_k.float().mean()))
# near-tie gradient spike: argmax != k by a hair
z = torch.tensor([[2.0, 0.0, 0.0, 2.0 - 2e-7, 0.0]], requires_grad=True)
l = CustomLoss(3, C)(z, torch.tensor([0]))
l.backward()
out['customloss_near_tie_grad_norm'] = float(z.grad.norm())
z2 = torch.tensor([[2.0, 0.0, 0.0, 1.0, 0.0]], requires_grad=True)
CustomLoss(3, C)(z2, torch.tensor([0])).backward()
out['customloss_normal_grad_norm'] = float(z2.grad.norm())

# ---- 6. dF: his sympy function vs closed form; size of the C update
from update_weights import calculate_F_and_derivative
upd = []
for nkp in (60, 76, 77, 90, 120, 150, 200):
    Fv, dF = calculate_F_and_derivative(76.0, float(nkp), 100.0)
    upd.append(dict(N_K_p=nkp, F=float(Fv), dF=float(dF), C_increment=float(8 / 600 * dF)))
out['F_and_update'] = upd

(HERE / 'audit_checks.json').write_text(json.dumps(out, indent=1, default=str))
print(json.dumps({k: v for k, v in out.items() if k != 'knee_stored'}, indent=1, default=str))
print(json.dumps({k: v for k, v in out['knee_stored'].items()}, indent=1, default=str))
