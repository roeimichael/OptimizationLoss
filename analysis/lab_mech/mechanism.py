"""Offline logit-space mechanism analysis of the count constraint (development labels, finished runs only).

Why can a count penalty on grade 3 not move the capped boundary in the correct direction?

  1 SIGN      the count logit gradient g_i = a p_i3 (e3 - p_i), a > 0. Its descent direction changes item i's
              grade-3 log-odds at rate  -a * w_i,  w_i = p_i3 |e3 - p_i|^2 / (1 - p_i3) >= 0.  Every item is demoted,
              none is promoted. Write w_i = p_i3 (1 - p_i3) c_i with c_i = |e3 - p_i|^2 / (1 - p_i3)^2 in [1 + 1/(K-1), 2].
              If c_i were constant the log-odds flow dL/dt = -a sig(L)(1-sig(L)) c would be one autonomous 1-D ODE for every
              item, which preserves order: the logit-space flow can reorder items ONLY through c_i heterogeneity.
  2 INFO      AUC (wrong vs correct occupants of the top-cap p3 slots) of w, c, -p3, and of the targeted step's real
              per-item log-odds change.
  3 ORACLE    cosine in the 826x5 centred-logit space between the count descent direction and the oracle
              (demote wrong-inside, promote correct-outside), vs random, the post-hoc cut and the real steps.
  4 ERRORS    who the wrong occupants are, and the reachable prize near the cut.
  5 TRANSFER  far-item (rank >= 3*cap) motion under tralo_target vs sham_target.

python mechanism.py OUT.json RUN_ROOT [RUN_ROOT ...]
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.stats import kendalltau, rankdata, spearmanr

from tralo.global_clipper import allocate
from tralo.streamed_constraint import count_logit_gradient

G = 3
K = 5
RNG = np.random.default_rng(20260926)
N_BOOT = 2000


def auc(score, positive):
    """P(score_positive > score_negative), ties count 1/2. None if a class is empty."""
    positive = np.asarray(positive, bool)
    n1, n0 = positive.sum(), (~positive).sum()
    if n1 == 0 or n0 == 0:
        return None
    r = rankdata(score)
    return float((r[positive].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def logodds(p):
    p = p.astype(np.float64)
    other = np.delete(p, G, axis=1).sum(1)
    return np.log(p[:, G]) - np.log(other)


def centred_logits(p):
    z = np.log(p.astype(np.float64))
    return z - z.mean(1, keepdims=True)


def softmax(z):
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def order(p, ids):
    """capped_first order for one capped class: -p3, then sample_id."""
    return np.array(sorted(range(len(p)), key=lambda i: (-p[i, G], ids[i])))


def demotion(p):
    p = p.astype(np.float64)
    e3 = np.zeros(K)
    e3[G] = 1
    sq = ((e3 - p) ** 2).sum(1)
    one_minus = np.delete(p, G, axis=1).sum(1)
    w = p[:, G] * sq / one_minus
    c = sq / one_minus ** 2
    return w, c


def unit_rows(m):
    n = np.linalg.norm(m, axis=1, keepdims=True)
    return np.where(n > 0, m / np.maximum(n, 1e-300), 0.0)


def cos(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return None
    return float((a * b).sum() / (na * nb))


def boot(values_by_seed):
    """Mean over seeds of per-seed values and a 95% seed-bootstrap CI. values_by_seed: list of floats (None dropped)."""
    v = np.array([x for x in values_by_seed if x is not None and np.isfinite(x)], float)
    if len(v) == 0:
        return dict(n=0)
    idx = RNG.integers(0, len(v), size=(N_BOOT, len(v)))
    b = v[idx].mean(1)
    return dict(n=int(len(v)), mean=float(v.mean()), sd=float(v.std(ddof=1)) if len(v) > 1 else None,
                ci95=[float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))])


def per_seed_mean(rows, key):
    by = {}
    for r in rows:
        if r.get(key) is not None:
            by.setdefault(r['seed'], []).append(r[key])
    return [float(np.mean(v)) for v in by.values()]


def summarise(rows, keys, by_epoch=True):
    out = dict(pooled={k: boot(per_seed_mean(rows, k)) for k in keys})
    if by_epoch:
        for ep in sorted({r['epoch'] for r in rows}):
            sub = [r for r in rows if r['epoch'] == ep]
            out['epoch%02d' % ep] = {k: boot(per_seed_mean(sub, k)) for k in keys}
    return out


def residual_auc(y_score, x, positive):
    """AUC of y after removing its least-squares linear fit on x (the part of y not explained by x)."""
    A = np.c_[np.ones_like(x), x]
    beta, *_ = np.linalg.lstsq(A, y_score, rcond=None)
    return auc(y_score - A @ beta, positive)


def logit_flow_to_cap(p, cap, fixed_direction):
    """Idealised per-item logit descent on the count until the hard argmax count reaches cap (no network coupling).
    Direction p3 (e3 - p) is the verified count gradient up to the positive scalar a."""
    e3 = np.eye(K)[G]
    z = centred_logits(p)
    q = softmax(z)
    g0 = q[:, [G]] * (e3 - q)
    for _ in range(50000):
        if (q.argmax(1) == G).sum() <= cap:
            return q
        g = g0 if fixed_direction else q[:, [G]] * (e3 - q)
        z = z - g * (0.01 / np.abs(g).max())
        q = softmax(z)
    return None


def caps(v):
    c = [None] * K
    c[G] = v
    return c


def study(root):
    root = Path(root)
    seeds = sorted(d for d in root.glob('seed*') if d.is_dir() and (d / 'summary.json').exists())
    cap = json.loads((seeds[0] / 'config.json').read_text())['caps'][G]
    sign, info, oracle, errors, transfer, flow = [], [], [], [], [], []
    for d in seeds:
        seed = int(d.name[4:])
        config = json.loads((d / 'config.json').read_text())
        assert config['caps'][G] == cap
        val = [r for r in json.loads((d / 'manifest.json').read_text())['rows'] if r['split'] == 'val']
        labels = np.array([r['label'] for r in val])
        ids = [r['sample_id'] for r in val]
        true3 = labels == G

        def epochs(arm):
            out = []
            for line in open(d / arm / 'events.jsonl'):
                e = json.loads(line)
                if e.get('event') == 'epoch' and 'hard_counts_before' in e:
                    out.append(e)
            return out

        # ---------- tralo_null: 1 SIGN, 2 INFO (static), 3 ORACLE, 4 ERRORS, idealised flow ----------
        null_snaps = [(e['epoch'], torch.load(d / 'tralo_null' / ('epoch%02d_before_constraint.pt' % e['epoch']),
                                               weights_only=True).numpy()) for e in epochs('tralo_null')]
        null_snaps.append((99, torch.load(d / 'tralo_null' / 'final_probabilities.pt', weights_only=True).numpy()))
        for ep, p in null_snaps:
            p64 = p.astype(np.float64)
            p64 = p64 / p64.sum(1, keepdims=True)
            o = order(p64, ids)
            slots = o[:cap]
            if ep == null_snaps[0][0]:
                cf = np.array(allocate(p.tolist(), config['caps'], ids, 'capped_first'))
                assert set(np.flatnonzero(cf == G)) == set(slots), 'capped_first set mismatch'
            rank = np.empty(len(o), int)
            rank[o] = np.arange(len(o))
            inside = rank < cap
            hard = int((p64.argmax(1) == G).sum())
            w, c = demotion(p64)
            L = logodds(p64)
            # 1 SIGN: library gradient == a p3 (e3 - p); finite logit step lowers every log-odds; formula matches
            g = count_logit_gradient(torch.tensor(p64), caps(0), torch.ones(K, dtype=torch.float64), 0.0).numpy()
            e3 = np.zeros(K)
            e3[G] = 1
            base = p64[:, [G]] * (e3 - p64)
            a = float((g * base).sum() / (base * base).sum())
            eps = 1e-4 / np.abs(g).max()
            L_step = logodds(softmax(centred_logits(p64) - eps * g))
            dL = L_step - L
            pred = -eps * a * w
            if ep != 99:
                sign.append(dict(seed=seed, epoch=ep, a=a, rel_formula_err=float(np.abs(g - a * base).max() / np.abs(g).max()),
                                 n_raised=int((dL > 1e-15).sum()), max_dL=float(dL.max()),
                                 dz3_max=float((-g[:, G]).max()), dz_other_min=float(np.delete(-g, G, 1).min()),
                                 fo_rel_err=float(np.abs(dL - pred).max() / np.abs(pred).max()),
                                 c_min=float(c.min()), c_max=float(c.max()),
                                 w_share_slots=float(w[inside].sum() / w.sum()),
                                 w_share_wrong_slots=float(w[inside & ~true3].sum() / w.sum()),
                                 w_share_far=float(w[rank >= 3 * cap].sum() / w.sum()),
                                 w_share_true3_outside=float(w[~inside & true3].sum() / w.sum()),
                                 w_q_slots=np.percentile(w[inside], [5, 25, 50, 75, 95]).tolist(),
                                 w_q_near_out=np.percentile(w[(rank >= cap) & (rank < 2 * cap)], [5, 25, 50, 75, 95]).tolist(),
                                 w_q_far=np.percentile(w[rank >= 3 * cap], [5, 25, 50, 75, 95]).tolist(),
                                 w_top_slot_over_last_slot=float(w[o[0]] / w[o[cap - 1]]),
                                 w_argmax_rank=int(rank[np.argmax(w)]), p3_at_argmax_w=float(p64[np.argmax(w), G])))
            # 2 INFO among occupants
            wrong = ~true3[slots]
            rec = dict(seed=seed, epoch=ep, binding=hard > cap, n_wrong=int(wrong.sum()),
                       auc_w=auc(w[slots], wrong), auc_c=auc(c[slots], wrong), auc_negp3=auc(-p64[slots, G], wrong),
                       auc_c_resid=residual_auc(np.log(c[slots]), L[slots], wrong) if wrong.any() and (~wrong).any() else None,
                       auc_w_resid=residual_auc(np.log(w[slots]), L[slots], wrong) if wrong.any() and (~wrong).any() else None)
            if rec['auc_w'] is not None:
                rec['auc_w_minus_negp3'] = rec['auc_w'] - rec['auc_negp3']
            info.append(rec)
            # 3 ORACLE
            D = -g  # descent direction, rows sum to 0
            U = unit_rows((e3 - p64) / np.delete(p64, G, 1).sum(1, keepdims=True))  # d logodds / dz, unit rows
            wi = [i for i in o[:cap][::-1] if not true3[i]]          # wrong inside, lowest p3 first
            co = [i for i in o[cap:] if true3[i]]                     # correct outside, highest p3 first
            orec = dict(seed=seed, epoch=ep, binding=hard > cap)
            for tag, win in (('all', None), ('k20', 20)):
                a_in = [i for i in wi if win is None or rank[i] >= cap - win]
                a_out = [i for i in co if win is None or rank[i] < cap + win]
                m = min(len(a_in), len(a_out))
                if m == 0:
                    continue
                O = np.zeros_like(p64)
                O[a_in[:m]] = -U[a_in[:m]]
                O[a_out[:m]] = U[a_out[:m]]
                Odown = np.where(O[:, [G]] < 0, O, 0.0)
                orec['m_' + tag] = m
                orec['cos_count_' + tag] = cos(D, O)
                orec['cos_count_down_part_' + tag] = float((D * Odown).sum() / (np.linalg.norm(D) * np.linalg.norm(O)))
                orec['cos_count_up_part_' + tag] = float((D * (O - Odown)).sum() / (np.linalg.norm(D) * np.linalg.norm(O)))
                orec['ceiling_demote_only_' + tag] = float(np.linalg.norm(Odown) / np.linalg.norm(O))
                R = RNG.standard_normal((200,) + p64.shape)
                R = R - R.mean(2, keepdims=True)
                rc = np.array([cos(r, O) for r in R])
                orec['cos_random_mean_' + tag] = float(rc.mean())
                orec['cos_random_sd_' + tag] = float(rc.std(ddof=1))
                cut = [i for i in range(len(p64)) if p64[i].argmax() == G and not inside[i]]
                if cut:
                    P = np.zeros_like(p64)
                    P[cut] = -U[cut]
                    orec['cos_posthoc_' + tag] = cos(P, O)
            oracle.append(orec)
            # 4 ERRORS
            erec = dict(seed=seed, epoch=ep, binding=hard > cap, hard=hard,
                        wrong_by_grade={str(k): int(((labels[slots] == k)).sum()) for k in range(K) if k != G},
                        n_wrong=int((~true3[slots]).sum()), n_correct_out=int((true3 & ~inside).sum()),
                        p3_wrong_q=np.percentile(p64[slots][~true3[slots], G], [10, 50, 90]).tolist() if (~true3[slots]).any() else None,
                        p3_correct_q=np.percentile(p64[slots][true3[slots], G], [10, 50, 90]).tolist(),
                        wrong_in_bottom_half=int((~true3[o[cap // 2:cap]]).sum()),
                        prize_global=min(int((~true3[slots]).sum()), int((true3 & ~inside).sum())))
            for kk in (10, 20):
                W = int((~true3[o[cap - kk:cap]]).sum())
                C = int(true3[o[cap:cap + kk]].sum())
                erec['wrong_in_k%d' % kk] = W
                erec['correct_out_k%d' % kk] = C
                erec['inversion_pairs_k%d' % kk] = W * C
                erec['prize_k%d' % kk] = min(W, C)
            errors.append(erec)
            # idealised per-item logit flow to the cap: does it pick the post-hoc cut's set?
            if hard > cap and ep != 99:
                frec = dict(seed=seed, epoch=ep, hard=hard)
                cf_correct = int(true3[slots].sum())
                for tag, fixed in (('linear', True), ('flow', False)):
                    q = logit_flow_to_cap(p64, cap, fixed)
                    if q is None:
                        continue
                    kept = set(np.flatnonzero(q.argmax(1) == G))
                    raw = set(np.flatnonzero(p64.argmax(1) == G))
                    cut_post = raw - set(slots)
                    cut_flow = raw - kept
                    frec[tag + '_n_kept'] = len(kept)
                    frec[tag + '_admitted'] = len(kept - raw)
                    frec[tag + '_evict_overlap_with_posthoc'] = len(cut_flow & cut_post) / max(len(cut_flow), 1)
                    frec[tag + '_kept_correct_minus_raw_topcap_correct'] = int(true3[list(kept)].sum()) - int(
                        true3[list(raw & set(slots))].sum()) if kept else None
                    # re-rank after flow under capped_first: correct slots vs capped_first on the pre-step probs
                    o2 = order(q, ids)
                    frec[tag + '_capped_correct_delta'] = int(true3[o2[:cap]].sum()) - cf_correct
                    frec[tag + '_topcap_set_overlap'] = len(set(o2[:cap]) & set(slots)) / cap
                flow.append(frec)

        # ---------- tralo_target / sham_target: 2 INFO (real step), 3 cosine of real step, 5 TRANSFER ----------
        for arm in ('tralo_target', 'sham_target'):
            for e in epochs(arm):
                t = e.get('targeted') or {}
                if not t.get('applied'):
                    continue
                ep = e['epoch']
                pb = torch.load(d / arm / ('epoch%02d_before_constraint.pt' % ep), weights_only=True).numpy().astype(np.float64)
                pa = torch.load(d / arm / ('epoch%02d_after_constraint.pt' % ep), weights_only=True).numpy().astype(np.float64)
                pb /= pb.sum(1, keepdims=True)
                pa /= pa.sum(1, keepdims=True)
                ob, oa = order(pb, ids), order(pa, ids)
                rb, ra = np.empty(len(ob), int), np.empty(len(oa), int)
                rb[ob] = np.arange(len(ob))
                ra[oa] = np.arange(len(oa))
                Lb, La = logodds(pb), logodds(pa)
                dL = La - Lb
                w, c = demotion(pb)
                slots = ob[:cap]
                wrong = ~true3[slots]
                far = rb >= 3 * cap
                near = rb < 3 * cap
                entered = set(oa[:cap]) - set(slots)
                left = set(slots) - set(oa[:cap])
                rec = dict(seed=seed, epoch=ep, arm=arm, hard_before=t['hard_before'], hard_after=t.get('hard_after'),
                           auc_step=auc(-dL[slots], wrong), auc_negp3=auc(-pb[slots, G], wrong),
                           auc_after_negp3=auc(-pa[slots, G], wrong),
                           auc_step_resid=residual_auc(-dL[slots], Lb[slots], wrong) if wrong.any() and (~wrong).any() else None,
                           frac_items_raised=float((dL > 0).mean()), frac_slots_raised=float((dL[slots] > 0).mean()),
                           raised_true3_frac=float(true3[dL > 0].mean()) if (dL > 0).any() else None,
                           mean_dL_all=float(dL.mean()), mean_dL_slots=float(dL[slots].mean()),
                           corr_dL_vs_minus_w_all=float(spearmanr(dL, -w)[0]),
                           corr_dL_vs_minus_w_far=float(spearmanr(dL[far], -w[far])[0]),
                           corr_dL_vs_minus_w_near=float(spearmanr(dL[near], -w[near])[0]),
                           tau_all=float(kendalltau(pb[:, G], pa[:, G])[0]),
                           tau_far=float(kendalltau(pb[far, G], pa[far, G])[0]),
                           tau_near=float(kendalltau(pb[near, G], pa[near, G])[0]),
                           mean_abs_drank_far=float(np.abs(ra[far] - rb[far]).mean()),
                           mean_abs_drank_near=float(np.abs(ra[near] - rb[near]).mean()),
                           frac_far_down=float((dL[far] < 0).mean()), mean_dL_far=float(dL[far].mean()),
                           sd_dL_far=float(dL[far].std(ddof=1)),
                           coherence_far=float(abs(dL[far].mean()) / dL[far].std(ddof=1)),
                           topcap_entered=len(entered), topcap_entered_true3=int(true3[list(entered)].sum()) if entered else 0,
                           topcap_left=len(left), topcap_left_wrong=int((~true3[list(left)]).sum()) if left else 0,
                           capped_correct_delta=int(true3[oa[:cap]].sum()) - int(true3[slots].sum()))
                # real step direction in centred-logit space vs the oracle built on the before snapshot
                Z = centred_logits(pa) - centred_logits(pb)
                g = count_logit_gradient(torch.tensor(pb), caps(0), torch.ones(K, dtype=torch.float64), 0.0).numpy()
                U = unit_rows((np.eye(K)[G] - pb) / np.delete(pb, G, 1).sum(1, keepdims=True))
                wi = [i for i in slots[::-1] if not true3[i]]
                co = [i for i in ob[cap:] if true3[i]]
                m = min(len(wi), len(co))
                if m:
                    O = np.zeros_like(pb)
                    O[wi[:m]] = -U[wi[:m]]
                    O[co[:m]] = U[co[:m]]
                    rec['cos_realstep_oracle'] = cos(Z, O)
                    rec['cos_countdir_oracle'] = cos(-g, O)
                rec['cos_realstep_countdir'] = cos(Z, -g)
                transfer.append(rec)

    def sub(rows, **kw):
        return [r for r in rows if all(r.get(k) == v for k, v in kw.items())]

    grade_totals = {}
    for r in sub(errors, epoch=99):
        for k, v in r['wrong_by_grade'].items():
            grade_totals.setdefault(k, []).append(v)
    constraint_errors = [r for r in errors if r['epoch'] != 99]
    for r in constraint_errors:
        for k, v in r['wrong_by_grade'].items():
            r['wrong_grade_' + k] = v
    out = dict(
        root=str(root), cap=cap, n_seeds=len(seeds),
        sign=dict(n_snapshots=len(sign),
                  max_rel_formula_err=max(r['rel_formula_err'] for r in sign),
                  a_min=min(r['a'] for r in sign),
                  items_raised_total=sum(r['n_raised'] for r in sign),
                  max_dL=max(r['max_dL'] for r in sign),
                  max_dz3=max(r['dz3_max'] for r in sign), min_dz_other=min(r['dz_other_min'] for r in sign),
                  max_first_order_rel_err=max(r['fo_rel_err'] for r in sign),
                  c_range=[min(r['c_min'] for r in sign), max(r['c_max'] for r in sign)],
                  summary=summarise(sign, ['w_share_slots', 'w_share_wrong_slots', 'w_share_far', 'w_share_true3_outside',
                                           'w_top_slot_over_last_slot', 'w_argmax_rank', 'p3_at_argmax_w'], by_epoch=False),
                  w_quantiles_5_25_50_75_95=dict(
                      slots=np.mean([r['w_q_slots'] for r in sign], 0).tolist(),
                      near_outside=np.mean([r['w_q_near_out'] for r in sign], 0).tolist(),
                      far=np.mean([r['w_q_far'] for r in sign], 0).tolist())),
        info_null=summarise([r for r in info if r['epoch'] != 99],
                            ['auc_w', 'auc_c', 'auc_negp3', 'auc_w_minus_negp3', 'auc_c_resid', 'auc_w_resid', 'n_wrong']),
        info_null_binding_only=summarise([r for r in info if r['epoch'] != 99 and r['binding']],
                                         ['auc_w', 'auc_c', 'auc_negp3', 'auc_w_minus_negp3', 'auc_c_resid'], by_epoch=False),
        info_null_final=summarise(sub(info, epoch=99), ['auc_w', 'auc_c', 'auc_negp3', 'auc_c_resid'], by_epoch=False),
        oracle_null=summarise([r for r in oracle if r['epoch'] != 99],
                              [k for k in ('m_all', 'cos_count_all', 'cos_count_down_part_all', 'cos_count_up_part_all',
                                           'ceiling_demote_only_all', 'cos_random_mean_all', 'cos_random_sd_all', 'cos_posthoc_all',
                                           'm_k20', 'cos_count_k20', 'cos_count_down_part_k20', 'cos_count_up_part_k20',
                                           'ceiling_demote_only_k20', 'cos_random_sd_k20', 'cos_posthoc_k20')]),
        errors_null_constraint_epochs=summarise(constraint_errors,
                                                ['hard', 'n_wrong', 'wrong_grade_0', 'wrong_grade_1', 'wrong_grade_2', 'wrong_grade_4',
                                                 'n_correct_out', 'wrong_in_bottom_half', 'prize_global',
                                                 'wrong_in_k10', 'correct_out_k10', 'inversion_pairs_k10', 'prize_k10',
                                                 'wrong_in_k20', 'correct_out_k20', 'inversion_pairs_k20', 'prize_k20']),
        errors_null_final=dict(
            wrong_by_grade_mean={k: float(np.mean(v)) for k, v in grade_totals.items()},
            **summarise(sub(errors, epoch=99), ['n_wrong', 'prize_global', 'prize_k10', 'prize_k20',
                                                'inversion_pairs_k10', 'inversion_pairs_k20'], by_epoch=False),
            p3_wrong_q10_50_90=np.mean([r['p3_wrong_q'] for r in sub(errors, epoch=99) if r['p3_wrong_q']], 0).tolist(),
            p3_correct_q10_50_90=np.mean([r['p3_correct_q'] for r in sub(errors, epoch=99)], 0).tolist()),
        p3_wrong_vs_correct_constraint_epochs=dict(
            wrong_q10_50_90=np.mean([r['p3_wrong_q'] for r in constraint_errors if r['p3_wrong_q']], 0).tolist(),
            correct_q10_50_90=np.mean([r['p3_correct_q'] for r in constraint_errors], 0).tolist()),
        idealised_logit_flow=dict(n_snapshots=len(flow), **summarise(
            flow, [k for k in ('linear_n_kept', 'linear_admitted', 'linear_evict_overlap_with_posthoc',
                               'linear_capped_correct_delta', 'linear_topcap_set_overlap',
                               'flow_n_kept', 'flow_admitted', 'flow_evict_overlap_with_posthoc',
                               'flow_capped_correct_delta', 'flow_topcap_set_overlap')], by_epoch=False)),
    )
    tkeys = ['auc_step', 'auc_negp3', 'auc_after_negp3', 'auc_step_resid', 'frac_items_raised', 'frac_slots_raised',
             'raised_true3_frac', 'mean_dL_all', 'mean_dL_slots', 'corr_dL_vs_minus_w_all', 'corr_dL_vs_minus_w_far',
             'corr_dL_vs_minus_w_near', 'tau_all', 'tau_far', 'tau_near', 'mean_abs_drank_far', 'mean_abs_drank_near',
             'frac_far_down', 'mean_dL_far', 'sd_dL_far', 'coherence_far', 'topcap_entered', 'topcap_entered_true3',
             'topcap_left', 'topcap_left_wrong', 'capped_correct_delta', 'cos_realstep_oracle', 'cos_countdir_oracle',
             'cos_realstep_countdir']
    for arm in ('tralo_target', 'sham_target'):
        rows = sub(transfer, arm=arm)
        out['steps_' + arm] = dict(n_steps=len(rows), **summarise(rows, tkeys, by_epoch=False))
    # paired target - sham per seed (per-seed means over applied steps)
    diffs = {}
    for k in tkeys:
        a = {s: v for s, v in zip(*_seed_means(sub(transfer, arm='tralo_target'), k))}
        b = {s: v for s, v in zip(*_seed_means(sub(transfer, arm='sham_target'), k))}
        diffs[k] = boot([a[s] - b[s] for s in a if s in b])
    out['target_minus_sham_paired_by_seed'] = diffs
    return out


def _seed_means(rows, key):
    by = {}
    for r in rows:
        if r.get(key) is not None:
            by.setdefault(r['seed'], []).append(r[key])
    s = sorted(by)
    return s, [float(np.mean(by[x])) for x in s]


def main():
    out = [study(r) for r in sys.argv[2:]]
    Path(sys.argv[1]).write_text(json.dumps(out, indent=1))
    print('wrote', sys.argv[1])


if __name__ == '__main__':
    main()
