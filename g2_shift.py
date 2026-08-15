"""GEOM probe 2 -- how much of what the constraint DID was a global per-class
shift (the free, AP-neutral escape route), and did ONE competitor class absorb it?

Matched pairs at warm-up 50: `heuristic` never runs a constraint phase, so its
score vector IS the cached warm-up model that the trained arm started from
(bit-identical base_model_id).  D = logP_after - logP_before, row-centred (the
softmax gauge), decomposes as

    D = 1 mu^T  +  R          mu = column mean  (a per-class bias shift applied
                                                 uniformly to the whole pool)

  frac_shift = N||mu||^2 / ||D||_F^2      how much of the action is a pure shift
  top1_share = max_j>0 mu_j / sum_j max(mu_j,0)   is the inflation concentrated
                                                  on ONE competitor class?
Counterfactual: apply mu alone to the warm-up logits and re-measure the hard
count and AP.  If the shift alone already satisfies the cap, the constraint
never needed to touch the representation at all.
"""
import glob, json, os, sys
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

ROOT = os.path.expanduser("~/OptimizationLoss")
sys.path.insert(0, ROOT)
os.chdir(ROOT)
from src.utils.constants import UNLIMITED
from src.training.constraints import compute_global_constraints

TRAINED = ("tralo", "tralo_bounded", "fioretto_ldf", "hounie_rcl")
POST = ("heuristic", "danits_lp")


def load(run_dir):
    cfg = json.load(open(os.path.join(run_dir, "config.json")))
    raw = os.path.join(run_dir, "final_predictions_raw.csv")
    if not os.path.exists(raw):
        return None
    t = pd.read_csv(raw)
    cols = sorted((x for x in t.columns if x.startswith("Prob_Class_")),
                  key=lambda x: int(x.rsplit("_", 1)[1]))
    if not cols:
        return None
    P = t[cols].to_numpy(float)
    y = t["True_Label"].to_numpy(int)
    dc = cfg.get("dataset_config") or {}
    cl = dc.get("constrained_class")
    if cl is None:
        return None
    c = int(cl[0] if isinstance(cl, (list, tuple)) else cl)
    lp, gp = cfg["constraint"]
    G = compute_global_constraints(pd.DataFrame({"label": y}), "label", gp,
                                   constrained_class=[c], num_classes=P.shape[1])
    if G[c] >= UNLIMITED:
        return None
    return dict(P=P, y=y, c=c, K=int(G[c]), cfg=cfg)


cells = {}
for cp in glob.glob("results/pending_runs/**/config.json", recursive=True):
    try:
        cfg = json.load(open(cp))
    except Exception:
        continue
    if cfg.get("status") != "completed":
        continue
    hp = cfg.get("hyperparams") or {}
    if hp.get("warmup_epochs") != 50:
        continue
    m = cfg.get("methodology")
    if m not in TRAINED + POST:
        continue
    camp = "/".join(cp.split("/")[:3])
    key = (camp, cfg.get("dataset_mode"), cfg.get("constraint_tag"),
           cfg.get("model_name"), hp.get("seed"))
    cells.setdefault(key, {}).setdefault(m, []).append(os.path.dirname(cp))

rows = []
seen = {}
for key, bym in sorted(cells.items()):
    posts = [p for m in POST for p in bym.get(m, [])]
    if not posts:
        continue
    k3 = key[1:4]
    if seen.get(k3, 0) >= 16:
        continue
    b = load(sorted(posts)[0])
    if b is None:
        continue
    seen[k3] = seen.get(k3, 0) + 1
    Pb, y, c, K = b["P"], b["y"], b["c"], b["K"]
    N, C = Pb.shape
    Lb = np.log(np.clip(Pb, 1e-12, 1))
    Lb = Lb - Lb.mean(1, keepdims=True)
    pos = (y == c).astype(int)
    ap_b = average_precision_score(pos, Pb[:, c])
    for meth in TRAINED:
        for rd in bym.get(meth, []):
            a = load(rd)
            if a is None or a["P"].shape != Pb.shape or not np.array_equal(a["y"], y):
                continue
            Pa = a["P"]
            La = np.log(np.clip(Pa, 1e-12, 1))
            La = La - La.mean(1, keepdims=True)
            D = La - Lb
            fro2 = float((D ** 2).sum())
            if fro2 < 1e-12:
                continue
            mu = D.mean(0)
            shift2 = float(N * (mu ** 2).sum())
            posmu = np.maximum(mu, 0)
            # counterfactual: warm-up logits + the global shift only
            Ls = Lb + mu
            Ps = np.exp(Ls); Ps /= Ps.sum(1, keepdims=True)
            cnt_shift = int((Ps.argmax(1) == c).sum())
            ap_shift = average_precision_score(pos, Ps[:, c])
            rows.append(dict(
                campaign=key[0], dataset=key[1], cap=key[2], model=key[3], seed=key[4],
                method=meth, N=N, K=K,
                frac_shift=shift2 / fro2,
                mu_c=float(mu[c]),
                mu_top1=float(posmu.max()),
                top1_share=float(posmu.max() / max(posmu.sum(), 1e-12)),
                top1_class=int(np.argmax(posmu)),
                n_infl_classes=int((posmu > 0.2 * posmu.max()).sum()),
                dL2=float(np.sqrt(fro2 / N)),
                cnt_before=int((Pb.argmax(1) == c).sum()),
                cnt_after=int((Pa.argmax(1) == c).sum()),
                cnt_shift_only=cnt_shift,
                AP_before=ap_b, AP_after=average_precision_score(pos, Pa[:, c]),
                AP_shift_only=ap_shift))

d = pd.DataFrame(rows)
d.to_csv(os.path.expanduser("~/OptimizationLoss/newdirections/arm_geom/g2_shift.csv"), index=False)
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40)
print("pairs:", len(d))
print(d.groupby("method").agg(
    n=("frac_shift", "size"), frac_shift=("frac_shift", "median"),
    mu_c=("mu_c", "median"), mu_top1=("mu_top1", "median"),
    top1_share=("top1_share", "median"), n_infl=("n_infl_classes", "median"),
    cnt_b=("cnt_before", "median"), cnt_a=("cnt_after", "median"),
    cnt_shift=("cnt_shift_only", "median"), K=("K", "median"),
    APb=("AP_before", "median"), APa=("AP_after", "median"),
    APshift=("AP_shift_only", "median")).round(4).to_string())
print()
print(d.groupby(["dataset", "method"]).agg(
    n=("frac_shift", "size"), frac_shift=("frac_shift", "median"),
    top1_share=("top1_share", "median"), n_infl=("n_infl_classes", "median"),
    cnt_b=("cnt_before", "median"), cnt_a=("cnt_after", "median"),
    cnt_shift=("cnt_shift_only", "median"), K=("K", "median"),
    APb=("AP_before", "median"), APa=("AP_after", "median"),
    APshift=("AP_shift_only", "median")).round(4).to_string())
