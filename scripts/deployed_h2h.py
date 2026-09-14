"""Inspect deployed predictions; fresh identity and metric validation remain release gates."""

import argparse
import json
import os
from pathlib import Path
import sys
from scripts import pred_integrity

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
    allc = sorted(set(y.unique()) | set(p.unique()))
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
        vals.append(2.0 * d["TP"] / den if den else float("nan"))
    return sum(vals) / len(vals) if vals else float("nan")


def macrof1(all_per):
    vals = []
    for c in sorted(all_per):
        d = all_per[c]
        den = d["K"] + d["n"]
        vals.append(2.0 * d["TP"] / den if den else 0.0)
    return sum(vals) / len(vals) if vals else float("nan")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--campaign", nargs="+", required=True)
    ap.add_argument("--json", dest="output")
    args = ap.parse_args()
    records = []
    try:
        for root in args.campaign:
            paths = sorted(Path(root).rglob("final_predictions.csv"))
            if not paths:
                raise ValueError("no deployed predictions under " + root)
            if pred_integrity.audit([root]):
                raise ValueError("prediction integrity failed")
            for p in paths:
                cfg = json.loads((p.parent / "config.json").read_text(encoding="utf-8"))
                if cfg.get("status") != "completed":
                    raise ValueError("run is not completed: " + str(p.parent))
                rec = read_run(str(p.parent))
                if rec is None:
                    raise ValueError("invalid or off-recipe run: " + str(p.parent))
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
                        per_class=rec["all_per"],
                    )
                )
        if not records:
            raise ValueError("no completed runs")
    except (OSError, ValueError, TypeError, KeyError) as exc:
        print("FAIL deployed_h2h: %s" % exc)
        return 1
    text = json.dumps(records, indent=2)
    if args.output:
        Path(args.output).write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
