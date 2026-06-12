"""Build paper-track archive: master manifest + per-axis views + README.

Server-side only. Does NOT move/copy raw cells. Builds:
  archive/
    README.md                      - top-level guide
    MASTER_INDEX.csv               - one row per cell, all fields
    by_axis/
      per_dataset.md               - tissue / derm / aider summaries
      per_model.md                 - backbone summaries
      per_method.md                - tralo / fioretto / hounie / danits / heuristic
      per_tightness.md             - L20..L80, sym vs asym
      per_sweep.md                 - what each sweep was for
    tables/
      pivot_ds_model_method.csv    - counts by (ds, model, method)
      pivot_ds_tight_method.csv    - counts by (ds, tightness, method)
      methodology_means.csv        - mean macro_f1 / sat / flips per (ds, model, method)

Run server-side: python scripts/build_archive.py
"""
import csv
import glob
import json
import os
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path("results/pending_runs")
OUT = Path("archive")
(OUT / "by_axis").mkdir(parents=True, exist_ok=True)
(OUT / "tables").mkdir(parents=True, exist_ok=True)

# Active paper-track filters (CLAUDE.md + REJECTED.md). Cells outside these
# are excluded even if they live in an included sweep (e.g. blackwell_new_backbones
# touched ResNet18 during corroboration; we keep only the active backbones).
ACTIVE_DATASETS = {"tissuemnist", "dermmnist", "aider",
                   "octmnist", "cifar100", "retinamnist", "bloodmnist"}
ACTIVE_MODELS   = {"MobileNetV3", "MobileNetV2", "RegNetY400MF", "ShuffleNetV2"}
ACTIVE_METHODS  = {"tralo", "fioretto_ldf", "hounie_rcl", "danits_lp", "heuristic"}

# Paper-track sweeps with one-line purpose. Order = section grouping in README.
SWEEPS = [
    # Headline contamination grid + clean baseline
    ("contamination_clean",          "Clean sigma=0 anchor for contamination grid (3 ds x 4 tight x 5 methods x 2 seeds)"),
    ("contamination_tissuemnist",    "TissueMNIST contamination grid (sigma in {0.10,0.20,0.30} x 4 tight x 5 methods x 2 seeds)"),
    ("contamination_dermmnist",      "DermMNIST contamination grid (sigma in {0.10,0.20,0.30} x 4 tight x 5 methods x 2 seeds)"),
    ("contamination_aider",          "AIDER contamination grid (sigma in {0.10,0.20,0.30} x 4 tight x 5 methods x 2 seeds)"),

    # Paper headline backbones sweep
    ("paper_backbones",              "Headline 3-backbone x 3-dataset x 5-tight x 6-method x 4-seed sweep (G1+G5)"),
    ("paper400_baselines",           "400-config paper baseline grid: TraLO + 4 baselines x 3 ds (Turing era)"),
    ("paper400_tralofix",            "TraLO-fix rerun of paper400 set (undershoot_hinge + reset_optimizer_at_sat)"),

    # Asymmetric + multiclass
    ("g2_asym_tissue_aider",         "G2: asymmetric (L != G) tightness on tissue+aider, 4 methods"),
    ("g3_multiclass_tissue",         "G3: multiclass cap (multiple constrained classes) on tissuemnist"),
    ("g4_table_b_backfill",          "G4: backfill cells missing from Table B in paper v1"),
    ("aider_asym",                   "AIDER asymmetric tightness, extension seeds"),

    # Component ablations
    ("g5_component_ablation",        "G5: ablate undershoot_hinge / reset_optimizer_at_sat / lambda_toggle (the TraLO-fix components)"),
    ("component_ablation",           "Component ablation: lambda_toggle, KL, rho schedule, optimizer reset"),
    ("kl_ablation",                  "KL drift damper ablation (alpha_kl in {0, 0.05, 0.1, ...})"),

    # Blackwell validation + arch
    ("blackwell_validation",         "Blackwell 8-seed paired validation of Turing winners (MobileNetV2/V3 x derm/aider)"),
    ("blackwell_new_backbones",      "Blackwell extension: RegNetY16GF + DenseNet121 corroboration"),
    ("arch_validation",              "Turing vs Blackwell cell ranking comparison (architecture independence check)"),

    # Failure-regime probes
    ("derm_cripple",                 "Crippled derm warmup (low train acc) -- headroom hypothesis test"),
    ("aider_cripple",                "Crippled aider warmup -- saturated-warmup regime probe"),
    ("derm_backbone_weak",           "Weak-backbone search for clearer TraLO win on derm"),

    # Warmup + LR + model search
    ("warmup_confirm",               "Confirm warmup sweet-spot for TraLO F1 edge"),
    ("warmup_probe",                 "Initial warmup-quality probe (train_acc range)"),
    ("lr_hp_smoke",                  "LR sweep + HP variants (warmup, rho, lambda_step, alpha_kl) on derm sigma=0.20"),
    ("model_search",                 "8-backbone x 5-dataset search for clean TraLO winners"),

    # Expansion baselines (paper-track baseline backfills)
    ("expansion_baselines",          "Expansion baselines (heuristic + danits_lp) for full grid coverage"),
    ("expansion_aider_baselines",    "AIDER baseline expansion"),
    ("expansion_dermmnist_baselines","DermMNIST baseline expansion"),

    # Paper v2 phases (the 8-block paper pipeline)
    ("paperv2_phase1",               "Paper v2 phase 1: TraLO vs trained baselines, sym tightness"),
    ("paperv2_phase2",               "Paper v2 phase 2: post-hoc baselines"),
    ("paperv2_phase3",               "Paper v2 phase 3: asymmetric tightness"),
    ("paperv2_phase3_v3",            "Paper v2 phase 3 rev3: asymmetric refit"),
    ("paperv2_phase4",               "Paper v2 phase 4: multiclass constraint"),
    ("paperv2_phase5",               "Paper v2 phase 5: corroboration backbones"),
    ("paperv2_phase6",               "Paper v2 phase 6: component ablation"),

    # AIDER seed extension
    ("aider_seed_ext",               "AIDER seed extension for paired stats"),

    # New 2026-06-02: class-rotation + OctMNIST + CIFAR-100 small-train
    ("class_rotation",               "Alternate constrained-class rotation: tissue/derm/aider (3 alt classes each); confirms universal claim across cap-class choice"),
    ("octmnist_smoke",               "OctMNIST smoke probe (drusen as constrained class, 12 cells)"),
    ("octmnist_expansion",           "OctMNIST 60-cell full panel: 5 methods x 4 seeds x 3 tightness; CLEAN WIN at L30_G30 vs trained baselines"),
    ("cifar100_smalltrain",          "CIFAR-100 train-data-quantity headroom test: subset50/10/5 train samples/class"),
    ("new_dataset_probes",           "Retina/Blood/CIFAR-100 smoke probes (paper extension)"),
]
ACTIVE_DATASETS_EXTRA = {"octmnist", "cifar100", "retinamnist", "bloodmnist"}


def parse_tight(tag):
    """L40_G70 -> (40, 70, asym). L50_G50 -> (50,50,sym). Robust to L20-only."""
    if not isinstance(tag, str): return (None, None, None)
    L, G, asym = None, None, None
    parts = tag.split("_")
    for p in parts:
        if p.startswith("L"):
            try: L = int(p[1:])
            except ValueError: pass
        elif p.startswith("G"):
            try: G = int(p[1:])
            except ValueError: pass
    if L is not None and G is not None:
        asym = int(L != G)
    return (L, G, asym)


def parse_sigma(data_dir):
    """data/dermmnist_sigma20/slice_1 -> 0.20.  data/dermmnist/slice_1 -> 0.0"""
    if not isinstance(data_dir, str): return 0.0
    for s in ("sigma30","sigma20","sigma10"):
        if s in data_dir: return float(s[5:]) / 100.0
    return 0.0


def parse_metrics(p):
    out = {}
    try:
        with open(p) as f:
            for row in csv.DictReader(f):
                out[row["Metric"]] = row["Value"]
    except Exception:
        return None
    return out


def main():
    rows = []
    sweep_counts = Counter()
    missing_metrics = Counter()
    for sweep, purpose in SWEEPS:
        sdir = ROOT / sweep
        if not sdir.is_dir():
            sweep_counts[sweep + " [MISSING]"] = 0
            continue
        for cfg_p in glob.glob(str(sdir / "**" / "config.json"), recursive=True):
            try:
                with open(cfg_p) as f: cfg = json.load(f)
            except Exception: continue
            metrics_p = cfg_p.replace("config.json","evaluation_metrics.csv")
            m = parse_metrics(metrics_p)
            if m is None:
                missing_metrics[sweep] += 1
                continue
            try:
                ds = cfg["dataset_mode"]
                model = cfg["model_name"]
                method = cfg["methodology"]
                tight = cfg.get("constraint_tag", "")
                cls = cfg["dataset_config"]["constrained_class"]
                n_cls = cfg["dataset_config"]["num_classes"]
                data_dir = cfg["dataset_config"]["data_dir"]
                hp = cfg.get("hyperparams", {})
                seed = hp.get("seed", "")
                pretrained = hp.get("pretrained", True)
            except Exception:
                continue
            # Active-only filter
            if ds not in ACTIVE_DATASETS: continue
            if model not in ACTIVE_MODELS: continue
            if method not in ACTIVE_METHODS: continue
            L, G, is_asym = parse_tight(tight)
            sigma = parse_sigma(data_dir)
            row = {
                "sweep": sweep,
                "sweep_purpose": purpose,
                "cell_path": cfg_p.replace("/config.json",""),
                "dataset": ds,
                "model": model,
                "method": method,
                "constraint_tag": tight,
                "tight_L_pct": L,
                "tight_G_pct": G,
                "is_asymmetric": is_asym,
                "sigma": sigma,
                "constrained_class": cls,
                "num_classes": n_cls,
                "data_dir": data_dir,
                "seed": seed,
                "pretrained": int(bool(pretrained)),
                "warmup_epochs": hp.get("warmup_epochs",""),
                "constraint_epochs": hp.get("constraint_epochs",""),
                "learning_rate": hp.get("learning_rate",""),
                "batch_size": hp.get("batch_size",""),
                "lambda_step": hp.get("lambda_step",""),
                "rho_step": hp.get("rho_step",""),
                "rho_target": hp.get("rho_target",""),
                "alpha_kl": hp.get("alpha_kl",""),
                "disable_lambda_toggle": int(bool(hp.get("disable_lambda_toggle", False))),
                "reset_optimizer_at_sat": int(bool(hp.get("reset_optimizer_at_sat", False))),
                "undershoot_hinge": int(bool(hp.get("undershoot_hinge", False))),
                "restore_best_satisfied": int(bool(hp.get("restore_best_satisfied", False))),
            }
            # Metrics: dump all common ones, normalized
            def get_f(k, default=None):
                v = m.get(k, default)
                try: return float(v)
                except (TypeError, ValueError): return default
            row["macro_f1"] = get_f("F1 (Macro)")
            row["accuracy"] = get_f("Accuracy")
            row["raw_all_satisfied"] = int(m.get("Raw All Satisfied", "0") == "1")
            row["flips_required"] = get_f("Flips Required")
            row["ece"] = get_f("ECE")
            row["brier"] = get_f("Brier")
            row["train_acc_final"] = get_f("Train Accuracy")
            # per-class
            for c in range(n_cls):
                row[f"F1_Class{c}"] = get_f(f"F1_Class{c}")
                row[f"Support_Class{c}"] = get_f(f"Support_Class{c}")
            # constrained-class shortcut
            row["f1_cstr"] = get_f(f"F1_Class{cls}")
            unc = [get_f(f"F1_Class{c}") for c in range(n_cls)
                   if c != cls and get_f(f"F1_Class{c}") is not None]
            row["f1_unc_mean"] = float(np.mean(unc)) if unc else None
            rows.append(row)
            sweep_counts[sweep] += 1

    if not rows:
        print("ERROR: no rows collected — check sweep paths"); return

    # Determine union of fields, fixed leading order
    leading = ["sweep","sweep_purpose","dataset","model","method","constraint_tag",
               "tight_L_pct","tight_G_pct","is_asymmetric","sigma",
               "constrained_class","num_classes","seed","pretrained",
               "macro_f1","accuracy","f1_cstr","f1_unc_mean",
               "raw_all_satisfied","flips_required","ece","brier","train_acc_final",
               "warmup_epochs","constraint_epochs","learning_rate","batch_size",
               "lambda_step","rho_step","rho_target","alpha_kl",
               "disable_lambda_toggle","reset_optimizer_at_sat",
               "undershoot_hinge","restore_best_satisfied",
               "data_dir","cell_path"]
    all_keys = set()
    for r in rows: all_keys.update(r.keys())
    extra = sorted(k for k in all_keys if k not in leading)
    fields = leading + extra

    master = OUT / "MASTER_INDEX.csv"
    with open(master, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: ("" if r.get(k) is None else r.get(k)) for k in fields})
    print(f"WROTE {master}  ({len(rows)} cells, {len(fields)} fields)")

    # === per-axis views ===
    write_axis_views(rows, sweep_counts, missing_metrics)

    # === pivots ===
    write_pivots(rows)

    # === paired tables ===
    write_paired(rows)

    # === README ===
    write_readme(rows, sweep_counts, missing_metrics)


def write_paired(rows):
    """For each baseline, build per-cell (tralo - baseline) deltas. Same seed,
    same (sweep, dataset, model, constraint_tag, data_dir, constrained_class).

    Output one CSV per baseline + one combined summary CSV.
    """
    # cell key (excluding method)
    key_fields = ["sweep","dataset","model","constraint_tag",
                  "data_dir","constrained_class","seed"]
    cells = defaultdict(dict)
    for r in rows:
        k = tuple(r[f] for f in key_fields)
        cells[k][r["method"]] = r
    baselines = ["fioretto_ldf","hounie_rcl","danits_lp","heuristic"]
    summary_rows = []
    for bl in baselines:
        out = []
        for k, by_m in cells.items():
            if "tralo" not in by_m or bl not in by_m: continue
            t, b = by_m["tralo"], by_m[bl]
            def diff(field):
                a = t.get(field); c = b.get(field)
                if a is None or c is None: return ""
                try: return float(a) - float(c)
                except Exception: return ""
            row = {**{f: t[f] for f in key_fields},
                   "tight_L_pct": t["tight_L_pct"],
                   "tight_G_pct": t["tight_G_pct"],
                   "is_asymmetric": t["is_asymmetric"],
                   "sigma": t["sigma"],
                   "tralo_macro_f1": t.get("macro_f1",""),
                   f"{bl}_macro_f1": b.get("macro_f1",""),
                   "d_macro_f1": diff("macro_f1"),
                   "d_f1_cstr":  diff("f1_cstr"),
                   "d_f1_unc":   diff("f1_unc_mean"),
                   "d_acc":      diff("accuracy"),
                   "d_flips":    diff("flips_required"),
                   "d_ece":      diff("ece"),
                   "tralo_sat":  t.get("raw_all_satisfied",""),
                   f"{bl}_sat":  b.get("raw_all_satisfied",""),
                   "tralo_path": t["cell_path"],
                   f"{bl}_path": b["cell_path"]}
            out.append(row)
        if not out: continue
        fields = list(out[0].keys())
        path = OUT / f"tables/paired_tralo_vs_{bl}.csv"
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields); w.writeheader()
            for o in out: w.writerow(o)
        print(f"WROTE {path}  ({len(out)} paired cells)")
        # summary
        for ds in ("tissuemnist","dermmnist","aider"):
            for asym in (0, 1):
                sub = [o for o in out if o["dataset"]==ds and o["is_asymmetric"]==asym]
                if not sub: continue
                dmac = np.array([o["d_macro_f1"] for o in sub if isinstance(o["d_macro_f1"], float)])
                if len(dmac)==0: continue
                w = int(np.sum(dmac > 1e-4))
                l = int(np.sum(dmac < -1e-4))
                summary_rows.append({
                    "baseline": bl, "dataset": ds,
                    "asym": "asym" if asym else "sym",
                    "n": len(dmac),
                    "wins": w, "losses": l, "ties": len(dmac)-w-l,
                    "mean_d_macro_f1": f"{np.mean(dmac):+.4f}",
                    "median_d_macro_f1": f"{np.median(dmac):+.4f}",
                })
    if summary_rows:
        with open(OUT/"tables/paired_summary.csv","w",newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
            w.writeheader()
            for r in summary_rows: w.writerow(r)
        print(f"WROTE {OUT/'tables/paired_summary.csv'}  ({len(summary_rows)} rows)")


def write_axis_views(rows, sweep_counts, missing_metrics):
    """Write per-axis breakdown markdowns."""
    # per dataset
    by_ds = defaultdict(list)
    for r in rows: by_ds[r["dataset"]].append(r)
    lines = ["# Per-dataset breakdown\n"]
    for ds, rs in sorted(by_ds.items()):
        lines.append(f"\n## {ds}  (n={len(rs)})\n")
        models = Counter(r["model"] for r in rs)
        methods = Counter(r["method"] for r in rs)
        tights = Counter(r["constraint_tag"] for r in rs)
        sigmas = Counter(r["sigma"] for r in rs)
        sweeps = Counter(r["sweep"] for r in rs)
        lines.append(f"- **Models** ({len(models)}): " + ", ".join(f"`{k}`={v}" for k,v in models.most_common()))
        lines.append(f"- **Methods**: " + ", ".join(f"`{k}`={v}" for k,v in methods.most_common()))
        lines.append(f"- **Tightness** ({len(tights)} unique): " + ", ".join(f"`{k}`={v}" for k,v in tights.most_common(10)))
        lines.append(f"- **Contamination sigma**: " + ", ".join(f"`{k}`={v}" for k,v in sorted(sigmas.items())))
        lines.append(f"- **Sweeps contributing**: " + ", ".join(f"`{k}`={v}" for k,v in sweeps.most_common()))
    (OUT/"by_axis/per_dataset.md").write_text("\n".join(lines))
    print(f"WROTE {OUT/'by_axis/per_dataset.md'}")

    # per model
    by_m = defaultdict(list)
    for r in rows: by_m[r["model"]].append(r)
    lines = ["# Per-backbone breakdown\n"]
    for m_, rs in sorted(by_m.items(), key=lambda kv: -len(kv[1])):
        lines.append(f"\n## {m_}  (n={len(rs)})\n")
        dss = Counter(r["dataset"] for r in rs)
        meths = Counter(r["method"] for r in rs)
        sweeps = Counter(r["sweep"] for r in rs)
        pre = Counter(r["pretrained"] for r in rs)
        lines.append(f"- **Datasets**: " + ", ".join(f"`{k}`={v}" for k,v in dss.most_common()))
        lines.append(f"- **Methods**: " + ", ".join(f"`{k}`={v}" for k,v in meths.most_common()))
        lines.append(f"- **Pretrained {{0,1}}**: " + ", ".join(f"`{k}`={v}" for k,v in pre.most_common()))
        lines.append(f"- **Sweeps**: " + ", ".join(f"`{k}`={v}" for k,v in sweeps.most_common()))
    (OUT/"by_axis/per_model.md").write_text("\n".join(lines))
    print(f"WROTE {OUT/'by_axis/per_model.md'}")

    # per method
    by_meth = defaultdict(list)
    for r in rows: by_meth[r["method"]].append(r)
    lines = ["# Per-method breakdown\n"]
    for me, rs in sorted(by_meth.items(), key=lambda kv: -len(kv[1])):
        dss = Counter(r["dataset"] for r in rs)
        sweeps = Counter(r["sweep"] for r in rs)
        mf1 = [r["macro_f1"] for r in rs if r["macro_f1"] is not None]
        sat = [r["raw_all_satisfied"] for r in rs]
        lines.append(f"\n## {me}  (n={len(rs)})\n")
        lines.append(f"- **Mean macro-F1**: {np.mean(mf1):.4f}  (n_valid={len(mf1)})")
        lines.append(f"- **Mean satisfaction rate**: {np.mean(sat):.3f}")
        lines.append(f"- **Datasets**: " + ", ".join(f"`{k}`={v}" for k,v in dss.most_common()))
        lines.append(f"- **Sweeps**: " + ", ".join(f"`{k}`={v}" for k,v in sweeps.most_common(8)))
    (OUT/"by_axis/per_method.md").write_text("\n".join(lines))
    print(f"WROTE {OUT/'by_axis/per_method.md'}")

    # per tightness
    by_t = defaultdict(list)
    for r in rows: by_t[r["constraint_tag"]].append(r)
    lines = ["# Per-tightness breakdown\n",
             "(`L` = local cap %, `G` = global cap %. Asymmetric = L != G.)\n"]
    for t, rs in sorted(by_t.items()):
        dss = Counter(r["dataset"] for r in rs)
        meths = Counter(r["method"] for r in rs)
        asym = rs[0].get("is_asymmetric")
        lines.append(f"\n## {t}  (n={len(rs)})  asym={asym}")
        lines.append(f"- **Datasets**: " + ", ".join(f"`{k}`={v}" for k,v in dss.most_common()))
        lines.append(f"- **Methods**: " + ", ".join(f"`{k}`={v}" for k,v in meths.most_common()))
    (OUT/"by_axis/per_tightness.md").write_text("\n".join(lines))
    print(f"WROTE {OUT/'by_axis/per_tightness.md'}")

    # per sweep
    lines = ["# Per-sweep manifest\n"]
    sweep_to_purpose = {s: p for s, p in SWEEPS}
    by_sw = defaultdict(list)
    for r in rows: by_sw[r["sweep"]].append(r)
    for sw, _ in SWEEPS:
        rs = by_sw.get(sw, [])
        purp = sweep_to_purpose[sw]
        if not rs:
            lines.append(f"\n## {sw}\n- **Purpose**: {purp}\n- **Status**: MISSING or empty\n")
            continue
        dss = Counter(r["dataset"] for r in rs)
        models = Counter(r["model"] for r in rs)
        meths = Counter(r["method"] for r in rs)
        tights = Counter(r["constraint_tag"] for r in rs)
        seeds = sorted({r["seed"] for r in rs})
        missing = missing_metrics.get(sw, 0)
        lines.append(f"\n## {sw}\n- **Purpose**: {purp}")
        lines.append(f"- **Cells**: {len(rs)}  (missing metrics: {missing})")
        lines.append(f"- **Datasets**: " + ", ".join(f"`{k}`={v}" for k,v in dss.most_common()))
        lines.append(f"- **Models**: " + ", ".join(f"`{k}`={v}" for k,v in models.most_common()))
        lines.append(f"- **Methods**: " + ", ".join(f"`{k}`={v}" for k,v in meths.most_common()))
        lines.append(f"- **Tightness**: " + ", ".join(f"`{k}`={v}" for k,v in tights.most_common(8)))
        lines.append(f"- **Seeds**: {seeds}")
    (OUT/"by_axis/per_sweep.md").write_text("\n".join(lines))
    print(f"WROTE {OUT/'by_axis/per_sweep.md'}")


def write_pivots(rows):
    """Two count pivots + one means pivot."""
    # (ds, model, method) -> count
    triple = Counter((r["dataset"], r["model"], r["method"]) for r in rows)
    with open(OUT/"tables/pivot_ds_model_method.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["dataset","model","method","n_cells"])
        for (ds, m, me), n in sorted(triple.items()):
            w.writerow([ds, m, me, n])
    print(f"WROTE {OUT/'tables/pivot_ds_model_method.csv'}")

    # (ds, tight, method) -> count
    triple2 = Counter((r["dataset"], r["constraint_tag"], r["method"]) for r in rows)
    with open(OUT/"tables/pivot_ds_tight_method.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["dataset","constraint_tag","method","n_cells"])
        for (ds, t, me), n in sorted(triple2.items()):
            w.writerow([ds, t, me, n])
    print(f"WROTE {OUT/'tables/pivot_ds_tight_method.csv'}")

    # Methodology means: per (ds, model, method) mean macro_f1, sat, flips
    groups = defaultdict(list)
    for r in rows:
        groups[(r["dataset"], r["model"], r["method"])].append(r)
    with open(OUT/"tables/methodology_means.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["dataset","model","method","n","macro_f1_mean","macro_f1_std",
                    "satisfaction_rate","flips_mean","ece_mean"])
        for k, rs in sorted(groups.items()):
            mf1 = [r["macro_f1"] for r in rs if r["macro_f1"] is not None]
            fl = [r["flips_required"] for r in rs if r["flips_required"] is not None]
            ece = [r["ece"] for r in rs if r["ece"] is not None]
            sat = [r["raw_all_satisfied"] for r in rs]
            w.writerow([k[0], k[1], k[2], len(rs),
                        f"{np.mean(mf1):.4f}" if mf1 else "",
                        f"{np.std(mf1):.4f}" if len(mf1)>1 else "",
                        f"{np.mean(sat):.3f}",
                        f"{np.mean(fl):.2f}" if fl else "",
                        f"{np.mean(ece):.4f}" if ece else ""])
    print(f"WROTE {OUT/'tables/methodology_means.csv'}")


def write_readme(rows, sweep_counts, missing_metrics):
    n = len(rows)
    dss = Counter(r["dataset"] for r in rows)
    models = Counter(r["model"] for r in rows)
    methods = Counter(r["method"] for r in rows)
    sweep_count_real = Counter(r["sweep"] for r in rows)
    miss = sum(missing_metrics.values())
    lines = [
        "# Paper-track results archive",
        "",
        f"Built by `scripts/build_archive.py`. Lossless manifest over paper-track sweeps only.",
        f"Probe / dropped-dataset / failed sweeps are intentionally excluded.",
        "",
        "## Coverage",
        "",
        f"- **Cells indexed**: {n}",
        f"- **Cells with missing evaluation_metrics.csv (skipped)**: {miss}",
        f"- **Datasets**: " + ", ".join(f"`{k}` ({v})" for k,v in dss.most_common()),
        f"- **Models**: " + ", ".join(f"`{k}` ({v})" for k,v in models.most_common()),
        f"- **Methods**: " + ", ".join(f"`{k}` ({v})" for k,v in methods.most_common()),
        f"- **Sweeps**: {len(sweep_count_real)}",
        "",
        "## Files",
        "",
        "- `MASTER_INDEX.csv` — one row per cell, all fields. Source of truth.",
        "- `by_axis/per_dataset.md` — breakdown by dataset",
        "- `by_axis/per_model.md` — breakdown by backbone",
        "- `by_axis/per_method.md` — breakdown by methodology + mean macro-F1",
        "- `by_axis/per_tightness.md` — breakdown by constraint tightness",
        "- `by_axis/per_sweep.md` — what each sweep was for, cell counts",
        "- `tables/pivot_ds_model_method.csv` — cell counts (dataset, model, method)",
        "- `tables/pivot_ds_tight_method.csv` — cell counts (dataset, tightness, method)",
        "- `tables/methodology_means.csv` — mean macro_f1 / sat / flips / ECE per (ds, model, method)",
        "- `tables/paired_tralo_vs_<baseline>.csv` — per-cell (TraLO − baseline) deltas, same-seed paired",
        "- `tables/paired_summary.csv` — W/L/T + mean delta per (baseline, dataset, sym/asym) — paper-table-ready",
        "",
        "## How to use",
        "",
        "Filter the master CSV. E.g. with pandas:",
        "```python",
        "import pandas as pd",
        "df = pd.read_csv('archive/MASTER_INDEX.csv')",
        "# Headline: tissue MobileNetV3 sym tightness, all methods",
        "df[(df.dataset=='tissuemnist') & (df.model=='MobileNetV3') &",
        "   (df.is_asymmetric==0)].groupby('method').macro_f1.agg(['mean','std','count'])",
        "```",
        "",
        "Paired comparison (TraLO vs each baseline, same seed):",
        "```python",
        "key = ['sweep','dataset','model','constraint_tag','seed','data_dir']",
        "wide = df.pivot_table(index=key, columns='method', values='macro_f1')",
        "wide['d_vs_hounie']   = wide['tralo'] - wide['hounie_rcl']",
        "wide['d_vs_fioretto'] = wide['tralo'] - wide['fioretto_ldf']",
        "wide['d_vs_danits']   = wide['tralo'] - wide['danits_lp']",
        "```",
        "",
        "## Rebuilding",
        "",
        "```bash",
        "cd ~/OptimizationLoss && python scripts/build_archive.py",
        "```",
        "",
        "Idempotent — overwrites this archive on each run. Cell paths in MASTER_INDEX",
        "are relative to repo root; raw artifacts (config.json, evaluation_metrics.csv,",
        "final_predictions.csv, training_log.csv) remain in `results/pending_runs/...`.",
    ]
    (OUT/"README.md").write_text("\n".join(lines))
    print(f"WROTE {OUT/'README.md'}")


if __name__ == "__main__":
    main()
