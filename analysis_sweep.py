"""
Comprehensive analysis of TissueMNIST experiment sweep.
Compares our_approach (constraint optimization) vs heuristic (greedy top-K).
"""
import json
import csv
import os
from collections import defaultdict
from pathlib import Path

BASE = Path(r"C:\Users\roeym\Desktop\projects\OptimizationLoss\results\pending_runs")

MODELS = ["MobileNetV3", "EfficientNetB0", "ConvNeXtTiny"]
CONSTRAINTS = ["L80_G80", "L50_G50", "L20_G80", "L20_G50"]
HP_CONFIGS = ["kl05_lr5e6", "kl01_lr5e6", "kl00_lr5e6", "kl01_lr1e5"]
CLASS_NAMES = ["CDI", "CDS", "CST", "EPI", "GE", "PTC", "STR", "TUB"]
CONSTRAINED_CLASS = 4  # GE

def load_json(path):
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception:
        return None

def load_csv_rows(path):
    try:
        with open(path, "r") as f:
            reader = csv.DictReader(f)
            rows = []
            for row in reader:
                # Skip duplicate header rows embedded in data
                first_val = list(row.values())[0] if row else None
                first_key = list(row.keys())[0] if row else None
                if first_val == first_key:
                    continue
                rows.append(row)
            return rows
    except Exception:
        return None

def load_eval_metrics(path):
    try:
        with open(path, "r") as f:
            reader = csv.reader(f)
            header = next(reader)
            metrics = {}
            for row in reader:
                metrics[row[0]] = float(row[1])
            return metrics
    except Exception:
        return None


def get_experiment_path(model, constraint, hp_config=None):
    """Get path to experiment directory."""
    if hp_config is None:
        dirname = f"{model}_{constraint}_heuristic"
        return BASE / dirname / "heuristic"
    else:
        dirname = f"{model}_{constraint}_{hp_config}"
        return BASE / dirname / "our_approach"


def safe_float(v, default=None):
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


# ============================================================================
# SECTION 1: SUMMARY TABLE
# ============================================================================
def section1_summary_table():
    print("=" * 120)
    print("SECTION 1: SUMMARY TABLE -- Our Approach vs Heuristic")
    print("=" * 120)
    print()

    # Collect heuristic baselines
    heuristic_data = {}
    for model in MODELS:
        for constraint in CONSTRAINTS:
            path = get_experiment_path(model, constraint)
            cfg = load_json(path / "config.json")
            if cfg and cfg.get("status") == "completed" and cfg.get("results"):
                r = cfg["results"]
                heuristic_data[(model, constraint)] = {
                    "f1_macro": r.get("f1_macro"),
                    "accuracy": r.get("accuracy"),
                    "precision_macro": r.get("precision_macro"),
                    "recall_macro": r.get("recall_macro"),
                    "samples_adjusted": r.get("samples_adjusted", 0),
                }

    header = f"{'Experiment':<45} {'Status':<10} {'F1-M':>7} {'Acc':>7} {'F1-H':>7} {'Acc-H':>7} {'dF1':>7} {'dAcc':>7} {'Adj':>5} {'ChkSrc':>14} {'BrkEp':>6}"
    print(header)
    print("-" * 120)

    all_rows = []
    for hp_config in HP_CONFIGS:
        for model in MODELS:
            for constraint in CONSTRAINTS:
                exp_path = get_experiment_path(model, constraint, hp_config)
                cfg = load_json(exp_path / "config.json")
                exp_name = f"{model}_{constraint}_{hp_config}"

                if cfg is None:
                    print(f"{exp_name:<45} {'NO_CFG':<10}")
                    continue

                status = cfg.get("status", "unknown")
                if status != "completed":
                    print(f"{exp_name:<45} {status:<10}")
                    continue

                r = cfg.get("results", {})
                f1 = r.get("f1_macro")
                acc = r.get("accuracy")
                adj = r.get("samples_adjusted", "?")
                chk_src = r.get("checkpoint_source", "N/A")
                brk_ep = r.get("bracket_epoch", "N/A")

                h = heuristic_data.get((model, constraint), {})
                f1_h = h.get("f1_macro")
                acc_h = h.get("accuracy")

                df1 = (f1 - f1_h) if (f1 is not None and f1_h is not None) else None
                dacc = (acc - acc_h) if (acc is not None and acc_h is not None) else None

                row = {
                    "model": model, "constraint": constraint, "hp_config": hp_config,
                    "f1": f1, "acc": acc, "f1_h": f1_h, "acc_h": acc_h,
                    "df1": df1, "dacc": dacc, "adj": adj,
                    "chk_src": chk_src, "brk_ep": brk_ep,
                }
                all_rows.append(row)

                def fmt(v, w=7):
                    return f"{v:>{w}.4f}" if v is not None else f"{'N/A':>{w}}"

                sign_df1 = f"{df1:>+7.4f}" if df1 is not None else f"{'N/A':>7}"
                sign_dacc = f"{dacc:>+7.4f}" if dacc is not None else f"{'N/A':>7}"

                print(f"{exp_name:<45} {status:<10} {fmt(f1)} {fmt(acc)} {fmt(f1_h)} {fmt(acc_h)} {sign_df1} {sign_dacc} {str(adj):>5} {str(chk_src):>14} {str(brk_ep):>6}")
        print()

    return all_rows, heuristic_data


# ============================================================================
# SECTION 2: PER HP-CONFIG ANALYSIS
# ============================================================================
def section2_hp_analysis(all_rows):
    print()
    print("=" * 120)
    print("SECTION 2: PER HP-CONFIG ANALYSIS -- Average Deltas")
    print("=" * 120)
    print()

    hp_stats = defaultdict(lambda: {"df1_list": [], "dacc_list": [], "f1_list": [], "adj_list": []})

    for row in all_rows:
        hp = row["hp_config"]
        if row["df1"] is not None:
            hp_stats[hp]["df1_list"].append(row["df1"])
        if row["dacc"] is not None:
            hp_stats[hp]["dacc_list"].append(row["dacc"])
        if row["f1"] is not None:
            hp_stats[hp]["f1_list"].append(row["f1"])
        if row.get("adj") is not None and row["adj"] != "?":
            hp_stats[hp]["adj_list"].append(int(row["adj"]))

    print(f"{'HP Config':<20} {'N':>3} {'Mean dF1':>10} {'Mean dAcc':>10} {'Mean F1':>10} {'Min dF1':>10} {'Max dF1':>10} {'Mean Adj':>10}")
    print("-" * 85)

    best_hp = None
    best_df1 = -999
    for hp in HP_CONFIGS:
        s = hp_stats[hp]
        n = len(s["df1_list"])
        if n == 0:
            print(f"{hp:<20} {'0':>3} {'N/A':>10} {'N/A':>10} {'N/A':>10} {'N/A':>10} {'N/A':>10} {'N/A':>10}")
            continue
        mean_df1 = sum(s["df1_list"]) / n
        mean_dacc = sum(s["dacc_list"]) / len(s["dacc_list"]) if s["dacc_list"] else 0
        mean_f1 = sum(s["f1_list"]) / len(s["f1_list"]) if s["f1_list"] else 0
        min_df1 = min(s["df1_list"])
        max_df1 = max(s["df1_list"])
        mean_adj = sum(s["adj_list"]) / len(s["adj_list"]) if s["adj_list"] else 0

        if mean_df1 > best_df1:
            best_df1 = mean_df1
            best_hp = hp

        print(f"{hp:<20} {n:>3} {mean_df1:>+10.4f} {mean_dacc:>+10.4f} {mean_f1:>10.4f} {min_df1:>+10.4f} {max_df1:>+10.4f} {mean_adj:>10.1f}")

    print()
    print(f"  >>> BEST HP CONFIG by mean F1 delta: {best_hp} (mean dF1 = {best_df1:+.4f})")

    # Also break down by model
    print()
    print("  Breakdown by model:")
    print(f"  {'HP Config':<20} {'Model':<15} {'Mean dF1':>10} {'N':>3}")
    print("  " + "-" * 55)
    for hp in HP_CONFIGS:
        for model in MODELS:
            vals = [r["df1"] for r in all_rows if r["hp_config"] == hp and r["model"] == model and r["df1"] is not None]
            if vals:
                mean = sum(vals) / len(vals)
                print(f"  {hp:<20} {model:<15} {mean:>+10.4f} {len(vals):>3}")

    # Breakdown by constraint
    print()
    print("  Breakdown by constraint pair:")
    print(f"  {'HP Config':<20} {'Constraint':<12} {'Mean dF1':>10} {'N':>3}")
    print("  " + "-" * 50)
    for hp in HP_CONFIGS:
        for constraint in CONSTRAINTS:
            vals = [r["df1"] for r in all_rows if r["hp_config"] == hp and r["constraint"] == constraint and r["df1"] is not None]
            if vals:
                mean = sum(vals) / len(vals)
                print(f"  {hp:<20} {constraint:<12} {mean:>+10.4f} {len(vals):>3}")

    return best_hp


# ============================================================================
# SECTION 3: TRAINING PROGRESS
# ============================================================================
def section3_training_progress():
    print()
    print("=" * 120)
    print("SECTION 3: TRAINING PROGRESS ANALYSIS")
    print("=" * 120)
    print()

    header = (f"{'Experiment':<45} {'Epochs':>6} {'1stSat':>6} {'ConvEp':>6} {'Converged':>9} "
              f"{'LamToggles':>10} {'MaxLamG':>8} {'MaxLamL':>8} {'BrkBest':>7} {'BrkPrev':>7}")
    print(header)
    print("-" * 130)

    for hp_config in HP_CONFIGS:
        for model in MODELS:
            for constraint in CONSTRAINTS:
                exp_name = f"{model}_{constraint}_{hp_config}"
                exp_path = get_experiment_path(model, constraint, hp_config)

                cfg = load_json(exp_path / "config.json")
                if cfg is None or cfg.get("status") != "completed":
                    status = cfg.get("status", "no_cfg") if cfg else "no_cfg"
                    print(f"{exp_name:<45} {'(' + status + ')'}")
                    continue

                rows = load_csv_rows(exp_path / "training_log.csv")
                if not rows:
                    print(f"{exp_name:<45} (no training log)")
                    continue

                epochs = [int(r["Epoch"]) for r in rows]
                n_epochs = len(epochs)
                max_epoch = max(epochs) if epochs else 0

                # Track constraint satisfaction
                first_satisfied_epoch = None
                consecutive_satisfied = 0
                convergence_epoch = None
                lambda_toggles = 0
                prev_global_sat = None

                max_lam_g = 0
                max_lam_l = 0

                for r in rows:
                    g_sat = int(r.get("Global_Satisfied", 0))
                    l_sat = int(r.get("Local_Satisfied", 0))
                    both_sat = (g_sat == 1 and l_sat == 1)

                    lam_g = safe_float(r.get("Lambda_Global"), 0)
                    lam_l = safe_float(r.get("Lambda_Local"), 0)
                    max_lam_g = max(max_lam_g, lam_g)
                    max_lam_l = max(max_lam_l, lam_l)

                    if both_sat and first_satisfied_epoch is None:
                        first_satisfied_epoch = int(r["Epoch"])

                    if both_sat:
                        consecutive_satisfied += 1
                        if consecutive_satisfied >= 5 and convergence_epoch is None:
                            convergence_epoch = int(r["Epoch"])
                    else:
                        consecutive_satisfied = 0

                    # Lambda toggle: satisfaction flips
                    if prev_global_sat is not None and g_sat != prev_global_sat:
                        lambda_toggles += 1
                    prev_global_sat = g_sat

                # Bracket info from config
                r = cfg.get("results", {})
                chk_src = r.get("checkpoint_source", "N/A")
                brk_ep = r.get("bracket_epoch", "N/A")
                rc = cfg.get("results_comparison", {})
                brk_best_f1 = rc.get("bracket_best", {}).get("f1_macro")
                brk_prev_f1 = rc.get("bracket_previous", {}).get("f1_macro")

                converged_str = "YES" if convergence_epoch else "NO"
                first_sat_str = str(first_satisfied_epoch) if first_satisfied_epoch else "NEVER"
                conv_ep_str = str(convergence_epoch) if convergence_epoch else "-"

                brk_best_str = f"{brk_best_f1:.4f}" if brk_best_f1 else "N/A"
                brk_prev_str = f"{brk_prev_f1:.4f}" if brk_prev_f1 else "N/A"

                print(f"{exp_name:<45} {n_epochs:>6} {first_sat_str:>6} {conv_ep_str:>6} {converged_str:>9} "
                      f"{lambda_toggles:>10} {max_lam_g:>8.3f} {max_lam_l:>8.3f} {brk_best_str:>7} {brk_prev_str:>7}")
        print()


# ============================================================================
# SECTION 4: PER-CLASS PRECISION/RECALL COMPARISON
# ============================================================================
def section4_per_class_analysis(best_hp, heuristic_data):
    print()
    print("=" * 120)
    print(f"SECTION 4: PER-CLASS PRECISION/RECALL -- Best HP Config: {best_hp}")
    print("=" * 120)
    print()

    def compute_per_class_metrics(preds_rows, n_classes=8):
        """Compute per-class precision, recall, F1 from final_predictions.csv rows."""
        tp = [0] * n_classes
        fp = [0] * n_classes
        fn = [0] * n_classes
        total = [0] * n_classes

        for row in preds_rows:
            true_label = int(row["True_Label"])
            pred_label = int(row["Predicted_Label"])
            total[true_label] += 1
            if true_label == pred_label:
                tp[true_label] += 1
            else:
                fp[pred_label] += 1
                fn[true_label] += 1

        metrics = {}
        for c in range(n_classes):
            prec = tp[c] / (tp[c] + fp[c]) if (tp[c] + fp[c]) > 0 else 0
            rec = tp[c] / (tp[c] + fn[c]) if (tp[c] + fn[c]) > 0 else 0
            f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
            metrics[c] = {"precision": prec, "recall": rec, "f1": f1, "support": total[c],
                          "predicted_count": tp[c] + fp[c]}
        return metrics

    for model in MODELS:
        for constraint in CONSTRAINTS:
            exp_path = get_experiment_path(model, constraint, best_hp)
            heur_path = get_experiment_path(model, constraint)

            cfg = load_json(exp_path / "config.json")
            if cfg is None or cfg.get("status") != "completed":
                continue

            our_preds = load_csv_rows(exp_path / "final_predictions.csv")
            heur_preds = load_csv_rows(heur_path / "final_predictions.csv")

            if not our_preds or not heur_preds:
                continue

            our_metrics = compute_per_class_metrics(our_preds)
            heur_metrics = compute_per_class_metrics(heur_preds)

            print(f"  {model} / {constraint}")
            print(f"  {'Class':<6} {'Prec-O':>7} {'Prec-H':>7} {'dPrec':>7} {'Rec-O':>7} {'Rec-H':>7} {'dRec':>7} {'F1-O':>7} {'F1-H':>7} {'dF1':>7} {'PredCnt-O':>10} {'PredCnt-H':>10} {'Support':>8}")
            print("  " + "-" * 110)

            for c in range(8):
                o = our_metrics[c]
                h = heur_metrics[c]
                dp = o["precision"] - h["precision"]
                dr = o["recall"] - h["recall"]
                df = o["f1"] - h["f1"]
                marker = " ***" if c == CONSTRAINED_CLASS else ""
                print(f"  {CLASS_NAMES[c]:<6} {o['precision']:>7.4f} {h['precision']:>7.4f} {dp:>+7.4f} "
                      f"{o['recall']:>7.4f} {h['recall']:>7.4f} {dr:>+7.4f} "
                      f"{o['f1']:>7.4f} {h['f1']:>7.4f} {df:>+7.4f} "
                      f"{o['predicted_count']:>10} {h['predicted_count']:>10} {o['support']:>8}{marker}")

            # Macro averages
            our_f1_macro = sum(our_metrics[c]["f1"] for c in range(8)) / 8
            heur_f1_macro = sum(heur_metrics[c]["f1"] for c in range(8)) / 8
            print(f"  {'MACRO':<6} {'':>7} {'':>7} {'':>7} {'':>7} {'':>7} {'':>7} "
                  f"{our_f1_macro:>7.4f} {heur_f1_macro:>7.4f} {our_f1_macro - heur_f1_macro:>+7.4f}")
            print()

    # Summary: average GE-class delta across all model/constraint combos
    print("  SUMMARY: GE (constrained class) impact across models/constraints:")
    print(f"  {'Model':<15} {'Constraint':<12} {'GE F1-O':>8} {'GE F1-H':>8} {'dGE-F1':>8} {'Other F1-O':>10} {'Other F1-H':>10} {'dOther':>8}")
    print("  " + "-" * 85)

    ge_deltas = []
    other_deltas = []

    for model in MODELS:
        for constraint in CONSTRAINTS:
            exp_path = get_experiment_path(model, constraint, best_hp)
            heur_path = get_experiment_path(model, constraint)

            our_preds = load_csv_rows(exp_path / "final_predictions.csv")
            heur_preds = load_csv_rows(heur_path / "final_predictions.csv")

            if not our_preds or not heur_preds:
                continue

            our_m = compute_per_class_metrics(our_preds)
            heur_m = compute_per_class_metrics(heur_preds)

            ge_f1_o = our_m[CONSTRAINED_CLASS]["f1"]
            ge_f1_h = heur_m[CONSTRAINED_CLASS]["f1"]
            dge = ge_f1_o - ge_f1_h

            other_f1_o = sum(our_m[c]["f1"] for c in range(8) if c != CONSTRAINED_CLASS) / 7
            other_f1_h = sum(heur_m[c]["f1"] for c in range(8) if c != CONSTRAINED_CLASS) / 7
            dother = other_f1_o - other_f1_h

            ge_deltas.append(dge)
            other_deltas.append(dother)

            print(f"  {model:<15} {constraint:<12} {ge_f1_o:>8.4f} {ge_f1_h:>8.4f} {dge:>+8.4f} "
                  f"{other_f1_o:>10.4f} {other_f1_h:>10.4f} {dother:>+8.4f}")

    if ge_deltas:
        print()
        print(f"  Average GE F1 delta: {sum(ge_deltas)/len(ge_deltas):+.4f}")
        print(f"  Average Other F1 delta (spillover): {sum(other_deltas)/len(other_deltas):+.4f}")


# ============================================================================
# SECTION 5: SAMPLE-LEVEL ANALYSIS
# ============================================================================
def section5_sample_analysis(best_hp, all_rows):
    print()
    print("=" * 120)
    print("SECTION 5: SAMPLE-LEVEL ANALYSIS -- GE Predictions Overlap")
    print("=" * 120)
    print()

    # Find the best-performing completed experiment for best_hp
    completed = [r for r in all_rows if r["hp_config"] == best_hp and r["f1"] is not None]
    if not completed:
        print("  No completed experiments for best HP config.")
        return

    best_row = max(completed, key=lambda r: r["f1"])
    model = best_row["model"]
    constraint = best_row["constraint"]
    print(f"  Best experiment: {model}_{constraint}_{best_hp} (F1={best_row['f1']:.4f})")
    print()

    exp_path = get_experiment_path(model, constraint, best_hp)
    heur_path = get_experiment_path(model, constraint)

    our_preds = load_csv_rows(exp_path / "final_predictions.csv")
    heur_preds = load_csv_rows(heur_path / "final_predictions.csv")

    if not our_preds or not heur_preds:
        print("  Missing prediction files.")
        return

    # Build sample index -> prediction
    n_samples = len(our_preds)
    our_ge = set()
    heur_ge = set()
    true_ge = set()

    for i in range(n_samples):
        if int(our_preds[i]["Predicted_Label"]) == CONSTRAINED_CLASS:
            our_ge.add(i)
        if int(heur_preds[i]["Predicted_Label"]) == CONSTRAINED_CLASS:
            heur_ge.add(i)
        if int(our_preds[i]["True_Label"]) == CONSTRAINED_CLASS:
            true_ge.add(i)

    overlap = our_ge & heur_ge
    only_ours = our_ge - heur_ge
    only_heur = heur_ge - our_ge

    print(f"  Total test samples: {n_samples}")
    print(f"  True GE samples: {len(true_ge)}")
    print(f"  Our approach predicted GE: {len(our_ge)}")
    print(f"  Heuristic predicted GE: {len(heur_ge)}")
    print()
    print(f"  Overlap (both predict GE): {len(overlap)}")
    print(f"    - Of these, actually GE: {len(overlap & true_ge)} ({len(overlap & true_ge)/len(overlap)*100:.1f}%)" if overlap else "")
    print(f"  Only our approach predicts GE: {len(only_ours)}")
    print(f"    - Of these, actually GE: {len(only_ours & true_ge)} ({len(only_ours & true_ge)/len(only_ours)*100:.1f}%)" if only_ours else "")
    print(f"  Only heuristic predicts GE: {len(only_heur)}")
    print(f"    - Of these, actually GE: {len(only_heur & true_ge)} ({len(only_heur & true_ge)/len(only_heur)*100:.1f}%)" if only_heur else "")
    print()

    # Confidence analysis: for the unique picks, what's the avg GE probability?
    def avg_ge_prob(indices, preds):
        if not indices:
            return 0
        return sum(float(preds[i]["Prob_Class_4"]) for i in indices) / len(indices)

    print(f"  Avg GE probability for overlapping samples (ours): {avg_ge_prob(overlap, our_preds):.4f}")
    print(f"  Avg GE probability for overlapping samples (heur): {avg_ge_prob(overlap, heur_preds):.4f}")
    print(f"  Avg GE probability for only-ours picks (ours): {avg_ge_prob(only_ours, our_preds):.4f}")
    print(f"  Avg GE probability for only-heur picks (heur): {avg_ge_prob(only_heur, heur_preds):.4f}")
    print()

    # What did the other method predict for the unique picks?
    if only_ours:
        pred_dist = defaultdict(int)
        for i in only_ours:
            pred_dist[int(heur_preds[i]["Predicted_Label"])] += 1
        print(f"  For samples ONLY our approach predicted as GE, heuristic predicted:")
        for c in sorted(pred_dist.keys()):
            print(f"    {CLASS_NAMES[c]}: {pred_dist[c]}")

    if only_heur:
        pred_dist = defaultdict(int)
        for i in only_heur:
            pred_dist[int(our_preds[i]["Predicted_Label"])] += 1
        print(f"  For samples ONLY heuristic predicted as GE, our approach predicted:")
        for c in sorted(pred_dist.keys()):
            print(f"    {CLASS_NAMES[c]}: {pred_dist[c]}")

    # Repeat for ALL model/constraint combos with best_hp
    print()
    print("  AGGREGATE across all models/constraints (best HP config):")
    print(f"  {'Model':<15} {'Constraint':<12} {'Our#GE':>6} {'Heur#GE':>7} {'Overlap':>7} {'OnlyOurs':>8} {'OnlyHeur':>8} "
          f"{'OursCorrect%':>12} {'HeurCorrect%':>12}")
    print("  " + "-" * 100)

    for model in MODELS:
        for constraint in CONSTRAINTS:
            exp_path = get_experiment_path(model, constraint, best_hp)
            heur_path = get_experiment_path(model, constraint)
            cfg = load_json(exp_path / "config.json")
            if not cfg or cfg.get("status") != "completed":
                continue

            our_p = load_csv_rows(exp_path / "final_predictions.csv")
            heur_p = load_csv_rows(heur_path / "final_predictions.csv")
            if not our_p or not heur_p:
                continue

            n = len(our_p)
            o_ge = {i for i in range(n) if int(our_p[i]["Predicted_Label"]) == CONSTRAINED_CLASS}
            h_ge = {i for i in range(n) if int(heur_p[i]["Predicted_Label"]) == CONSTRAINED_CLASS}
            t_ge = {i for i in range(n) if int(our_p[i]["True_Label"]) == CONSTRAINED_CLASS}

            ovlp = o_ge & h_ge
            oo = o_ge - h_ge
            oh = h_ge - o_ge

            oo_correct = len(oo & t_ge) / len(oo) * 100 if oo else 0
            oh_correct = len(oh & t_ge) / len(oh) * 100 if oh else 0

            print(f"  {model:<15} {constraint:<12} {len(o_ge):>6} {len(h_ge):>7} {len(ovlp):>7} {len(oo):>8} {len(oh):>8} "
                  f"{oo_correct:>11.1f}% {oh_correct:>11.1f}%")


# ============================================================================
# SECTION 6: CHECKPOINT ANALYSIS
# ============================================================================
def section6_checkpoint_analysis():
    print()
    print("=" * 120)
    print("SECTION 6: CHECKPOINT / BRACKET ANALYSIS")
    print("=" * 120)
    print()

    print(f"{'Experiment':<45} {'ChkSrc':<16} {'BrkEp':>6} {'Final F1':>9} {'BrkBest F1':>11} {'BrkPrev F1':>11} {'Best-Final':>11} {'Adj-F':>6} {'Adj-B':>6} {'Adj-P':>6}")
    print("-" * 140)

    has_bracket_data = False
    src_counts = defaultdict(int)
    improvements = []

    for hp_config in HP_CONFIGS:
        for model in MODELS:
            for constraint in CONSTRAINTS:
                exp_name = f"{model}_{constraint}_{hp_config}"
                exp_path = get_experiment_path(model, constraint, hp_config)
                cfg = load_json(exp_path / "config.json")

                if cfg is None or cfg.get("status") != "completed":
                    continue

                rc = cfg.get("results_comparison", {})
                r = cfg.get("results", {})
                chk_src = r.get("checkpoint_source", "N/A")
                brk_ep = r.get("bracket_epoch", "N/A")
                src_counts[chk_src] += 1

                final = rc.get("final", {})
                brk_best = rc.get("bracket_best", {})
                brk_prev = rc.get("bracket_previous", {})

                f1_final = final.get("f1_macro")
                f1_best = brk_best.get("f1_macro")
                f1_prev = brk_prev.get("f1_macro")

                adj_f = final.get("adjusted", "?")
                adj_b = brk_best.get("adjusted", "?")
                adj_p = brk_prev.get("adjusted", "?")

                if f1_best is not None and f1_final is not None:
                    improvement = f1_best - f1_final
                    improvements.append(improvement)
                    has_bracket_data = True

                    def fmt(v):
                        return f"{v:.4f}" if v is not None else "N/A"

                    imp_str = f"{improvement:+.4f}" if improvement is not None else "N/A"

                    print(f"{exp_name:<45} {chk_src:<16} {str(brk_ep):>6} {fmt(f1_final):>9} {fmt(f1_best):>11} "
                          f"{fmt(f1_prev):>11} {imp_str:>11} {str(adj_f):>6} {str(adj_b):>6} {str(adj_p):>6}")

    print()
    print("  Checkpoint source distribution:")
    for src, cnt in sorted(src_counts.items()):
        print(f"    {src}: {cnt}")

    if improvements:
        pos = [x for x in improvements if x > 0]
        neg = [x for x in improvements if x < 0]
        zero = [x for x in improvements if x == 0]
        print()
        print(f"  Bracket best vs final:")
        print(f"    Bracket better: {len(pos)} experiments (avg improvement: {sum(pos)/len(pos):+.4f})" if pos else "    Bracket better: 0 experiments")
        print(f"    Final better: {len(neg)} experiments (avg: {sum(neg)/len(neg):+.4f})" if neg else "    Final better: 0 experiments")
        print(f"    Equal: {len(zero)} experiments")
        print(f"    Overall mean bracket-best minus final: {sum(improvements)/len(improvements):+.4f}")


# ============================================================================
# BONUS: HEURISTIC BASELINE SUMMARY
# ============================================================================
def section_bonus_heuristic_summary(heuristic_data):
    print()
    print("=" * 120)
    print("BONUS: HEURISTIC BASELINE SUMMARY")
    print("=" * 120)
    print()

    print(f"{'Model':<15} {'Constraint':<12} {'F1-Macro':>9} {'Accuracy':>9} {'Precision':>9} {'Recall':>9} {'Adjusted':>9}")
    print("-" * 75)

    for model in MODELS:
        for constraint in CONSTRAINTS:
            h = heuristic_data.get((model, constraint))
            if h:
                print(f"{model:<15} {constraint:<12} {h['f1_macro']:>9.4f} {h['accuracy']:>9.4f} "
                      f"{h['precision_macro']:>9.4f} {h['recall_macro']:>9.4f} {h['samples_adjusted']:>9}")


# ============================================================================
# MAIN
# ============================================================================
if __name__ == "__main__":
    print()
    print("#" * 120)
    print("#  COMPREHENSIVE ANALYSIS: TissueMNIST Experiment Sweep")
    print(f"#  {len(MODELS)} models x {len(CONSTRAINTS)} constraints x {len(HP_CONFIGS)} HP configs = {len(MODELS)*len(CONSTRAINTS)*len(HP_CONFIGS)} our_approach + {len(MODELS)*len(CONSTRAINTS)} heuristic")
    print("#" * 120)
    print()

    all_rows, heuristic_data = section1_summary_table()

    section_bonus_heuristic_summary(heuristic_data)

    best_hp = section2_hp_analysis(all_rows)

    section3_training_progress()

    section4_per_class_analysis(best_hp, heuristic_data)

    section5_sample_analysis(best_hp, all_rows)

    section6_checkpoint_analysis()

    print()
    print("=" * 120)
    print("ANALYSIS COMPLETE")
    print("=" * 120)
