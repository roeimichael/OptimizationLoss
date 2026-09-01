"""STAGE 5 -- WHAT THE OPTIMISER ACTUALLY DID, from `training_log.csv` and
`config.json`, DURING and AFTER a run. The bucket that catches CE saturation, a
lost dose, terminal collapse and divergence. Sources are cited per gate:
reachability.py, log_health.py, diagnose_run.py, dose_landed.py, FRAMEWORK 2(u)
and 3(0c), PLAYBOOK 2 and 3, REJECTED_full, audit_config.py.

The detectors are pure functions here so the gate is self-contained and runs in
CI with no dataset and no run directory. Every one of them separates a MEASURED
broken input from a MEASURED healthy one -- a gate that has never failed has
never been shown to work (conftest, rule 2).
"""
import ast
import glob
import io
import json
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from conftest import rel, report                                # noqa: E402

pytestmark = pytest.mark.stage5_trainlog

# Every constant is lifted from the tool that measured it, not invented.
SATURATED_ACC, FLAT_GAIN = 0.93, 0.005    # log_health
CE_FLOOR = 0.05                           # diagnose_run, section 4
COLLAPSE_DROP = 0.02                      # log_health, ~10x a converged wobble
REACHABLE = 0.040                         # reachability
DOSE_TOL = 0.05                           # dose_landed.DOSE_FRACTION_TOLERANCE
WIDE = ["Epoch", "Train_Acc", "L_CE", "L_Global", "L_Local", "Grad_Norm",
        "Lambda_Global", "Lambda_Local", "Global_Satisfied", "Local_Satisfied",
        "Limit_Class2", "Hard_Class2", "Soft_Class2"]      # src/training/logging.py
NARROW = ["epoch", "train_acc", "ce_loss", "constraint_loss", "total_excess",
          "all_satisfied", "max_lambda_g"]                 # fioretto_ldf/train.py


def write_run(tmp, name, acc, ce=None, **cfg):
    """A real run directory on disk, so the detectors read what the scripts do.
    conftest forbids a FALLBACK to synthetic data; a toy built ON PURPOSE as a
    negative control is the only way to show a gate can fail."""
    d = os.path.join(str(tmp), name)
    os.makedirs(d, exist_ok=True)
    n = len(acc)
    row = dict({c: [0.0] * n for c in WIDE},
               Epoch=list(range(1, n + 1)), Train_Acc=acc,
               L_CE=ce if ce is not None else [0.5] * n,
               Lambda_Local=[0.01] * n, Limit_Class2=[352.0] * n,
               Hard_Class2=[400.0] * n)
    pd.DataFrame({c: row[c] for c in WIDE}).to_csv(
        os.path.join(d, "training_log.csv"), index=False)
    hp = dict({"constraint_epochs": 29, "lambda_local": 0.01},
              **cfg.pop("hyperparams", {}))
    io.open(os.path.join(d, "config.json"), "w", encoding="utf-8").write(
        json.dumps(dict({"status": "completed", "hyperparams": hp,
                         "results": {}}, **cfg)))
    return d


def read_run(d):
    return (pd.read_csv(os.path.join(d, "training_log.csv")),
            json.load(io.open(os.path.join(d, "config.json"), encoding="utf-8")))


# --- the detectors. The "why" for each lives in the gate that exercises it. ---
def saturation_verdict(d):
    """log_health (both halves) + diagnose_run 4 (the CE tell) + the post-hoc
    exemption: a clipper runs no constraint phase for saturation to gate off."""
    df, cfg = read_run(d)
    if (cfg.get("hyperparams") or {}).get("constraint_epochs", 0) == 0:
        return "post_hoc"
    a = pd.to_numeric(df["Train_Acc"], errors="coerce").dropna()
    c = pd.to_numeric(df["L_CE"], errors="coerce").dropna()
    both = float(a.iloc[0]) >= SATURATED_ACC and abs(
        float(a.iloc[-1]) - float(a.iloc[0])) <= FLAT_GAIN
    return "SATURATED" if (both or float(c.iloc[-1]) < CE_FLOOR) else "live"


def cut_slope(p):
    """`sum`'s per-item gradient at the cut. reachability.slope_at, mode=sum."""
    return p * (1.0 - p)


def dose_problems(per):
    """dose_landed.report as strings. `per` is arm -> (applied, attempted)."""
    out, frac = [], {}
    for arm, (app, att) in sorted(per.items()):
        if att <= 0:
            continue
        frac[arm] = app / float(att)
        if app != att:
            out.append("%s lost %d of %d steps" % (arm, att - app, att))
    if len(frac) > 1 and max(frac.values()) - min(frac.values()) > DOSE_TOL:
        out.append("arms did not run at the same dose")
    return out


def dose_diagnosis(per):
    """ONE arm low = the loss shape. EVERY arm low = the host. dose_landed."""
    frac = {a: app / float(att) for a, (app, att) in per.items() if att > 0}
    if all(f >= 1.0 for f in frac.values()):
        return "ok"
    low = [a for a in frac if frac[a] < max(frac.values()) - DOSE_TOL]
    return "loss_shape" if len(low) == 1 else "host"


def fp16_signature(frac, amp):
    """FP16 + GradScaler skips an overflowing step at roughly 25-31%; BF16 has
    float32's exponent range and does not. FRAMEWORK 2(u)."""
    return amp == "float16" and 0.24 <= 1.0 - frac <= 0.33


def schema_kind(df):
    """tralo* write the capitalised wide schema, the duals a lowercase narrow
    one -- 76 / 16 / 15 / 14 / 34 columns. FRAMEWORK 3(0c)."""
    return ("wide" if "Train_Acc" in df.columns
            else "narrow" if "train_acc" in df.columns else "unknown")


def comparable(dfs):
    """Two logs are comparable only if their columns match."""
    return len({tuple(df.columns) for df in dfs}) == 1


def count_source_ok(source, question):
    """PLAYBOOK 3(a),(b): the log is never a count source; `_raw` is the MODEL's
    argmax and not budget-equalized; the plain file is post-allocator, exactly K."""
    return {("final_predictions_raw.csv", "model"): True,
            ("final_predictions.csv", "allocator"): True}.get(
                (source, question), False)


def terminal_collapse(acc):
    """The pipeline keeps the LAST epoch, so this IS the scored model."""
    a = [float(x) for x in acc]
    return (a[-2], a[-1]) if len(a) >= 2 and a[-1] < a[-2] - COLLAPSE_DROP else None


def late_move_z(series):
    """How far the last epoch moved, in the run's OWN within-run sds."""
    v = np.asarray([float(x) for x in series], dtype=float)
    body = v[:-1]
    sd = float(np.std(body, ddof=1))
    if sd <= 0:
        return float("inf") if v[-1] != body[-1] else 0.0
    return abs(v[-1] - float(np.median(body))) / sd


def nonfinite_cols(df):
    """log_health: an ALL-blank column was NOT LOGGED and the warm-up row
    predates the constraint object. Only a NaN beside real values diverged."""
    scan = df[pd.to_numeric(df["Epoch"], errors="coerce") >= 2]
    bad = {}
    for c in scan.select_dtypes(include=[np.number]).columns:
        f = np.isfinite(scan[c].to_numpy(dtype=float))
        if f.any() and not f.all():
            bad[c] = int((~f).sum())
    return bad


def config_keys_read(root):
    """Every literal config key the code actually READS, via the AST."""
    keys = set()

    class V(ast.NodeVisitor):
        def visit_Subscript(self, n):
            self._lit([n.slice])
            self.generic_visit(n)

        def visit_Call(self, n):
            f = n.func
            if (f.attr if isinstance(f, ast.Attribute)
                    else getattr(f, "id", None)) in ("get", "pop", "_required"):
                self._lit(n.args[:2])
            self.generic_visit(n)

        def _lit(self, nodes):
            keys.update(a.value for a in nodes
                        if isinstance(a, ast.Constant)
                        and isinstance(a.value, str))

    for r, dirs, files in os.walk(root):
        dirs[:] = [x for x in dirs if x != "__pycache__"]
        for f in (x for x in files if x.endswith(".py")):
            V().visit(ast.parse(io.open(os.path.join(r, f),
                                        encoding="utf-8").read()))
    return keys


# ================================ THE GATES ================================
def test_ce_saturation_is_refused_and_warmup_1_survives(tmp_path):
    """GATE 1, the strongest here. Warm-up 50 saturates CE, the gradient gates
    off and every method becomes identical: the REGIME is worth ~8 pp and the
    METHOD ~0.1 pp (REJECTED_full 9, 69 -- train acc 0.998 with `L_CE` 0.0 at
    the last epoch in 150/150 runs). NEGATIVE CONTROL, and the whole point: a
    HEALTHY iwildcam warm-up-1 run ALSO starts at 0.956, so a gate on accuracy
    alone refuses the only live regime. Both halves; post-hoc arms exempt."""
    hi, live = [0.998] * 6, [0.956, 0.962, 0.968, 0.971, 0.980, 0.986]
    bad = []
    for name, acc, ce, n_con, want in [
            ("warmup1_iwildcam", live, [.42, .36, .30, .25, .21, .18], 29, "live"),
            ("warmup50_dead_regime", hi, [0.004] * 6, 29, "SATURATED"),
            ("high_acc_still_moving", [.94, .95, .96, .96, .97, .97],
             [.30, .27, .22, .19, .15, .12], 29, "live"),
            ("flat_but_unconverged", [0.71] * 6, [0.90] * 5 + [0.88], 29, "live"),
            ("ce_floor_only", [.88, .881, .882, .883, .884, .885],
             [.60, .31, .11, .03, .01, .004], 29, "SATURATED"),
            ("posthoc_clip", hi, [0.004] * 6, 0, "post_hoc")]:
        got = saturation_verdict(write_run(
            tmp_path, name, acc, ce, hyperparams={"constraint_epochs": n_con}))
        if got != want:
            bad.append("%s: verdict %s, expected %s" % (name, got, want))
    # The mechanism, and why the log-level proxy is only a POINTER: what decides
    # reachability is p(1-p) at the cut. reachability.py, dermmnist x ViTB16.
    for label, p, want in [("L50_G30 warm-up, 4/4 seeds responded", .9389, True),
                           ("L30_G20 warm-up, 0/4 seeds responded", .9730, False),
                           ("L50_G30 at 30 epochs, converged", .9990, False)]:
        if (cut_slope(p) >= REACHABLE) != want:
            bad.append("%s: p(1-p)=%.4f vs REACHABLE=%.3f, expected reachable=%s"
                       % (label, cut_slope(p), REACHABLE, want))
    ratio = cut_slope(0.9389) / cut_slope(0.9990)
    if not 55.0 <= ratio <= 70.0:
        bad.append("converging must drop p(1-p) at the cut ~60x, got %.1fx" % ratio)
    report(bad, "CE-saturation gate failures")


def test_a_completed_run_can_have_landed_three_percent_of_its_dose(tmp_path):
    """GATE 2. `finish_constraint_step` drops an update whose constraint
    gradient is non-finite: the epoch ran, nothing landed, the run still writes
    `status: completed`. FRAMEWORK 2(u). NEGATIVE CONTROL: two run directories
    on disk, identical in `status`, at 1/29 and 29/29 -- `status` cannot
    separate them and the gate must."""
    bad = []
    for name, per, want in [
            ("uniform1: 1/29 beside 29/29", {"tralo": (29, 29),
             "tralo_head": (29, 29), "tralo_uniform": (1, 29)}, 2),
            ("iwc3: 716/1044", {"tralo": (716, 1044)}, 1),
            ("taskwin1, no --constraint-fp32: 20/29", {"tralo": (20, 29)}, 1),
            ("taskwin2, --constraint-fp32: 29/29", {"tralo": (29, 29)}, 0),
            ("dom1: every arm 29/29", {"tralo": (29, 29), "alm": (29, 29)}, 0)]:
        n = len(dose_problems(per))
        if n != want:
            bad.append("%s: %d problem(s), expected exactly %d" % (name, n, want))
    got = {}
    for k, applied in (("underdosed", 1), ("dosed", 29)):
        cfg = read_run(write_run(tmp_path, k, [0.95] * 4, results={
            "constraint_steps_applied": applied,
            "constraint_steps_attempted": 29}))[1]
        if cfg["status"] != "completed":
            bad.append("the control is void: %s must report `completed`" % k)
        got[k] = len(dose_problems({"tralo": (
            cfg["results"]["constraint_steps_applied"],
            cfg["results"]["constraint_steps_attempted"])}))
    if got["underdosed"] == 0:
        bad.append("a 1/29 run reporting `completed` was passed as healthy")
    if got["dosed"] != 0:
        bad.append("a 29/29 run was refused: the gate does not separate")
    report(bad, "dose-landed gate failures")


def test_one_arm_low_is_the_loss_shape_every_arm_low_is_the_host():
    """GATE 3. The two diagnoses have opposite fixes -- clamp a probability, or
    move host / set `--constraint-fp32` -- so they must not be conflated; `amp`
    tells them apart. FRAMEWORK 2(u), PLAYBOOK 2. NEGATIVE CONTROL: the same
    1/29 fraction moved from ONE arm to EVERY arm must flip `loss_shape` to
    `host`, and the bfloat16 rows must NOT match the fp16 signature."""
    bad = []
    for name, per, want in [
            ("uniform1: tralo_uniform alone at 3.4%", {"tralo": (29, 29),
             "tralo_head": (29, 29), "tralo_uniform": (1, 29)}, "loss_shape"),
            ("iwc3: both arms ~68.6%, agreeing with each other",
             {"tralo": (716, 1044), "alm": (720, 1044)}, "host"),
            ("xfam1: bfloat16, nothing lost",
             {"tralo": (29, 29), "fioretto": (29, 29)}, "ok"),
            ("the loss moved to EVERY arm", {"tralo": (1, 29),
             "tralo_head": (1, 29), "tralo_uniform": (1, 29)}, "host")]:
        got = dose_diagnosis(per)
        if got != want:
            bad.append("%s: diagnosed %s, expected %s" % (name, got, want))
    for name, frac, amp, want in [
            ("iwc1", 0.688, "float16", True), ("iwc2", 0.746, "float16", True),
            ("iwc3", 0.686, "float16", True),
            ("taskwin1", 20 / 29.0, "float16", True),
            ("uniform1 tralo_uniform", 1 / 29.0, "bfloat16", False),
            ("dom1", 1.0, "bfloat16", False)]:
        if fp16_signature(frac, amp) != want:
            bad.append("%s: fp16 signature %s at %.3f on %s, expected %s"
                       % (name, not want, frac, amp, want))
    report(bad, "loss-shape vs host diagnosis failures")


def test_the_log_is_never_the_state(tmp_path):
    """GATE 4. The arms write different SCHEMAS and, for every TRAINED arm, the
    last logged `Hard_Class2` disagrees with the predictions (`alm` logs 340,
    emits 467, 0/24) while both nulls agree 24/24 -- reading that table gave the
    EXACT OPPOSITE of the truth on dom1 (FRAMEWORK 3(0c), PLAYBOOK 3(a),(b)).
    Same defect class: `Lambda_Local` was a hardcoded 0.0 in older worktrees
    while the real init was 0.01, so the column described the FORMATTER.
    NEGATIVE CONTROLS: the nulls DO agree, making the disagreement specific to
    arms taking constraint steps -- yet they are still refused as a source,
    because the rule is about the SOURCE; and a genuine lambda=0 twin, where log
    and config agree on 0.0, must read as off rather than as a display bug."""
    bad, K = [], 352
    wide, narrow = pd.DataFrame(columns=WIDE), pd.DataFrame(columns=NARROW)
    for name, dfs, want in [("tralo vs fioretto", [wide, narrow], False),
                            ("tralo vs tralo", [wide, wide.copy()], True),
                            ("fioretto vs hounie", [narrow, narrow.copy()], True)]:
        if comparable(dfs) != want:
            bad.append("%s: comparable=%s, expected %s" % (name, not want, want))
    if schema_kind(wide) != "wide" or schema_kind(narrow) != "narrow":
        bad.append("the two shipped schemas are no longer recognised")

    logged = {"tralo_null": 393, "alm_null": 393, "tralo": 428, "alm": 340,
              "fioretto": 343, "hounie": 332}
    emitted = {"tralo_null": 393, "alm_null": 393, "tralo": 419, "alm": 467,
               "fioretto": 402, "hounie": 413}
    trained = ["tralo", "alm", "fioretto", "hounie"]
    for a in ("tralo_null", "alm_null"):
        if logged[a] != emitted[a]:
            bad.append("%s is the control and must agree 24/24" % a)
    for a in trained:
        if logged[a] == emitted[a]:
            bad.append("%s: log and predictions are stated to disagree 0/24" % a)
    by_log = sorted(trained, key=lambda a: abs(logged[a] - K))
    by_pred = sorted(trained, key=lambda a: abs(emitted[a] - K))
    if by_log == by_pred:
        bad.append("the log ordering reproduced the prediction ordering; the "
                   "measured fact is that it reverses")
    if not (by_log.index("alm") <= 1 and by_pred.index("alm") == 3):
        bad.append("`alm` is the receipt: 2nd-closest to K in the log (340 vs "
                   "K=352) and LAST in the predictions (467)")
    for src, q, want in [("training_log.csv", "model", False),
                         ("training_log.csv", "allocator", False),
                         ("final_predictions_raw.csv", "model", True),
                         ("final_predictions_raw.csv", "allocator", False),
                         ("final_predictions.csv", "allocator", True),
                         ("final_predictions.csv", "model", False)]:
        if count_source_ok(src, q) != want:
            bad.append("%s for a %s count: allowed=%s, expected %s"
                       % (src, q, not want, want))
    for name, shown, cfg_v, want in [("cosmetic_zero", 0.0, 0.01, "disagrees"),
                                     ("healthy", 0.01, 0.01, "agrees"),
                                     ("genuine_lambda0_twin", 0.0, 0.0, "agrees")]:
        p = os.path.join(write_run(tmp_path, name, [0.95] * 4,
                                   hyperparams={"lambda_local": cfg_v}),
                         "training_log.csv")
        df = pd.read_csv(p)
        df["Lambda_Local"] = [shown] * 4
        df.to_csv(p, index=False)
        df, cfg = read_run(os.path.dirname(p))
        col = pd.to_numeric(df["Lambda_Local"], errors="coerce").dropna()
        got = ("disagrees" if float(col.max()) == 0.0 and float(
            cfg["hyperparams"]["lambda_local"]) != 0.0 else "agrees")
        if got != want:
            bad.append("Lambda_Local %s: %s, expected %s" % (name, got, want))
    report(bad, "log-is-not-the-state failures")


def test_terminal_collapse_in_a_control_arm_reverses_a_headline():
    """GATE 5. The pipeline keeps the LAST epoch, not the best. A `clip` seed
    ended 0.9934 -> 0.9116 and that one collapsed CONTROL reversed a whole
    comparison, so the final epoch of the CONTROL arms is what must be read
    (PLAYBOOK 3.1). NEGATIVE CONTROL: ordinary wobble, a rising tail and a
    mid-run dip that recovers must all pass -- only the last epoch is scored."""
    bad = []
    for name, acc, want in [
            ("clip seed 4, measured", [.9928, .9931, .9934, .9116], True),
            ("ordinary wobble", [.9928, .9931, .9934, .9927], False),
            ("rising tail", [.980, .984, .988, .993], False),
            ("mid-run dip that recovers", [.991, .870, .988, .992], False)]:
        if (terminal_collapse(acc) is not None) != want:
            bad.append("%s: collapse=%s, expected %s" % (name, not want, want))
    treated, control = [0.60] * 4, [0.62] * 4
    honest = float(np.mean(treated) - np.mean(control))
    broken = float(np.mean(treated) - np.mean(control[:3] + [0.30]))
    if not honest < 0 < broken:
        bad.append("the control case is void: a collapsed control must flip the "
                   "contrast's sign (%.3f -> %.3f)" % (honest, broken))
    if terminal_collapse(control[:3] + [0.30]) is None:
        bad.append("the collapse that manufactured the win was not detected")
    report(bad, "terminal-collapse gate failures")


def test_divergence_is_read_on_the_runs_own_scale(tmp_path):
    """GATE 6. A late-epoch move is a defect against the run's OWN within-run sd,
    never an absolute threshold: 0.02 on a flat trajectory is ~140 sds and 0.30
    on a noisy one is under 1. NEGATIVE CONTROL: an absolute 0.02 rule fires on
    BOTH, which is why it is not the rule. Plus log_health's two exclusions --
    an all-blank reserved column was NOT LOGGED, and the warm-up row predates
    the constraint object."""
    bad = []
    flat = [.9930, .9928, .9931, .9929, .9932, .9700]
    noisy = [.31, .88, .42, .79, .35, .60]
    for name, v, want in [("flat run, 0.02 drop", flat, True),
                          ("noisy run, 0.30 move", noisy, False)]:
        if (late_move_z(v) >= 6.0) != want:
            bad.append("%s: z=%.1f, expected flagged=%s"
                       % (name, late_move_z(v), want))
    if [abs(v[-1] - v[-2]) > 0.02 for v in (flat, noisy)] != [True, True]:
        bad.append("the absolute rule is supposed to fire on BOTH runs, which is "
                   "the reason it is not the rule")
    df = pd.read_csv(os.path.join(
        write_run(tmp_path, "diverged", [0.95] * 4), "training_log.csv"))
    df["Group0_Hard_Class4"] = [np.nan] * 4      # reserved, never logged
    df.loc[0, "Limit_Class2"] = np.nan           # warm-up row, no constraint yet
    df.loc[3, "L_Global"] = np.inf               # real divergence
    found = nonfinite_cols(df)
    if "Group0_Hard_Class4" in found:
        bad.append("an all-blank reserved column was read as divergence")
    if "Limit_Class2" in found:
        bad.append("the warm-up row was read as divergence")
    if found.get("L_Global") != 1:
        bad.append("a non-finite value beside real ones was missed: %s" % found)
    report(bad, "divergence-detection failures")


def test_rho_step_is_log_only_so_use_the_ast_never_grep():
    """GATE 7. `rho_step` is named in a log-format string, so a grep reports it
    as read while it is a HALLUCINATED config key. Any "is this read" claim must
    walk the AST (audit_config.py). NEGATIVE CONTROL: keys that ARE read must be
    found, or a walker that finds nothing proves nothing -- plus the textual
    count, which shows grep really would have been fooled here."""
    bad, src = [], rel("src")
    keys = config_keys_read(src)
    textual = 0
    for r, dirs, files in os.walk(src):
        dirs[:] = [x for x in dirs if x != "__pycache__"]
        textual += sum("rho_step" in io.open(os.path.join(r, f),
                                             encoding="utf-8").read()
                       for f in files if f.endswith(".py"))
    if textual == 0:
        bad.append("the control is void: nothing under src/ names `rho_step`, so "
                   "grep would not have been fooled here")
    for dead in ("rho_step", "alpha_kl", "base_loss", "enable_ce_skip"):
        if dead in keys:
            bad.append("`%s` is documented as deleted or log-only and the AST "
                       "finds it read" % dead)
    for live in ("constraint_epochs", "warmup_epochs", "soft_count_mode",
                 "lambda_local", "constraint_fp32"):
        if live not in keys:
            bad.append("`%s` is read and the AST walker missed it -- the gate "
                       "cannot prove absence" % live)
    report(bad, "config-key AST failures")


def test_the_detectors_run_on_real_logs_when_any_are_present():
    """The synthetic schemas above are worth only what their fidelity is worth.
    When real runs exist, assert every log classifies as one of the two shipped
    schemas and that the detectors read it without raising. SKIPS otherwise:
    this stage must gate in CI with no data, and a silent fallback to a toy
    reports a pass about nothing (conftest, `slice_dir`)."""
    logs = sorted(glob.glob(rel("results", "**", "training_log.csv"),
                            recursive=True))[:40]
    if not logs:
        pytest.skip("no real training_log.csv on this machine -- gate NOT run")
    bad = []
    for p in logs:
        try:
            df = pd.read_csv(p)
            kind = schema_kind(df)
            if kind == "unknown":
                bad.append("%s: neither shipped schema -- the synthetic logs "
                           "above no longer match reality" % p)
                continue
            acc = pd.to_numeric(df["Train_Acc" if kind == "wide" else "train_acc"],
                                errors="coerce").dropna().tolist()
            terminal_collapse(acc)
            if kind == "wide" and "Epoch" in df.columns:
                nonfinite_cols(df)
            if len(acc) > 2:
                late_move_z(acc)
        except Exception as e:                                    # noqa: BLE001
            bad.append("%s: a detector raised on a real log (%s)" % (p, e))
    report(bad, "real-log compatibility failures")
