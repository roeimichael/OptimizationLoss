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


# ==========================================================================
#   THE SENSITIVITY SCREEN -- can this cell separate two methods at all?
#   Source: scripts/sensitivity_screen.py, measured over dom1 + taskwin2 +
#   equaldose1 on 2026-09-04 (28 cells: SENSITIVE 0, UNDER-POWERED 27,
#   SATURATED 1).
# ==========================================================================
def test_the_sensitivity_screen_separates_saturated_from_underpowered_from_live():
    """Four verdicts, and collapsing any two of them is the defect.

    "Nothing moved" and "we could not have seen it move" are opposite
    conclusions from the same table. This project's standing rule is that a
    tie must always be resolved into one or the other, and the screen is the
    instrument that does it per cell.

    NEGATIVE CONTROLS, all in this test:
      * a live, well-separated cell MUST come back SENSITIVE -- a screen that
        only ever says no has not been shown to work;
      * saturation must OUTRANK the seed count, because no number of seeds
        rescues a cell where nothing could have moved;
      * the same spread must read UNDER-POWERED at 4 seeds and NOT
        DIFFERENTIATED at 600, or the two are being collapsed;
      * a floor resting on too few observations must refuse to decide, which
        is the case on EVERY campaign in the corpus today (one `_null`/
        `_reseed` pair at 4 seeds = 4 observations against the 8 bar);
      * and it must refuse under its OWN name, `FLOOR UNMEASURED`, never as
        `UNDER-POWERED`. The two were one label until 2026-09-10 and call for
        opposite remedies -- seeds on the TREATED arms vs seeds or a third
        STREAM on the lambda=0 arms -- so a tally that merges them points the
        reader at the wrong fix. FRAMEWORK 2(z70).
    """
    from scripts.sensitivity_screen import (BAND_MIN, GRAD_MIN,  # noqa: E402
                                            MIN_FLOOR_OBS, classify)
    LIVE, DEAD = 0.20, GRAD_MIN / 10.0
    BAND, N = 50, 16
    cases = [
        # label,                      grad, band,      spread, floor, seeds, expected
        ("LIVENESS: live and separated", LIVE, BAND,    12.0,   4.0,  N, "SENSITIVE"),
        ("cut in dead territory",        DEAD, BAND,    12.0,   4.0,  N, "SATURATED"),
        ("no contestable items",         LIVE, 1,       12.0,   4.0,  N, "SATURATED"),
        ("saturation beats 10k seeds",   DEAD, BAND,     0.5,   4.0, 10000, "SATURATED"),
        ("small spread, few seeds",      LIVE, BAND,     0.5,   4.0,  N, "UNDER-POWERED"),
        ("same spread, many seeds",      LIVE, BAND,     0.5,   4.0, 600, "NOT DIFFERENTIATED"),
        ("no floor at all",              LIVE, BAND,    12.0,  None,  N, "NO DATA"),
        ("no gradient reading",          None, BAND,    12.0,   4.0,  N, "NO DATA"),
    ]
    bad = []
    for lbl, g, b, sp, fl, ns, want in cases:
        got, why = classify(g, b, sp, fl, ns, n_floor=N)
        if got != want:
            bad.append("%s: got %r, expected %r (%s)" % (lbl, got, want, why))

    # The thin-floor branch, which is what the whole corpus trips today.
    got, why = classify(LIVE, BAND, 12.0, 4.0, 4, n_floor=MIN_FLOOR_OBS - 1)
    if got != "FLOOR UNMEASURED" or "RNG floor itself" not in why:
        bad.append("a floor resting on %d observations must refuse to decide "
                   "AS `FLOOR UNMEASURED`, not as `UNDER-POWERED`; got %r (%s)"
                   % (MIN_FLOOR_OBS - 1, got, why))
    # NEGATIVE CONTROL: the two verdicts must not be interchangeable. A SPREAD
    # that is genuinely under a WELL-ESTIMATED floor is UNDER-POWERED, and the
    # remedy differs -- seeds on the TREATED arms there, seeds or a third
    # STREAM on the lambda=0 arms above. They were one label until 2026-09-10,
    # which sent the reader to the wrong fix and made the TALLY unreadable.
    got, why = classify(LIVE, BAND, 0.5, 4.0, 4, n_floor=MIN_FLOOR_OBS)
    if got != "UNDER-POWERED" or "RNG floor itself" in why:
        bad.append("a small spread under a WELL-ESTIMATED floor must stay "
                   "UNDER-POWERED, or the split is a rename; got %r (%s)"
                   % (got, why))
    # ...and it must NOT refuse once the floor is properly estimated, or the
    # bar is simply a blanket refusal wearing a statistic.
    got, _ = classify(LIVE, BAND, 12.0, 4.0, 4, n_floor=MIN_FLOOR_OBS)
    if got != "SENSITIVE":
        bad.append("at exactly MIN_FLOOR_OBS=%d the screen still refuses, so "
                   "the bar is off by one or is a blanket refusal"
                   % MIN_FLOOR_OBS)

    # A SATURATED verdict must name WHICH point it means. FRAMEWORK section 4:
    # rank K and the decision boundary are different items, and "the gradient
    # cannot reach the cut" is wrong when stated without one of them.
    _v, why = classify(DEAD, BAND, 12.0, 4.0, N, n_floor=N, grad_bd=0.248)
    if "DECISION BOUNDARY" not in why or "CUT-PLACEMENT" not in why:
        bad.append("a dead cut beside a live boundary must be reported as a "
                   "cut-placement result, not as a saturated model: %s" % why)

    if BAND_MIN <= 0 or GRAD_MIN <= 0:
        bad.append("the bars must be positive; GRAD_MIN=%s BAND_MIN=%s"
                   % (GRAD_MIN, BAND_MIN))
    report(bad, "sensitivity-screen verdict failures")


def test_the_cross_arm_spread_is_a_pairwise_statistic_not_a_range():
    """A max-min RANGE over k arms is not comparable to a two-arm floor.

    Under pure noise the range of k samples grows like `sd*sqrt(2 ln k)`
    (~3.1*sd at k=10) while the two-arm floor it would be measured against is
    `E|X-Y| = 1.13*sd`. So `range >= floor` certifies a cell of pure noise as
    differentiated, at a ratio of ~2.7x, before any method does anything.

    Measured independently on the real corpus the same day: raw
    `range/floor` reads a healthy median 2.51 over 50 cells, and the SAME
    cells read 0.97 once the range is divided by E[range of n]. An sd-based
    estimator agrees at 0.94. The raw ratio was an artifact of arm count.

    This test is arithmetic, not a fixture, so it cannot rot.
    """
    rng = np.random.RandomState(20260904)
    N, TRIALS = 4, 4000
    ranges, pairs = [], []
    for _ in range(TRIALS):
        arms = rng.normal(0.0, 1.0, (10, N))       # 10 arms, ONE noise law
        means = arms.mean(axis=1)
        ranges.append(means.max() - means.min())
        a, b = rng.normal(0.0, 1.0, N), rng.normal(0.0, 1.0, N)
        pairs.append(abs(a.mean() - b.mean()))
    r, p = float(np.median(ranges)), float(np.median(pairs))
    bad = []
    if not r / p > 2.0:
        bad.append("the range/pairwise inflation is %.2fx, under the 2.0x this "
                   "gate exists to document -- re-derive it before trusting "
                   "any spread-vs-floor ratio" % (r / p))
    # And the pairwise statistic must NOT be inflated: two arms of pure noise
    # compared to two other arms of pure noise is 1.0 by construction.
    q = float(np.median([abs(rng.normal(0, 1, N).mean()
                             - rng.normal(0, 1, N).mean())
                         for _ in range(TRIALS)]))
    if not 0.8 < p / q < 1.25:
        bad.append("two pairwise draws of the SAME noise disagree by %.2fx; "
                   "the pairwise statistic is not scale-free" % (p / q))
    report(bad, "spread-statistic failures")


def test_a_THIRD_rng_stream_is_read_by_the_floor_and_kept_OUT_of_the_spread():
    """`tralo_reseed2` was invisible to `sensitivity_screen`, in both directions.

    The test was `arm.endswith("_reseed")`, which is False for `tralo_reseed2`.
    So the third lambda=0 stream -- the one `classify`'s own UNDER-POWERED
    message tells the reader to buy -- was treated as a rival ARM, widening the
    cross-arm spread it exists to define the floor for, while the floor itself
    paired only `<fam>_null` with `<fam>_reseed` and saw 4 of the 12
    observations three streams yield at 4 seeds. Both errors push the same way:
    a larger spread against a smaller, under-powered floor.

    `deployed_h2h` was fixed on 2026-09-06 and `sensitivity_screen` kept its own
    copy of the rule, which is exactly the drift `scripts/floors.py` was created
    to stop. The predicate now lives there and both import it.

    `dualprop1` is the first campaign carrying three streams, so this was about
    to be load-bearing rather than hypothetical.
    """
    from scripts.floors import is_lambda0_stream, stream_family, stream_pairs
    from scripts.sensitivity_screen import _is_floor_control

    # The exact string the old rule got wrong. This assertion FAILS on
    # `arm.endswith("_reseed")`, which is what makes it a control and not a
    # restatement of the code.
    assert is_lambda0_stream("tralo_reseed2")
    assert _is_floor_control("tralo_reseed2"), (
        "the third RNG stream must not be scored as a rival arm")
    assert _is_floor_control("tralo_reseed")

    # A `_null` is a lambda=0 stream for FLOOR purposes and still a legitimate
    # comparison arm, so it must NOT be excluded from the spread.
    assert is_lambda0_stream("tralo_null")
    assert not _is_floor_control("tralo_null")

    # NEGATIVE CONTROL: `<fam>_lam0` keeps `lambda_step` and takes real
    # constraint steps. Treating it as an RNG stream would put a TREATED arm
    # into the noise floor and make every cell look quiet.
    assert not is_lambda0_stream("tralo_lam0"), (
        "`_lam0` is not a lambda=0 stream -- it takes real constraint steps")
    assert not is_lambda0_stream("tralo")
    assert not is_lambda0_stream("alm")

    assert stream_family("alm_reseed2") == "alm"
    assert stream_family("tralo") is None

    # THREE streams give C(3,2) = 3 pairs; ONE gives none, which is why a count
    # of streams is not a count of observations.
    three = stream_pairs(["tralo", "tralo_null", "tralo_reseed", "tralo_reseed2"])
    assert len(three) == 3, three
    assert ("tralo_null", "tralo_reseed") in three
    assert ("tralo_null", "tralo_reseed2") in three
    assert ("tralo_reseed", "tralo_reseed2") in three

    # vitdual2's real arm set: four singleton families plus one pair.
    v2 = stream_pairs(["alm", "alm_null", "clip", "fioretto", "fioretto_null",
                       "focal_clip", "hounie", "hounie_null", "tralo",
                       "tralo_null", "tralo_reseed"])
    assert v2 == [("tralo_null", "tralo_reseed")], v2

    # Families never cross: an `alm` stream cannot bound `tralo`'s RNG noise,
    # even though at lambda=0 both are plain CE.
    mixed = stream_pairs(["tralo_null", "tralo_reseed", "alm_null", "alm_reseed"])
    assert len(mixed) == 2 and all(a.split("_")[0] == b.split("_")[0]
                                   for a, b in mixed), mixed


def test_a_RAGGED_cell_is_counted_by_its_SHARED_seeds_not_the_max():
    """The fixture every self-test in this project was missing.

    2(z52): not one scorer's self-test carried a ragged-coverage fixture --
    every one gave every arm all four seeds. That is why a whole class of defect
    survived in eight of thirteen scorers until a real unfinished campaign hit
    it. The campaign in the fixtures was the campaign we wished we had.

    `vitdual2` is the shape that matters and it is not hypothetical: `alm` ran
    seeds {1, 3} while `tralo` and `hounie` ran {1, 2}, so the cell has three
    distinct seeds, a max-over-arms of 2, and exactly ONE seed behind any
    arm-vs-arm statement.
    """
    from scripts.sensitivity_screen import shared_seed_count

    ragged = {
        "tralo": {1: 600.0, 2: 604.0},
        "alm":   {1: 603.0, 3: 611.0},
        "hounie": {1: 606.0, 2: 605.0},
    }
    arms = sorted(ragged)

    # The defect: max is 2 and the union is 3, but only seed 1 is shared.
    assert max(len(v) for v in ragged.values()) == 2
    assert len({s for v in ragged.values() for s in v}) == 3
    assert shared_seed_count(ragged, arms) == 1, (
        "a ragged cell must be counted by the seeds EVERY compared arm ran")

    # NEGATIVE CONTROL: a square cell is unchanged. The fix must be a no-op on
    # every complete campaign, or it would silently restate the whole corpus.
    square = {a: {1: 1.0, 2: 2.0, 3: 3.0, 4: 4.0}
              for a in ("tralo", "alm", "hounie", "clip")}
    assert shared_seed_count(square, sorted(square)) == 4

    # NEGATIVE CONTROL: disjoint arms share nothing, and that is 0, not 2.
    disjoint = {"tralo": {1: 1.0, 2: 2.0}, "alm": {3: 3.0, 4: 4.0}}
    assert shared_seed_count(disjoint, sorted(disjoint)) == 0

    # An arm present in the roster but with no usable runs must not drag the
    # intersection to zero -- it is absent, not disagreeing.
    with_empty = dict(square)
    with_empty["fioretto"] = {}
    assert shared_seed_count(with_empty, sorted(with_empty)) == 4
    assert shared_seed_count({}, []) == 0


def test_latch_probe_REFUSES_a_ratchet_mode_it_cannot_reconstruct():
    """The instrument was about to condemn a live arm.

    `weight_rankings` RECONSTRUCTS the multiplier as `lam0 + step * (epochs
    violated)`. That formula IS the constant ratchet, so handed any other arm it
    reports what a constant ratchet WOULD have built. On 2026-09-07 it was
    pointed at `tralo_dualprop`, whose ratchet is PROPORTIONAL, and returned a
    range of exactly 24.3x = (0.01 + 29*0.05)/(0.01 + 1*0.05) -- the constant
    ratchet's own algebraic ceiling, and the precise value FRAMEWORK 2(z51) had
    pre-registered as the arm's INERTNESS FALSIFIER.

    The arm was live: its LOGGED Lambda_Global reached 41.5 against `tralo`'s
    0.885, and its per-seed starting lambda differed, which a constant ratchet
    cannot produce. FRAMEWORK 2(z53).

    A probe that silently answers the wrong question is worse than one that
    refuses, because its output is indistinguishable from a real measurement.
    """
    import pytest

    from scripts.latch_probe import weight_rankings

    # (satisfied, {scope: excess}, {scope: limit}) per epoch
    log = [(False, {"a": 10.0, "b": 1.0}, {"a": 5, "b": 5}),
           (False, {"a": 8.0, "b": 0.0}, {"a": 5, "b": 5})]

    # The constant ratchet is what it models, and it still works.
    lam, mag = weight_rankings(log, None, 0.01, 0.05, "constant")
    assert lam["a"] == pytest.approx(0.01 + 2 * 0.05)
    assert lam["b"] == pytest.approx(0.01 + 1 * 0.05)
    assert mag["a"] == pytest.approx(18.0)

    # Default stays "constant", so every existing caller is unaffected.
    lam2, _ = weight_rankings(log, None, 0.01, 0.05)
    assert lam2 == lam

    # THE FIX: anything else must RAISE, not silently reconstruct.
    with pytest.raises(ValueError) as e:
        weight_rankings(log, None, 0.01, 0.05, "proportional")
    assert "constant" in str(e.value).lower()

    # NEGATIVE CONTROL: an unrecognised mode must also refuse rather than be
    # treated as constant by default -- that is the failure mode being fixed.
    with pytest.raises(ValueError):
        weight_rankings(log, None, 0.01, 0.05, "something_new")
