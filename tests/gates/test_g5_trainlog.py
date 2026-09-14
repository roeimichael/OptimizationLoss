"""Maintained behavioral regression fixtures."""

import glob

import io

import json

import os

import numpy as np

import pandas as pd

import pytest

from .conftest import rel, report

pytestmark = pytest.mark.stage5_trainlog

(SATURATED_ACC, FLAT_GAIN) = (0.93, 0.005)

CE_FLOOR = 0.05

COLLAPSE_DROP = 0.02

REACHABLE = 0.04

DOSE_TOL = 0.05

WIDE = [
    "Epoch",
    "Train_Acc",
    "L_CE",
    "L_Global",
    "L_Local",
    "Grad_Norm",
    "Lambda_Global",
    "Lambda_Local",
    "Global_Satisfied",
    "Local_Satisfied",
    "Limit_Class2",
    "Hard_Class2",
    "Soft_Class2",
]

NARROW = [
    "epoch",
    "train_acc",
    "ce_loss",
    "constraint_loss",
    "total_excess",
    "all_satisfied",
    "max_lambda_g",
]


def write_run(tmp, name, acc, ce=None, **cfg):
    d = os.path.join(str(tmp), name)
    os.makedirs(d, exist_ok=True)
    n = len(acc)
    row = dict(
        {c: [0.0] * n for c in WIDE},
        Epoch=list(range(1, n + 1)),
        Train_Acc=acc,
        L_CE=ce if ce is not None else [0.5] * n,
        Lambda_Local=[0.01] * n,
        Limit_Class2=[352.0] * n,
        Hard_Class2=[400.0] * n,
    )
    pd.DataFrame({c: row[c] for c in WIDE}).to_csv(
        os.path.join(d, "training_log.csv"), index=False
    )
    hp = dict(
        {"constraint_epochs": 29, "lambda_local": 0.01}, **cfg.pop("hyperparams", {})
    )
    io.open(os.path.join(d, "config.json"), "w", encoding="utf-8").write(
        json.dumps(
            dict({"status": "completed", "hyperparams": hp, "results": {}}, **cfg)
        )
    )
    return d


def read_run(d):
    return (
        pd.read_csv(os.path.join(d, "training_log.csv")),
        json.load(io.open(os.path.join(d, "config.json"), encoding="utf-8")),
    )


def saturation_verdict(d):
    (df, cfg) = read_run(d)
    if (cfg.get("hyperparams") or {}).get("constraint_epochs", 0) == 0:
        return "post_hoc"
    a = pd.to_numeric(df["Train_Acc"], errors="coerce").dropna()
    c = pd.to_numeric(df["L_CE"], errors="coerce").dropna()
    both = (
        float(a.iloc[0]) >= SATURATED_ACC
        and abs(float(a.iloc[-1]) - float(a.iloc[0])) <= FLAT_GAIN
    )
    return "SATURATED" if both or float(c.iloc[-1]) < CE_FLOOR else "live"


def cut_slope(p):
    return p * (1.0 - p)


def dose_problems(per):
    (out, frac) = ([], {})
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
    frac = {a: app / float(att) for (a, (app, att)) in per.items() if att > 0}
    if all((f >= 1.0 for f in frac.values())):
        return "ok"
    low = [a for a in frac if frac[a] < max(frac.values()) - DOSE_TOL]
    return "loss_shape" if len(low) == 1 else "host"


def fp16_signature(frac, amp):
    return amp == "float16" and 0.24 <= 1.0 - frac <= 0.33


def schema_kind(df):
    return (
        "wide"
        if "Train_Acc" in df.columns
        else "narrow"
        if "train_acc" in df.columns
        else "unknown"
    )


def comparable(dfs):
    return len({tuple(df.columns) for df in dfs}) == 1


def count_source_ok(source, question):
    return {
        ("final_predictions_raw.csv", "model"): True,
        ("final_predictions.csv", "allocator"): True,
    }.get((source, question), False)


def terminal_collapse(acc):
    a = [float(x) for x in acc]
    return (a[-2], a[-1]) if len(a) >= 2 and a[-1] < a[-2] - COLLAPSE_DROP else None


def late_move_z(series):
    v = np.asarray([float(x) for x in series], dtype=float)
    body = v[:-1]
    sd = float(np.std(body, ddof=1))
    if sd <= 0:
        return float("inf") if v[-1] != body[-1] else 0.0
    return abs(v[-1] - float(np.median(body))) / sd


def nonfinite_cols(df):
    scan = df[pd.to_numeric(df["Epoch"], errors="coerce") >= 2]
    bad = {}
    for c in scan.select_dtypes(include=[np.number]).columns:
        f = np.isfinite(scan[c].to_numpy(dtype=float))
        if f.any() and (not f.all()):
            bad[c] = int((~f).sum())
    return bad


def test_ce_saturation_is_refused_and_warmup_1_survives(tmp_path):
    (hi, live) = ([0.998] * 6, [0.956, 0.962, 0.968, 0.971, 0.98, 0.986])
    bad = []
    for name, acc, ce, n_con, want in [
        ("warmup1_iwildcam", live, [0.42, 0.36, 0.3, 0.25, 0.21, 0.18], 29, "live"),
        ("warmup50_dead_regime", hi, [0.004] * 6, 29, "SATURATED"),
        (
            "high_acc_still_moving",
            [0.94, 0.95, 0.96, 0.96, 0.97, 0.97],
            [0.3, 0.27, 0.22, 0.19, 0.15, 0.12],
            29,
            "live",
        ),
        ("flat_but_unconverged", [0.71] * 6, [0.9] * 5 + [0.88], 29, "live"),
        (
            "ce_floor_only",
            [0.88, 0.881, 0.882, 0.883, 0.884, 0.885],
            [0.6, 0.31, 0.11, 0.03, 0.01, 0.004],
            29,
            "SATURATED",
        ),
        ("posthoc_clip", hi, [0.004] * 6, 0, "post_hoc"),
    ]:
        got = saturation_verdict(
            write_run(tmp_path, name, acc, ce, hyperparams={"constraint_epochs": n_con})
        )
        if got != want:
            bad.append("%s: verdict %s, expected %s" % (name, got, want))
    for label, p, want in [
        ("L50_G30 warm-up, 4/4 seeds responded", 0.9389, True),
        ("L30_G20 warm-up, 0/4 seeds responded", 0.973, False),
        ("L50_G30 at 30 epochs, converged", 0.999, False),
    ]:
        if (cut_slope(p) >= REACHABLE) != want:
            bad.append(
                "%s: p(1-p)=%.4f vs REACHABLE=%.3f, expected reachable=%s"
                % (label, cut_slope(p), REACHABLE, want)
            )
    ratio = cut_slope(0.9389) / cut_slope(0.999)
    if not 55.0 <= ratio <= 70.0:
        bad.append("converging must drop p(1-p) at the cut ~60x, got %.1fx" % ratio)
    report(bad, "CE-saturation gate failures")


def test_a_completed_run_can_have_landed_three_percent_of_its_dose(tmp_path):
    bad = []
    for name, per, want in [
        (
            "uniform1: 1/29 beside 29/29",
            {"tralo": (29, 29), "tralo_head": (29, 29), "tralo_uniform": (1, 29)},
            2,
        ),
        ("iwc3: 716/1044", {"tralo": (716, 1044)}, 1),
        ("taskwin1, no --constraint-fp32: 20/29", {"tralo": (20, 29)}, 1),
        ("taskwin2, --constraint-fp32: 29/29", {"tralo": (29, 29)}, 0),
        ("dom1: every arm 29/29", {"tralo": (29, 29), "alm": (29, 29)}, 0),
    ]:
        n = len(dose_problems(per))
        if n != want:
            bad.append("%s: %d problem(s), expected exactly %d" % (name, n, want))
    got = {}
    for k, applied in (("underdosed", 1), ("dosed", 29)):
        cfg = read_run(
            write_run(
                tmp_path,
                k,
                [0.95] * 4,
                results={
                    "constraint_steps_applied": applied,
                    "constraint_steps_attempted": 29,
                },
            )
        )[1]
        if cfg["status"] != "completed":
            bad.append("the control is void: %s must report `completed`" % k)
        got[k] = len(
            dose_problems(
                {
                    "tralo": (
                        cfg["results"]["constraint_steps_applied"],
                        cfg["results"]["constraint_steps_attempted"],
                    )
                }
            )
        )
    if got["underdosed"] == 0:
        bad.append("a 1/29 run reporting `completed` was passed as healthy")
    if got["dosed"] != 0:
        bad.append("a 29/29 run was refused: the gate does not separate")
    report(bad, "dose-landed gate failures")


def test_one_arm_low_is_the_loss_shape_every_arm_low_is_the_host():
    bad = []
    for name, per, want in [
        (
            "uniform1: tralo_uniform alone at 3.4%",
            {"tralo": (29, 29), "tralo_head": (29, 29), "tralo_uniform": (1, 29)},
            "loss_shape",
        ),
        (
            "iwc3: both arms ~68.6%, agreeing with each other",
            {"tralo": (716, 1044), "alm": (720, 1044)},
            "host",
        ),
        (
            "xfam1: bfloat16, nothing lost",
            {"tralo": (29, 29), "fioretto": (29, 29)},
            "ok",
        ),
        (
            "the loss moved to EVERY arm",
            {"tralo": (1, 29), "tralo_head": (1, 29), "tralo_uniform": (1, 29)},
            "host",
        ),
    ]:
        got = dose_diagnosis(per)
        if got != want:
            bad.append("%s: diagnosed %s, expected %s" % (name, got, want))
    for name, frac, amp, want in [
        ("iwc1", 0.688, "float16", True),
        ("iwc2", 0.746, "float16", True),
        ("iwc3", 0.686, "float16", True),
        ("taskwin1", 20 / 29.0, "float16", True),
        ("uniform1 tralo_uniform", 1 / 29.0, "bfloat16", False),
        ("dom1", 1.0, "bfloat16", False),
    ]:
        if fp16_signature(frac, amp) != want:
            bad.append(
                "%s: fp16 signature %s at %.3f on %s, expected %s"
                % (name, not want, frac, amp, want)
            )
    report(bad, "loss-shape vs host diagnosis failures")


def test_the_log_is_never_the_state(tmp_path):
    (bad, K) = ([], 352)
    (wide, narrow) = (pd.DataFrame(columns=WIDE), pd.DataFrame(columns=NARROW))
    for name, dfs, want in [
        ("tralo vs fioretto", [wide, narrow], False),
        ("tralo vs tralo", [wide, wide.copy()], True),
        ("fioretto vs hounie", [narrow, narrow.copy()], True),
    ]:
        if comparable(dfs) != want:
            bad.append("%s: comparable=%s, expected %s" % (name, not want, want))
    if schema_kind(wide) != "wide" or schema_kind(narrow) != "narrow":
        bad.append("the two shipped schemas are no longer recognised")
    logged = {
        "tralo_null": 393,
        "alm_null": 393,
        "tralo": 428,
        "alm": 340,
        "fioretto": 343,
        "hounie": 332,
    }
    emitted = {
        "tralo_null": 393,
        "alm_null": 393,
        "tralo": 419,
        "alm": 467,
        "fioretto": 402,
        "hounie": 413,
    }
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
        bad.append(
            "the log ordering reproduced the prediction ordering; the measured fact is that it reverses"
        )
    if not (by_log.index("alm") <= 1 and by_pred.index("alm") == 3):
        bad.append(
            "`alm` is the receipt: 2nd-closest to K in the log (340 vs K=352) and LAST in the predictions (467)"
        )
    for src, q, want in [
        ("training_log.csv", "model", False),
        ("training_log.csv", "allocator", False),
        ("final_predictions_raw.csv", "model", True),
        ("final_predictions_raw.csv", "allocator", False),
        ("final_predictions.csv", "allocator", True),
        ("final_predictions.csv", "model", False),
    ]:
        if count_source_ok(src, q) != want:
            bad.append(
                "%s for a %s count: allowed=%s, expected %s" % (src, q, not want, want)
            )
    for name, shown, cfg_v, want in [
        ("cosmetic_zero", 0.0, 0.01, "disagrees"),
        ("healthy", 0.01, 0.01, "agrees"),
        ("genuine_lambda0_twin", 0.0, 0.0, "agrees"),
    ]:
        p = os.path.join(
            write_run(tmp_path, name, [0.95] * 4, hyperparams={"lambda_local": cfg_v}),
            "training_log.csv",
        )
        df = pd.read_csv(p)
        df["Lambda_Local"] = [shown] * 4
        df.to_csv(p, index=False)
        (df, cfg) = read_run(os.path.dirname(p))
        col = pd.to_numeric(df["Lambda_Local"], errors="coerce").dropna()
        got = (
            "disagrees"
            if float(col.max()) == 0.0
            and float(cfg["hyperparams"]["lambda_local"]) != 0.0
            else "agrees"
        )
        if got != want:
            bad.append("Lambda_Local %s: %s, expected %s" % (name, got, want))
    report(bad, "log-is-not-the-state failures")


def test_terminal_collapse_in_a_control_arm_reverses_a_headline():
    bad = []
    for name, acc, want in [
        ("clip seed 4, measured", [0.9928, 0.9931, 0.9934, 0.9116], True),
        ("ordinary wobble", [0.9928, 0.9931, 0.9934, 0.9927], False),
        ("rising tail", [0.98, 0.984, 0.988, 0.993], False),
        ("mid-run dip that recovers", [0.991, 0.87, 0.988, 0.992], False),
    ]:
        if (terminal_collapse(acc) is not None) != want:
            bad.append("%s: collapse=%s, expected %s" % (name, not want, want))
    (treated, control) = ([0.6] * 4, [0.62] * 4)
    honest = float(np.mean(treated) - np.mean(control))
    broken = float(np.mean(treated) - np.mean(control[:3] + [0.3]))
    if not honest < 0 < broken:
        bad.append(
            "the control case is void: a collapsed control must flip the contrast's sign (%.3f -> %.3f)"
            % (honest, broken)
        )
    if terminal_collapse(control[:3] + [0.3]) is None:
        bad.append("the collapse that manufactured the win was not detected")
    report(bad, "terminal-collapse gate failures")


def test_divergence_is_read_on_the_runs_own_scale(tmp_path):
    bad = []
    flat = [0.993, 0.9928, 0.9931, 0.9929, 0.9932, 0.97]
    noisy = [0.31, 0.88, 0.42, 0.79, 0.35, 0.6]
    for name, v, want in [
        ("flat run, 0.02 drop", flat, True),
        ("noisy run, 0.30 move", noisy, False),
    ]:
        if (late_move_z(v) >= 6.0) != want:
            bad.append("%s: z=%.1f, expected flagged=%s" % (name, late_move_z(v), want))
    if [abs(v[-1] - v[-2]) > 0.02 for v in (flat, noisy)] != [True, True]:
        bad.append(
            "the absolute rule is supposed to fire on BOTH runs, which is the reason it is not the rule"
        )
    df = pd.read_csv(
        os.path.join(write_run(tmp_path, "diverged", [0.95] * 4), "training_log.csv")
    )
    df["Group0_Hard_Class4"] = [np.nan] * 4
    df.loc[0, "Limit_Class2"] = np.nan
    df.loc[3, "L_Global"] = np.inf
    found = nonfinite_cols(df)
    if "Group0_Hard_Class4" in found:
        bad.append("an all-blank reserved column was read as divergence")
    if "Limit_Class2" in found:
        bad.append("the warm-up row was read as divergence")
    if found.get("L_Global") != 1:
        bad.append("a non-finite value beside real ones was missed: %s" % found)
    report(bad, "divergence-detection failures")


def test_the_detectors_run_on_real_logs_when_any_are_present():
    logs = sorted(glob.glob(rel("results", "**", "training_log.csv"), recursive=True))[
        :40
    ]
    if not logs:
        pytest.skip("no real training_log.csv on this machine -- gate NOT run")
    bad = []
    for p in logs:
        try:
            df = pd.read_csv(p)
            kind = schema_kind(df)
            if kind == "unknown":
                bad.append(
                    "%s: neither shipped schema -- the synthetic logs above no longer match reality"
                    % p
                )
                continue
            acc = (
                pd.to_numeric(
                    df["Train_Acc" if kind == "wide" else "train_acc"], errors="coerce"
                )
                .dropna()
                .tolist()
            )
            terminal_collapse(acc)
            if kind == "wide" and "Epoch" in df.columns:
                nonfinite_cols(df)
            if len(acc) > 2:
                late_move_z(acc)
        except Exception as e:
            bad.append("%s: a detector raised on a real log (%s)" % (p, e))
    report(bad, "real-log compatibility failures")
