"""Derive compact, checkable report statistics from preserved score and event files."""

from __future__ import annotations

import hashlib
import json
import re
import statistics
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
AUDIT = Path("C:/Users/roeym/.codex/rebuild-audit-20260922")
OUT = ROOT / "experiments" / "tralo_pdf_evidence_20261001.json"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def summarize(values: list[float]) -> dict:
    return {
        "n": len(values),
        "mean": statistics.mean(values),
        "seed_sd": statistics.stdev(values) if len(values) > 1 else None,
    }


def score_block(filename: str) -> dict:
    path = AUDIT / filename
    doc = json.loads(path.read_text(encoding="utf-8"))
    assert len(doc["seeds"]) == 12
    assert len({s["seed"] for s in doc["seeds"]}) == 12
    result = {"source": str(path), "sha256": digest(path), "caps": {}}
    for divisor, cap in [("10", 167), ("20", 83)]:
        rows = [s["caps"][divisor] for s in doc["seeds"]]
        assert {r["quota"]["global_cap"] for r in rows} == {cap}
        arms = rows[0]["arms"].keys()
        cap_summary = {"cap": cap, "arms": {}}
        for arm in arms:
            cap_summary["arms"][arm] = {}
            for metric in ("cc_f1", "accuracy", "macro_f1", "weighted_f1"):
                vals = [r["arms"][arm]["allocated"][metric] for r in rows]
                cap_summary["arms"][arm][metric] = summarize(vals)
        for arm in ("ens_joint", "ens_phr_local", "ens_global_dose", "ens_sham"):
            if arm in arms:
                cap_summary.setdefault("paired_vs_pto", {})[arm] = {
                    metric: summarize([
                        r["arms"][arm]["allocated"][metric]
                        - r["arms"]["ens_pto"]["allocated"][metric]
                        for r in rows
                    ])
                    for metric in ("cc_f1", "accuracy", "macro_f1", "weighted_f1")
                }
        result["caps"][str(cap)] = cap_summary
    return result


def global_block(filename: str) -> dict:
    path = ROOT / "analysis" / filename
    raw = path.read_text(encoding="utf-8")
    start = raw.index("per seed (cc-F1")
    rows = []
    for line in raw[start:].splitlines()[1:]:
        match = re.match(
            r"\s*(\d{4})\s+([\d.]+)\s*/\s*([\d.]+)\s*/\s*([\d.]+)\s*\|", line
        )
        if not match:
            if rows:
                break
            continue
        seed, pto, tralo, sham = match.groups()
        rows.append((int(seed), float(pto), float(tralo), float(sham)))
    assert len(rows) in (48, 72), (filename, len(rows))
    return {
        "source": str(path.relative_to(ROOT)),
        "sha256": digest(path),
        "ens_pto": summarize([r[1] for r in rows]),
        "ens_tralo": summarize([r[2] for r in rows]),
        "ens_sham": summarize([r[3] for r in rows]),
        "tralo_minus_pto": summarize([r[2] - r[1] for r in rows]),
        "tralo_minus_sham": summarize([r[2] - r[3] for r in rows]),
    }


def vit_log_curve() -> dict:
    paths = [AUDIT / f"vit{seed}.events.jsonl" for seed in range(6601, 6613)]
    per_seed = []
    for path in paths:
        events = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        epochs = [e for e in events if e["event"] == "epoch"]
        assert len(epochs) >= 6
        per_seed.append(epochs)
    curve = {}
    for epoch in range(1, 7):
        rows = [next(e for e in events if e["epoch"] == epoch) for events in per_seed]
        curve[str(epoch)] = {
            "training_loss": summarize([e["training_loss"] for e in rows]),
            "stop_loss": summarize([e["stop_loss"] for e in rows]),
        }
    return {
        "sources": [{"path": str(p), "sha256": digest(p)} for p in paths],
        "epochs_1_to_6_all_12_seeds": curve,
    }


def main() -> None:
    result = {
        "description": "Development evidence only; compact summaries contain no labels or predictions.",
        "global_knee": {
            "mobilenetv3": global_block("stepens_mn3_score_20260928.txt"),
            "efficientnet_b5": global_block("stepens_b5_score_20260928.txt"),
        },
        "local_fmow2": {
            "mobilenetv3": score_block("fmow_boundary_full_score_31d50d05_20261001.json"),
            "vit_b16": score_block("fmow_vit_full_score_24a78b8e_20261001.json"),
        },
        "vit_training_curve": vit_log_curve(),
    }
    OUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(OUT)
    for name, block in result["global_knee"].items():
        print(name, block["ens_pto"], block["ens_tralo"], block["tralo_minus_pto"])
    for name, block in result["local_fmow2"].items():
        for cap, stats in block["caps"].items():
            print(name, cap, stats["arms"]["ens_pto"]["cc_f1"],
                  stats["arms"]["ens_joint"]["cc_f1"],
                  stats["paired_vs_pto"]["ens_joint"]["cc_f1"])


if __name__ == "__main__":
    main()
