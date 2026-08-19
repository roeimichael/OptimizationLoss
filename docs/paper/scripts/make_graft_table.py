"""Generate tables/tab_graft.tex from the review_graft campaign CSV.

Per tight-cap cell (cap x backbone): mean cc-F1 over seeds 1-4 for the two
dual hosts, their graft arms (+RH = optimizer reset + undershoot hinge), the
anti-windup arm (Fioretto + dual restart), and TraLO, all re-run on identical
hardware with identical warmups. Bold = best in the row.
"""
# Re-homed to docs/paper/scripts/ on 2026-08-19. This file previously lived
# only in the gitignored archive/legacy/final_AAAI_PAPER/scripts/, so no clone
# of this repository could regenerate the float it emits. Paths below resolve
# against docs/paper/ -- ROOT is this file's parent's parent.


import os
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

CAPS = [("L30_G30", "L30"), ("L40_G40", "L40")]
BBP = {"MobileNetV3": "MNetV3", "RegNetY400MF": "RegNet", "ViTB16": "ViT-B/16"}
COLS = ["fioretto_ldf", "fioretto_rh", "fioretto_restart",
        "hounie_rcl", "hounie_rh", "tralo"]


def main():
    df = pd.read_csv(os.path.join(ROOT, "data", "corpus", "review_graft_2026-07.csv"))
    L = []
    L.append(r"\begin{table}[t]\centering")
    # Caption and label BEFORE the tabular, and 5pt not 2.6pt. Both were
    # hand-applied to the committed tab_graft.tex and reverted on every
    # regeneration; PROVENANCE listed them as known drift for six days.
    L.append(r"\caption{\textbf{Graft and anti-windup arms on the OctMNIST tight caps}")
    L.append(r"(mean cc-F1 over seeds $1$--$4$; \textbf{bold} = best in row). $+$R{+}H grafts")
    L.append(r"TraLO's optimizer reset and undershoot hinge onto the host; $+$restart resets")
    L.append(r"all dual multipliers to zero at soft feasibility \citep{gallego2022controlled}.")
    L.append(r"All $144$ runs share the frozen recipe, warmup caches, and GPU model with the")
    L.append(r"re-run hosts; the re-run hosts reproduce the main-grid values to $\pm 0.0013$.}")
    L.append(r"\label{tab:graft}")
    L.append(r"{\small")
    L.append(r"\setlength{\tabcolsep}{5pt}")
    L.append(r"\begin{tabular}{ll cc c cc c}")
    L.append(r"\toprule")
    L.append(r" & & \multicolumn{3}{c}{Fioretto-LDF} & \multicolumn{2}{c}{Hounie-RCL} & \\")
    L.append(r"\cmidrule(lr){3-5}\cmidrule(lr){6-7}")
    L.append(r"Cap & Backbone & host & $+$R{+}H & $+$restart & host & $+$R{+}H & TraLO \\")
    L.append(r"\midrule")
    for tag, cap in CAPS:
        for i, (bb, bbp) in enumerate(BBP.items()):
            sub = df[(df.tag == tag) & (df.model == bb)]
            vals = {}
            for m in COLS:
                s = sub[sub.method == m].cc_f1
                assert len(s) == 4, (tag, bb, m, len(s))
                vals[m] = s.mean()
            best = max(round(v, 3) for v in vals.values())
            def fmt(m):
                v = round(vals[m], 3)
                s = f"{v:.3f}"
                return r"\textbf{" + s + "}" if v == best else s
            cap_cell = cap if i == 0 else ""
            L.append(f"{cap_cell} & {bbp} & {fmt('fioretto_ldf')} & {fmt('fioretto_rh')} & "
                     f"{fmt('fioretto_restart')} & {fmt('hounie_rcl')} & {fmt('hounie_rh')} & "
                     f"{fmt('tralo')} \\\\")
        if tag == CAPS[0][0]:
            L.append(r"\midrule")
    L.append(r"\bottomrule")
    L.append(r"\end{tabular}}")
    L.append(r"\end{table}")
    out = os.path.join(ROOT, "tables", "tab_graft.tex")
    with open(out, "w") as f:
        f.write("\n".join(L) + "\n")
    print("wrote", out)
    # console summary for the prose
    for tag, cap in CAPS:
        for bb in BBP:
            sub = df[(df.tag == tag) & (df.model == bb)]
            g = lambda m: sub[sub.method == m].cc_f1.mean()
            print(f"{cap} {bb:13s} fio {g('fioretto_ldf'):.3f} rh {g('fioretto_rh'):.3f} "
                  f"restart {g('fioretto_restart'):.3f} hou {g('hounie_rcl'):.3f} "
                  f"hrh {g('hounie_rh'):.3f} tralo {g('tralo'):.3f}")


if __name__ == "__main__":
    main()
