"""Replace embedded base64 PNGs in paper/presentation.html with the latest
on-disk PNG bytes. Updates §5 (convergence), §6 (F1-tightness), §7 (satisfaction)
after fair_evaluation_metrics regeneration.

The HTML order of <h2> tags ↔ image embeds is fixed:
  [1, 5, 6, 7] → [proposal_fig1_penalty, fig_convergence, fig_f1_tightness, fig_satisfaction]

We DO NOT replace §1's penalty figure — it's the proposal sketch and unchanged.
"""
import base64
import re
from pathlib import Path

HTML = Path("paper/presentation.html")
FIG = Path("paper/figures")

# Order matters — matches the 4 embeds in current HTML.
SLOTS = [
    None,  # §1 penalty fig — keep
    FIG / "fig_convergence.png",     # §5
    FIG / "fig_f1_tightness.png",    # §6
    FIG / "fig_satisfaction.png",    # §7
]


def main():
    html = HTML.read_text(encoding="utf-8")
    # Match each "data:image/png;base64,<chars>" prefix-and-payload.
    pat = re.compile(r"data:image/png;base64,[A-Za-z0-9+/=]+")
    matches = list(pat.finditer(html))
    print(f"Found {len(matches)} image embeds in {HTML}")
    if len(matches) != len(SLOTS):
        raise SystemExit(
            f"Expected {len(SLOTS)} embeds, got {len(matches)}. Aborting.")

    out = []
    last = 0
    for i, m in enumerate(matches):
        out.append(html[last:m.start()])
        target = SLOTS[i]
        if target is None:
            out.append(m.group(0))  # keep original
        else:
            if not target.exists():
                raise SystemExit(f"Missing {target}")
            b64 = base64.b64encode(target.read_bytes()).decode()
            out.append(f"data:image/png;base64,{b64}")
            print(f"  slot {i} -> {target.name} "
                  f"({len(target.read_bytes())} bytes -> {len(b64)} b64)")
        last = m.end()
    out.append(html[last:])
    HTML.write_text("".join(out), encoding="utf-8")
    print(f"Patched {HTML} ({HTML.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
