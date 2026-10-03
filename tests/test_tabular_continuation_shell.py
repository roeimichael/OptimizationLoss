from pathlib import Path


def test_embedded_python_in_continuation_is_executable() -> None:
    script = (Path(__file__).resolve().parents[1] / "tools" /
              "tabular_calibrated_fixed_after_gate.sh")
    lines = script.read_text(encoding="utf-8").splitlines()
    snippets = []
    index = 0
    while index < len(lines):
        if "<<'PY'" not in lines[index]:
            index += 1
            continue
        start = index + 1
        index = start
        while index < len(lines) and lines[index] != "PY":
            index += 1
        assert index < len(lines), f"unclosed Python heredoc at line {start}"
        snippets.append((start, "\n".join(lines[start:index])))
        index += 1
    assert snippets
    for line, body in snippets:
        compile(body, f"{script}:{line}", "exec")
