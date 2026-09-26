import json
import subprocess
import sys
from pathlib import Path

import torch

SCORER = Path(__file__).resolve().parents[1] / 'analysis' / 'score_step_probe.py'
N, CAP = 60, 5


def probs(top, argmax3):
    """Top-CAP by p3 on `top`; argmax 3 on `argmax3`."""
    z = torch.zeros(N, 5, dtype=torch.float64)
    z[:, 0] = 1.0
    z[list(argmax3), 3] = 2.0
    z[list(top), 3] = 5.0
    return z.softmax(1)


def write_seed(root, seed, tralo_applied=True):
    d = root / ('seed%d' % seed)
    d.mkdir()
    rows = [dict(split='val', label=3 if i < 10 else 0, sample_id='s%03d' % i) for i in range(N)]
    (d / 'manifest.json').write_text(json.dumps(dict(rows=rows)))
    torch.save(probs(range(10, 15), range(10, 20)), d / 'state_S.pt')      # base: 0 correct slots
    out = []
    for f in (1.0, 0.5):
        torch.save(probs(range(0, 5), range(0, 5)), d / ('tralo_f%s.pt' % f))  # 5 correct slots
        torch.save(probs(range(10, 15), range(10, 20)), d / ('sham_f%s.pt' % f))
        out.append(dict(kind='tralo', fraction=f, target=5, file='tralo_f%s.pt' % f, applied=tralo_applied,
                        hard_before=10, hard_after=5 if tralo_applied else 10, radius=0.1 * f))
        out.append(dict(kind='sham', fraction=f, target=5, file='sham_f%s.pt' % f, applied=True,
                        hard_before=10, hard_after=10, radius=0.1 * f))
    (d / 'probe.json').write_text(json.dumps(dict(cap=CAP, rows=out)))


def run(root):
    return subprocess.run([sys.executable, str(SCORER), str(root)], capture_output=True, text=True, check=True).stdout


def test_known_swap_is_scored_as_plus_five_with_sign_and_pairing(tmp_path):
    for s in (2601, 2602, 2603):
        write_seed(tmp_path, s)
    out = run(tmp_path)
    assert 'seeds scored: 3' in out
    assert 'tralo-sham +5.00' in out
    assert 'f=1.0 mean +5.00' in out


def test_unapplied_step_excludes_the_seed_and_missing_seeds_are_listed(tmp_path):
    write_seed(tmp_path, 2601)
    write_seed(tmp_path, 2602)
    write_seed(tmp_path, 2603, tralo_applied=False)
    out = run(tmp_path)
    assert 'seeds scored: 2' in out
    assert 'EXCLUDED seed2603' in out
    assert 'MISSING seeds' in out and '2604' in out
