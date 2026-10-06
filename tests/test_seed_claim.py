"""Fictitious registry examples; no scientific seed or campaign authorization."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import threading

import pytest


def claim(*args):
    from tralo.seed_claim import claim_seed
    return claim_seed(*args)


def test_claim_preserves_declared_identity_without_granting_campaign_permission(tmp_path):
    identity = {'fixture': 'FICTITIOUS', 'source_sha256': 'a' * 64, 'ordinal': 3}
    result = claim(tmp_path, 9490001, identity)
    raw = (tmp_path / 'seed-9490001.json').read_bytes()
    record = json.loads(raw)
    assert record['seed'] == 9490001 and record['declared_identity'] == identity
    assert record['campaign_permission'] is False
    assert record['scope'] == 'exclusive creation in this registry only; historical freshness, device, source and budget certification remain external'
    assert result['sha256'] == hashlib.sha256(raw).hexdigest()
    assert result['file'] == str(tmp_path / 'seed-9490001.json')
    identity['fixture'] = 'CHANGED AFTER CLAIM'
    assert json.loads(raw)['declared_identity']['fixture'] == 'FICTITIOUS'


@pytest.mark.parametrize('seed', [0, -1, True, 1.5, '9490002'])
def test_invalid_seed_refuses_before_any_claim_file(tmp_path, seed):
    with pytest.raises(ValueError):
        claim(tmp_path, seed, {'fixture': 'FICTITIOUS'})
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize('identity', [{}, [], {'bad': float('nan')}, {'bad': object()}])
def test_invalid_identity_refuses_before_any_claim_file(tmp_path, identity):
    with pytest.raises((ValueError, TypeError)):
        claim(tmp_path, 9490002, identity)
    assert list(tmp_path.iterdir()) == []


def test_any_existing_claim_including_incomplete_failure_is_permanently_refused(tmp_path):
    path = tmp_path / 'seed-9490002.json'
    path.write_bytes(b'PRESERVED INCOMPLETE ORIGINAL')
    with pytest.raises(FileExistsError):
        claim(tmp_path, 9490002, {'fixture': 'FICTITIOUS NEW'})
    assert path.read_bytes() == b'PRESERVED INCOMPLETE ORIGINAL'


def test_concurrent_contenders_have_exactly_one_kernel_exclusive_winner(tmp_path):
    barrier = threading.Barrier(12)
    def contend(index):
        barrier.wait(timeout=5)
        try:
            claim(tmp_path, 9490003, {'fixture': 'FICTITIOUS', 'contender': index})
            return index
        except FileExistsError:
            return None
    with ThreadPoolExecutor(max_workers=12) as pool:
        values = list(pool.map(contend, range(12)))
    winners = [value for value in values if value is not None]
    assert len(winners) == 1
    assert json.loads((tmp_path / 'seed-9490003.json').read_bytes())['declared_identity']['contender'] == winners[0]


def test_write_failure_burns_the_exclusive_name_without_rollback_or_retry(tmp_path, monkeypatch):
    import os
    def failed_write(fd, data):
        raise OSError('FICTITIOUS WRITE FAILURE')
    with monkeypatch.context() as patch:
        patch.setattr(os, 'write', failed_write)
        with pytest.raises(OSError, match='FICTITIOUS WRITE FAILURE'):
            claim(tmp_path, 9490004, {'fixture': 'FICTITIOUS'})
    path = tmp_path / 'seed-9490004.json'
    assert path.is_file() and path.read_bytes() == b''
    with pytest.raises(FileExistsError):
        claim(tmp_path, 9490004, {'fixture': 'FICTITIOUS RETRY'})


def test_claim_is_scoped_to_a_registry_and_does_not_prove_global_historical_nonuse(tmp_path):
    a, b = tmp_path / 'a', tmp_path / 'b'
    a.mkdir(); b.mkdir()
    claim(a, 9490005, {'fixture': 'FICTITIOUS A'})
    claim(b, 9490005, {'fixture': 'FICTITIOUS B'})
    assert json.loads((a / 'seed-9490005.json').read_bytes())['campaign_permission'] is False
    assert json.loads((b / 'seed-9490005.json').read_bytes())['campaign_permission'] is False


def test_uncreated_registry_is_not_silently_created(tmp_path):
    missing = tmp_path / 'missing'
    with pytest.raises(FileNotFoundError):
        claim(missing, 9490006, {'fixture': 'FICTITIOUS'})
    assert not missing.exists()
