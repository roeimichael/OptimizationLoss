"""New7107 config and immediate device receipt before dependency/data access."""
import json
from pathlib import Path
import sys

import pytest


def test_explicit_cache_amendment_pilot_7107_config():
    from tralo.celeba_snapshot_run import campaign_config,validate_config
    root=Path(__file__).parents[1]/'experiments/configs/celeba_shared_snapshot_20261006'
    config=json.loads((root/'seed7107.json').read_text())
    assert campaign_config(7107)==config and config['pilot'] is True
    validate_config(config)
    prior=json.loads((root/'seed7106.json').read_text())
    assert {k:v for k,v in prior.items() if k!='seed'}=={k:v for k,v in config.items() if k!='seed'}
    with pytest.raises(ValueError):validate_config({**config,'seed':7108})
    with pytest.raises(ValueError):validate_config({**config,'pilot':False})


def test_device_receipt_survives_next_import_failure(tmp_path,monkeypatch):
    from tralo.celeba_snapshot_run import campaign_config,run
    from tralo import knee_snapshot_device
    device={'fictitious_CPU_test_provider':True,'requested_gpu_uuid':'test-only'}
    monkeypatch.setattr(knee_snapshot_device,'observe_single_device',lambda uuid:device)
    monkeypatch.setitem(sys.modules,'tralo.tabular_backbones',None)
    monkeypatch.setenv('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    monkeypatch.setenv('DECLARED_SOURCE','fictitious-test-source')
    config=tmp_path/'config.json';config.write_text(json.dumps(campaign_config(7107)))
    with pytest.raises(ModuleNotFoundError):run(tmp_path/'absent_public',config,tmp_path/'output','test-only')
    events=[json.loads(line) for line in (tmp_path/'output/events.jsonl').read_text().splitlines()]
    assert [e['event'] for e in events]==['device_observed','failed']
    assert events[0]['device']==device and events[0]['seed']==7107
    assert events[0]['source']=='fictitious-test-source'
    assert len(events[0]['config_sha256'])==64
    assert not (tmp_path/'output/manifest.json').exists()
