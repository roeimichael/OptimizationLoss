"""The separately declared post-namespace-fix pilot reaches the device gate."""
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


def test_declared_7106_cli_checks_device_before_any_cohort_access(tmp_path):
    from tralo.celeba_snapshot_run import campaign_config
    config=campaign_config(7106)
    assert config['pilot'] is True
    declared=Path(__file__).parents[1]/'experiments/configs/celeba_shared_snapshot_20261006/seed7106.json'
    assert json.loads(declared.read_text())==config
    path=tmp_path/'config.json';path.write_text(json.dumps(config))
    env=os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES='',CUBLAS_WORKSPACE_CONFIG=':4096:8',OMP_NUM_THREADS='1')
    output=tmp_path/'output'
    result=subprocess.run([sys.executable,'-B','-m','tralo.celeba_snapshot_run',
        str(tmp_path/'absent_public'),str(path),str(output),'bad-device'],
        env=env,capture_output=True,timeout=30)
    assert result.returncode!=0 and b'complete physical GPU UUID' in result.stderr
    events=[json.loads(line) for line in (output/'events.jsonl').read_text().splitlines()]
    assert len(events)==1 and events[0]['event']=='failed'
    assert not (output/'manifest.json').exists()


def test_no_implicit_pilot_seed_or_science_amendment():
    from tralo.celeba_snapshot_run import campaign_config,validate_config
    config=campaign_config(7106);validate_config(config)
    for change in ({'seed':7105},{'seed':7107},{'pilot':False},
                   {'epochs':5},{'dose_rtol':1e-3}):
        with pytest.raises(ValueError):validate_config({**config,**change})


def test_paired_study_seed_recipe_remains_identical():
    from tralo.celeba_snapshot_run import campaign_config,validate_config
    root=Path(__file__).parents[1]/'experiments/configs/celeba_shared_snapshot_20261006'
    for seed in range(7101,7105):
        config=json.loads((root/f'seed{seed}.json').read_text())
        assert campaign_config(seed)==config and config['pilot'] is False
        validate_config(config)
