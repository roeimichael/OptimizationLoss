"""Fictitious bytecode substitutions must not override authenticated source.

-B disables writes; it does not by itself prevent existing .pyc reads. These
independent examples substitute code while retaining accepted cache headers.
"""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import py_compile
import struct
import subprocess
import sys

import pytest

from tralo.knee_snapshot_boundary import cpu_command


PATHS = dict(runtime='/opt/fixture/runtime', release='/opt/fixture/release',
             public='/opt/fixture/public', weights='/opt/fixture/weights.pth',
             operator='/opt/fixture/operator.py', output='/opt/fixture/output')


def substituted_cache(root, mode):
    source = root / 'fixture_module.py'
    source.write_bytes(b"VALUE = 'authenticated fictitious source'\n")
    counterfeit = root / 'counterfeit.py'
    counterfeit.write_bytes(b"VALUE = 'substituted fictitious cache'\n")
    # The child negative control has no prefix. Do not inherit the parent's
    # prefix when placing the cache whose acceptance that child must expose.
    cache = root / '__pycache__' / (source.stem + '.' + sys.implementation.cache_tag + '.pyc')
    cache.parent.mkdir()
    py_compile.compile(str(counterfeit), cfile=str(cache), doraise=True,
                       invalidation_mode=mode)
    data = bytearray(cache.read_bytes())
    if mode == py_compile.PycInvalidationMode.TIMESTAMP:
        data[8:16] = struct.pack('<II', int(source.stat().st_mtime), source.stat().st_size)
    else:
        data[8:16] = importlib.util.source_hash(source.read_bytes())
    cache.write_bytes(data)
    return source, cache


def invoke(root, flags):
    code = ("import json,sys; sys.path.insert(0,sys.argv[1]); import fixture_module; "
            "print(json.dumps({'value':fixture_module.VALUE,'prefix':sys.pycache_prefix,"
            "'dont_write':sys.dont_write_bytecode}))")
    env = {'PATH': os.defpath, 'CUDA_VISIBLE_DEVICES': ''}
    if os.name == 'nt': env['SYSTEMROOT'] = os.environ['SYSTEMROOT']
    completed = subprocess.run([sys.executable, *flags, '-c', code, str(root)],
                               env=env, capture_output=True, timeout=10, check=True)
    return json.loads(completed.stdout)


@pytest.mark.parametrize('mode', list(py_compile.PycInvalidationMode))
def test_negative_control_B_alone_accepts_substituted_existing_cache(tmp_path, mode):
    source, cache = substituted_cache(tmp_path, mode)
    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    result = invoke(tmp_path, ['-I', '-B'])
    assert result['value'] == 'substituted fictitious cache'
    assert result['dont_write'] is True
    assert hashlib.sha256(source.read_bytes()).hexdigest() == source_hash


@pytest.mark.parametrize('mode', list(py_compile.PycInvalidationMode))
def test_namespace_flags_ignore_substituted_release_cache(tmp_path, mode):
    source, cache = substituted_cache(tmp_path, mode)
    before = cache.read_bytes()
    command = cpu_command(**PATHS)
    flags = command[command.index('--') + 2:-1]
    fresh = tmp_path / 'fresh_namespace_tmp_cache'
    # Map only the namespace's fresh /tmp destination into this local fixture.
    flags = [f'pycache_prefix={fresh}' if x.startswith('pycache_prefix=') else x for x in flags]
    result = invoke(tmp_path, flags)
    assert result['value'] == 'authenticated fictitious source'
    assert result['prefix'] == str(fresh)
    assert result['dont_write'] is True
    assert not fresh.exists()
    assert cache.read_bytes() == before


def test_cache_prefix_is_inside_fresh_namespace_tmp():
    command = cpu_command(**PATHS)
    assert command[command.index('--tmpfs') + 1] == '/tmp'
    flags = command[command.index('--') + 2:-1]
    assert flags == ['-I', '-B', '-X', 'pycache_prefix=/tmp/python-bytecode']
    assert '--share-net' not in command and '--dev-bind' not in command
