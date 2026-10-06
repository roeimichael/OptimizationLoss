"""New cache-environment examples derived from the preserved import failure."""
from tralo.knee_snapshot_boundary import cpu_command

PATHS=dict(runtime='/opt/cache/runtime',release='/opt/cache/release',
           public='/opt/cache/public',weights='/opt/cache/weights.pth',
           operator='/opt/cache/operator.py',output='/opt/cache/output')


def environment(command):
    return {command[i+1]:command[i+2] for i,value in enumerate(command) if value=='--setenv'}


def test_compiler_cache_has_an_explicit_path_in_fresh_namespace_tmp():
    command=cpu_command(**PATHS)
    assert environment(command)['TORCHINDUCTOR_CACHE_DIR']=='/tmp/torchinductor-cache'
    assert command[command.index('--tmpfs')+1]=='/tmp'
    assert '--clearenv' in command


def test_parent_compiler_cache_or_identity_is_not_forwarded(monkeypatch):
    monkeypatch.setenv('TORCHINDUCTOR_CACHE_DIR','/opt/cache/fictitious-private-cache')
    monkeypatch.setenv('USER','fictitious-host-user')
    selected=environment(cpu_command(**PATHS))
    assert selected['TORCHINDUCTOR_CACHE_DIR']=='/tmp/torchinductor-cache'
    assert 'USER' not in selected
    assert '/opt/cache/fictitious-private-cache' not in selected.values()
