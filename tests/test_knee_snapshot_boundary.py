"""CPU-only examples for a closed-mount namespace and an owned finite child."""
import json
import sys

import pytest

from tralo.knee_snapshot_boundary import cpu_command, finite_child


PATHS=dict(runtime='/opt/fixture/runtime',release='/opt/fixture/release',
           public='/opt/fixture/public',weights='/opt/fixture/weights.pth',
           operator='/opt/fixture/operator.py',output='/opt/fixture/output')


def mounts(command, flag):
    return [(command[i+1],command[i+2]) for i,x in enumerate(command) if x==flag]


def test_namespace_mounts_only_declared_public_sources_and_exclusive_output():
    command=cpu_command(**PATHS)
    assert command[0]=='/usr/bin/bwrap'
    assert {'--unshare-all','--new-session','--die-with-parent','--clearenv'}<=set(command)
    assert mounts(command,'--bind')==[(PATHS['output'],'/output')]
    declared=dict(mounts(command,'--ro-bind'))
    assert declared[PATHS['public']]=='/public'
    assert declared[PATHS['release']]=='/release'
    assert declared[PATHS['weights']]=='/weights.pth'
    assert declared[PATHS['operator']]=='/operator.py'
    assert set(declared)=={'/usr','/lib','/lib64','/etc',PATHS['runtime'],PATHS['public'],
                           PATHS['release'],PATHS['weights'],PATHS['operator']}
    assert '--dev-bind' not in command and '--share-net' not in command and '--keep-fd' not in command
    assert command[-6:]==[PATHS['runtime']+'/bin/python','-I','-B',
                         '-X','pycache_prefix=/tmp/python-bytecode','/operator.py']
    environment={command[i+1]:command[i+2] for i,x in enumerate(command) if x=='--setenv'}
    assert environment['CUDA_VISIBLE_DEVICES']=='' and environment['HOME']=='/tmp'
    assert environment['OMP_NUM_THREADS']==environment['MKL_NUM_THREADS']=='1'


@pytest.mark.parametrize('field,value',[
    ('public','relative/public'),('output','/'),('release','/opt/fixture/../release'),
    ('weights','/usr/weights.pth'),('operator','/opt/fixture/operator.py\n'),
    ('output','/opt/fixture/public/run'),('public','/opt/fixture/runtime/public'),
    ('weights','/opt/fixture/public/weights.pth'),('operator','/opt/fixture/output/child.py'),
])
def test_unsafe_or_overlapping_sources_are_refused(field,value):
    paths=dict(PATHS,**{field:value})
    with pytest.raises(ValueError): cpu_command(**paths)


@pytest.mark.parametrize('seconds',[0,-1,float('nan'),float('inf'),True,21601])
def test_invalid_wall_bound_refuses_before_artifacts(tmp_path,seconds):
    output=tmp_path/'uncreated'
    with pytest.raises(ValueError): finite_child([sys.executable,'-c','raise AssertionError'],output,seconds)
    assert not output.exists()


def test_normal_child_receipt_binds_output_and_terminal_status(tmp_path):
    output=tmp_path/'normal'
    result=finite_child([sys.executable,'-c',"print('FICTITIOUS CPU OUTPUT')"],output,5)
    assert result['status']=='completed' and result['returncode']==0 and not result['wall_timeout']
    assert result['own_child_reaped'] and result['close_fds'] and result['pass_fds']==[]
    assert (output/'stdout').read_bytes().strip()==b'FICTITIOUS CPU OUTPUT'
    assert json.loads((output/'receipt.json').read_bytes())==result
    with pytest.raises(FileExistsError): finite_child([sys.executable,'-c','print(1)'],output,5)


def test_failed_child_preserves_negative_status(tmp_path):
    result=finite_child([sys.executable,'-c',"import sys; print('preserved failure',file=sys.stderr); sys.exit(7)"],tmp_path/'failure',5)
    assert result['status']=='child_failed' and result['returncode']==7
    assert result['own_child_reaped'] and not result['wall_timeout']
    assert (tmp_path/'failure/stderr').read_bytes().strip()==b'preserved failure'


def test_child_does_not_inherit_parent_startup_or_private_environment(tmp_path,monkeypatch):
    monkeypatch.setenv('TRALO_FICTITIOUS_PRIVATE','fictitious-secret')
    monkeypatch.setenv('PYTHONPATH',str(tmp_path/'untrusted'))
    monkeypatch.setenv('PYTHONSTARTUP',str(tmp_path/'untrusted.py'))
    script="import os; assert not any(k in os.environ for k in ['TRALO_FICTITIOUS_PRIVATE','PYTHONPATH','PYTHONSTARTUP']); print('clean startup')"
    result=finite_child([sys.executable,'-c',script],tmp_path/'environment',5)
    assert result['status']=='completed' and result['parent_environment_allowlisted']
    assert (tmp_path/'environment/stdout').read_bytes().strip()==b'clean startup'


def test_wall_timeout_kills_and_reaps_only_the_created_child(tmp_path):
    script="import time; print('owned child started',flush=True); time.sleep(20)"
    result=finite_child([sys.executable,'-c',script],tmp_path/'timeout',.5)
    assert result['status']=='wall_timeout' and result['wall_timeout']
    assert result['returncode']!=0 and result['own_child_reaped']
    assert result['elapsed_wall_seconds']<8
    assert result['signal_target']=='only_created_Popen_child'
    assert (tmp_path/'timeout/stdout').read_bytes().strip()==b'owned child started'
    assert result['descendant_termination_certified'] is False
