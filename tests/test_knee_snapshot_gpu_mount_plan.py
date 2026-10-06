"""Fictitious mount plans only; never observe, mount or import a real GPU."""
import os

import pytest

from tralo.knee_snapshot_boundary import cpu_command, gpu_command

PATHS=dict(runtime='/opt/fixture/runtime',release='/opt/fixture/release',
 public='/opt/fixture/public',weights='/opt/fixture/weights.pth',
 operator='/opt/fixture/operator.py',output='/opt/fixture/output')
UUID='GPU-00000000-0000-0000-0000-000000000001'


def test_one_requested_physical_node_and_control_nodes_are_the_only_new_mounts(monkeypatch):
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','fictitious-untrusted-host-selection')
    monkeypatch.setenv('NVIDIA_VISIBLE_DEVICES','all')
    before=dict(os.environ)
    cpu=cpu_command(**PATHS)
    command=gpu_command(**PATHS,gpu_uuid=UUID,physical_device_node='/dev/nvidia7')
    mounts=[(command[i+1],command[i+2]) for i,x in enumerate(command) if x=='--dev-bind']
    assert mounts==[(p,p) for p in ['/dev/nvidia7','/dev/nvidiactl','/dev/nvidia-uvm','/dev/nvidia-uvm-tools']]
    assert ('/dev','/dev') not in mounts and '--share-net' not in command and '--keep-fd' not in command
    environment={command[i+1]:command[i+2] for i,x in enumerate(command) if x=='--setenv'}
    assert environment['CUDA_VISIBLE_DEVICES']==UUID and 'NVIDIA_VISIBLE_DEVICES' not in environment
    # Removing exactly the declared device mounts/selection recovers the CPU command.
    stripped=[];i=0
    while i<len(command):
        if command[i]=='--dev-bind':i+=3;continue
        stripped.append(command[i]);i+=1
    index=stripped.index('CUDA_VISIBLE_DEVICES');stripped[index+1]=''
    assert stripped==cpu and dict(os.environ)==before


@pytest.mark.parametrize('uuid',['','0','GPU-abcd',UUID+','+UUID,'MIG-'+UUID[4:],'GPU-00000000-0000-0000-0000-000000000000'])
def test_incomplete_multiple_MIG_or_zero_requests_are_refused(uuid):
    with pytest.raises(ValueError):gpu_command(**PATHS,gpu_uuid=uuid,physical_device_node='/dev/nvidia7')


@pytest.mark.parametrize('node',['/dev','/dev/nvidiactl','/dev/nvidia-uvm','/dev/nvidia-caps/nvidia-cap1',
 '/dev/nvidia7/../nvidia8','/dev/nvidia7\n','/tmp/nvidia7'])
def test_ambiguous_or_nonphysical_device_nodes_are_refused(node):
    with pytest.raises(ValueError):gpu_command(**PATHS,gpu_uuid=UUID,physical_device_node=node)


def test_public_private_mount_rules_are_not_relaxed_for_gpu_plan():
    with pytest.raises(ValueError):gpu_command(**dict(PATHS,public='/opt/fixture/runtime/private'),
        gpu_uuid=UUID,physical_device_node='/dev/nvidia7')
