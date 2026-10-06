"""Namespace command preparation, never a scientific campaign authorization.

The command builder exposes only explicit read-only public sources and one
writable output. The finite child collector closes inherited descriptors and
signals only its own unreaped Popen child. A separate real launcher still needs
source/data/device/seed/cost and certified budget gates; both CLIs stay closed.
"""
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import subprocess
import time


def cpu_command(*, runtime, release, public, weights, operator, output):
    """Build an isolated CPU command; do not open sources or execute anything.

    Callers must authenticate source bytes and refuse source symlinks before
    running this preparation command. No private tree or host GPU is mounted.
    Python isolated mode excludes user-site and PYTHONPATH startup injection.
    A cache prefix in fresh /tmp avoids existing release/library bytecode caches;
    -B alone prevents writes, but still accepts existing .pyc files.
    An operator importing product code must explicitly add /release to sys.path.
    """
    paths=dict(runtime=runtime,release=release,public=public,weights=weights,operator=operator,output=output)
    reserved=[PurePosixPath(p) for p in ['/usr','/lib','/lib64','/etc','/proc','/dev','/tmp',
                                       '/release','/public','/output','/operator.py','/weights.pth']]
    parsed={}
    for role,value in paths.items():
        if not isinstance(value,str) or any(ord(c)<32 for c in value):
            raise ValueError('invalid namespace source path: '+role)
        path=PurePosixPath(value)
        if (not path.is_absolute() or str(path)!=value or path==PurePosixPath('/')
                or '..' in path.parts or any(path==r or r in path.parents or path in r.parents for r in reserved)):
            raise ValueError('unsafe namespace source path: '+role)
        if any(path==p or p in path.parents or path in p.parents for p in parsed.values()):
            raise ValueError('namespace source/output paths overlap')
        parsed[role]=path
    command=['/usr/bin/bwrap','--unshare-all','--new-session','--die-with-parent','--clearenv',
             '--cap-drop','ALL']
    for library in ['/usr','/lib','/lib64','/etc']:
        command+=['--ro-bind',library,library]
    command+=['--proc','/proc','--dev','/dev','--tmpfs','/tmp']
    for role,destination in [('runtime',runtime),('release','/release'),('public','/public'),
                              ('weights','/weights.pth'),('operator','/operator.py')]:
        command+=['--ro-bind',paths[role],destination]
    command+=['--bind',output,'/output']
    environment=dict(PATH='/usr/bin:/bin',HOME='/tmp',TMPDIR='/tmp',LC_ALL='C.UTF-8',
                     CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',
                     TORCHINDUCTOR_CACHE_DIR='/tmp/torchinductor-cache',
                     OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
    for name,value in environment.items(): command+=['--setenv',name,value]
    return command+['--chdir','/release','--',runtime+'/bin/python','-I','-B',
                   '-X','pycache_prefix=/tmp/python-bytecode','/operator.py']


def gpu_command(*, gpu_uuid, physical_device_node, **paths):
    """Prepare one requested physical-device mount plan, without executing it.

    Caller-supplied UUID and device node are declarations, not an authenticated
    mapping or ownership observation. Before any real use, an external launcher
    must authenticate their same-host mapping and source/private/seed boundaries,
    fresh both-host exclusive ownership and certified finite budget. This builder
    neither observes CUDA nor authorizes entry; scientific CLIs remain closed.
    Control/UVM nodes are shared driver interfaces. Restricting device mounts
    alone is not a proof of CUDA/driver isolation from other physical devices.
    """
    import re
    import uuid
    pattern=r'GPU-[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}'
    if (type(gpu_uuid) is not str or re.fullmatch(pattern,gpu_uuid) is None
            or uuid.UUID(gpu_uuid[4:]).int==0
            or type(physical_device_node) is not str
            or re.fullmatch(r'/dev/nvidia(?:0|[1-9][0-9]*)',physical_device_node) is None):
        raise ValueError('one complete nonzero physical UUID and explicit physical device node are required')
    command=cpu_command(**paths)
    command[command.index('CUDA_VISIBLE_DEVICES')+1]=gpu_uuid
    devices=[physical_device_node,'/dev/nvidiactl','/dev/nvidia-uvm','/dev/nvidia-uvm-tools']
    bindings=[item for device in devices for item in ('--dev-bind',device,device)]
    index=command.index('--tmpfs')  # /dev already exists; other host devices stay unmounted.
    command[index:index]=bindings
    return command


def finite_child(command, output, wall_seconds):
    """Collect one new OWN child with an explicit finite wall and bounded cleanup.

    This generic collector is not a CPU/device or approval check. Its namespace
    command and authenticated mount plan must be checked by the caller. A normal
    wrapper exit does not certify descendant resource usage or termination.
    """
    if (type(wall_seconds) not in (int,float) or not math.isfinite(wall_seconds)
            or not 0<wall_seconds<=21600 or not isinstance(command,(list,tuple))
            or not command or any(not isinstance(x,str) for x in command)):
        raise ValueError('invalid command or finite wall bound')
    output=Path(output)
    output.mkdir(parents=True,exist_ok=False)
    environment=dict(PATH=os.defpath,LANG='C.UTF-8',LC_ALL='C.UTF-8',CUDA_VISIBLE_DEVICES='')
    if os.name=='nt' and 'SYSTEMROOT' in os.environ:
        environment['SYSTEMROOT']=os.environ['SYSTEMROOT']
    started=time.monotonic()
    child=None; timed_out=False; cleanup_unknown=False; stdout=stderr=b''; error=None
    try:
        child=subprocess.Popen(command,stdin=subprocess.DEVNULL,stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE,close_fds=True,start_new_session=True,env=environment)
        try:
            stdout,stderr=child.communicate(timeout=wall_seconds)
        except subprocess.TimeoutExpired as exc:
            timed_out=True
            stdout,stderr=exc.output or b'',exc.stderr or b''
            child.kill()  # Only the created Popen child; never an arbitrary PID/group.
            try:
                stdout,stderr=child.communicate(timeout=5)
            except subprocess.TimeoutExpired as cleanup:
                cleanup_unknown=True
                stdout,stderr=cleanup.output or stdout,cleanup.stderr or stderr
                child.poll()
        status=('cleanup_unverified' if cleanup_unknown else 'wall_timeout' if timed_out
                else 'completed' if child.returncode==0 else 'child_failed')
    except BaseException as exc:
        error=dict(exception=type(exc).__name__,reason=str(exc))
        status='collector_failed'
        if child is not None and child.poll() is None:
            child.kill()
            try: stdout,stderr=child.communicate(timeout=5)
            except subprocess.TimeoutExpired as cleanup:
                stdout,stderr=cleanup.output or stdout,cleanup.stderr or stderr
                cleanup_unknown=True
    for name,data in [('stdout',stdout),('stderr',stderr)]:
        with (output/name).open('xb') as stream: stream.write(data)
    record=dict(status=status,command=list(command),wall_seconds=wall_seconds,
                elapsed_wall_seconds=time.monotonic()-started,wall_timeout=timed_out,
                created_child_PID=None if child is None else child.pid,
                returncode=None if child is None else child.returncode,
                own_child_reaped=child is not None and child.returncode is not None,
                close_fds=True,pass_fds=[],signal_target='only_created_Popen_child',
                parent_environment_allowlisted=True,parent_environment_keys=sorted(environment),
                cleanup_unverified=cleanup_unknown,descendant_termination_certified=False,
                stdout_sha256=hashlib.sha256(stdout).hexdigest(),stderr_sha256=hashlib.sha256(stderr).hexdigest(),
                error=error,limitation='Owned child/outer wall only, not descendant resource usage, device ownership, GPUtime or campaign permission.')
    with (output/'receipt.json').open('x',encoding='utf-8') as stream:
        json.dump(record,stream,sort_keys=True,indent=2,allow_nan=False)
    return record
