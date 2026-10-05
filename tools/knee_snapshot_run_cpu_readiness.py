"""Fictitious four-run rehearsal of the new training/artifact/scoring components.

Usage: python tools/knee_snapshot_run_cpu_readiness.py EXCLUSIVE_OUTPUT
No real inputs, pretrained weights, scientific seeds or GPU work are used.
"""

import builtins
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time

os.environ['CUDA_VISIBLE_DEVICES'] = ''
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def run(output):
    from PIL import Image
    import torch
    from analysis.score_knee_snapshot_local import score_panel
    from tralo.knee_snapshot_data import encode, prepare, sha256
    from tralo.knee_snapshot_local import RECIPE
    from tralo.knee_snapshot_run import fit_public, verify_run
    try:
        import resource
    except ImportError:
        resource = None
    torch.set_num_threads(1)
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    source, public, private = output/'fictitious_source', output/'public', output/'private.json'
    for split,count,offset in [('train',40,1000000),('val',100,2000000)]:
        for i in range(count):
            path = source/split/str(i%5)/f'{offset+i:07d}L.png'
            path.parent.mkdir(parents=True, exist_ok=True)
            Image.new('RGB',(7,9),(i,offset//1000000,i//2)).save(path)
    (source/'test').mkdir()
    (source/'test'/'SEALED_FICTITIOUS_SENTINEL').write_text('must not be read')
    pin = prepare(source,public,private,{'train':40,'development':100})['public_manifest_sha256']
    private_pin = sha256(private.read_bytes())  # Trusted preparation of fictitious labels.
    commit = subprocess.check_output(['git','rev-parse','HEAD'],cwd=Path(__file__).resolve().parents[1],text=True).strip()
    original_open, original_path_open = builtins.open, Path.open
    private_allowed, private_reads, public_checks = False, 0, 0
    def allowed(path):
        nonlocal private_reads, public_checks
        if isinstance(path,(str,bytes,os.PathLike)):
            candidate = Path(path).resolve()
            if source == candidate or source in candidate.parents:
                raise RuntimeError('runtime accessed original/test source')
            if candidate == private:
                if not private_allowed:
                    raise RuntimeError('runtime accessed private development labels')
                private_reads += 1
            public_checks += 1
    def guarded_open(path,*args,**kwargs):
        allowed(path)
        return original_open(path,*args,**kwargs)
    def guarded_path_open(path,*args,**kwargs):
        allowed(path)
        return original_path_open(path,*args,**kwargs)
    builtins.open, Path.open = guarded_open, guarded_path_open
    try:
        runs, activations = [], {}
        for seed in (81,82,83,84):
            model = torch.nn.Sequential(torch.nn.AdaptiveAvgPool2d(1),torch.nn.Flatten(),torch.nn.Linear(3,5))
            with torch.no_grad():
                model[-1].weight.zero_(); model[-1].bias.zero_(); model[-1].bias[3] = .01
            directory = output/f'fictitious_{seed}'
            completion = fit_public(model,public,pin,dict(RECIPE,seed=seed,max_epochs=2,patience=2),directory,
                                    source_commit=commit,simulation=True)
            checked = verify_run(directory,completion,allow_simulation=True)
            activations[str(seed)] = {epoch:record['arms']['joint_local']['applied']
                                     for epoch,record in checked['result']['snapshots'].items()}
            runs.append((directory,completion))
        if private_reads:
            raise RuntimeError('private read occurred during training/verification')
        private_allowed = True  # Only the complete-panel scorer role now receives fictitious targets.
        scores = score_panel(runs,private,private_pin,allow_simulation=True)
        if private_reads != 1 or torch.cuda.is_initialized():
            raise RuntimeError('private routing or CPU boundary failed')
        with (output/'fictitious_scores.json').open('xb') as stream:
            stream.write(encode(scores))
        receipt = dict(status='passed', host=socket.gethostname(), source_commit=commit,
            source_sha256={'tools/knee_snapshot_run_cpu_readiness.py':sha256(Path(__file__).read_bytes()),
                           **checked['result']['source_sha256'],
                           'analysis/score_knee_snapshot_local.py':sha256((Path(__file__).resolve().parents[1]/'analysis/score_knee_snapshot_local.py').read_bytes())},
            fictitious_images_targets=True, scientific_seed_claims=0, cuda_initialized=False, gpu_hours=0,
            run_completion_sha256={str(81+i):pin for i,(_,pin) in enumerate(runs)},
            private_reads_during_runtime=0, private_reads_in_complete_scorer=private_reads,
            guarded_open_calls=public_checks, joint_activations=activations,
            primary_contrasts_checked=len(scores['contrasts']),
            seconds=time.monotonic()-started, peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss if resource else None,
            limitations='Tiny CPU models and two-epoch fictitious fixtures. No authentic pretrained/real-data/GPU/dose/cost/OS-isolation/seed/budget/campaign certificate.')
        with (output/'receipt.json').open('xb') as stream:
            stream.write(encode(receipt))
        print(json.dumps({key:receipt[key] for key in ['status','host','seconds','peak_rss_kib','primary_contrasts_checked']}))
    finally:
        builtins.open, Path.open = original_open, original_path_open


if __name__ == '__main__':
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    run(sys.argv[1])
