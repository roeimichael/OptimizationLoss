"""Read-only dense replay: CACHE REFERENCE_ARM OUTPUT. No evaluation labels in training."""
import json
import math
from pathlib import Path
import sys
from .global_comparison import train_arm, audited_arm_log
from .knee_experiment import digest, save, cuda_setup, source


class SnapshotObserver:
    def __init__(self, features, directory):
        self.features, self.directory, self.index = features, Path(directory), []

    def __call__(self, stage, phase, epoch, batch, model, optimizer):
        import torch
        # Every task batch end, epoch start, and both sides of each constraint step.
        if phase == 'task' and stage == 'before' and batch != 0:
            return
        with torch.no_grad():
            logits = model(self.features).detach().cpu().clone()
        if not torch.isfinite(logits).all(): raise RuntimeError('nonfinite snapshot')
        payload = {'logits': logits}
        if phase == 'constraint':
            payload.update(parameters={n:p.detach().cpu().clone() for n,p in model.named_parameters()},
                gradients={n:p.grad.detach().cpu().clone() for n,p in model.named_parameters()},
                optimizer=optimizer.state_dict())
        name = '%05d.pt' % len(self.index)
        path = self.directory/name
        with path.open('xb') as f: torch.save(payload,f)
        self.index.append(dict(file=name,sha256=digest(path),stage=stage,phase=phase,
                               epoch=epoch,batch=batch))


def run(cache, reference, output):
    import torch
    cache, reference, output = map(Path,(cache,reference,output))
    config=json.loads((reference.parent/'config.json').read_text())
    caps=json.loads((reference.parent/'caps.json').read_text())
    seed_text,arm=reference.name.split('_',1);seed=int(seed_text)
    previous=[json.loads(s) for s in (reference/'events.jsonl').read_text().splitlines()][-1]
    if previous['event']!='completed':raise ValueError('incomplete reference')
    for name,sha in previous['artifacts'].items():
        if digest(reference/name)!=sha:raise ValueError('reference hash mismatch '+name)
    if digest(cache/'events.jsonl')!=config['cache_events_sha256']:raise ValueError('cache identity')
    end=[json.loads(s) for s in (cache/'events.jsonl').read_text().splitlines()][-1]
    if end['event']!='completed':raise ValueError('incomplete cache')
    for name,sha in end['artifacts'].items():
        if digest(cache/name)!=sha:raise ValueError('cache artifact '+name)
    manifest=json.loads((cache/'manifest.json').read_text())
    training=[r for r in manifest['rows'] if r['split']=='train']
    development=[r for r in manifest['rows'] if r['split']=='val']
    saved=json.loads((reference/'predictions.json').read_text())
    if saved['sample_ids'] != [r['sample_id'] for r in development]:raise ValueError('row identity')
    cuda_setup()
    x=torch.load(cache/'train.pt',weights_only=True).cuda()
    u=torch.load(cache/'val.pt',weights_only=True).cuda()
    y=torch.tensor([r['label'] for r in training],dtype=torch.long,device='cuda')
    if len(x)!=len(training) or len(u)!=len(development):raise ValueError('feature rows')
    output.mkdir(parents=True,exist_ok=False)
    with audited_arm_log(output/'events.jsonl') as log:
        log.emit('started',source_sha256=source(),reference=str(reference),cache=str(cache),
            reference_events_sha256=digest(reference/'events.jsonl'),config=config,caps=caps,
            device=str(torch.cuda.get_device_name()),precision='fp32',seed=seed,arm=arm)
        # A fresh same-host unobserved reference distinguishes hardware drift from observer effects.
        baseline=train_arm(x,y,u,caps,config,seed,arm,lambda r:None)
        snapshots=output/'snapshots';snapshots.mkdir()
        observer=SnapshotObserver(u,snapshots)
        traced=train_arm(x,y,u,caps,config,seed,arm,
            lambda row:log.emit(row['event'],**{k:v for k,v in row.items() if k!='event'}),observer)
        if not torch.equal(baseline['probabilities'],traced['probabilities']):
            raise RuntimeError('instrumentation changed probabilities')
        for name in baseline['state']:
            if not torch.equal(baseline['state'][name],traced['state'][name]):
                raise RuntimeError('instrumentation changed parameters')
        for key in ('warmup_sha256','batch_sha256','task_updates','constraint_updates'):
            if baseline[key]!=traced[key]:raise RuntimeError('instrumentation changed '+key)
        if traced['task_updates_skipped'] or traced['constraint_updates_skipped']:
            raise RuntimeError('skipped updates')
        expected_task=config['epochs']*math.ceil(len(training)/config['batch_size'])
        if traced['task_updates']!=expected_task:raise RuntimeError('task dose')
        historic=torch.load(reference/'head.pt',map_location='cpu',weights_only=True)
        historic_equal=all(torch.equal(historic[k],traced['state'][k]) for k in historic)
        historic_prob_equal=torch.equal(torch.tensor(saved['probabilities']),traced['probabilities'])
        save(output/'snapshot_index.json',observer.index)
        save(output/'identity.json',dict(sample_ids=saved['sample_ids'],caps=caps,reference=str(reference)))
        torch.save(traced['state'],output/'head.pt')
        torch.save(traced['probabilities'],output/'probabilities.pt')
        log.emit('completed',instrumentation_exact=True,historical_parameters_exact=historic_equal,
            historical_probabilities_exact=historic_prob_equal,snapshots=len(observer.index),
            task_updates=traced['task_updates'],constraint_updates=traced['constraint_updates'],
            artifacts={p.name:digest(p) for p in output.iterdir() if p.is_file() and p.name!='events.jsonl'})


if __name__=='__main__':
    if len(sys.argv)!=4:raise SystemExit(__doc__)
    run(*sys.argv[1:])
