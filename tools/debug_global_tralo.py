"""Exact replay and two single-change diagnostics; not a hyperparameter search."""
import copy
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from tralo.global_comparison import train_arm
from tralo.global_report import evaluate_global
from tralo.events import EventLog


def run(cache, original, output):
    import torch
    from torchvision.datasets import CIFAR100
    cache, original, output = map(Path,(cache,original,output))
    output.mkdir(parents=True,exist_ok=False)
    if not torch.cuda.is_available(): raise RuntimeError('CUDA required')
    torch.backends.cuda.matmul.allow_tf32=False
    torch.use_deterministic_algorithms(True)
    config=json.loads((original/'config.json').read_text())
    caps=json.loads((original/'caps.json').read_text())['global_caps']
    assert hashlib.sha256((cache/'events.jsonl').read_bytes()).hexdigest()==config['cache_events_sha256']
    completion=json.loads((cache/'events.jsonl').read_text().splitlines()[-1])
    for name,digest in completion['artifacts'].items():
        assert hashlib.sha256((cache/name).read_bytes()).hexdigest()==digest,name
    prior=json.loads((cache/'config.json').read_text())
    for name,digest in json.loads((cache/'dataset_hashes.json').read_text()).items():
        assert hashlib.sha256((Path(prior['data_root'])/name).read_bytes()).hexdigest()==digest,name
    split=json.loads((cache/'split_indices.json').read_text())
    dataset=CIFAR100(prior['data_root'],train=True,download=False)
    x=torch.load(cache/'train_features.pt',weights_only=True,map_location='cpu').cuda()
    u=torch.load(cache/'development_features.pt',weights_only=True,map_location='cpu').cuda()
    y=torch.tensor([dataset.targets[i] for i in split['train']],device='cuda')
    saved=json.loads((cache/'development_probabilities.json').read_text())
    seed=config['seeds'][0]
    generator=torch.Generator().manual_seed(seed+1)
    orders=[torch.randperm(len(x),generator=generator) for _ in range(config['epochs'])]
    results={}

    def measure(model):
        with torch.no_grad():
            p=model(u).softmax(dim=1)
            return {'hard_counts':torch.bincount(p.argmax(1),minlength=len(caps)).tolist(),
                    'soft_counts':p.sum(0).tolist()}

    for variant in ('exact_replay','constant_rho','isolated_optimizer_state'):
        cfg=dict(config)
        if variant=='constant_rho': cfg['rho_target']=cfg['rho_initial']
        traces=[]; epochs=[]; task_state=None; constraint_state=None; before=None
        def observer(stage,phase,epoch,batch,model,optimizer):
            nonlocal task_state,constraint_state,before
            # Diagnostic intervention: separate persistent Adam states while
            # retaining the very same model, gradients, LR, batches and schedule.
            if variant=='isolated_optimizer_state' and phase=='constraint':
                if stage=='before':
                    task_state=copy.deepcopy(optimizer.state_dict())
                    if constraint_state is None:
                        constraint_state=copy.deepcopy(task_state)
                        constraint_state['state']={}
                    optimizer.load_state_dict(constraint_state)
                else:
                    constraint_state=copy.deepcopy(optimizer.state_dict())
                    optimizer.load_state_dict(task_state)
            if not (phase=='constraint' or (epoch>=7 and batch<5)):
                return
            if stage=='before':
                before=[p.detach().clone() for p in model.parameters()]
                row={'epoch':epoch,'phase':phase,'batch':batch,'before':measure(model),
                     'gradient_norm':sum(float(p.grad.square().sum()) for p in model.parameters())**.5,
                     'first_moment_norm':sum(float(optimizer.state[p].get('exp_avg',torch.zeros_like(p)).square().sum()) for p in model.parameters())**.5,
                     'second_moment_norm':sum(float(optimizer.state[p].get('exp_avg_sq',torch.zeros_like(p)).square().sum()) for p in model.parameters())**.5}
                if variant=='exact_replay' and phase=='task' and epoch==10 and batch==0:
                    branches={}
                    ids=orders[epoch-1][:cfg['batch_size']].cuda()
                    initial_loss=float(torch.nn.functional.cross_entropy(model(x[ids]),y[ids]))
                    for state_change in ('unchanged','zero_first_moment','fresh_adam'):
                        clone=copy.deepcopy(model)
                        opt=torch.optim.Adam(clone.parameters(),lr=cfg['lr'])
                        if state_change!='fresh_adam':
                            state=copy.deepcopy(optimizer.state_dict())
                            if state_change=='zero_first_moment':
                                for item in state['state'].values(): item['exp_avg'].zero_()
                            opt.load_state_dict(state)
                        for p,q in zip(clone.parameters(),model.parameters()): p.grad=q.grad.detach().clone()
                        opt.step()
                        with torch.no_grad():
                            loss=float(torch.nn.functional.cross_entropy(clone(x[ids]),y[ids]))
                            dot=sum(float((p-q).mul(q.grad).sum()) for p,q in zip(clone.parameters(),model.parameters()))
                        branches[state_change]={'task_loss_before':initial_loss,'task_loss_after':loss,
                                                'gradient_dot_displacement':dot,**measure(clone)}
                    row['one_step_counterfactuals']=branches
                traces.append(row)
            else:
                row=traces[-1]
                row['after']=measure(model)
                row['gradient_dot_displacement']=sum(float((p.detach()-b).mul(p.grad).sum()) for p,b in zip(model.parameters(),before))
        result=train_arm(x,y,u,caps,cfg,seed,'tralo',epochs.append,observer=observer)
        directory=output/variant;directory.mkdir()
        probabilities=result['probabilities'].tolist()
        if variant=='exact_replay':
            expected=torch.tensor(json.loads((original/f'{seed}_tralo/predictions.json').read_text())['probabilities'])
            assert torch.equal(result['probabilities'],expected),'diagnostic replay differs from original'
        scores=evaluate_global(probabilities,saved['labels'],caps,saved['sample_ids'])
        (directory/'trace.json').write_text(json.dumps(traces,indent=2,allow_nan=False))
        (directory/'epochs.json').write_text(json.dumps(epochs,indent=2,allow_nan=False))
        (directory/'predictions.json').write_text(json.dumps({'probabilities':probabilities,'labels':saved['labels'],'sample_ids':saved['sample_ids']},allow_nan=False))
        (directory/'report.json').write_text(json.dumps(scores,allow_nan=False))
        results[variant]={'seed':seed,'scores':{p:{k:r['metrics'][k] for k in ('accuracy','macro_f1','cc_f1')} for p,r in scores.items()},
                          'raw_counts':scores['raw']['counts'][:10],
                          'constraint_updates':result['constraint_updates'],
                          'warmup_sha256':result['warmup_sha256'],'batch_sha256':result['batch_sha256']}
        with EventLog(directory/'receipt.jsonl') as log:
            log.emit('completed',variant=variant,config=cfg,artifacts={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in directory.iterdir() if p.name!='receipt.jsonl'})
    assert len({r['warmup_sha256'] for r in results.values()})==1
    assert len({r['batch_sha256'] for r in results.values()})==1
    (output/'summary.json').write_text(json.dumps(results,indent=2,allow_nan=False))
    print(json.dumps(results,indent=2),flush=True)


if __name__=='__main__':
    if len(sys.argv)!=4: raise SystemExit('usage: debug_global_tralo.py CACHE ORIGINAL_COMPARISON OUTPUT')
    run(*sys.argv[1:])
