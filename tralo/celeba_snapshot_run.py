"""CelebA CE/focal trajectories with isolated bounded-loss snapshots and common output.

Usage: python -m tralo.celeba_snapshot_run PUBLIC_ROOT CONFIG NEW_OUTPUT UUID
Private development targets are never read by this runner.
"""
import copy
import hashlib
import io
import json
import math
import os
from pathlib import Path
import sys
import time

import torch
import torch.nn.functional as F

from .celeba_snapshot_core import (InputModel,average_epochs,encode_groups,predict,
                                   snapshot_arms,state_hash)


STUDY='celeba_shared_snapshot_v1'
# 7106 is separately declared after failed 7100/7105; lifecycle claims remain exclusive.
PILOT_SEEDS=(7100,7106)
PRETRAINED_SHA='5c1a416349c4cf298f2a6a5e2600ed0ee55e604713578f5e74e6bc8bcaef7997'
RECIPE=dict(study=STUDY,dataset='celeba',backbone='mobilenet_v3_large',epochs=6,
            warmup_epochs=1,ensemble_epochs=[4,5,6],batch_size=32,lr=1e-4,weight_decay=0.,
            rho=.5,max_displacement=.1,dose_atol=1e-6,dose_rtol=1e-4,
            modalities=['image','image_male'],focal_gamma=2.,
            correction_scales={'level1':{'tralo':.0045,'phr':.0027},
                               'level2':{'tralo':.0074,'phr':.0112}})


def campaign_config(seed):
    return {**copy.deepcopy(RECIPE),'seed':seed,'pilot':seed in PILOT_SEEDS}


def validate_config(config):
    if (set(config)!=set(RECIPE)|{'seed','pilot'} or type(config['seed']) is not int
            or type(config['pilot']) is not bool):
        raise ValueError('unregistered CelebA configuration')
    if (config['seed'] not in (*range(7101,7105),*PILOT_SEEDS)
            or config['pilot']!=(config['seed'] in PILOT_SEEDS)):
        raise ValueError('seed/pilot outside the prospective CelebA block')
    for key,value in RECIPE.items():
        if type(config[key]) is not type(value) or config[key]!=value:
            raise ValueError('registered CelebA setting changed: '+key)


def _sha(data):return hashlib.sha256(data).hexdigest()


def _save_json(path,value):
    with Path(path).open('x',encoding='utf-8') as f:json.dump(value,f,indent=2,sort_keys=True,allow_nan=False)


def _save_probabilities(path,probabilities,ids):
    with path.open('xb') as stream:torch.save({'probabilities':probabilities,'sample_ids':ids},stream)
    return dict(file=path.name,sha256=_sha(path.read_bytes()))


def _batches(data,order,batch_size,device):
    for start in range(0,len(order),batch_size):
        items=[data[i] for i in order[start:start+batch_size]]
        images=torch.stack([r[0] for r in items]).to(device)
        labels=torch.tensor([r[1] for r in items],dtype=torch.long,device=device)
        groups=[r[2] for r in items];ids=[r[3] for r in items]
        yield (images,encode_groups(groups,device=device,dtype=images.dtype)),labels,ids


def _pool(data,batch_size,device):
    def batches():
        for inputs,labels,ids in _batches(data,list(range(len(data))),batch_size,device):
            if (labels!=-1).any():raise RuntimeError('development target entered constraint pool')
            yield inputs,ids
    return batches


def _stop(model,data,batch_size,device):
    model.eval();total=0.
    with torch.no_grad():
        for inputs,labels,_ in _batches(data,list(range(len(data))),batch_size,device):
            total+=float(F.cross_entropy(model(inputs),labels,reduction='sum'))
    return total/len(data)


def _task_epoch(model,optimizer,data,order,seed,epoch,config,loss_kind):
    model.train();device=next(model.parameters()).device
    total=0.;updates=0;maximum=0.;first_hash=None
    for batch,start in enumerate(range(0,len(order),config['batch_size'])):
        stream_seed=seed*1000003+epoch*10007+2*batch
        torch.manual_seed(stream_seed)
        selected=order[start:start+config['batch_size']]
        inputs,labels,ids=next(_batches(data,selected,len(selected),device))
        if first_hash is None:first_hash=state_hash(inputs[0])
        torch.manual_seed(stream_seed+1)
        optimizer.zero_grad(set_to_none=True);logits=model(inputs)
        if loss_kind=='ce':loss=F.cross_entropy(logits,labels)
        else:
            logp=logits.log_softmax(1).gather(1,labels[:,None]).squeeze(1)
            loss=(-(1-logp.exp()).pow(config.get('focal_gamma',2.))*logp).mean()
        if not torch.isfinite(loss):raise RuntimeError('nonfinite task loss')
        loss.backward()
        norm=math.sqrt(sum(float(p.grad.detach().double().square().sum()) for p in model.parameters() if p.grad is not None))
        if not math.isfinite(norm):raise RuntimeError('nonfinite task gradient')
        optimizer.step();updates+=1;maximum=max(maximum,norm);total+=float(loss.detach())*len(selected)
    return dict(training_loss=total/len(order),task_updates=updates,max_task_gradient_norm=maximum,
                first_image_batch_sha256=first_hash,
                sample_order_sha256=_sha(json.dumps([data.rows[i]['sample_id'] for i in order],separators=(',',':')).encode()))


def fit_condition(initial,datasets,quotas,config,directory,*,modality,loss_kind,emit):
    """Complete a general declared task schedule; tests may provide small fixed datasets."""
    if loss_kind not in ('ce','focal'):raise ValueError('unknown task loss')
    if any('label' in row for row in datasets['development_pool'].rows):
        raise RuntimeError('development target present before task training')
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=False)
    model=InputModel(copy.deepcopy(initial),modality);device=next(model.parameters()).device
    optimizer=torch.optim.Adam(model.parameters(),lr=config['lr'],weight_decay=0.)
    image_initial=state_hash(model.base.state_dict())
    initial_hash=state_hash(model.state_dict())
    generator=torch.Generator(device='cpu').manual_seed(config['seed']+105729)
    orders=[torch.randperm(len(datasets['train']),generator=generator).tolist() for _ in range(config['epochs'])]
    pool=_pool(datasets['development_pool'],config['batch_size'],device)
    groups=[r['group'] for r in datasets['development_pool'].rows]
    expected_ids=[r['sample_id'] for r in datasets['development_pool'].rows]
    duals={level:{k:0. for k in ('global',*q['local_caps'])} for level,q in quotas.items()}
    rows=[];epoch_values={}
    cuda_devices=[device.index] if device.type=='cuda' else []
    with torch.random.fork_rng(devices=cuda_devices):
        for epoch,order in enumerate(orders,1):
            started=time.monotonic()
            row=_task_epoch(model,optimizer,datasets['train'],order,config['seed'],epoch,config,loss_kind)
            row.update(epoch=epoch,model_sha256=state_hash(model.state_dict()),optimizer_sha256=state_hash(optimizer.state_dict()))
            epoch_directory=directory/f'epoch{epoch:02d}';epoch_directory.mkdir()
            values={};corrections={}
            stop=lambda side:_stop(side,datasets['stop'],config['batch_size'],device)
            if loss_kind=='focal':
                side=copy.deepcopy(model);side.eval()
                values['focal_clip'],ids=predict(side,pool);row['stop_loss']=stop(side);del side
                row['pto_unchanged']=True
            else:
                for level,quota in quotas.items():
                    scales=config['correction_scales'][level]
                    result=snapshot_arms(model,optimizer,pool,groups,quota,
                        tralo_scale=scales['tralo'],phr_scale=scales['phr'],dual=duals[level],rho=config['rho'],
                        maximum=config['max_displacement'],stop_loss=stop,
                        sham_seed=config['seed']*1000003+epoch*10007+int(level[-1])*701,
                        enabled=epoch>config['warmup_epochs'],atol=config['dose_atol'],rtol=config['dose_rtol'],
                        emit=lambda event,**fields:emit(event,modality=modality,loss_kind=loss_kind,level=level,epoch=epoch,**fields))
                    duals[level]=result['next_dual'];ids=result['sample_ids']
                    if 'null' in values and not torch.equal(values['null'],result['probabilities']['null']):
                        raise RuntimeError('cap-dependent null output')
                    values['null']=result['probabilities']['null']
                    values.update({level+'_'+arm:p for arm,p in result['probabilities'].items() if arm!='null'})
                    corrections[level]=result['records'];row['stop_loss']=result['records']['null']['pre_stop_loss']
                row['pto_unchanged']=True
            if ids!=expected_ids:raise RuntimeError('epoch output IDs differ from public pool')
            artifacts={arm:_save_probabilities(epoch_directory/(arm+'.pt'),p,ids) for arm,p in values.items()}
            epoch_values[epoch]=values;row.update(artifacts=artifacts,corrections=corrections,elapsed_seconds=time.monotonic()-started)
            rows.append(row);emit('task_epoch_completed',modality=modality,loss_kind=loss_kind,**row)
    averaged=average_epochs(epoch_values,config['ensemble_epochs'])
    outputs={arm:_save_probabilities(directory/(arm+'.pt'),p,expected_ids) for arm,p in averaged.items()}
    with (directory/'task_final_model.pt').open('xb') as f:
        torch.save({k:v.detach().cpu().clone() for k,v in model.state_dict().items()},f)
    result=dict(modality=modality,loss_kind=loss_kind,initial_image_model_sha256=image_initial,
                initial_model_sha256=initial_hash,ensemble_epochs=config['ensemble_epochs'],epochs=rows,
                task_updates=sum(r['task_updates'] for r in rows),outputs=outputs,
                sample_ids_sha256=_sha(json.dumps(expected_ids,separators=(',',':')).encode()),
                task_final_model_sha256=_sha((directory/'task_final_model.pt').read_bytes()))
    _save_json(directory/'summary.json',result)
    return result


def make_initial(weight_path,seed):
    """Authenticate and load the SAME cached ImageNet bytes; no downloads or old model replay."""
    from torchvision import models
    path=Path(weight_path)
    if path.is_symlink() or not path.is_file():raise RuntimeError('pinned weight missing/linked')
    raw=path.read_bytes()
    if _sha(raw)!=PRETRAINED_SHA:raise RuntimeError('pinned ImageNet V2 bytes changed')
    torch.manual_seed(seed)
    model=models.mobilenet_v3_large(weights=None)
    state=torch.load(io.BytesIO(raw),weights_only=True,map_location='cpu')
    model.load_state_dict(state,strict=True)
    model.classifier[3]=torch.nn.Linear(model.classifier[3].in_features,2)
    return model,dict(sha256=_sha(raw),bytes=len(raw),constructor='native 1000-way strict V2 load then fresh 2-way head')


def run(public_root,config_path,output,uuid):
    from .events import EventLog
    from .knee_snapshot_device import observe_single_device
    config_path=Path(config_path);raw=config_path.read_bytes();config=json.loads(raw);validate_config(config)
    if os.environ.get('CUBLAS_WORKSPACE_CONFIG')!=':4096:8':
        raise RuntimeError('deterministic CUDA requires declared CUBLAS_WORKSPACE_CONFIG=:4096:8 before initialization')
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    with EventLog(output/'events.jsonl') as log:
        try:
            device_info=observe_single_device(uuid)
            from .tabular_backbones import configure_fp32,image_transforms
            from .tabular_image_data import PreparedImageRows,load_runner_cohort
            from .tabular_quota_policy import caps_for_unlabeled_pool
            configure_fp32();torch.use_deterministic_algorithms(True)
            manifest,rows=load_runner_cohort(public_root,'celeba')
            if manifest.get('target')!='Smiling' or manifest.get('group_attribute')!='Male':
                raise RuntimeError('CelebA task/group identity changed')
            quotas=caps_for_unlabeled_pool('celeba',[r['group'] for r in rows['development_pool']])
            train_tf,eval_tf=image_transforms('mobilenet_v3_large')
            data={k:PreparedImageRows(manifest['image_dir'],v,train_tf if k=='train' else eval_tf) for k,v in rows.items()}
            initial,weight=make_initial(Path(torch.hub.get_dir())/'checkpoints'/'mobilenet_v3_large-5c1a4163.pth',config['seed'])
            initial=initial.to(torch.device('cuda:0'))
            source_hashes={str(p.relative_to(Path(__file__).parent)): _sha(p.read_bytes())
                           for p in Path(__file__).parent.glob('*.py')}
            identity=dict(config_sha256=_sha(raw),public_manifest_sha256=_sha((Path(public_root)/'manifest.json').read_bytes()),
                runner_files_sha256=manifest['files_sha256'],source_hashes=source_hashes,
                weight=weight,device=device_info,precision='fp32_tf32_off',development_targets_loaded=False,
                cublas_workspace_config=os.environ['CUBLAS_WORKSPACE_CONFIG'],
                initialization_sha256=state_hash(initial.state_dict()),split_sizes={k:len(v) for k,v in rows.items()})
            _save_json(output/'manifest.json',dict(identity=identity,config=config,quotas=quotas))
            (output/'config.json').write_bytes(raw);log.emit('started',**identity)
            results={}
            for modality in config['modalities']:
                for loss in ('ce','focal'):
                    name=modality+'_'+loss
                    log.emit('condition_started',condition=name)
                    results[name]=fit_condition(initial,data,quotas,config,output/name,
                                                modality=modality,loss_kind=loss,emit=log.emit)
                    log.emit('condition_completed',condition=name,summary_sha256=_sha((output/name/'summary.json').read_bytes()))
            for modality in config['modalities']:
                ce=results[modality+'_ce'];focal=results[modality+'_focal']
                for a,b in zip(ce['epochs'],focal['epochs']):
                    if any(a[k]!=b[k] for k in ('task_updates','sample_order_sha256','first_image_batch_sha256')):
                        raise RuntimeError('CE/focal task input schedule differs')
            _save_json(output/'summary.json',dict(config=config,identity=identity,quotas=quotas,conditions=results))
            log.emit('completed',summary_sha256=_sha((output/'summary.json').read_bytes()))
        except BaseException as exc:
            log.emit('failed',exception=type(exc).__name__,reason=str(exc));raise


if __name__=='__main__':
    if len(sys.argv)!=5:raise SystemExit(__doc__)
    run(*sys.argv[1:])
