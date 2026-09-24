"""Small matched global-only TraLO comparison on verified frozen features."""
import hashlib
import json
import math
from pathlib import Path
import sys
import time
from contextlib import contextmanager

from .events import EventLog
from .global_report import evaluate_global


def validate_config(config):
    keys = {'seeds', 'epochs', 'warmup_epochs', 'batch_size', 'lr',
            'lambda_initial', 'lambda_step', 'rho_initial', 'rho_target', 'cache_events_sha256'}
    optional = {'constraint_optimizer','constraint_lr','supervised_auxiliary','auxiliary_weight','auxiliary_margin','alm_rho','alm_lambda_initial','arms'}
    if not keys <= set(config) or set(config) - keys - optional:
        raise ValueError('comparison config keys must match the declared contract')
    if 'arms' in config and (not isinstance(config['arms'],list) or not config['arms'] or len(set(config['arms']))!=len(config['arms']) or any(a not in ('clipper','tralo_null','tralo','alm','alm_null') for a in config['arms'])):
        raise ValueError('invalid arms')
    for key in ('alm_rho','alm_lambda_initial'):
        if key in config and (type(config[key]) not in (int,float) or not math.isfinite(config[key]) or config[key]<0 or (key=='alm_rho' and config[key]==0)):
            raise ValueError('invalid '+key)
    if config.get('constraint_optimizer', 'shared') not in ('shared', 'separate'):
        raise ValueError('constraint_optimizer must be shared or separate')
    if 'constraint_lr' in config:
        value=config['constraint_lr']
        if (config.get('constraint_optimizer')!='separate' or type(value) not in (int,float)
                or not math.isfinite(value) or value<=0):
            raise ValueError('constraint_lr requires separate optimizer and positive finite value')
    auxiliary = config.get('supervised_auxiliary','none')
    if auxiliary not in ('none','margin','false_positive','far_error'):
        raise ValueError('unknown supervised_auxiliary')
    for key, default in [('auxiliary_weight',0.),('auxiliary_margin',1.)]:
        value = config.get(key,default)
        if type(value) not in (int,float) or not math.isfinite(value) or value < 0:
            raise ValueError(key+' must be finite and nonnegative')
    if auxiliary == 'none' and config.get('auxiliary_weight',0.) != 0:
        raise ValueError('nonzero auxiliary weight requires a named term')
    if (type(config['cache_events_sha256']) is not str or len(config['cache_events_sha256'])!=64 or
            any(c not in '0123456789abcdef' for c in config['cache_events_sha256'])):
        raise ValueError('cache completion provenance must be pinned by SHA-256')
    if (not isinstance(config['seeds'], list) or not config['seeds'] or
            any(type(s) is not int or s < 0 for s in config['seeds']) or
            len(set(config['seeds'])) != len(config['seeds'])):
        raise ValueError('seeds must be distinct nonnegative integers')
    for key in ('epochs', 'warmup_epochs', 'batch_size'):
        if type(config[key]) is not int or config[key] <= 0:
            raise ValueError(key + ' must be a positive integer')
    if config['warmup_epochs'] >= config['epochs']:
        raise ValueError('warmup must precede a nonempty constraint phase')
    for key in ('lr', 'lambda_initial', 'lambda_step', 'rho_initial', 'rho_target'):
        value = config[key]
        if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
            raise ValueError(key + ' must be finite and nonnegative')
    if config['lr'] == 0 or config['rho_target'] < config['rho_initial']:
        raise ValueError('positive lr and nondecreasing rho required')


def _state_hash(model):
    digest = hashlib.sha256()
    for name, value in model.state_dict().items():
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


@contextmanager
def audited_arm_log(path):
    with EventLog(path) as log:
        try:
            yield log
        except Exception as error:
            log.emit('failed',error_type=type(error).__name__,message=str(error))
            raise


def train_arm(train_x, train_y, unlabelled_x, caps, config, seed, arm, emit, observer=None):
    """No evaluation-label argument. All arms use identical supervised batches."""
    import torch
    from .global_constraint import bounded_count_penalty, advance_controller
    from .sample_losses import sample_loss
    validate_config(config)
    if arm not in ('clipper', 'tralo_null', 'tralo', 'alm', 'alm_null'):
        raise ValueError('unknown arm')
    if (train_x.ndim != 2 or unlabelled_x.ndim != 2 or
            train_x.shape[1] != unlabelled_x.shape[1] or
            len(train_x) != len(train_y) or len(train_x) == 0 or len(unlabelled_x) == 0 or
            train_y.dtype != torch.long or train_y.ndim != 1 or
            not torch.isfinite(train_x).all() or not torch.isfinite(unlabelled_x).all() or
            (train_y < 0).any() or (train_y >= len(caps)).any()):
        raise ValueError('invalid training features or labels')
    if arm in ('alm','alm_null') and config.get('auxiliary_weight',0.) != 0:
        raise ValueError('ALM comparison does not mix supervised auxiliary terms')
    if arm in ('alm','alm_null') and not {'alm_rho','alm_lambda_initial'} <= set(config):
        raise ValueError('ALM settings must be explicit')
    from .alm import augmented_penalty, residuals, update_dual
    device = train_x.device
    devices = [torch.cuda.current_device()] if device.type == 'cuda' else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(seed)
        model = torch.nn.Linear(train_x.shape[1], len(caps), device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])
    constraint_optimizer_mode = config.get('constraint_optimizer', 'shared')
    # Separate moments isolate task updates from large constraint gradients.
    # Shared remains the original comparison; compare both before choosing a mode.
    constraint_optimizer = (torch.optim.Adam(model.parameters(), lr=config.get('constraint_lr',config['lr']))
                            if constraint_optimizer_mode == 'separate' else None)
    generator = torch.Generator().manual_seed(seed + 1)
    multipliers = [float(config['lambda_initial'])] * len(caps)
    rho, frozen = float(config['rho_initial']), False
    rho_step = (config['rho_target'] - rho) / (config['epochs'] - config['warmup_epochs'])
    dual = train_x.new_full((sum(k is not None for k in caps),),config.get('alm_lambda_initial',0.))
    joint_updates = dual_updates = 0
    task_updates = constraint_updates = 0
    task_attempts = constraint_attempts = 0
    planned_task = config['epochs'] * math.ceil(len(train_x)/config['batch_size'])
    auxiliary = config.get('supervised_auxiliary','none')
    auxiliary_weight = config.get('auxiliary_weight',0.)
    emit({'event':'started','seed':seed,'arm':arm,'task_updates_planned':planned_task,
          'method_definition':'joint inequality PHR ALM' if arm=='alm' else arm,
          'alm_rho':config.get('alm_rho'), 'alm_lambda_initial':config.get('alm_lambda_initial'),
          'joint_constraint_updates_planned':(config['epochs']-config['warmup_epochs'])*math.ceil(len(train_x)/config['batch_size']) if arm=='alm' else 0,
          'constraint_optimizer':constraint_optimizer_mode,
          'constraint_lr':config.get('constraint_lr',config['lr']),
          'supervised_auxiliary':auxiliary if arm != 'clipper' else 'none',
          'auxiliary_weight':auxiliary_weight if arm != 'clipper' else 0.,
          'constraint_opportunities':config['epochs']-config['warmup_epochs'] if arm=='tralo' else 0})
    batch_hash = hashlib.sha256()
    warmup_hash = None

    def abort(message):
        emit({'event':'update_failed','message':message,'task_attempted':task_attempts,
              'task_applied':task_updates,'constraint_attempted':constraint_attempts,
              'constraint_applied':constraint_updates})
        raise RuntimeError(message)

    def step(loss, phase, epoch, batch, active_optimizer):
        if not torch.isfinite(loss):
            abort('nonfinite loss; no optimizer update applied')
        active_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if any(p.grad is None or not torch.isfinite(p.grad).all() for p in model.parameters()):
            abort('missing/nonfinite gradient; no optimizer update applied')
        norm = float(torch.sqrt(sum(p.grad.square().sum() for p in model.parameters())).item())
        before = [p.detach().clone() for p in model.parameters()]
        if observer is not None:
            observer('before',phase,epoch+1,batch,model,active_optimizer)
        active_optimizer.step()
        if observer is not None:
            observer('after',phase,epoch+1,batch,model,active_optimizer)
        if any(not torch.isfinite(p).all() for p in model.parameters()):
            abort('nonfinite parameter after optimizer update; run invalid')
        displacement = float(torch.sqrt(sum((p.detach()-b).square().sum()
                                           for p,b in zip(model.parameters(),before))).item())
        return norm, displacement

    for epoch in range(config['epochs']):
        if epoch == config['warmup_epochs'] and arm != 'clipper':
            optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])
            emit({'event':'optimizer_reset', 'epoch':epoch+1})
        order = torch.randperm(len(train_x), generator=generator)
        batch_hash.update(order.numpy().tobytes())
        loss_sum = ce_sum = auxiliary_sum = 0.0
        task_gradient_max = task_displacement_max = 0.0
        alm_sum = 0.0
        for start in range(0,len(order),config['batch_size']):
            indices = order[start:start+config['batch_size']].to(device)
            logits = model(train_x[indices])
            ce = torch.nn.functional.cross_entropy(logits,train_y[indices])
            loss = ce
            extra = 0.
            if arm != 'clipper' and epoch >= config['warmup_epochs'] and auxiliary_weight > 0:
                extra = sample_loss(logits,train_y[indices],caps,auxiliary,
                                    config.get('auxiliary_margin',1.))
                loss = ce+auxiliary_weight*extra
            if arm == 'alm' and epoch >= config['warmup_epochs']:
                term=augmented_penalty(residuals(model(unlabelled_x),caps),dual,config['alm_rho'])
                loss=loss+term
                alm_sum += float(term.detach())*len(indices)
            task_attempts += 1
            grad_norm, displacement = step(loss,'task',epoch,start//config['batch_size'],optimizer)
            task_gradient_max = max(task_gradient_max,grad_norm)
            task_displacement_max = max(task_displacement_max,displacement)
            task_updates += 1
            if arm == 'alm' and epoch >= config['warmup_epochs']:
                joint_updates += 1
            loss_sum += float(loss.item())*len(indices)
            ce_sum += float(ce.item())*len(indices)
            auxiliary_sum += (float(extra.detach().item())
                              if isinstance(extra,torch.Tensor) else float(extra))*len(indices)
        alm_row={}
        if arm in ('alm','alm_null') and epoch >= config['warmup_epochs']:
            with torch.no_grad():
                g=residuals(model(unlabelled_x),caps)
                previous=dual.tolist()
                if arm=='alm':
                    dual=update_dual(g,dual,config['alm_rho'])
                    dual_updates += 1
            alm_row=dict(alm_penalty=alm_sum/len(train_x),alm_signed_residuals=g.tolist(),
                         alm_multipliers_before=previous,alm_multipliers_after=dual.tolist(),
                         alm_rho=config['alm_rho'],joint_constraint_updates_cumulative=joint_updates)
        if epoch+1 == config['warmup_epochs']:
            warmup_hash = _state_hash(model)
        row = {'event':'epoch', 'epoch':epoch+1, 'task_loss':loss_sum/len(train_x),
               'task_ce_loss':ce_sum/len(train_x), 'auxiliary_loss':auxiliary_sum/len(train_x),
               'task_gradient_norm_max':task_gradient_max,'task_displacement_norm_max':task_displacement_max,
               'task_updates_cumulative':task_updates, 'constraint_updates_cumulative':constraint_updates}
        if epoch >= config['warmup_epochs']:
            logits = model(unlabelled_x)
            probs = logits.softmax(dim=1)
            hard = torch.bincount(probs.argmax(dim=1),minlength=len(caps)).tolist()
            loss = bounded_count_penalty(logits,caps,
                    torch.tensor(multipliers,device=device,dtype=logits.dtype),rho)
            row.update(soft_counts_before=probs.detach().sum(dim=0).cpu().tolist(),
                       hard_counts_before=hard, penalty=float(loss.item()), rho=rho,
                       penalty_definition='TraLO bounded-count diagnostic; not ALM objective',
                       multipliers=list(multipliers), constraint_attempted=False,
                       constraint_applied=False, gradient_norm=0., displacement_norm=0.)
            # A zero-valued penalty gets no Adam step: existing momentum alone
            # must not move the model when the intervention is disabled/inactive.
            if arm == 'tralo' and loss.item() > 0:
                row['constraint_attempted'] = True
                constraint_attempts += 1
                norm, displacement = step(loss,'constraint',epoch,None,
                                          constraint_optimizer if constraint_optimizer is not None else optimizer)
                constraint_updates += 1
                row.update(constraint_applied=True, gradient_norm=norm, displacement_norm=displacement)
            if arm == 'tralo':
                multipliers, rho, frozen = advance_controller(
                    hard,caps,multipliers,rho,rho_step,config['lambda_step'],frozen)
                row.update(multipliers_after=list(multipliers),rho_after=rho,controller_frozen=frozen)
            with torch.no_grad():
                after = model(unlabelled_x).softmax(dim=1)
                row['hard_counts_after'] = torch.bincount(after.argmax(dim=1),minlength=len(caps)).tolist()
            row['constraint_updates_cumulative'] = constraint_updates
        row.update(alm_row)
        emit(row)
    with torch.no_grad():
        probabilities = model(unlabelled_x).softmax(dim=1).cpu()
    return {'state':{k:v.detach().cpu() for k,v in model.state_dict().items()},
            'probabilities':probabilities, 'warmup_sha256':warmup_hash,
            'batch_sha256':batch_hash.hexdigest(), 'task_updates':task_updates,
            'constraint_updates':constraint_updates, 'joint_constraint_updates':joint_updates,
            'dual_updates':dual_updates, 'task_updates_planned':planned_task,
            'task_updates_attempted':task_attempts,'task_updates_skipped':task_attempts-task_updates,
            'constraint_updates_attempted':constraint_attempts,
            'constraint_updates_skipped':constraint_attempts-constraint_updates}


def run(config_path, cache, caps_path, output):
    import torch
    from torchvision.datasets import CIFAR100
    config = json.loads(Path(config_path).read_text())
    validate_config(config)
    cache, output = Path(cache), Path(output)
    output.mkdir(parents=True,exist_ok=False)
    with EventLog(output/'events.jsonl') as log:
        try:
            if not torch.cuda.is_available():
                raise RuntimeError('CUDA required')
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            torch.use_deterministic_algorithms(True)
            log.emit('started', config=config, cache=str(cache), precision='fp32',
                     torch_version=str(torch.__version__), device=str(torch.cuda.get_device_name()),
                     source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest()
                                    for p in sorted(Path(__file__).parent.glob('*.py'))})
            cache_events=(cache/'events.jsonl').read_bytes()
            if hashlib.sha256(cache_events).hexdigest()!=config['cache_events_sha256']:
                raise ValueError('cache provenance differs from the registered feature run')
            events=[json.loads(x) for x in cache_events.decode().splitlines()]
            if events[-1]['event']!='completed': raise ValueError('feature cache incomplete')
            for name,digest in events[-1]['artifacts'].items():
                if hashlib.sha256((cache/name).read_bytes()).hexdigest()!=digest:
                    raise ValueError('cache hash mismatch: '+name)
            prior=json.loads((cache/'config.json').read_text())
            data_hashes=json.loads((cache/'dataset_hashes.json').read_text())
            for name,digest in data_hashes.items():
                if hashlib.sha256((Path(prior['data_root'])/name).read_bytes()).hexdigest()!=digest:
                    raise ValueError('dataset identity mismatch: '+name)
            split=json.loads((cache/'split_indices.json').read_text())
            saved=json.loads((cache/'development_probabilities.json').read_text())
            caps=json.loads(Path(caps_path).read_text())['global_caps']
            dataset=CIFAR100(prior['data_root'],train=True,download=False)
            from .image_baseline import sample_ids
            if (set(split['train']) & set(split['development']) or
                    len(set(split['train']))!=len(split['train']) or
                    sample_ids(split['development'])!=saved['sample_ids'] or
                    [dataset.targets[i] for i in split['development']]!=saved['labels']):
                raise ValueError('split/label identity mismatch')
            train_x=torch.load(cache/'train_features.pt',map_location='cpu',weights_only=True).to('cuda')
            unlabelled_x=torch.load(cache/'development_features.pt',map_location='cpu',weights_only=True).to('cuda')
            if len(train_x)!=len(split['train']) or len(unlabelled_x)!=len(split['development']):
                raise ValueError('feature count mismatch')
            train_y=torch.tensor([dataset.targets[i] for i in split['train']],device='cuda',dtype=torch.long)
            (output/'config.json').write_text(json.dumps(config,indent=2))
            (output/'caps.json').write_bytes(Path(caps_path).read_bytes())
            results=[]
            for seed in config['seeds']:
                identities=[]
                null_probabilities=None
                for arm in config.get('arms',('clipper','tralo_null','tralo')):
                    directory=output/f'{seed}_{arm}'; directory.mkdir()
                    started=time.monotonic()
                    with audited_arm_log(directory/'events.jsonl') as arm_log:
                        result=train_arm(train_x,train_y,unlabelled_x,caps,config,seed,arm,
                                         lambda row:arm_log.emit(row['event'],**{k:v for k,v in row.items() if k!='event'}))
                        from .comparison_checks import check_dose, check_report
                        check_dose(result,config,arm)
                        if arm=='tralo_null': null_probabilities=result['probabilities'].clone()
                        if arm=='alm_null' and null_probabilities is not None and not torch.equal(null_probabilities,result['probabilities']):
                            raise RuntimeError('matched null predictions differ')
                        predictions={'sample_ids':saved['sample_ids'],'labels':saved['labels'],
                                     'probabilities':result['probabilities'].tolist()}
                        (directory/'predictions.json').write_text(json.dumps(predictions,allow_nan=False))
                        torch.save(result['state'],directory/'head.pt')
                        scores=evaluate_global(predictions['probabilities'],predictions['labels'],caps,predictions['sample_ids'])
                        if 'alm' in config.get('arms',[]): check_report(scores,predictions['labels'],caps)
                        (directory/'report.json').write_text(json.dumps(scores,allow_nan=False))
                        identities.append((result['warmup_sha256'],result['batch_sha256']))
                        row={k:v for k,v in result.items() if k not in ('state','probabilities')}
                        row.update(seed=seed,arm=arm,seconds=time.monotonic()-started,
                                   scores={policy:{'accuracy':s['metrics']['accuracy'],
                                                   'macro_f1':s['metrics']['macro_f1'],
                                                   'cc_f1':s['metrics']['cc_f1'],
                                                   'feasible':s['feasible'],'changed':s['changed']}
                                           for policy,s in scores.items()})
                        arm_log.emit('completed',**row,artifacts={p.name:hashlib.sha256(p.read_bytes()).hexdigest()
                                                               for p in directory.iterdir() if p.name!='events.jsonl'})
                        results.append(row)
                        print(json.dumps(row),flush=True)
                if len(set(identities))!=1: raise RuntimeError('matched warmup or batches diverged')
            (output/'summary.json').write_text(json.dumps(results,indent=2,allow_nan=False))
            log.emit('completed',summary_sha256=hashlib.sha256((output/'summary.json').read_bytes()).hexdigest())
        except Exception as error:
            log.emit('failed',error_type=type(error).__name__,message=str(error))
            raise


if __name__=='__main__':
    if len(sys.argv)!=5: raise SystemExit('usage: python -m tralo.global_comparison CONFIG CACHE CAPS OUTPUT')
    run(*sys.argv[1:])
