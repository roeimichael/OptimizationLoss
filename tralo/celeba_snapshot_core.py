"""Shared-PTO binary snapshot corrections, metadata inputs and common averaging.

The bounded scalar and native PHR equations are reused without modification.
Matched PHR/sham are explicit comparator controllers, not renamed native PHR.
"""
import copy
import hashlib
import json
import math
import random

import numpy as np
import torch

from .tabular_constraint_gradient import apply_fixed_correction,streaming_parameter_gradient


ARMS=('null','tralo','native_phr','matched_phr','sham')


def state_hash(value):
    """Bind model/optimizer/gradient tensor values, shapes and dtypes, not pickle IDs."""
    h=hashlib.sha256()
    def visit(x):
        if isinstance(x,torch.Tensor):
            v=x.detach().cpu().contiguous()
            h.update(str((str(v.dtype),tuple(v.shape))).encode());h.update(v.numpy().tobytes())
        elif isinstance(x,dict):
            h.update(b'dict')
            for k in sorted(x,key=lambda k:(type(k).__name__,str(k))):visit(k);visit(x[k])
        elif isinstance(x,(list,tuple)):
            h.update(str((type(x).__name__,len(x))).encode())
            for item in x:visit(item)
        else:h.update(json.dumps([type(x).__name__,x],allow_nan=False).encode())
    visit(value)
    return h.hexdigest()


def encode_groups(groups,*,device,dtype):
    if not groups or any(g not in ('female','male') for g in groups):
        raise ValueError('CelebA metadata needs female/male groups')
    return torch.tensor([[float(g=='male')] for g in groups],device=device,dtype=dtype)


class InputModel(torch.nn.Module):
    """Image logits plus a zero-initialized Male coefficient for the metadata condition."""
    def __init__(self,base,modality):
        super().__init__()
        if modality not in ('image','image_male'):raise ValueError('unknown input modality')
        self.base=base;self.modality=modality
        if modality=='image_male':
            p=next(base.parameters())
            self.metadata_weight=torch.nn.Parameter(torch.zeros(2,1,device=p.device,dtype=p.dtype))

    def forward(self,value):
        if isinstance(value,tuple):images,metadata=value
        else:images,metadata=value,None
        logits=self.base(images)
        if self.modality=='image_male':
            if metadata is None or metadata.shape!=(len(logits),1):
                raise ValueError('metadata input missing or misaligned')
            if not torch.isfinite(metadata).all() or not ((metadata==0)|(metadata==1)).all():
                raise ValueError('metadata must encode the supplied Male annotation')
            logits=logits+metadata@self.metadata_weight.T
        return logits


def predict(model,batches):
    model.eval();values=[];ids=[]
    with torch.no_grad():
        for inputs,identities in batches():
            p=model(inputs).softmax(1).detach().cpu()
            if p.shape!=(len(identities),2) or not torch.isfinite(p).all():
                raise RuntimeError('invalid binary pool probabilities')
            values.append(p);ids.extend(identities)
    if not values or len(set(ids))!=len(ids):raise RuntimeError('empty/duplicate pool IDs')
    return torch.cat(values),ids


def _norm(model):
    # Preserve per-parameter float64 sums and Python addition order, with one transfer.
    squared=[p.grad.detach().double().square().sum()
             for p in model.parameters() if p.grad is not None]
    return math.sqrt(sum(torch.stack(squared).cpu().tolist())) if squared else 0.


def _no_step(reason,norm=0.):
    return dict(applied=False,reason=reason,raw_gradient_norm=norm,
                proposed_displacement_norm=0.,actual_displacement_norm=0.)


def apply_matched_correction(model,reference,maximum,*,atol,rtol):
    if (not all(math.isfinite(x) for x in (reference,maximum,atol,rtol)) or
            reference<0 or maximum<=0 or atol<0 or rtol<0):
        raise ValueError('invalid matched displacement settings')
    norm=_norm(model)
    if not math.isfinite(norm):raise RuntimeError('nonfinite matching direction')
    if reference==0:return _no_step('zero_reference',norm)
    if norm==0:raise RuntimeError('positive reference has zero matching direction')
    result=apply_fixed_correction(model,reference/norm,maximum)
    if (not result['applied'] or
            not math.isclose(result['actual_displacement_norm'],reference,abs_tol=atol,rel_tol=rtol)):
        raise RuntimeError('realized dose match failed')
    return result


def observations(probabilities,groups,quota):
    result={}
    scopes={'global':(list(range(len(groups))),quota['global_cap']),**{
        g:([i for i,x in enumerate(groups) if x==g],k) for g,k in quota['local_caps'].items()}}
    for name,(indices,cap) in scopes.items():
        soft=float(probabilities[indices,1].sum());hard=int((probabilities[indices].argmax(1)==1).sum())
        result[name]=dict(cap=cap,soft=soft,hard=hard,signed_residual=(soft-cap)/max(cap,1),hard_excess=max(0,hard-cap))
    return result


def snapshot_arms(model,optimizer,batches,groups,quota,*,tralo_scale,phr_scale,dual,
                  rho,maximum,stop_loss,sham_seed,enabled=True,atol=1e-6,rtol=1e-4,emit=None):
    """Observe corrections on isolated copies; leave the full shared task state unchanged."""
    def guard():
        return state_hash({'model':model.state_dict(),'optimizer':optimizer.state_dict(),
                           'gradients':[p.grad for p in model.parameters()],
                           'modes':[(n,m.training) for n,m in model.named_modules()]})
    original=guard();origin_hash=state_hash(model.state_dict())
    buffer_hash=state_hash(dict(model.named_buffers()))
    rng_python=random.getstate();rng_numpy=np.random.get_state()
    device=next(model.parameters()).device
    cuda_devices=[device.index] if device.type=='cuda' else []
    records={};probabilities={};next_dual=copy.deepcopy(dual);reference=0.
    phr_gradients=None;phr_record=None
    emit=emit or (lambda event,**fields:None)

    def measure(side):
        before=state_hash(side.state_dict())
        with torch.no_grad():value=float(stop_loss(side))
        if not math.isfinite(value):raise RuntimeError('nonfinite stopping loss')
        if state_hash(side.state_dict())!=before:raise RuntimeError('stopping observation changed model state')
        return value

    try:
        with torch.random.fork_rng(devices=cuda_devices):
            common_pre_loss=None;common_values=None;common_ids=None
            for arm in ARMS:
                emit('snapshot_arm_started',arm=arm,pre_model_sha256=origin_hash)
                side=copy.deepcopy(model);side.eval()
                try:
                    if state_hash(side.state_dict())!=origin_hash:raise RuntimeError('snapshot copy changed origin')
                    if arm=='null':
                        common_pre_loss=measure(side);common_values,common_ids=predict(side,batches)
                        if len(common_ids)!=len(groups):raise RuntimeError('pool/group alignment changed')
                        if state_hash(side.state_dict())!=origin_hash:
                            raise RuntimeError('origin observation changed model state')
                    gradient={'parameter_gradient_norm':0.,'active':False,'next_dual':None}
                    correction=_no_step('scheduled_zero_step')
                    evaluated=enabled and arm!='null'
                    if evaluated and arm in ('tralo','native_phr','matched_phr'):
                        method='tralo' if arm=='tralo' else 'phr'
                        if arm=='matched_phr':
                            gradient=copy.deepcopy(phr_record)
                            gradient['reused_shared_origin_phr_gradient']=True
                            for p,g in zip(side.parameters(),phr_gradients):
                                p.grad=None if g is None else g.to(p.device).clone()
                        else:
                            gradient=streaming_parameter_gradient(side,batches,groups,quota,method,rho=rho,
                                multipliers={k:1. for k in ('global',*quota['local_caps'])} if method=='tralo' else None,
                                dual=dual if method=='phr' else None,
                                first_pass=(common_values,common_ids))
                        if arm=='tralo':
                            correction=apply_fixed_correction(side,tralo_scale,maximum)
                            reference=correction['actual_displacement_norm']
                        elif arm=='native_phr':
                            phr_gradients=[None if p.grad is None else p.grad.detach().cpu().clone() for p in side.parameters()]
                            phr_record=copy.deepcopy(gradient)
                            correction=apply_fixed_correction(side,phr_scale,maximum)
                            next_dual=gradient['next_dual']
                        else:
                            if gradient['next_dual']!=next_dual:raise RuntimeError('PHR control dual histories differ')
                            correction=apply_matched_correction(side,reference,maximum,atol=atol,rtol=rtol)
                    elif evaluated and arm=='sham':
                        generator=torch.Generator(device='cpu').manual_seed(sham_seed)
                        for p in side.parameters():
                            p.grad=(torch.randn(p.shape,generator=generator,dtype=p.dtype,device='cpu').to(p.device)
                                    if p.requires_grad else None)
                        gradient['parameter_gradient_norm']=_norm(side)
                        gradient['active']=gradient['parameter_gradient_norm']>0
                        correction=apply_matched_correction(side,reference,maximum,atol=atol,rtol=rtol)
                    post_loss=common_pre_loss if arm=='null' else measure(side)
                    values,ids=(common_values,common_ids) if arm=='null' else predict(side,batches)
                    if ids!=common_ids:raise RuntimeError('snapshot sample IDs differ')
                    if state_hash(dict(side.named_buffers()))!=buffer_hash:
                        raise RuntimeError('evaluation/correction changed BN buffers')
                    if not correction['applied'] and not torch.equal(values,common_values):
                        raise RuntimeError('inactive snapshot differs from null')
                    record=dict(pre_model_sha256=origin_hash,post_model_sha256=state_hash(side.state_dict()),
                        pre_buffer_sha256=buffer_hash,post_buffer_sha256=state_hash(dict(side.named_buffers())),
                        pre_stop_loss=common_pre_loss,post_stop_loss=post_loss,
                        stop_loss_change=post_loss-common_pre_loss,gradient=gradient,
                        gradient_evaluated=evaluated,correction=correction,
                        dual_before=copy.deepcopy(dual) if 'phr' in arm else None,
                        dual_after=gradient['next_dual'] if 'phr' in arm and enabled else
                            (copy.deepcopy(dual) if 'phr' in arm else None),
                        before=observations(common_values,groups,quota),after=observations(values,groups,quota),
                        rho=rho,multipliers={k:1. for k in ('global',*quota['local_caps'])} if arm=='tralo' else None,
                        planned_updates=int(enabled and arm!='null'),attempted_updates=int(evaluated),
                        applied_updates=int(correction['applied']),skipped_updates=int(evaluated and not correction['applied']))
                    records[arm]=record;probabilities[arm]=values
                    emit('snapshot_arm_completed',arm=arm,**record)
                except BaseException as exc:
                    emit('snapshot_arm_failed',arm=arm,exception=type(exc).__name__,reason=str(exc))
                    raise
                finally:del side
    finally:
        random.setstate(rng_python);np.random.set_state(rng_numpy)
        if guard()!=original:raise RuntimeError('snapshot changed shared PTO state/optimizer/gradients/modes')
    return dict(records=records,probabilities=probabilities,next_dual=next_dual,
                sample_ids=common_ids,pto_unchanged=True,shared_guard_sha256=original)


def average_epochs(records,window):
    if not window or len(set(window))!=len(window) or any(epoch not in records for epoch in window):
        raise ValueError('missing or repeated common epoch')
    arms=set(records[window[0]])
    if any(set(records[e])!=arms for e in window):raise ValueError('common window has missing arm')
    result={}
    for arm in sorted(arms):
        values=[records[e][arm] for e in window]
        shape=values[0].shape
        if any(v.shape!=shape or v.ndim!=2 or v.shape[1]!=2 or not torch.isfinite(v).all()
               or (v<0).any() or not torch.allclose(v.sum(1),v.new_ones(len(v)),atol=1e-6,rtol=0)
               for v in values):raise ValueError('invalid common probabilities')
        result[arm]=torch.stack(values).mean(0)
    return result
