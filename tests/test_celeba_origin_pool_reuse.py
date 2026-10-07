"""New cases for reusing the exact unchanged-origin constraint probability pass."""
import copy

import pytest
import torch

from tralo import celeba_snapshot_core as core
from tralo.tabular_constraint_gradient import streaming_parameter_gradient


GROUPS=['female','male','female','male','female']
QUOTA={'global_cap':1,'local_caps':{'female':1,'male':1}}
INPUTS=torch.tensor([[.2,-.8],[1.1,.4],[-.7,1.3],[.9,-1.2],[.5,.6]])
IDS=['new-reuse-a','new-reuse-b','new-reuse-c','new-reuse-d','new-reuse-e']


def fixed_model():
    model=torch.nn.Sequential(torch.nn.Linear(2,3),torch.nn.BatchNorm1d(3),
                              torch.nn.Tanh(),torch.nn.Linear(3,2))
    with torch.no_grad():
        for index,parameter in enumerate(model.parameters()):
            parameter.copy_(torch.arange(parameter.numel()).reshape(parameter.shape)*.03+.07*(index+1))
    return model.eval()


def pool(counter):
    def batches():
        counter.append('opened')
        for start in (0,2,4):yield INPUTS[start:start+2],IDS[start:start+2]
    return batches


@pytest.mark.parametrize('method',['tralo','phr'])
def test_origin_probability_reuse_preserves_exact_gradient_dual_and_model(method):
    origin=fixed_model();reference=copy.deepcopy(origin);reused=copy.deepcopy(origin)
    before=core.state_hash(reused.state_dict())
    first,ids=core.predict(origin,pool([]));first_before=first.clone()
    options={'multipliers':{'global':1.,'female':1.,'male':1.}} if method=='tralo' else {'dual':{'global':.2,'female':.1,'male':.3}}
    reference_calls=[];reused_calls=[]
    expected=streaming_parameter_gradient(reference,pool(reference_calls),GROUPS,QUOTA,method,**options)
    actual=streaming_parameter_gradient(reused,pool(reused_calls),GROUPS,QUOTA,method,
                                        first_pass=(first,ids),**options)
    assert len(reference_calls)==2 and len(reused_calls)==1
    assert {k:v for k,v in actual.items() if k!='first_pass_reused'}==expected
    assert actual['first_pass_reused'] is True
    assert all(torch.equal(a.grad,b.grad) for a,b in zip(reference.parameters(),reused.parameters()))
    assert core.state_hash(reused.state_dict())==before and torch.equal(first,first_before)


def test_reused_origin_rejects_changed_model_probabilities():
    model=fixed_model();first,ids=core.predict(model,pool([]))
    with torch.no_grad():model[-1].bias[1].add_(.2)
    with pytest.raises(RuntimeError,match='changed probabilities'):
        streaming_parameter_gradient(model,pool([]),GROUPS,QUOTA,'phr',
            dual={'global':0.,'female':0.,'male':0.},first_pass=(first,ids))


def test_reused_origin_rejects_changed_pool_order():
    model=fixed_model();first,ids=core.predict(model,pool([]))
    with pytest.raises(RuntimeError,match='order changed'):
        streaming_parameter_gradient(model,pool([]),GROUPS,QUOTA,'phr',
            dual={'global':0.,'female':0.,'male':0.},first_pass=(first,list(reversed(ids))))


def test_reused_origin_rejects_nonfinite_probabilities():
    model=fixed_model();first,ids=core.predict(model,pool([]));first[0,0]=float('nan')
    with pytest.raises(RuntimeError,match='invalid reused'):
        streaming_parameter_gradient(model,pool([]),GROUPS,QUOTA,'phr',
            dual={'global':0.,'female':0.,'male':0.},first_pass=(first,ids))


def test_snapshot_uses_null_origin_without_extra_coefficient_forward_passes(monkeypatch):
    model=fixed_model();optimizer=torch.optim.Adam(model.parameters(),lr=.0001)
    for p in model.parameters():p.grad=torch.full_like(p,.125)
    shared_before=core.state_hash((model.state_dict(),optimizer.state_dict(),[p.grad for p in model.parameters()]))
    actual_calls=[]
    kwargs=dict(tralo_scale=.0045,phr_scale=.0027,dual={'global':.2,'female':.1,'male':.3},
                rho=.5,maximum=.1,stop_loss=lambda side:side(INPUTS).square().sum()/len(INPUTS),sham_seed=123)
    actual=core.snapshot_arms(model,optimizer,pool(actual_calls),GROUPS,QUOTA,**kwargs)
    assert len(actual_calls)==7
    implementation=core.streaming_parameter_gradient
    def original_first_pass(*args,**kwargs):
        kwargs.pop('first_pass',None)
        return implementation(*args,**kwargs)
    monkeypatch.setattr(core,'streaming_parameter_gradient',original_first_pass)
    reference_calls=[]
    expected=core.snapshot_arms(model,optimizer,pool(reference_calls),GROUPS,QUOTA,**kwargs)
    assert len(reference_calls)==9
    for arm in core.ARMS:
        assert torch.equal(actual['probabilities'][arm],expected['probabilities'][arm])
        record=copy.deepcopy(actual['records'][arm]);record['gradient'].pop('first_pass_reused',None)
        assert record==expected['records'][arm]
    assert actual['next_dual']==expected['next_dual'] and actual['sample_ids']==expected['sample_ids']
    assert core.state_hash((model.state_dict(),optimizer.state_dict(),[p.grad for p in model.parameters()]))==shared_before
    assert not torch.cuda.is_initialized()
