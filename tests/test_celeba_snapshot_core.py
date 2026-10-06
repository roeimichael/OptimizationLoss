"""New native fixed CPU contracts for the CelebA shared-snapshot comparison.

Infrastructure RNG seed9497001 is declared before output; no real data or quality.
"""
import copy
import math
import random
from fractions import Fraction

import pytest
import torch


class Margins(torch.nn.Module):
    def __init__(self, probabilities=(.8,.6), dtype=torch.float64):
        super().__init__()
        q=torch.tensor(probabilities,dtype=dtype)
        self.margin=torch.nn.Parameter(torch.log(q/(1-q)))

    def forward(self,x):
        value=x@self.margin
        return torch.stack((torch.zeros_like(value),value),dim=1)


def fixture():
    torch.manual_seed(9497001)
    model=Margins()
    optimizer=torch.optim.Adam(model.parameters(),lr=1e-4)
    optimizer.state[model.margin]={'step':torch.tensor(3.),
                                   'exp_avg':torch.tensor([.2,.3],dtype=torch.float64),
                                   'exp_avg_sq':torch.tensor([.4,.5],dtype=torch.float64)}
    model.margin.grad=torch.tensor([7.,8.],dtype=torch.float64)
    def pool():yield torch.eye(2,dtype=torch.float64),['a','b']
    quota={'global_cap':1,'local_caps':{'female':0,'male':1}}
    def stop(side):
        return float(torch.nn.functional.cross_entropy(
            side(torch.eye(2,dtype=torch.float64)),torch.tensor([1,0])).detach())
    return model,optimizer,pool,quota,stop


def test_actual_bounded_step_and_matched_controls_share_origin_and_preserve_pto():
    # Catches stepping the shared model, changing Adam/grad/RNG, or normalizing TraLO itself.
    from tralo.celeba_snapshot_core import snapshot_arms
    model,opt,pool,quota,stop=fixture()
    state=copy.deepcopy(model.state_dict());adam=copy.deepcopy(opt.state_dict())
    gradients=model.margin.grad.clone();rng=torch.get_rng_state().clone();py=random.getstate()
    result=snapshot_arms(model,opt,pool,['female','male'],quota,
                         tralo_scale=.0045,phr_scale=.0027,dual={'global':0.,'female':0.,'male':0.},
                         rho=.5,maximum=.1,stop_loss=stop,sham_seed=9497001)
    assert set(result['probabilities'])=={'null','tralo','native_phr','matched_phr','sham'}
    cg=Fraction(25,49)+Fraction(250,841)
    cf=Fraction(25,81)+Fraction(500,1681)
    expected=.0045*math.hypot(float(Fraction(4,25)*(cg+cf)),float(Fraction(6,25)*cg))
    records=result['records'];D=records['tralo']['correction']['actual_displacement_norm']
    assert D==pytest.approx(expected,abs=1e-12)
    assert records['native_phr']['correction']['actual_displacement_norm']==pytest.approx(.0027*math.hypot(.096,.048),abs=1e-12)
    assert records['matched_phr']['correction']['actual_displacement_norm']==pytest.approx(D,abs=1e-12)
    assert records['sham']['correction']['actual_displacement_norm']==pytest.approx(D,abs=1e-12)
    assert result['next_dual']==pytest.approx({'global':.2,'female':.4,'male':0.},abs=1e-12)
    assert records['native_phr']['dual_after']==records['matched_phr']['dual_after']
    assert all(r['pre_model_sha256']==records['null']['pre_model_sha256'] for r in records.values())
    assert all(r['pre_stop_loss']==records['null']['pre_stop_loss'] for r in records.values())
    assert all(math.isfinite(r['post_stop_loss']) for r in records.values())
    assert torch.equal(model.margin,state['margin']) and torch.equal(model.margin.grad,gradients)
    assert model.training and torch.equal(torch.get_rng_state(),rng) and random.getstate()==py
    assert opt.state_dict()['param_groups']==adam['param_groups']
    for k,v in adam['state'][0].items():assert torch.equal(opt.state_dict()['state'][0][k],v)
    assert result['pto_unchanged']


def test_warmup_slot_has_no_correction_and_all_outputs_equal_null():
    # Catches taking a correction in the declared warm-up or falsely logging an evaluated gradient.
    from tralo.celeba_snapshot_core import snapshot_arms
    m,o,p,q,s=fixture()
    result=snapshot_arms(m,o,p,['female','male'],q,tralo_scale=.0045,phr_scale=.0027,
                         dual={'global':0.,'female':0.,'male':0.},rho=.5,maximum=.1,
                         stop_loss=s,sham_seed=9497001,enabled=False)
    for name,values in result['probabilities'].items():
        assert torch.equal(values,result['probabilities']['null'])
        assert not result['records'][name]['gradient_evaluated']
        assert result['records'][name]['correction']['actual_displacement_norm']==0
    assert result['next_dual']=={'global':0.,'female':0.,'male':0.}


def test_observer_failure_preserves_shared_model_gradients_and_rng():
    # Catches failing to restore global RNG/state when a genuine side-path observation raises.
    from tralo.celeba_snapshot_core import snapshot_arms
    m,o,p,q,_=fixture();before=m.margin.detach().clone();grad=m.margin.grad.clone()
    rng=torch.get_rng_state().clone()
    def fail(side):
        torch.rand(3)
        raise RuntimeError('fixed observer failure')
    with pytest.raises(RuntimeError,match='fixed observer failure'):
        snapshot_arms(m,o,p,['female','male'],q,tralo_scale=.0045,phr_scale=.0027,
                      dual={'global':0.,'female':0.,'male':0.},rho=.5,maximum=.1,
                      stop_loss=fail,sham_seed=9497001)
    assert torch.equal(m.margin,before) and torch.equal(m.margin.grad,grad)
    assert torch.equal(torch.get_rng_state(),rng) and m.training


def test_positive_reference_with_zero_direction_is_unmatchable():
    # Catches replacing a missing parameter direction with a fabricated vector/dose.
    from tralo.celeba_snapshot_core import apply_matched_correction
    m=Margins();m.margin.grad=torch.zeros_like(m.margin)
    with pytest.raises(RuntimeError,match='zero.*direction'):
        apply_matched_correction(m,.02,.1,atol=1e-6,rtol=1e-4)


def test_realized_fp32_mismatch_refused_even_when_below_ceiling():
    # Catches comparing intended radius instead of actual rounded parameter displacement.
    from tralo.celeba_snapshot_core import apply_matched_correction
    m=torch.nn.Module();m.p=torch.nn.Parameter(torch.tensor([2.**20,.125],dtype=torch.float32))
    m.p.grad=torch.tensor([.8,.6],dtype=torch.float32)
    with pytest.raises(RuntimeError,match='dose match'):
        apply_matched_correction(m,.0625,.1,atol=1e-6,rtol=1e-4)


def test_zero_reference_disables_active_direction_without_division():
    # Catches applying a native active gradient when the matched reference is zero.
    from tralo.celeba_snapshot_core import apply_matched_correction
    m=Margins();m.margin.grad=torch.ones_like(m.margin);before=m.margin.detach().clone()
    result=apply_matched_correction(m,0.,.1,atol=1e-6,rtol=1e-4)
    assert result['actual_displacement_norm']==0 and not result['applied']
    assert torch.equal(before,m.margin)


def test_metadata_forward_initially_matches_images_and_uses_groups_for_all_consumers():
    # Catches random metadata initialization, ignored metadata inputs or missing group routing.
    from tralo.celeba_snapshot_core import InputModel,encode_groups,predict
    torch.manual_seed(9497001)
    base=torch.nn.Linear(2,2);images=torch.tensor([[1.,2.],[3.,4.]])
    rng=torch.get_rng_state().clone()
    image=InputModel(copy.deepcopy(base),'image')
    metadata=InputModel(copy.deepcopy(base),'image_male')
    assert torch.equal(torch.get_rng_state(),rng)
    groups=encode_groups(['female','male'],device='cpu',dtype=images.dtype)
    assert groups.tolist()==[[0.],[1.]]
    assert torch.equal(image((images,groups)),metadata((images,groups)))
    with torch.no_grad():metadata.metadata_weight[1,0]=1
    a=metadata((images,groups));b=image((images,groups))
    assert torch.equal(a[0],b[0]) and float((a[1,1]-b[1,1]).detach())==pytest.approx(1.)
    def pool():yield (images,groups),['female-case','male-case']
    probabilities,ids=predict(metadata,pool)
    assert ids==['female-case','male-case'] and probabilities.shape==(2,2)
    with pytest.raises(ValueError,match='group'):encode_groups(['Smiling'],device='cpu',dtype=images.dtype)
    with pytest.raises(ValueError,match='metadata'):metadata(images)


def test_common_window_retains_corrected_outputs_without_arm_selection():
    # Catches selecting an arm-specific stopping minimum or averaging the wrong epochs.
    from tralo.celeba_snapshot_core import average_epochs
    epochs={i:{'null':torch.tensor([[1-i/10.,i/10.]]),
               'tralo':torch.tensor([[i/10.,1-i/10.]])} for i in range(1,7)}
    result=average_epochs(epochs,[4,5,6])
    assert torch.allclose(result['null'],torch.tensor([[.5,.5]]))
    assert torch.allclose(result['tralo'],torch.tensor([[.5,.5]]))
    with pytest.raises(ValueError,match='epoch'):average_epochs({1:epochs[1]},[4,5,6])


def test_common_window_refuses_nonfinite_or_missing_arm():
    # Catches silently dropping failed/missing arms from the shared ensemble.
    from tralo.celeba_snapshot_core import average_epochs
    values={1:{'null':torch.tensor([[.4,.6]]),'tralo':torch.tensor([[.3,.7]])},
            2:{'null':torch.tensor([[.5,.5]])}}
    with pytest.raises(ValueError,match='arm'):average_epochs(values,[1,2])
    values[2]['tralo']=torch.tensor([[float('nan'),.7]])
    with pytest.raises(ValueError,match='probab'):average_epochs(values,[1,2])


def test_next_task_update_matches_unobserved_reference_with_adam_bn_dropout():
    # Catches side work leaking into the next supervised Adam/BN/dropout update.
    from tralo.celeba_snapshot_core import snapshot_arms
    torch.manual_seed(9497001)
    m=torch.nn.Sequential(torch.nn.Linear(2,4),torch.nn.BatchNorm1d(4),
                          torch.nn.ReLU(),torch.nn.Dropout(.2),torch.nn.Linear(4,2))
    o=torch.optim.Adam(m.parameters(),lr=1e-4)
    x=torch.tensor([[1.,0.],[0.,1.],[1.,1.],[2.,1.]]);y=torch.tensor([0,1,1,0])
    def task(model,optimizer):
        model.train();optimizer.zero_grad(set_to_none=True)
        loss=torch.nn.functional.cross_entropy(model(x),y);loss.backward();optimizer.step()
        return float(loss.detach())
    task(m,o)
    reference=copy.deepcopy(m);ro=torch.optim.Adam(reference.parameters(),lr=1e-4)
    ro.load_state_dict(copy.deepcopy(o.state_dict()))
    def pool():yield x,['a','b','c','d']
    def stop(side):return float(torch.nn.functional.cross_entropy(side(x),y).detach())
    rng=torch.get_rng_state().clone()
    result=snapshot_arms(m,o,pool,['female','female','male','male'],
        {'global_cap':0,'local_caps':{'female':0,'male':0}},tralo_scale=.0045,phr_scale=.0027,
        dual={'global':0.,'female':0.,'male':0.},rho=.5,maximum=.1,stop_loss=stop,sham_seed=9497001)
    assert result['pto_unchanged'] and torch.equal(torch.get_rng_state(),rng)
    actual=task(m,o);torch.set_rng_state(rng);expected=task(reference,ro)
    assert actual==expected
    for k,v in reference.state_dict().items():assert torch.equal(m.state_dict()[k],v)
    for p,state in ro.state_dict()['state'].items():
        for k,v in state.items():assert torch.equal(o.state_dict()['state'][p][k],v)
