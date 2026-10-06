"""Distinct public PHR scopes, fixed native CPU examples without RNG."""
import pytest
torch=pytest.importorskip("torch")
from tralo.tabular_constraint_gradient import phr_local_logit_gradient, pooled_logit_gradient

def fixed(groups=("female","male"), cap=1):
    p=torch.tensor([[.2,.8],[.4,.6]],dtype=torch.float64)
    caps={groups[0]:0,groups[1]:1}
    dual={"global":0.,**{g:0. for g in caps}}
    return p,list(groups),cap,caps,dual,.5

def test_global_local_scope_is_refused_before_gradient_or_dual_return():
    with pytest.raises(ValueError,match="global.*reserved|reserved.*global"):
        phr_local_logit_gradient(*fixed(("global","male"))[:2],1,*fixed(("global","male"))[2:])

def test_unconstrained_global_group_also_cannot_alias_the_pooled_coefficient():
    p=torch.tensor([[.2,.8],[.4,.6]],dtype=torch.float64)
    with pytest.raises(ValueError,match="global.*reserved|reserved.*global"):
        phr_local_logit_gradient(p,["global","male"],1,1,{}, {"global":0.},.5)

def test_public_method_dispatch_preserves_the_reserved_group_refusal():
    p,groups,cap,caps,dual,rho=fixed(("global","male"))
    with pytest.raises(ValueError,match="global.*reserved|reserved.*global"):
        pooled_logit_gradient(p,groups,{"global_cap":cap,"local_caps":caps},"phr",dual=dual,rho=rho)

def test_two_scope_native_gradient_matches_independent_fixed_numbers():
    p,groups,cap,caps,dual,rho=fixed()
    grad,updated=phr_local_logit_gradient(p,groups,1,cap,caps,dual,rho)
    expected=torch.tensor([[-.096,.096],[-.048,.048]],dtype=torch.float64)
    assert torch.allclose(grad,expected,atol=1e-12,rtol=0)
    assert updated==pytest.approx({"global":.2,"female":.4,"male":0.},abs=1e-12)

def test_two_scope_native_gradient_matches_direct_scalar_autograd():
    p,groups,cap,caps,dual,rho=fixed()
    logits=p.log().detach().requires_grad_(True)
    q=logits.softmax(1)[:,1]
    pooled=torch.relu(rho*(q.sum()-1)).square()/(2*rho)
    female=torch.relu(rho*q[0]).square()/(2*rho)
    male=torch.relu(rho*(q[1]-1)).square()/(2*rho)
    (pooled+female+male).backward()
    grad,_=phr_local_logit_gradient(p,groups,1,cap,caps,dual,rho)
    assert torch.allclose(grad,logits.grad,atol=1e-12,rtol=0)

def test_valid_row_permutation_preserves_scope_math():
    p,groups,cap,caps,dual,rho=fixed()
    grad,updated=phr_local_logit_gradient(p,groups,1,cap,caps,dual,rho)
    reverse,next_dual=phr_local_logit_gradient(p.flip(0),groups[::-1],1,cap,caps,dual,rho)
    assert torch.equal(reverse.flip(0),grad) and updated==next_dual

def test_valid_group_relabel_preserves_exact_gradient_and_dual_values():
    p,groups,cap,caps,dual,rho=fixed()
    grad,updated=phr_local_logit_gradient(p,groups,1,cap,caps,dual,rho)
    p2,g2,c2,cs2,d2,r2=fixed(("H0","H1"))
    relabeled,next_dual=phr_local_logit_gradient(p2,g2,1,c2,cs2,d2,r2)
    assert torch.equal(grad,relabeled)
    assert [updated[k] for k in ["global","female","male"]]==[next_dual[k] for k in ["global","H0","H1"]]

def test_nonreserved_namespace_like_names_remain_valid_and_distinct():
    p,groups,cap,caps,dual,rho=fixed(("pooled","local:global"))
    grad,updated=phr_local_logit_gradient(p,groups,1,cap,caps,dual,rho)
    assert set(updated)=={"global","pooled","local:global"}
    assert torch.allclose(grad,torch.tensor([[-.096,.096],[-.048,.048]],dtype=torch.float64),atol=1e-12,rtol=0)

def test_slack_gradient_does_not_hide_the_reserved_scope_collision():
    p=torch.tensor([[.2,.8],[.4,.6]],dtype=torch.float64)
    with pytest.raises(ValueError,match="global.*reserved|reserved.*global"):
        phr_local_logit_gradient(p,["global","male"],1,100,{"global":100,"male":100},{"global":0.,"male":0.},.5)
