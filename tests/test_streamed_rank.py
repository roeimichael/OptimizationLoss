"""Exact one-step replay checks for combined training-rank and count gradients."""
import copy
import unittest

try:
    import torch
except ImportError:
    torch = None


@unittest.skipUnless(torch is not None, 'Torch is unavailable')
class StreamedRankTests(unittest.TestCase):
    def test_hard_pairs_streamed_step_matches_unchunked_adam(self):
        from tralo.streamed_rank import streamed_rank_count_step
        from tralo.hard_pair_rank import hard_pair_rank_loss
        from tralo.global_constraint import bounded_count_penalty
        torch.manual_seed(19)
        base=torch.nn.Linear(3,5,dtype=torch.float64).eval()
        train=torch.randn(5,3,dtype=torch.float64)
        val=torch.randn(4,3,dtype=torch.float64)
        labels=[3,0,3,1,2]
        ids=[f'h{i}' for i in range(5)]
        caps=[None,None,None,1,None]
        multipliers=torch.tensor([0.,0.,0.,2.,0.],dtype=torch.float64)
        oracle=copy.deepcopy(base)
        expected_optim=torch.optim.Adam(oracle.parameters(),lr=.001)
        expected_optim.zero_grad(set_to_none=True)
        rank,_=hard_pair_rank_loss(oracle(train),labels,ids,2)
        count=bounded_count_penalty(oracle(val),caps,multipliers,.5)
        (rank+count).backward()
        expected_optim.step()
        model=copy.deepcopy(base)
        optimizer=torch.optim.Adam(model.parameters(),lr=.001)
        result=streamed_rank_count_step(model,[train[:2],train[2:]],labels,ids,2,
            [val[:3],val[3:]],caps,multipliers,.5,optimizer,
            use_rank=True,use_count=True,rank_objective='hard_pairs')
        self.assertTrue(result['applied'])
        self.assertEqual(result['ranking']['active_pairs'],4)
        self.assertGreater(result['rank_logit_gradient_norm'],0)
        for actual,expected in zip(model.parameters(),oracle.parameters()):
            self.assertTrue(torch.allclose(actual,expected,atol=1e-12,rtol=1e-12))

    def test_combined_gradient_and_adam_step_match_unchunked(self):
        from tralo.streamed_rank import streamed_rank_count_step
        from tralo.cutoff_rank import cutoff_rank_loss
        from tralo.global_constraint import bounded_count_penalty
        torch.manual_seed(7)
        base=torch.nn.Sequential(torch.nn.Linear(3,8),torch.nn.BatchNorm1d(8),
                                 torch.nn.Tanh(),torch.nn.Linear(8,5)).double()
        train=torch.randn(6,3,dtype=torch.float64)
        val=torch.randn(4,3,dtype=torch.float64)
        ids=[f't{i}' for i in range(6)]
        with torch.no_grad():
            score=base.eval()(train)
            margin=score[:,3]-torch.logsumexp(torch.cat((score[:,:3],score[:,4:]),1),1)
            chosen=set(torch.topk(margin,2).indices.tolist())
        labels=[0]*6
        for index in [i for i in range(6) if i not in chosen][:2]:labels[index]=3
        caps=[None,None,None,1,None]
        multipliers=torch.tensor([0.,0.,0.,2.,0.],dtype=torch.float64)
        oracle=copy.deepcopy(base).eval()
        expected_optim=torch.optim.Adam(oracle.parameters(),lr=.001)
        expected_optim.zero_grad(set_to_none=True)
        rank,_=cutoff_rank_loss(oracle(train),labels,ids,2)
        count=bounded_count_penalty(oracle(val),caps,multipliers,.5)
        (rank+count).backward()
        expected_grad=[p.grad.clone() for p in oracle.parameters()]
        expected_optim.step()
        for chunk in (2,3):
            model=copy.deepcopy(base).train()
            bn_before=[b.clone() for b in model.buffers()]
            optimizer=torch.optim.Adam(model.parameters(),lr=.001)
            result=streamed_rank_count_step(model,
                [train[i:i+chunk] for i in range(0,len(train),chunk)],labels,ids,2,
                [val[i:i+chunk] for i in range(0,len(val),chunk)],caps,multipliers,.5,
                optimizer,use_rank=True,use_count=True)
            self.assertTrue(result['applied'])
            self.assertEqual(result['ranking']['active_pairs'],4)
            self.assertTrue(model.training)
            for p,expected,g in zip(model.parameters(),oracle.parameters(),expected_grad):
                self.assertTrue(torch.allclose(p,expected,atol=1e-12,rtol=1e-12))
                self.assertTrue(torch.allclose(p.grad,g,atol=1e-12,rtol=1e-12))
            for a,b in zip(model.buffers(),bn_before):self.assertTrue(torch.equal(a,b))

    def test_inactive_step_leaves_model_and_optimizer_untouched(self):
        from tralo.streamed_rank import streamed_rank_count_step
        model=torch.nn.Linear(2,5,dtype=torch.float64)
        before=[p.clone() for p in model.parameters()]
        optimizer=torch.optim.Adam(model.parameters(),lr=.01)
        train=[torch.tensor([[1.,0.],[0.,1.]],dtype=torch.float64)]
        val=[torch.tensor([[1.,1.]],dtype=torch.float64)]
        result=streamed_rank_count_step(model,train,[3,0],['a','b'],1,val,
            [None,None,None,99,None],torch.zeros(5,dtype=torch.float64),.5,optimizer,
            use_rank=True,use_count=True)
        self.assertFalse(result['applied'])
        self.assertEqual(len(optimizer.state),0)
        for p,expected in zip(model.parameters(),before):self.assertTrue(torch.equal(p,expected))


if __name__=='__main__':unittest.main()
