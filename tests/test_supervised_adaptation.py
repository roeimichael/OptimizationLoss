import unittest
import torch
from tralo.supervised_adaptation import adapt, TrainingImages


class AdaptationTests(unittest.TestCase):
    def test_logging_preserves_state_and_dose(self):
        x=torch.tensor([[1.,0.],[0.,1.],[-1.,0.],[0.,-1.]])
        y=torch.tensor([0,1,1,0]);loader=[(x[:2],y[:2]),(x[2:],y[2:])]
        outputs=[]
        for logging in [False,True]:
            torch.manual_seed(8);model=torch.nn.Linear(2,2);events=[]
            adapt(model,loader,3,.005,events.append if logging else lambda _:None)
            outputs.append(model.state_dict())
            if logging:self.assertEqual(events[-1]['applied_updates'],6)
        for k in outputs[0]:self.assertTrue(torch.equal(outputs[0][k],outputs[1][k]))

    def test_only_training_rows_are_selected(self):
        rows=[dict(split='val',label=99),dict(split='train',label=1),dict(split='test',label=98)]
        dataset=TrainingImages('.',rows,lambda x:x)
        self.assertEqual(dataset.rows,[rows[1]])

    def test_nonfinite_training_is_rejected(self):
        model=torch.nn.Linear(2,2)
        with self.assertRaisesRegex(RuntimeError,'nonfinite'):
            adapt(model,[(torch.full((1,2),float('nan')),torch.tensor([0]))],1,.01,lambda _:None)
