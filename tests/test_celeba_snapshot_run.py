"""New synthetic end-to-end dataset/batch/task/snapshot/artifact path contracts."""
import copy
import math
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch


class FeatureRows:
    def __init__(self,prefix,development=False):
        self.rows=[{'sample_id':prefix+str(i),'group':g,**({} if development else {'label':i})}
                   for i,g in enumerate(('female','male'))]
        self.development=development

    def __len__(self):return len(self.rows)

    def __getitem__(self,i):
        r=self.rows[i]
        return torch.eye(2)[i],r.get('label',-1),r['group'],r['sample_id']


def inputs():
    torch.manual_seed(9497001)
    initial=torch.nn.Linear(2,2)
    with torch.no_grad():initial.weight.zero_();initial.bias.copy_(torch.tensor([0.,2.]))
    config=dict(seed=9497001,epochs=2,warmup_epochs=1,ensemble_epochs=[1,2],batch_size=2,lr=1e-4,
                rho=.5,max_displacement=.1,dose_atol=1e-6,dose_rtol=1e-4,
                correction_scales={'level1':{'tralo':.0045,'phr':.0027},'level2':{'tralo':.0074,'phr':.0112}})
    datasets={k:FeatureRows(k,k=='development_pool') for k in ('train','stop','development_pool')}
    quotas={level:dict(global_cap=1,local_caps={'female':0,'male':1}) for level in ('level1','level2')}
    return initial,config,datasets,quotas


def test_full_new_path_uses_common_average_and_preserves_every_declared_arm(tmp_path):
    # Catches persistence/selection leakage, missing arm files or averaging the wrong saved outputs.
    from tralo.celeba_snapshot_run import fit_condition
    initial,config,data,quotas=inputs();events=[]
    result=fit_condition(initial,data,quotas,config,tmp_path/'ce',modality='image',loss_kind='ce',
                         emit=lambda event,**fields:events.append({'event':event,**fields}))
    assert result['ensemble_epochs']==[1,2] and result['task_updates']==2
    assert set(result['outputs'])=={'null',*[level+'_'+a for level in ('level1','level2')
                                             for a in ('tralo','native_phr','matched_phr','sham')]}
    for arm,artifact in result['outputs'].items():
        final=torch.load(tmp_path/'ce'/artifact['file'],weights_only=True)
        saved=[torch.load(tmp_path/'ce'/f'epoch{i:02d}'/(arm+'.pt'),weights_only=True)['probabilities'] for i in (1,2)]
        assert torch.equal(final['probabilities'],torch.stack(saved).mean(0))
        assert final['sample_ids']==['development_pool0','development_pool1']
    assert all(row['pto_unchanged'] for row in result['epochs'])
    assert all('optimizer_sha256' in row for row in result['epochs'])
    assert any(e['event']=='snapshot_arm_completed' and e['epoch']==2 for e in events)
    assert not initial.weight.grad and torch.equal(initial.bias,torch.tensor([0.,2.]))
    with pytest.raises(FileExistsError):
        fit_condition(initial,data,quotas,config,tmp_path/'ce',modality='image',loss_kind='ce',emit=lambda *a,**k:None)


def test_metadata_and_focal_have_matching_initialization_orders_and_image_streams(tmp_path):
    # Catches giving only TraLO new input/initialization/augmentation or a mismatched focal recipe.
    from tralo.celeba_snapshot_run import fit_condition
    initial,config,data,quotas=inputs()
    image=fit_condition(initial,data,quotas,config,tmp_path/'image',modality='image',loss_kind='ce',emit=lambda *a,**k:None)
    metadata=fit_condition(initial,data,quotas,config,tmp_path/'metadata',modality='image_male',loss_kind='ce',emit=lambda *a,**k:None)
    focal=fit_condition(initial,data,quotas,config,tmp_path/'focal',modality='image',loss_kind='focal',emit=lambda *a,**k:None)
    assert image['initial_image_model_sha256']==metadata['initial_image_model_sha256']==focal['initial_image_model_sha256']
    assert set(focal['outputs'])=={'focal_clip'}
    for a,b,c in zip(image['epochs'],metadata['epochs'],focal['epochs']):
        assert a['sample_order_sha256']==b['sample_order_sha256']==c['sample_order_sha256']
        assert a['first_image_batch_sha256']==b['first_image_batch_sha256']==c['first_image_batch_sha256']
        assert a['task_updates']==b['task_updates']==c['task_updates']


def test_development_target_refused_before_any_task_update(tmp_path):
    # Catches a real pool row with a target being consumed before the label boundary check.
    from tralo.celeba_snapshot_run import fit_condition
    initial,config,data,quotas=inputs();data['development_pool'].rows[0]['label']=1
    with pytest.raises(RuntimeError,match='development target'):
        fit_condition(initial,data,quotas,config,tmp_path/'bad',modality='image',loss_kind='ce',emit=lambda *a,**k:None)
    assert not (tmp_path/'bad').exists() and initial.weight.grad is None


def test_focal_gamma_two_loss_and_parameter_gradient_match_independent_formula():
    # Catches CE masquerading as focal, incorrect gamma, class reweighting or reduction.
    from tralo.celeba_snapshot_run import _task_epoch
    from tralo.celeba_snapshot_core import InputModel
    class DoubleRows(FeatureRows):
        def __getitem__(self,i):
            x,y,g,s=super().__getitem__(i)
            return x.double(),y,g,s
    base=torch.nn.Linear(2,2,dtype=torch.float64)
    with torch.no_grad():
        base.weight.copy_(torch.tensor([[0.,0.],[0.,math.log(3)]],dtype=torch.float64));base.bias.zero_()
    model=InputModel(base,'image');optimizer=torch.optim.SGD(model.parameters(),lr=0.)
    row=_task_epoch(model,optimizer,DoubleRows('train'),[0,1],9497001,1,
                    {'batch_size':2,'focal_gamma':2.},'focal')
    expected=(math.log(2)/4+math.log(4/3)/16)/2
    d0=.25*math.log(.5)-.125;d1=.09375*math.log(.75)-.015625
    assert row['training_loss']==pytest.approx(expected,abs=1e-12)
    assert torch.allclose(base.weight.grad,torch.tensor([[d0/2,-d1/2],[-d0/2,d1/2]],dtype=torch.float64),atol=1e-12,rtol=0)


def test_campaign_refuses_numeric_pilot_flag_and_unknown_scientific_settings():
    # Catches ambiguous claim/config identity, rather than prescribing generic optimizer choices.
    from tralo.celeba_snapshot_run import campaign_config,validate_config
    good=campaign_config(7100);validate_config(good)
    for invalid in ({**good,'pilot':1},{**good,'seed':True},{**good,'undeclared_feature':'Smiling'}):
        with pytest.raises(ValueError):validate_config(invalid)


def test_real_cli_refuses_bad_device_before_loading_dataset_and_preserves_failure(tmp_path):
    # Catches bypassing physical device validation or losing the exclusive failed event.
    from tralo.celeba_snapshot_run import campaign_config
    config=tmp_path/'config.json';config.write_text(json.dumps(campaign_config(7100)))
    output=tmp_path/'output';env=os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES='',CUBLAS_WORKSPACE_CONFIG=':4096:8',OMP_NUM_THREADS='1')
    run=subprocess.run([sys.executable,'-B','-m','tralo.celeba_snapshot_run',
                        str(tmp_path/'nonexistent_public'),str(config),str(output),'bad-device'],
                       env=env,capture_output=True,timeout=30)
    assert run.returncode!=0 and b'complete physical GPU UUID' in run.stderr
    events=[json.loads(line) for line in (output/'events.jsonl').read_text().splitlines()]
    assert events[-1]['event']=='failed' and not (output/'manifest.json').exists()
