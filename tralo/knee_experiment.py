"""Chen knee data: audited frozen ResNet18 features and matched global arms.

prepare DATA_ROOT CACHE [ADAPTATION_SEED]; compare CACHE CONFIG OUTPUT.
Test images are audited for split overlap but never featurized or scored here.
"""
import hashlib
import json
from pathlib import Path
import sys
from .events import EventLog
from .knee_data import audit
from .global_comparison import train_arm, audited_arm_log, validate_config
from .global_report import evaluate_global


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024*1024), b''): h.update(chunk)
    return h.hexdigest()


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False))


def cuda_setup():
    import torch
    if not torch.cuda.is_available(): raise RuntimeError('CUDA required')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)


def source():
    return {p.name:digest(p) for p in sorted(Path(__file__).parent.glob('*.py'))}


def prepare(data_root, output, adaptation_seed=None):
    import torch
    from torchvision import models, transforms
    from PIL import Image
    output = Path(output); output.mkdir(parents=True, exist_ok=False)
    with audited_arm_log(output/'events.jsonl') as log:
        cuda_setup()
        manifest = audit(data_root)
        save(output/'manifest.json', manifest)
        log.emit('data_audit_passed', counts=manifest['counts'], subjects=manifest['subjects'],
                 duplicates=manifest['within_split_pixel_duplicates'], source_sha256=source())
        weights = models.ResNet18_Weights.IMAGENET1K_V1
        if adaptation_seed is not None:torch.manual_seed(int(adaptation_seed))
        model = models.resnet18(weights=weights).cuda().eval()
        transform = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor(),
            transforms.Normalize((.485,.456,.406),(.229,.224,.225))])
        weight_path = Path(torch.hub.get_dir())/'checkpoints'/weights.url.rsplit('/',1)[-1]
        log.emit('model_loaded', backbone='ResNet18 ImageNet initialization', weights_sha256=digest(weight_path),
                 precision='fp32', device=str(torch.cuda.get_device_name()),
                 transform='RGB, resize224x224, ImageNet normalization, no augmentation')
        if adaptation_seed is not None:
            from .supervised_adaptation import adapt, TrainingImages
            model.fc=torch.nn.Linear(model.fc.in_features,5).cuda()
            dataset=TrainingImages(data_root,manifest['rows'],transform)
            loader=torch.utils.data.DataLoader(dataset,batch_size=32,shuffle=True,num_workers=0,
                generator=torch.Generator().manual_seed(int(adaptation_seed)+10000))
            log.emit('adaptation_config',seed=int(adaptation_seed),epochs=5,lr=.0001,
                     batch_size=32,scope='training only; final epoch; no validation selection')
            adapt(model,loader,5,.0001,
                  lambda row:log.emit(row['event'],**{k:v for k,v in row.items() if k!='event'}))
            torch.save(model.state_dict(),output/'adapted_backbone.pt')
        model.fc = torch.nn.Identity()
        model.requires_grad_(False)
        model.eval()
        for split in ('train','val'):
            rows = [r for r in manifest['rows'] if r['split']==split]
            features = []
            with torch.no_grad():
                for start in range(0,len(rows),64):
                    batch=[]
                    for row in rows[start:start+64]:
                        p=Path(data_root)/row['path']
                        if digest(p)!=row['sha256']: raise RuntimeError('image changed after audit')
                        with Image.open(p) as image: batch.append(transform(image.convert('RGB')))
                    value=model(torch.stack(batch).cuda()).cpu()
                    if value.shape != (len(batch),512) or not torch.isfinite(value).all():
                        raise RuntimeError('invalid features')
                    features.append(value)
                    log.emit('feature_batch', split=split, completed=start+len(batch), total=len(rows))
            torch.save(torch.cat(features),output/(split+'.pt'))
        log.emit('completed',artifacts={p.name:digest(p) for p in output.iterdir() if p.name!='events.jsonl'})


def compare(cache, config_path, output):
    import torch
    from sklearn.metrics import f1_score, accuracy_score
    cache, output = Path(cache), Path(output)
    config=json.loads(Path(config_path).read_text()); validate_config(config)
    if digest(cache/'events.jsonl')!=config['cache_events_sha256']: raise ValueError('cache identity mismatch')
    events=[json.loads(s) for s in (cache/'events.jsonl').read_text().splitlines()]
    if events[-1]['event']!='completed': raise ValueError('incomplete feature cache')
    for name, sha in events[-1]['artifacts'].items():
        if digest(cache/name)!=sha: raise ValueError('modified cache artifact: '+name)
    manifest=json.loads((cache/'manifest.json').read_text())
    training=[r for r in manifest['rows'] if r['split']=='train']
    development=[r for r in manifest['rows'] if r['split']=='val']
    cuda_setup()
    x=torch.load(cache/'train.pt',weights_only=True,map_location='cpu').cuda()
    u=torch.load(cache/'val.pt',weights_only=True,map_location='cpu').cuda()
    if len(x)!=len(training) or len(u)!=len(development): raise ValueError('feature/row mismatch')
    y=torch.tensor([r['label'] for r in training],dtype=torch.long,device='cuda')
    labels=[r['label'] for r in development]; ids=[r['sample_id'] for r in development]
    # Prespecified synthetic capacity fractions, independent of held-out labels.
    caps=[None,None,None,len(u)//10,len(u)//50]
    output.mkdir(parents=True,exist_ok=False)
    with audited_arm_log(output/'events.jsonl') as log:
        save(output/'config.json',config); save(output/'caps.json',caps)
        log.emit('started',source_sha256=source(),config=config,caps=caps,
                 device=str(torch.cuda.get_device_name()),precision='fp32',
                 scope='independent knee development comparison; not Yuval replication')
        summary=[]
        for seed in config['seeds']:
            identities=[]
            for arm in ('clipper','tralo_null','tralo'):
                directory=output/f'{seed}_{arm}'; directory.mkdir()
                with audited_arm_log(directory/'events.jsonl') as arm_log:
                    result=train_arm(x,y,u,caps,config,seed,arm,
                        lambda row:arm_log.emit(row['event'],**{k:v for k,v in row.items() if k!='event'}))
                    if result['task_updates_skipped'] or result['constraint_updates_skipped']:
                        raise RuntimeError('skipped update')
                    probabilities=result['probabilities'].tolist()
                    report=evaluate_global(probabilities,labels,caps,ids)
                    for policy, score in report.items():
                        predictions=score['predictions']
                        expected=f1_score(labels,predictions,labels=list(range(5)),average='macro',zero_division=0)
                        expected_cc=f1_score(labels,predictions,labels=[3,4],average='macro',zero_division=0)
                        if (abs(expected-score['metrics']['macro_f1'])>1e-12 or
                            abs(expected_cc-score['metrics']['cc_f1'])>1e-12 or
                            abs(accuracy_score(labels,predictions)-score['metrics']['accuracy'])>1e-12):
                            raise RuntimeError('independent metric disagreement')
                    save(directory/'predictions.json',dict(sample_ids=ids,labels=labels,probabilities=probabilities))
                    save(directory/'report.json',report); torch.save(result['state'],directory/'head.pt')
                    row={k:v for k,v in result.items() if k not in ('state','probabilities')}
                    row.update(seed=seed,arm=arm,scores={k:{a:v['metrics'][a] for a in ('accuracy','macro_f1','cc_f1')} for k,v in report.items()})
                    identities.append((result['warmup_sha256'],result['batch_sha256']))
                    arm_log.emit('completed',**row,artifacts={p.name:digest(p) for p in directory.iterdir() if p.name!='events.jsonl'})
                    summary.append(row)
                    print(json.dumps(row),flush=True)
            if len(set(identities))!=1: raise RuntimeError('warmup/batch mismatch')
        save(output/'summary.json',summary)
        log.emit('completed',summary_sha256=digest(output/'summary.json'))


if __name__=='__main__':
    if len(sys.argv) in (4,5) and sys.argv[1]=='prepare': prepare(*sys.argv[2:])
    elif len(sys.argv)==5 and sys.argv[1]=='compare': compare(*sys.argv[2:])
    else: raise SystemExit(__doc__)
