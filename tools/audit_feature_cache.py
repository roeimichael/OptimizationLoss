"""Rebuild every cached feature from original images; verify split and provenance."""
import hashlib
import json
from pathlib import Path
import sys


def run(cache, output):
    import torch
    from torchvision import datasets, models, transforms
    torch.set_num_threads(4)
    cache, output = Path(cache), Path(output)
    output.mkdir(parents=True,exist_ok=False)
    assert torch.cuda.is_available(), 'CUDA required'
    events = [json.loads(s) for s in (cache/'events.jsonl').read_text().splitlines()]
    assert events[-1]['event'] == 'completed'
    for name,digest in events[-1]['artifacts'].items():
        assert hashlib.sha256((cache/name).read_bytes()).hexdigest() == digest, name
    config = json.loads((cache/'config.json').read_text())
    assert config['precision'] == 'fp32'
    for name,digest in json.loads((cache/'dataset_hashes.json').read_text()).items():
        assert hashlib.sha256((Path(config['data_root'])/name).read_bytes()).hexdigest() == digest, name
    dataset = datasets.CIFAR100(config['data_root'],train=True,download=False)
    split = json.loads((cache/'split_indices.json').read_text())
    assert not set(split['train']) & set(split['development'])
    saved = json.loads((cache/'development_probabilities.json').read_text())
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
    from tralo.image_baseline import sample_ids
    assert saved['sample_ids'] == sample_ids(split['development'])
    assert saved['labels'] == [dataset.targets[i] for i in split['development']]
    hashes = {}
    for name,indices in split.items():
        assert len(indices) == len(set(indices))
        hashes[name] = [hashlib.sha256(dataset.data[i].tobytes()).hexdigest() for i in indices]
    assert not set(hashes['train']) & set(hashes['development']), 'cross-split exact duplicates'
    weights = models.ResNet18_Weights.IMAGENET1K_V1
    model = models.resnet18(weights=weights).cuda().eval()
    model.fc = torch.nn.Identity()
    weight_file = Path(torch.hub.get_dir())/'checkpoints'/weights.url.rsplit('/',1)[-1]
    model_record = next(e for e in events if e['event']=='model_loaded')
    assert hashlib.sha256(weight_file.read_bytes()).hexdigest() == model_record['weights_sha256']
    preprocess = transforms.Compose([transforms.Resize((config['image_size'],config['image_size'])),
        transforms.ToTensor(),transforms.Normalize((.485,.456,.406),(.229,.224,.225))])
    dataset.transform = preprocess
    checks = {}
    with torch.no_grad():
        for name,indices in split.items():
            features = torch.load(cache/(name+'_features.pt'),map_location='cpu',weights_only=True)
            assert features.shape == (len(indices),512) and torch.isfinite(features).all()
            maximum = 0.
            for start in range(0,len(indices),config['feature_batch_size']):
                batch = torch.stack([dataset[i][0] for i in indices[start:start+config['feature_batch_size']]]).cuda()
                actual = model(batch).cpu()
                expected = features[start:start+len(actual)]
                maximum = max(maximum,float((actual-expected).abs().max()))
                assert torch.allclose(actual,expected,atol=2e-5,rtol=2e-5), (name,start,maximum)
            checks[name] = dict(samples=len(indices),feature_shape=list(features.shape),
                reextraction_max_absolute_error=maximum,
                exact_duplicate_images_within_split=len(indices)-len(set(hashes[name])),
                class_support=[sum(dataset.targets[i]==c for i in indices) for c in range(100)])
    report = dict(status='passed',cache=str(cache),checks=checks,cross_split_exact_duplicates=0,
        device=torch.cuda.get_device_name(),torch_version=str(torch.__version__),
        cache_events_sha256=hashlib.sha256((cache/'events.jsonl').read_bytes()).hexdigest(),
        limits='No semantic near-duplicate or ImageNet pretraining overlap audit; development split already inspected.')
    (output/'audit.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report),flush=True)


if __name__ == '__main__':
    run(*sys.argv[1:])
