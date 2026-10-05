"""Independent synthetic constructor refusals and native architecture/RNG parity."""
import hashlib
import io

import pytest

from tralo import knee_snapshot_model as subject


@pytest.mark.parametrize('seed', [True, -1, 2**32, 1.5, None])
def test_invalid_seed_precedes_weight_access(tmp_path, seed):
    with pytest.raises(ValueError, match='uint32'):
        subject.make_model(tmp_path/'missing', seed)


def test_missing_weight_refused(tmp_path):
    with pytest.raises(ValueError, match='missing or linked'):
        subject.make_model(tmp_path/'missing', 55)


def test_wrong_bytes_do_not_reach_tensor_deserialization(tmp_path, monkeypatch):
    import torch
    path = tmp_path/'wrong.pth'; path.write_bytes(b'fictitious invalid weight')
    monkeypatch.setattr(torch, 'load', lambda *a, **kw: pytest.fail('unverified bytes deserialized'))
    with pytest.raises(ValueError, match='bytes differ'):
        subject.make_model(path, 55)


def test_committed_weight_identity():
    assert subject.WEIGHT_SHA256 == '5c1a416349c4cf298f2a6a5e2600ed0ee55e604713578f5e74e6bc8bcaef7997'
    assert subject.WEIGHT_ENUM == 'MobileNet_V3_Large_Weights.IMAGENET1K_V2'
    assert subject.WEIGHT_URL == 'https://download.pytorch.org/models/mobilenet_v3_large-5c1a4163.pth'


def test_native_model_head_and_rng_are_identical_from_one_authenticated_read(tmp_path, monkeypatch):
    import torch
    from torchvision import models
    torch.set_num_threads(1)
    torch.manual_seed(123)
    fictitious_weights = models.mobilenet_v3_large(weights=None).state_dict()
    path = tmp_path/'fictitious.pth'
    torch.save(fictitious_weights, path)
    data = path.read_bytes()
    # Fixture pins fictitious weights; production uses the immutable constant above.
    monkeypatch.setattr(subject, 'WEIGHT_SHA256', hashlib.sha256(data).hexdigest())
    monkeypatch.setattr(models.MobileNet_V3_Large_Weights, 'get_state_dict',
                        lambda *a, **kw: torch.load(io.BytesIO(data), map_location='cpu', weights_only=True))
    torch.manual_seed(55)
    reference = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.IMAGENET1K_V2)
    reference.classifier[3] = torch.nn.Linear(reference.classifier[3].in_features, 5)
    reference_rng = torch.get_rng_state().clone()
    original = type(path).read_bytes
    reads = []
    def read_once(p):
        if p == path:
            reads.append(str(p))
            if len(reads) > 1: pytest.fail('weight source reopened')
        return original(p)
    monkeypatch.setattr(type(path), 'read_bytes', read_once)
    actual, receipt = subject.make_model(path, 55)
    assert reads == [str(path)]
    assert set(actual.state_dict()) == set(reference.state_dict())
    assert all(torch.equal(value, reference.state_dict()[key]) for key, value in actual.state_dict().items())
    assert torch.equal(torch.get_rng_state(), reference_rng)
    assert all(p.requires_grad and p.dtype == torch.float32 and p.device.type == 'cpu' for p in actual.parameters())
    assert receipt['weight_source_reads'] == 1 and not receipt['implicit_download']
    assert receipt['head_classes'] == 5
    # CPU construction must not inherit an unrelated process-wide device default.
    torch.set_default_device('meta')
    try:
        reads.clear()
        on_cpu, _ = subject.make_model(path, 55)
        assert all(p.device.type == 'cpu' for p in on_cpu.parameters())
    finally:
        torch.set_default_device('cpu')


def test_malformed_authenticated_state_refused(tmp_path, monkeypatch):
    import torch
    path = tmp_path/'fictitious.pth'; torch.save({'unexpected': torch.ones(1)}, path)
    monkeypatch.setattr(subject, 'WEIGHT_SHA256', hashlib.sha256(path.read_bytes()).hexdigest())
    with pytest.raises(RuntimeError, match='state_dict'):
        subject.make_model(path, 55)
