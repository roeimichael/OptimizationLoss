"""New model/log bindings with fictitious devices; no real tensor import."""
import copy
import sys
from types import ModuleType, SimpleNamespace
import uuid

import pytest

from tralo.knee_snapshot_device import observe_model_device, verify_device_log

REQUEST = 'GPU-00000000-0000-0000-0000-000000000003'
OTHER = 'GPU-00000000-0000-0000-0000-000000000004'


def model(parameters, buffers=()):
    def values(items):
        return [SimpleNamespace(device=SimpleNamespace(type=kind, index=index)) for kind,index in items]
    return SimpleNamespace(parameters=lambda:iter(values(parameters)), buffers=lambda:iter(values(buffers)))


def provider(monkeypatch):
    calls = []
    fake = ModuleType('torch')
    def count(): calls.append('count'); return 1
    def properties(index):
        calls.append(('properties', index))
        return SimpleNamespace(uuid=SimpleNamespace(bytes=list(uuid.UUID(REQUEST[4:]).bytes)),
                               name='FICTITIOUS DEVICE', major=9, minor=1, total_memory=1024)
    fake.cuda = SimpleNamespace(device_count=count, get_device_properties=properties)
    monkeypatch.setitem(sys.modules, 'torch', fake)
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', REQUEST)
    return calls


def test_cpu_model_has_no_runtime_gpu_observation(monkeypatch):
    calls = provider(monkeypatch)
    assert observe_model_device(model([('cpu',None)], [('cpu',None)])) is None
    verify_device_log('cpu', False, None)
    assert calls == []


def test_all_parameters_and_buffers_bind_visible_index_zero(monkeypatch):
    calls = provider(monkeypatch)
    record = observe_model_device(model([('cuda',0),('cuda',0)], [('cuda',0)]), REQUEST)
    verify_device_log('cuda:0', True, record)
    assert record['requested_gpu_uuid'] == REQUEST == record['observed_gpu_uuid']
    assert calls == ['count',('properties',0)]


@pytest.mark.parametrize('parameters,buffers,selected_uuid',[
    ([],[],None), ([('cuda',0)],[],None), ([('cpu',None)],[],REQUEST),
    ([('cuda',1)],[],REQUEST), ([('cuda',0)],[('cuda',1)],REQUEST),
    ([('cuda',0)],[('cpu',None)],REQUEST), ([('cpu',None)],[('cuda',0)],None),
    ([('mps',0)],[],REQUEST), ([('cuda',None)],[],REQUEST), ([('cuda',False)],[],REQUEST),
])
def test_model_placement_refuses_before_runtime_probe(monkeypatch,parameters,buffers,selected_uuid):
    calls = provider(monkeypatch)
    with pytest.raises(RuntimeError, match='model|placement|request'):
        observe_model_device(model(parameters,buffers), selected_uuid)
    assert calls == []


@pytest.mark.parametrize('change',[
    {'observed_gpu_uuid':OTHER}, {'requested_gpu_uuid':'GPU-abcd'},
    {'visible_device_count':2}, {'visible_device_index':True},
    {'campaign_permission':True}, {'ownership_certified':True},
    {'capability':[True,1]}, {'total_memory_bytes':0}, {'device_name':''},
])
def test_rehashed_runtime_metadata_is_still_checked(monkeypatch,change):
    provider(monkeypatch)
    record = observe_model_device(model([('cuda',0)]), REQUEST)
    record.update(change)
    with pytest.raises(ValueError, match='device'):
        verify_device_log('cuda:0', True, record)


@pytest.mark.parametrize('device,initialized', [('cuda:1',True),('cuda:0',False),('cpu',True)])
def test_device_tag_cannot_disagree_with_runtime_record(monkeypatch,device,initialized):
    provider(monkeypatch)
    record = observe_model_device(model([('cuda',0)]), REQUEST)
    with pytest.raises(ValueError, match='device'):
        verify_device_log(device, initialized, copy.deepcopy(record))
