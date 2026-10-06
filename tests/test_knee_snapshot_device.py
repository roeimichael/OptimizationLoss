"""Fictitious CUDA provider only: no tensor library or device is imported."""
import builtins
import os
import sys
from types import ModuleType, SimpleNamespace
import uuid

import pytest

from tralo.knee_snapshot_device import observe_single_device

REQUESTED = 'GPU-00000000-0000-0000-0000-000000000001'
OTHER = 'GPU-00000000-0000-0000-0000-000000000002'


def provider(monkeypatch, *, count=1, raw=None, **changes):
    calls = []
    properties = SimpleNamespace(uuid=SimpleNamespace(bytes=list(uuid.UUID(REQUESTED[4:]).bytes)
        if raw is None else raw), name='FICTITIOUS GPU', major=9, minor=1, total_memory=123456789)
    for name,value in changes.items(): setattr(properties,name,value)
    fake = ModuleType('torch')
    def device_count():
        calls.append('count')
        return count
    def device_properties(index):
        calls.append(('properties',index))
        assert index == 0
        return properties
    fake.cuda = SimpleNamespace(device_count=device_count, get_device_properties=device_properties)
    monkeypatch.setitem(sys.modules,'torch',fake)
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES',REQUESTED)
    return calls


@pytest.mark.parametrize('selected',[None,'','0','GPU-abcd',REQUESTED+','+OTHER,'MIG-'+REQUESTED[4:]])
def test_invalid_selection_refuses_before_tensor_import(monkeypatch,selected):
    if selected is None: monkeypatch.delenv('CUDA_VISIBLE_DEVICES',raising=False)
    else: monkeypatch.setenv('CUDA_VISIBLE_DEVICES',selected)
    real = builtins.__import__
    def guarded(name,*args,**kwargs):
        if name == 'torch': pytest.fail('tensor library imported before selection validation')
        return real(name,*args,**kwargs)
    monkeypatch.setattr(builtins,'__import__',guarded)
    with pytest.raises(RuntimeError,match='complete physical'):
        observe_single_device(REQUESTED)


def test_request_cannot_differ_from_visible_environment(monkeypatch):
    calls=provider(monkeypatch)
    with pytest.raises(RuntimeError,match='requested selection'):
        observe_single_device(OTHER)
    assert calls == []


@pytest.mark.parametrize('count',[0,2,True])
def test_missing_multiple_or_invalid_visible_count_refuses_before_properties(monkeypatch,count):
    calls=provider(monkeypatch,count=count)
    with pytest.raises(RuntimeError,match='exactly one visible'):
        observe_single_device(REQUESTED)
    assert calls == ['count']


def test_request_is_distinct_from_runtime_uuid_and_not_ownership_permission(monkeypatch):
    calls=provider(monkeypatch)
    before=dict(os.environ)
    record=observe_single_device(REQUESTED)
    assert record['requested_gpu_uuid']==REQUESTED
    assert record['observed_gpu_uuid']==REQUESTED
    assert record['visible_device_index']==0 and record['visible_device_count']==1
    assert record['device_name']=='FICTITIOUS GPU' and record['capability']==[9,1]
    assert record['total_memory_bytes']==123456789
    assert record['ownership_certified'] is False and record['campaign_permission'] is False
    assert calls==['count',('properties',0)] and dict(os.environ)==before


def test_mismatched_observed_uuid_is_not_replaced_by_request(monkeypatch):
    provider(monkeypatch,raw=list(uuid.UUID(OTHER[4:]).bytes))
    with pytest.raises(RuntimeError,match='observed physical UUID differs'):
        observe_single_device(REQUESTED)


@pytest.mark.parametrize('raw',[[0]*16,[1]*15,[False]+[0]*15,[256]+[0]*15,'0000000000000001'])
def test_unavailable_or_malformed_raw_uuid_is_not_an_observation(monkeypatch,raw):
    provider(monkeypatch,raw=raw)
    with pytest.raises(RuntimeError,match='runtime UUID bytes'):
        observe_single_device(REQUESTED)


@pytest.mark.parametrize('changes',[{'name':''},{'major':True},{'minor':-1},{'total_memory':0}])
def test_invalid_hardware_fields_fail_closed(monkeypatch,changes):
    provider(monkeypatch,**changes)
    with pytest.raises(RuntimeError,match='runtime hardware properties'):
        observe_single_device(REQUESTED)


def test_complete_uuid_string_cannot_substitute_for_runtime_uuid_bytes(monkeypatch):
    provider(monkeypatch,uuid=REQUESTED)
    with pytest.raises(RuntimeError,match='runtime UUID bytes'):
        observe_single_device(REQUESTED)
