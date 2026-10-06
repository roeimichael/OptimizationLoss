"""Requested versus runtime GPU identity, not ownership or launch permission.

No tensor library is imported on module load. A future authenticated launcher
may call this only AFTER source/private/seed/exclusive ownership and certified
finite budget gates permit CUDA entry. Both scientific CLIs remain closed.
The current validation uses fictitious providers; actual CUDA is untested.
"""
import os
import re
import uuid


def observe_single_device(requested_uuid):
    """Refuse ambiguous selection; bind one visible device's raw UUID bytes.

    This operation can initialize CUDA. It does not select/change a device,
    allocate tensors, query other hosts, claim a seed or issue an approval.
    A matching physical UUID alone cannot establish exclusive ownership.
    """
    pattern = r'GPU-[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}'
    selected = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    if (type(requested_uuid) is not str or re.fullmatch(pattern, requested_uuid) is None
            or re.fullmatch(pattern, selected) is None):
        raise RuntimeError('one complete physical GPU UUID is required before tensor import')
    if selected != requested_uuid:
        raise RuntimeError('visible environment differs from requested selection')
    import torch
    count = torch.cuda.device_count()
    if type(count) is not int or count != 1:
        raise RuntimeError('runtime must expose exactly one visible CUDA device')
    properties = torch.cuda.get_device_properties(0)
    raw = getattr(getattr(properties, 'uuid', None), 'bytes', None)
    if (type(raw) not in (list, bytes) or len(raw) != 16 or not any(raw)
            or any(type(value) is not int or not 0 <= value <= 255 for value in raw)):
        raise RuntimeError('runtime UUID bytes are missing or malformed')
    observed = 'GPU-' + str(uuid.UUID(bytes=bytes(raw)))
    if uuid.UUID(requested_uuid[4:]) != uuid.UUID(observed[4:]):
        raise RuntimeError('observed physical UUID differs from requested selection')
    name = getattr(properties, 'name', None)
    major, minor = getattr(properties, 'major', None), getattr(properties, 'minor', None)
    memory = getattr(properties, 'total_memory', None)
    if (type(name) is not str or not name or type(major) is not int or major <= 0
            or type(minor) is not int or minor < 0 or type(memory) is not int or memory <= 0):
        raise RuntimeError('runtime hardware properties are missing or malformed')
    return dict(requested_gpu_uuid=requested_uuid, observed_gpu_uuid=observed,
                visible_device_index=0, visible_device_count=count, device_name=name,
                capability=[major, minor], total_memory_bytes=memory,
                ownership_certified=False, campaign_permission=False,
                limitation='Runtime identity only; exclusive ownership, source/private isolation, seed freshness, derivative/dose/cost and certified budget remain external.')


def observe_model_device(model, requested_uuid=None):
    """Bind a constructed model's entire placement to the runtime observation.

    CPU readiness does not query CUDA. GPU use requires external launch gates
    before this call, just like observe_single_device; no model is moved here.
    All parameters and buffers must occupy the one selected visible index zero.
    """
    parameters = list(model.parameters())
    tensors = parameters + list(model.buffers())
    placement = [(value.device.type, value.device.index) for value in tensors]
    if not parameters:
        raise RuntimeError('model must have parameters for placement observation')
    if all(kind == 'cpu' and index is None for kind,index in placement) and requested_uuid is None:
        return None
    if requested_uuid is None or any(kind != 'cuda' or type(index) is not int or index != 0 for kind,index in placement):
        raise RuntimeError('GPU model placement requires a request and every tensor on visible index zero')
    return observe_single_device(requested_uuid)


def verify_device_log(device, cuda_initialized, record):
    """Validate recorded bindings without importing Torch or observing a device.

    Self-consistent saved metadata is not authenticated runtime, ownership or
    permission evidence; the producing driver must obtain its own observation.
    """
    if device == 'cpu' and cuda_initialized is False and record is None:
        return
    keys = {'requested_gpu_uuid','observed_gpu_uuid','visible_device_index','visible_device_count',
            'device_name','capability','total_memory_bytes','ownership_certified','campaign_permission','limitation'}
    if device != 'cuda:0' or cuda_initialized is not True or type(record) is not dict or set(record) != keys:
        raise ValueError('device log does not bind the execution device')
    pattern = r'GPU-[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}'
    ids = [record[k] for k in ('requested_gpu_uuid','observed_gpu_uuid')]
    if (any(type(value) is not str or re.fullmatch(pattern,value) is None for value in ids)
            or uuid.UUID(ids[0][4:]).int == 0 or uuid.UUID(ids[0][4:]) != uuid.UUID(ids[1][4:])):
        raise ValueError('device UUID log mismatch')
    capability = record['capability']
    if (type(record['visible_device_index']) is not int or record['visible_device_index'] != 0
            or type(record['visible_device_count']) is not int or record['visible_device_count'] != 1
            or type(record['device_name']) is not str or not record['device_name']
            or type(capability) is not list or len(capability) != 2
            or any(type(value) is not int for value in capability) or capability[0] <= 0 or capability[1] < 0
            or type(record['total_memory_bytes']) is not int or record['total_memory_bytes'] <= 0
            or record['ownership_certified'] is not False or record['campaign_permission'] is not False
            or type(record['limitation']) is not str or not record['limitation']):
        raise ValueError('device hardware/permission log is malformed')
