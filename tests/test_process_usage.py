"""SELF counters are logging observations, never descendant/GPU cost receipts."""
import copy
import sys
import types

import pytest


def provider(monkeypatch, *, user=1.25, system=.5, rss=1234):
    monkeypatch.setattr(sys, 'platform', 'linux')
    resource = types.ModuleType('resource')
    resource.RUSAGE_SELF = 0
    def getrusage(who):
        assert who == resource.RUSAGE_SELF
        return types.SimpleNamespace(ru_utime=user, ru_stime=system, ru_maxrss=rss)
    resource.getrusage = getrusage
    monkeypatch.setitem(sys.modules, 'resource', resource)


def test_self_snapshot_normalizes_linux_lifetime_peak_rss(monkeypatch):
    from tralo.process_usage import self_usage
    provider(monkeypatch)
    value = self_usage()
    assert value['scope'] == 'process_self_not_children_or_gpu'
    assert value['status'] == 'observed' and value['pid'] > 0
    assert value['user_cpu_seconds'] == 1.25 and value['system_cpu_seconds'] == .5
    assert value['lifetime_peak_rss_bytes'] == 1234 * 1024


def test_unsupported_platform_records_unknown_not_zero(monkeypatch):
    from tralo.process_usage import self_usage, usage_delta
    monkeypatch.setattr(sys, 'platform', 'win32')
    value = self_usage()
    assert value['status'] == 'unavailable'
    assert all(value[k] is None for k in ('user_cpu_seconds', 'system_cpu_seconds', 'lifetime_peak_rss_bytes'))
    assert usage_delta(value, value) == dict(status='unavailable', user_cpu_seconds=None, system_cpu_seconds=None)


def test_delta_is_self_cpu_difference_without_subtracting_lifetime_rss(monkeypatch):
    from tralo.process_usage import self_usage, usage_delta
    provider(monkeypatch)
    before = self_usage()
    provider(monkeypatch, user=2.5, system=.75, rss=1500)
    after = self_usage()
    saved = copy.deepcopy((before, after))
    assert usage_delta(before, after) == dict(status='observed', user_cpu_seconds=1.25, system_cpu_seconds=.25)
    assert (before, after) == saved


@pytest.mark.parametrize('field,bad', [('ru_utime', float('nan')), ('ru_stime', float('inf')),
                                     ('ru_maxrss', -1), ('ru_utime', True)])
def test_invalid_native_counter_is_not_silently_zeroed(monkeypatch, field, bad):
    from tralo.process_usage import self_usage
    fields = dict(user=1., system=.5, rss=3)
    fields[{'ru_utime':'user','ru_stime':'system','ru_maxrss':'rss'}[field]] = bad
    provider(monkeypatch, **fields)
    with pytest.raises(ValueError): self_usage()


@pytest.mark.parametrize('field,bad', [('pid', 0), ('pid', True), ('scope', 'children'),
                                     ('user_cpu_seconds', -1.), ('system_cpu_seconds', float('nan')),
                                     ('lifetime_peak_rss_bytes', .5)])
def test_log_counter_tampering_refused(monkeypatch, field, bad):
    from tralo.process_usage import self_usage, usage_delta
    provider(monkeypatch)
    before = self_usage(); after = dict(before); after[field] = bad
    with pytest.raises(ValueError): usage_delta(before, after)


@pytest.mark.parametrize('change', ['pid', 'cpu_reversed', 'rss_reversed', 'status', 'platform'])
def test_delta_refuses_cross_process_or_inconsistent_observations(monkeypatch, change):
    from tralo.process_usage import self_usage, usage_delta
    provider(monkeypatch)
    before = self_usage(); after = dict(before)
    if change == 'pid': after['pid'] += 1
    if change == 'cpu_reversed': after['user_cpu_seconds'] = 0.
    if change == 'rss_reversed': after['lifetime_peak_rss_bytes'] -= 1
    if change == 'status': after['status'] = 'unavailable'
    if change == 'platform': after['platform'] = 'win32'
    with pytest.raises(ValueError): usage_delta(before, after)
