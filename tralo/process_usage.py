"""Linux process SELF observations; no Torch, child, device or budget accounting."""
import math
import os
import sys

SCOPE = 'process_self_not_children_or_gpu'
KEYS = {'scope', 'status', 'pid', 'platform', 'user_cpu_seconds',
        'system_cpu_seconds', 'lifetime_peak_rss_bytes'}


def _validate(value):
    if (not isinstance(value, dict) or set(value) != KEYS or value['scope'] != SCOPE
            or type(value['pid']) is not int or value['pid'] <= 0
            or not isinstance(value['platform'], str) or not value['platform']):
        raise ValueError('invalid process usage identity')
    if value['status'] == 'unavailable' and value['platform'] != 'linux':
        if any(value[k] is not None for k in ('user_cpu_seconds', 'system_cpu_seconds', 'lifetime_peak_rss_bytes')):
            raise ValueError('unavailable process usage must remain unknown')
    elif value['status'] == 'observed' and value['platform'] == 'linux':
        if (any(type(value[k]) not in (int, float) or not math.isfinite(value[k]) or value[k] < 0
                for k in ('user_cpu_seconds', 'system_cpu_seconds'))
                or type(value['lifetime_peak_rss_bytes']) is not int or value['lifetime_peak_rss_bytes'] < 0):
            raise ValueError('invalid process usage counters')
    else:
        raise ValueError('invalid process usage platform/status')


def self_usage():
    """Read cumulative SELF CPU and lifetime RSS; unsupported platforms are unknown."""
    value = dict(scope=SCOPE, pid=os.getpid(), platform=sys.platform, status='unavailable',
                 user_cpu_seconds=None, system_cpu_seconds=None, lifetime_peak_rss_bytes=None)
    if sys.platform == 'linux':
        import resource
        usage = resource.getrusage(resource.RUSAGE_SELF)
        if type(usage.ru_maxrss) is not int:
            raise ValueError('invalid native process usage RSS')
        value.update(status='observed', user_cpu_seconds=usage.ru_utime,
                     system_cpu_seconds=usage.ru_stime, lifetime_peak_rss_bytes=usage.ru_maxrss * 1024)
    _validate(value)
    return value


def usage_delta(start, end):
    """Validate same-process records and subtract CPU only; RSS remains lifetime peak."""
    _validate(start)
    _validate(end)
    if any(start[k] != end[k] for k in ('pid', 'platform', 'status', 'scope')):
        raise ValueError('process usage observations differ in identity/status')
    if start['status'] == 'unavailable':
        return dict(status='unavailable', user_cpu_seconds=None, system_cpu_seconds=None)
    if any(end[k] < start[k] for k in ('user_cpu_seconds', 'system_cpu_seconds', 'lifetime_peak_rss_bytes')):
        raise ValueError('process usage counters reversed')
    return dict(status='observed', user_cpu_seconds=end['user_cpu_seconds']-start['user_cpu_seconds'],
                system_cpu_seconds=end['system_cpu_seconds']-start['system_cpu_seconds'])
