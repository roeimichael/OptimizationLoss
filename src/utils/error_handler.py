# Error handling utilities: logger decorator and safe_execute wrapper.
# Provides consistent exception logging across the experiment pipeline.

import functools
import logging
import traceback
from datetime import datetime
from pathlib import Path

log = logging.getLogger(__name__)


def logger(reraise=True, **kwargs):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kw):
            try:
                return func(*args, **kw)
            except KeyboardInterrupt:
                log.warning("Interrupted in %s.%s", func.__module__, func.__name__)
                raise
            except Exception as e:
                log.error("Exception in %s.%s: %s: %s",
                          func.__module__, func.__name__, type(e).__name__, e)
                log.debug(traceback.format_exc())
                if reraise:
                    raise
                return None
        return wrapper
    return decorator


def log_exception(e, context="", experiment_path=None):
    log.error("%s: %s: %s", context, type(e).__name__, e)
    log.debug(traceback.format_exc())
    if experiment_path:
        _save_error_to_file(experiment_path, type(e).__name__, str(e), context)


def _save_error_to_file(experiment_path, exception_type, exception_msg, context):
    import json
    error_path = Path(experiment_path) / 'error_log.json'
    try:
        with open(error_path, 'w') as f:
            json.dump({
                'timestamp': datetime.now().isoformat(),
                'exception_type': exception_type,
                'exception_message': exception_msg,
                'context': context,
                'traceback': traceback.format_exc()
            }, f, indent=2)
    except Exception:
        pass


def safe_execute(func, *args, default=None, context="", **kwargs):
    try:
        return func(*args, **kwargs)
    except Exception as e:
        log.warning("%s failed: %s: %s", context or func.__name__, type(e).__name__, e)
        return default
