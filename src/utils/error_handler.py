"""
Centralized error handling and logging utilities.

This module provides a decorator-based approach to error handling that:
1. Catches and logs all exceptions with full stack traces
2. Preserves the original exception for proper propagation
3. Provides context about the function call that failed
4. Keeps error handling logic decoupled from core experiment logic
"""

import functools
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Callable, Any, Optional


class ExperimentError(Exception):
    """Base exception for experiment-related errors."""
    pass


class ConfigurationError(ExperimentError):
    """Raised when experiment configuration is invalid."""
    pass


class DataError(ExperimentError):
    """Raised when data loading or preprocessing fails."""
    pass


class TrainingError(ExperimentError):
    """Raised when training encounters an unrecoverable error."""
    pass


def format_exception_context(func_name: str, args: tuple, kwargs: dict) -> str:
    """Format the context of a function call for error reporting."""
    arg_strs = []
    for arg in args:
        if hasattr(arg, 'shape'):
            arg_strs.append(f"<tensor shape={arg.shape}>")
        elif isinstance(arg, (str, int, float, bool)):
            arg_strs.append(repr(arg)[:100])
        else:
            arg_strs.append(f"<{type(arg).__name__}>")

    for key, val in kwargs.items():
        if hasattr(val, 'shape'):
            arg_strs.append(f"{key}=<tensor shape={val.shape}>")
        elif isinstance(val, (str, int, float, bool)):
            arg_strs.append(f"{key}={repr(val)[:50]}")
        else:
            arg_strs.append(f"{key}=<{type(val).__name__}>")

    return f"{func_name}({', '.join(arg_strs)})"


def get_call_stack_summary(limit: int = 10) -> str:
    """Get a formatted summary of the call stack."""
    lines = []
    stack = traceback.extract_stack()[:-2]

    for frame in stack[-limit:]:
        filename = Path(frame.filename).name
        lines.append(f"  {filename}:{frame.lineno} in {frame.name}")
        if frame.line:
            lines.append(f"    {frame.line.strip()}")

    return "\n".join(lines)


def logger(
    reraise: bool = True,
    log_args: bool = True,
    include_locals: bool = False,
    experiment_path_arg: Optional[str] = None
) -> Callable:
    """
    Decorator for comprehensive error logging and stack trace reporting.

    This decorator wraps functions to provide:
    - Full exception type and message
    - Complete stack trace
    - Function call context (arguments)
    - Timestamp of failure

    Args:
        reraise: If True, re-raises the exception after logging. Default True.
        log_args: If True, logs function arguments in error context. Default True.
        include_locals: If True, attempts to log local variables (can be verbose).
        experiment_path_arg: Name of argument containing experiment path for status saving.

    Usage:
        @logger()
        def my_function(x, y):
            # function code
            pass

        @logger(reraise=False)
        def optional_function():
            # failures here won't crash the program
            pass
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            func_name = func.__name__
            module_name = func.__module__
            timestamp = datetime.now().isoformat()

            try:
                return func(*args, **kwargs)

            except KeyboardInterrupt:
                print(f"\n[INTERRUPT] {timestamp}")
                print(f"Function: {module_name}.{func_name}")
                print("User interrupted execution (Ctrl+C)")
                raise

            except SystemExit as e:
                print(f"\n[SYSTEM EXIT] {timestamp}")
                print(f"Function: {module_name}.{func_name}")
                print(f"Exit code: {e.code}")
                raise

            except Exception as e:
                exception_type = type(e).__name__
                exception_msg = str(e)

                separator = "=" * 80
                print(f"\n{separator}")
                print(f"[ERROR] {timestamp}")
                print(separator)
                print(f"Exception Type: {exception_type}")
                print(f"Exception Message: {exception_msg}")
                print(f"\nFunction: {module_name}.{func_name}")

                if log_args:
                    call_context = format_exception_context(func_name, args, kwargs)
                    print(f"Call: {call_context}")

                print(f"\n{'-' * 80}")
                print("STACK TRACE:")
                print("-" * 80)
                traceback.print_exc()

                print(f"\n{'-' * 80}")
                print("CALL CHAIN (most recent last):")
                print("-" * 80)
                print(get_call_stack_summary())

                print(separator)

                if reraise:
                    raise

                return None

        return wrapper
    return decorator


def log_exception(
    e: Exception,
    context: str = "",
    experiment_path: Optional[str] = None
) -> None:
    """
    Standalone function to log an exception with full context.

    Use this for manual exception logging in try-except blocks where
    you need custom handling but still want consistent error reporting.

    Args:
        e: The exception to log
        context: Additional context string to include
        experiment_path: Optional path to save error details
    """
    exception_type = type(e).__name__
    exception_msg = str(e)
    timestamp = datetime.now().isoformat()

    separator = "=" * 80
    print(f"\n{separator}")
    print(f"[EXCEPTION LOGGED] {timestamp}")
    if context:
        print(f"Context: {context}")
    print(separator)
    print(f"Exception Type: {exception_type}")
    print(f"Exception Message: {exception_msg}")
    print(f"\n{'-' * 80}")
    print("STACK TRACE:")
    print("-" * 80)
    traceback.print_exc()
    print(separator)

    if experiment_path:
        _save_error_to_file(experiment_path, exception_type, exception_msg, context)


def _save_error_to_file(
    experiment_path: str,
    exception_type: str,
    exception_msg: str,
    context: str
) -> None:
    """Save error details to a file in the experiment directory."""
    import json

    error_path = Path(experiment_path) / 'error_log.json'
    error_data = {
        'timestamp': datetime.now().isoformat(),
        'exception_type': exception_type,
        'exception_message': exception_msg,
        'context': context,
        'traceback': traceback.format_exc()
    }

    try:
        with open(error_path, 'w') as f:
            json.dump(error_data, f, indent=2)
    except Exception:
        pass


def safe_execute(
    func: Callable,
    *args,
    default: Any = None,
    context: str = "",
    **kwargs
) -> Any:
    """
    Execute a function safely, returning a default value on failure.

    Use this for operations that should not crash the program but
    where you still want error visibility.

    Args:
        func: Function to execute
        *args: Positional arguments for the function
        default: Value to return if function fails
        context: Description of what the function is doing
        **kwargs: Keyword arguments for the function

    Returns:
        Function result or default value on failure

    Example:
        result = safe_execute(
            load_cached_model,
            model_id,
            default=None,
            context="Loading cached model"
        )
    """
    try:
        return func(*args, **kwargs)
    except Exception as e:
        if context:
            print(f"[WARNING] {context} failed: {type(e).__name__}: {e}")
        else:
            print(f"[WARNING] {func.__name__} failed: {type(e).__name__}: {e}")
        return default
