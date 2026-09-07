"""The dispatcher must be launchable DETACHED, because that is how it is launched.

A campaign is partitioned by `EXPERIMENT_DIR` and started once per card:

    EXPERIMENT_DIR=results/<root> CUDA_VISIBLE_DEVICES=<n> \\
        setsid nohup python -u main.py > <root>.log 2>&1 < /dev/null &

On 2026-09-07 both dispatchers for `itemscale1` / `itemscale2` died in under a
second on `EOFError: EOF when reading a line`. `select_gpu()` printed its menu
and called `input()` unconditionally -- even though `CUDA_VISIBLE_DEVICES`
had already reduced `torch.cuda.device_count()` to exactly 1, so there was no
choice left to make. Nothing had been claimed, so `results/` looked untouched
and the only tell was the log.

One visible GPU is not a choice. Several IS one, and auto-picking there would
silently share a card with another user, which on dsisco02 crashes the host --
so the prompt stays for n > 1 and this test pins BOTH halves.
"""
import importlib
import sys

import pytest


def _load_main(monkeypatch, n_devices, available=True):
    """Import main.py with torch.cuda stubbed to report `n_devices`."""
    import torch
    monkeypatch.setattr(torch.cuda, "is_available", lambda: available)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: n_devices)
    monkeypatch.setattr(torch.cuda, "get_device_name",
                        lambda i=0: "StubGPU%d" % i)

    class _Props(object):
        total_memory = 24 * 1024 ** 3

    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda i: _Props())
    sys.modules.pop("main", None)
    return importlib.import_module("main")


def _forbid_input(monkeypatch, mod):
    def boom(*_a, **_kw):
        raise EOFError("EOF when reading a line")
    monkeypatch.setattr(mod, "input", boom, raising=False)
    monkeypatch.setitem(mod.__builtins__ if isinstance(mod.__builtins__, dict)
                        else vars(mod.__builtins__), "input", boom)


def test_one_visible_gpu_is_selected_without_prompting(monkeypatch):
    """The detached launch path: stdin is /dev/null, so input() must not run."""
    mod = _load_main(monkeypatch, 1)
    _forbid_input(monkeypatch, mod)
    assert mod.select_gpu() == 0


def test_several_visible_gpus_STILL_prompt(monkeypatch):
    """NEGATIVE CONTROL. Auto-picking one of several would share another
    user's card; on dsisco02 that crashes the host. The prompt must survive."""
    mod = _load_main(monkeypatch, 4)
    _forbid_input(monkeypatch, mod)
    with pytest.raises(EOFError):
        mod.select_gpu()


def test_no_gpu_returns_None_rather_than_prompting(monkeypatch):
    """The CPU path was already safe; pinned so the fix cannot regress it."""
    mod = _load_main(monkeypatch, 0, available=False)
    _forbid_input(monkeypatch, mod)
    assert mod.select_gpu() is None
