"""Utilities. Import the submodule you need -- `from src.utils.constants import
UNLIMITED`, not `from src.utils import ...`.

This file deliberately re-exports NOTHING. It used to pull in data_loader, which
imports src.training.constraints, which imports src.utils.constants -- a cycle
that only stayed quiet because the pipeline happened to import src.utils first.
Any entry point starting from src.training crashed on it.
"""
