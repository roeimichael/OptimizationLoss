"""Exclusive, append-only JSONL event logging."""

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, TextIO


_RESERVED = frozenset(("schema_version", "sequence", "timestamp", "event"))
_SCALAR_TYPES = (type(None), bool, int, float, str)


class EventLog:
    """Write immutable, finite JSON event records to a newly-created file."""

    def __init__(self, path: str | Path) -> None:
        self._stream: TextIO = open(path, "x", encoding="utf-8", newline="")
        self._sequence = 0

    def __enter__(self) -> "EventLog":
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        self._stream.close()

    def emit(self, event: str, /, **fields: Any) -> None:
        if type(event) is not str:
            raise TypeError("event must be a string")
        reserved = _RESERVED.intersection(fields)
        if reserved:
            raise ValueError(f"reserved event field: {next(iter(reserved))}")
        for name, value in fields.items():
            self._validate_value(value, name)

        timestamp = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        record = {
            "schema_version": 1,
            "sequence": self._sequence,
            "timestamp": timestamp,
            "event": event,
            **fields,
        }
        line = json.dumps(record, allow_nan=False, ensure_ascii=False) + "\n"
        self._stream.write(line)
        self._stream.flush()
        self._sequence += 1

    @staticmethod
    def _validate_value(value: Any, name: str) -> None:
        value_type = type(value)
        if value_type in _SCALAR_TYPES:
            return
        if value_type is list:
            for index, item in enumerate(value):
                EventLog._validate_value(item, f"{name}[{index}]")
            return
        if value_type is dict:
            for key, item in value.items():
                if type(key) is not str:
                    raise TypeError(f"field {name!r} has a non-string key")
                EventLog._validate_value(item, f"{name}.{key}")
            return
        raise TypeError(f"field {name!r} must be a plain JSON value")
