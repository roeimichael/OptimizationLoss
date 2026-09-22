import json
import math
import random
import tempfile
import unittest
from datetime import datetime
from pathlib import Path

from tralo.events import EventLog


class EventLogTests(unittest.TestCase):
    def test_writes_schema_sequence_timestamp_event_and_fields(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "events.jsonl"
            with EventLog(path) as log:
                log.emit("started", count=3, enabled=True, note=None)
                log.emit("finished", score=0.5)

            records = [json.loads(line) for line in path.read_text().splitlines()]
            self.assertEqual(records[0]["schema_version"], 1)
            self.assertEqual(records[0]["sequence"], 0)
            self.assertEqual(records[0]["event"], "started")
            self.assertEqual(records[0]["count"], 3)
            self.assertEqual(records[1]["sequence"], 1)
            self.assertEqual(records[1]["event"], "finished")
            for record in records:
                datetime.fromisoformat(record["timestamp"].replace("Z", "+00:00"))

    def test_opening_existing_file_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "events.jsonl"
            path.write_text("keep me\n")
            with self.assertRaises(FileExistsError):
                EventLog(path)
            self.assertEqual(path.read_text(), "keep me\n")

    def test_nested_json_values_are_written(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "events.jsonl"
            with EventLog(path) as log:
                log.emit("nested", values=[1, {"ok": True}], metadata={"name": "run"})
            record = json.loads(path.read_text())
            self.assertEqual(record["values"], [1, {"ok": True}])
            self.assertEqual(record["metadata"], {"name": "run"})

    def test_nested_nonfinite_values_write_no_partial_line(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "events.jsonl"
            with EventLog(path) as log:
                for value in ([math.nan], {"x": math.inf}, {"x": [-math.inf]}):
                    with self.assertRaises((TypeError, ValueError)):
                        log.emit("invalid", value=value)
                self.assertEqual(path.read_text(), "")

    def test_set_and_callable_values_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "events.jsonl"
            with EventLog(path) as log:
                for value in ({1}, lambda: None):
                    with self.assertRaises(TypeError):
                        log.emit("invalid", value=value)
                self.assertEqual(path.read_text(), "")

    def test_reserved_keys_are_rejected_without_writing(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "events.jsonl"
            with EventLog(path) as log:
                for key in ("schema_version", "sequence", "timestamp", "event"):
                    with self.assertRaises(ValueError):
                        log.emit("invalid", **{key: "supplied"})
                self.assertEqual(path.read_text(), "")

    def test_nested_event_snapshot_is_serialized_before_mutation(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "events.jsonl"
            value = {"items": ["before"]}
            with EventLog(path) as log:
                log.emit("snapshot", value=value)
                value["items"][0] = "after"
                value["new"] = True
            self.assertEqual(json.loads(path.read_text())["value"], {"items": ["before"]})

    def test_logging_does_not_change_global_random_state(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "events.jsonl"
            random.seed(12345)
            expected = random.getstate()
            random.setstate(expected)
            with EventLog(path) as log:
                log.emit("stable")
            actual = random.getstate()
            self.assertEqual(actual, expected)

    def test_write_after_close_propagates_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "events.jsonl"
            log = EventLog(path)
            with log:
                pass
            with self.assertRaises(ValueError):
                log.emit("closed")


if __name__ == "__main__":
    unittest.main()
