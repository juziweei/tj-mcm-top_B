"""Adapters and safety checks for the official CUMCM B simulator."""

from __future__ import annotations

from contextlib import closing
import json
from pathlib import Path
import sqlite3
import time
from typing import Any

from .client import SimulatorClient
from .geometry import Point
from .simulator import ClearObservation, MeasureObservation


def require_latest_practice_run(simulator_data_dir: str | Path) -> Path:
    """Return the latest behavior journal only when it is a practice run.

    The official UI separates practice and formal tests, while the local HTTP
    protocol does not.  Refusing to enter a non-practice journal prevents an
    accidental formal-attempt consumption at that irreversible boundary.
    """

    runs_dir = Path(simulator_data_dir) / "behavior-runs"
    journals = sorted(
        runs_dir.glob("run-*/behavior.journal.jsonl"),
        key=lambda path: path.stat().st_mtime_ns,
        reverse=True,
    )
    if not journals:
        raise RuntimeError(f"no official behavior journal found under {runs_dir}")

    journal = journals[0]
    lifecycle_events: list[str] = []
    with journal.open("r", encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("record_type") == "lifecycle":
                lifecycle_events.append(str(record.get("event", "")))

    if "practice_authorized" not in lifecycle_events:
        raise RuntimeError(
            "latest official simulator run is not proven to be practice; "
            "refusing to call /enter"
        )
    if "formal_authorized" in lifecycle_events:
        raise RuntimeError("latest journal contains a formal authorization")
    if "api_opened" not in lifecycle_events:
        raise RuntimeError("practice run exists, but its local API is not open yet")
    return journal


def latest_practice_statistics_id(database: str | Path) -> int:
    """Read the current queue watermark without creating or modifying the DB."""

    path = Path(database).resolve()
    if not path.is_file():
        return 0
    with closing(
        sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True)
    ) as connection:
        row = connection.execute(
            "SELECT COALESCE(MAX(id), 0) FROM practice_statistics_tasks"
        ).fetchone()
    return int(row[0]) if row else 0


def wait_for_practice_statistics(
    database: str | Path,
    *,
    after_id: int,
    team_no: str,
    problem_no: int,
    timeout_s: float = 10.0,
) -> dict[str, Any] | None:
    """Return the new authoritative practice row after the simulator exits."""

    path = Path(database).resolve()
    if not path.is_file():
        return None
    deadline = time.monotonic() + timeout_s
    latest: dict[str, Any] | None = None
    columns = (
        "id, practice_run_no, case_code, end_reason, cleared_jammer_count, "
        "measure_accepted_count, virtual_time_us, program_run_duration_ms, "
        "channel_switch_count, clear_failure_count, jammer_count, state"
    )
    while True:
        with closing(
            sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True)
        ) as connection:
            connection.row_factory = sqlite3.Row
            row = connection.execute(
                f"SELECT {columns} FROM practice_statistics_tasks "
                "WHERE id > ? AND team_no = ? AND problem_no = ? "
                "ORDER BY id DESC LIMIT 1",
                (after_id, team_no, problem_no),
            ).fetchone()
        if row is not None:
            latest = dict(row)
            if latest.get("state") == "confirmed":
                return latest
        if time.monotonic() >= deadline:
            return latest
        time.sleep(0.2)


class OfficialEnvironmentAdapter:
    """Expose the official HTTP simulator through the local environment API."""

    def __init__(self, client: SimulatorClient):
        self.client = client
        self.position: Point = (0.0, 0.0)
        self.current_channel = 1
        self.virtual_time_s = 0.0
        self.cleared_channels: set[int] = set()

    @property
    def cleared_count(self) -> int:
        return len(self.cleared_channels)

    @property
    def source_count(self) -> int:
        # The official API intentionally hides the source count.  At the end
        # of a run, the adapter can only report the number it cleared.
        return self.cleared_count

    def measure(self, position: Point, channel: int) -> MeasureObservation:
        before = self.virtual_time_s
        response = self.client.measure(position, channel)
        self.virtual_time_s = float(response["virtual_time_s"])
        duration = self.virtual_time_s - before
        self.position = (float(position[0]), float(position[1]))
        self.current_channel = int(channel)
        result = str(response["measure_result"])
        bearing = (
            float(response["svd_deg"])
            if result == "direction"
            else None
        )
        return MeasureObservation(
            result, self.position, channel, self.virtual_time_s, duration, bearing
        )

    def clear(self, position: Point, channel: int) -> ClearObservation:
        before = self.virtual_time_s
        response = self.client.clear(position, channel)
        self.virtual_time_s = float(response["virtual_time_s"])
        duration = self.virtual_time_s - before
        self.position = (float(position[0]), float(position[1]))
        result = str(response["clear_result"])
        if result == "success":
            self.cleared_channels.add(int(channel))
        return ClearObservation(
            result, self.position, channel, self.virtual_time_s, duration
        )

    def state_dict(self) -> dict[str, Any]:
        return {
            "position": {"x": self.position[0], "y": self.position[1]},
            "current_channel": self.current_channel,
            "virtual_time_s": self.virtual_time_s,
            "cleared_channels": sorted(self.cleared_channels),
        }
