"""HTTP+JSON client for the official B-problem simulator protocol."""

from __future__ import annotations

from collections.abc import Callable
import json
from pathlib import Path
import socket
import time
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
import uuid

from .geometry import Point

Transport = Callable[[str, bytes, float], tuple[int, bytes]]


class SimulatorProtocolError(RuntimeError):
    pass


def _default_transport(url: str, body: bytes, timeout_s: float) -> tuple[int, bytes]:
    request = Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlopen(request, timeout=timeout_s) as response:
            return response.status, response.read()
    except HTTPError as error:
        return error.code, error.read()


class SimulatorClient:
    """Serial client with idempotent timeout retries and JSONL logging."""

    def __init__(
        self,
        robot_id: str,
        *,
        base_url: str = "http://127.0.0.1:2026",
        arena_id: str = "default",
        timeout_s: float = 5.0,
        timeout_retries: int = 2,
        log_path: str | Path | None = None,
        transport: Transport | None = None,
    ):
        if not robot_id:
            raise ValueError("robot_id must be non-empty")
        if timeout_s <= 0:
            raise ValueError("timeout_s must be positive")
        if timeout_retries < 0:
            raise ValueError("timeout_retries must be non-negative")
        self.robot_id = robot_id
        self.base_url = base_url.rstrip("/")
        self.arena_id = arena_id
        self.timeout_s = timeout_s
        self.timeout_retries = timeout_retries
        self.log_path = Path(log_path) if log_path is not None else None
        self.transport = transport or _default_transport
        self._sequence = 0

    def _new_request_id(self, operation: str) -> str:
        self._sequence += 1
        return f"{operation}-{self._sequence}-{uuid.uuid4().hex[:12]}"

    def _base_payload(self, request_id: str) -> dict[str, Any]:
        return {
            "arena_id": self.arena_id,
            "robot_id": self.robot_id,
            "request_id": request_id,
        }

    def _append_log(self, record: dict[str, Any]) -> None:
        if self.log_path is None:
            return
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with self.log_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")))
            stream.write("\n")

    def _post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        url = self.base_url + path
        started_ns = time.time_ns()
        last_error: Exception | None = None
        for attempt in range(self.timeout_retries + 1):
            try:
                status, response_body = self.transport(url, encoded, self.timeout_s)
                try:
                    response = json.loads(response_body.decode("utf-8"))
                except (UnicodeDecodeError, json.JSONDecodeError) as error:
                    raise SimulatorProtocolError(
                        f"{path} returned a non-JSON response with HTTP {status}"
                    ) from error
                self._append_log(
                    {
                        "real_timestamp_ns": time.time_ns(),
                        "elapsed_real_ms": (time.time_ns() - started_ns) / 1_000_000,
                        "attempt": attempt + 1,
                        "path": path,
                        "http_status": status,
                        "request": payload,
                        "response": response,
                    }
                )
                if status != 200:
                    raise SimulatorProtocolError(
                        f"{path} failed with HTTP {status}: {response}"
                    )
                if response.get("accepted") is not True:
                    raise SimulatorProtocolError(
                        f"{path} was not accepted: {response}"
                    )
                return response
            except (TimeoutError, socket.timeout, URLError) as error:
                last_error = error
                self._append_log(
                    {
                        "real_timestamp_ns": time.time_ns(),
                        "elapsed_real_ms": (time.time_ns() - started_ns) / 1_000_000,
                        "attempt": attempt + 1,
                        "path": path,
                        "request": payload,
                        "transport_error": repr(error),
                    }
                )
                if attempt >= self.timeout_retries:
                    break
        raise SimulatorProtocolError(
            f"{path} timed out after {self.timeout_retries + 1} attempts"
        ) from last_error

    def enter(self, *, request_id: str | None = None) -> dict[str, Any]:
        request_id = request_id or self._new_request_id("enter")
        return self._post("/enter", self._base_payload(request_id))

    def measure(
        self,
        position: Point,
        channel: int,
        *,
        request_id: str | None = None,
    ) -> dict[str, Any]:
        request_id = request_id or self._new_request_id("measure")
        payload = self._base_payload(request_id)
        payload["position"] = {"x": position[0], "y": position[1]}
        payload["channel"] = channel
        response = self._post("/measure", payload)
        result = response.get("measure_result")
        if result not in {"no_signal", "near", "direction"}:
            raise SimulatorProtocolError(f"unknown measure_result: {result!r}")
        if result == "direction" and "svd_deg" not in response:
            raise SimulatorProtocolError("direction response is missing svd_deg")
        return response

    def clear(
        self,
        position: Point,
        channel: int,
        *,
        request_id: str | None = None,
    ) -> dict[str, Any]:
        request_id = request_id or self._new_request_id("clear")
        payload = self._base_payload(request_id)
        payload["position"] = {"x": position[0], "y": position[1]}
        payload["channel"] = channel
        response = self._post("/clear", payload)
        result = response.get("clear_result")
        if result not in {"success", "no_target_in_range"}:
            raise SimulatorProtocolError(f"unknown clear_result: {result!r}")
        return response

    def exit(self, *, request_id: str | None = None) -> dict[str, Any]:
        request_id = request_id or self._new_request_id("exit")
        return self._post("/exit", self._base_payload(request_id))

