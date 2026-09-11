"""A local, testable surrogate of the B-problem interaction rules.

The official simulator distribution is unknown.  This module implements the
published transition and observation semantics, while keeping source generation
and the spatial bearing-error field explicitly configurable for domain
randomization.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import random
from typing import Literal, Sequence

from .geometry import Point

MeasureKind = Literal["no_signal", "near", "direction"]
ClearKind = Literal["success", "no_target_in_range"]


def _normalize_degrees(angle: float) -> float:
    return angle % 360.0


def _signed_angle_difference(first: float, second: float) -> float:
    """Return ``first-second`` in [-180, 180)."""

    return (first - second + 180.0) % 360.0 - 180.0


@dataclass(frozen=True)
class Source:
    channel: int
    position: Point
    receive_radius: float
    directional: bool = False
    heading_deg: float = 0.0
    coverage_deg: float = 180.0
    error_coefficients: tuple[float, ...] = ()

    def __post_init__(self) -> None:
        if not 1 <= self.channel <= 20:
            raise ValueError("channel must be in 1..20")
        if not 1000.0 <= self.receive_radius <= 1500.0:
            raise ValueError("receive_radius must be in [1000, 1500]")
        if not 0.0 < self.coverage_deg <= 360.0:
            raise ValueError("coverage_deg must be in (0, 360]")


@dataclass(frozen=True)
class MeasureObservation:
    result: MeasureKind
    position: Point
    channel: int
    virtual_time_s: float
    action_duration_s: float
    bearing_deg: float | None = None


@dataclass(frozen=True)
class ClearObservation:
    result: ClearKind
    position: Point
    channel: int
    virtual_time_s: float
    action_duration_s: float


def _spatial_error_deg(source: Source, position: Point) -> float:
    """Return a deterministic, spatially correlated error in [-1, 1] degrees."""

    coefficients = source.error_coefficients
    if not coefficients:
        return 0.0
    if len(coefficients) % 4 != 0:
        raise ValueError("error_coefficients length must be a multiple of four")
    x, y = position
    total = 0.0
    amplitude_sum = 0.0
    for index in range(0, len(coefficients), 4):
        amplitude, kx, ky, phase = coefficients[index : index + 4]
        total += amplitude * math.sin(kx * x + ky * y + phase)
        amplitude_sum += abs(amplitude)
    if amplitude_sum <= 1e-12:
        return 0.0
    return max(-1.0, min(1.0, total / amplitude_sum))


def _source_visible(source: Source, observer: Point) -> bool:
    dx = observer[0] - source.position[0]
    dy = observer[1] - source.position[1]
    distance = math.hypot(dx, dy)
    if distance > source.receive_radius:
        return False
    if not source.directional or source.coverage_deg >= 360.0:
        return True
    source_to_observer = _normalize_degrees(math.degrees(math.atan2(dy, dx)))
    return abs(_signed_angle_difference(source_to_observer, source.heading_deg)) <= source.coverage_deg / 2.0 + 1e-10


def generate_sources(
    seed: int,
    *,
    count: int | None = None,
    target_radius: float = 1800.0,
    directional_probability: float = 0.5,
    error_modes: int = 4,
) -> list[Source]:
    """Generate one domain-randomized source configuration.

    Positions are uniform by area in the target disk.  Channels are sampled
    without replacement, preserving the official at-most-one-source-per-channel
    condition.
    """

    if count is not None and not 10 <= count <= 16:
        raise ValueError("count must be in [10, 16]")
    if not 0.0 <= directional_probability <= 1.0:
        raise ValueError("directional_probability must be in [0, 1]")
    if error_modes < 0:
        raise ValueError("error_modes must be non-negative")
    rng = random.Random(seed)
    source_count = count if count is not None else rng.randint(10, 16)
    channels = rng.sample(range(1, 21), source_count)
    sources: list[Source] = []
    for channel in channels:
        radius = target_radius * math.sqrt(rng.random())
        angle = rng.uniform(0.0, 2.0 * math.pi)
        position = (radius * math.cos(angle), radius * math.sin(angle))
        directional = rng.random() < directional_probability
        coefficients: list[float] = []
        for _ in range(error_modes):
            correlation_length = rng.uniform(80.0, 600.0)
            wave_angle = rng.uniform(0.0, 2.0 * math.pi)
            wave_number = 2.0 * math.pi / correlation_length
            coefficients.extend(
                (
                    rng.uniform(0.25, 1.0),
                    wave_number * math.cos(wave_angle),
                    wave_number * math.sin(wave_angle),
                    rng.uniform(0.0, 2.0 * math.pi),
                )
            )
        sources.append(
            Source(
                channel=channel,
                position=position,
                receive_radius=rng.uniform(1000.0, 1500.0),
                directional=directional,
                heading_deg=rng.uniform(0.0, 360.0),
                coverage_deg=180.0 if directional else 360.0,
                error_coefficients=tuple(coefficients),
            )
        )
    return sources


class InterferenceEnvironment:
    """Stateful surrogate implementing movement, measurement, and clearing."""

    speed_mps = 5.0
    switch_time_s = 1.0
    measure_time_s = 5.0
    precise_localization_time_s = 3.0
    clearing_time_s = 2.0
    near_radius_m = 5.0
    clear_radius_m = 20.0

    def __init__(self, sources: Sequence[Source]):
        channels = [source.channel for source in sources]
        if len(set(channels)) != len(channels):
            raise ValueError("each channel may contain at most one source")
        self._sources = {source.channel: source for source in sources}
        self.position: Point = (0.0, 0.0)
        self.current_channel = 1
        self.virtual_time_s = 0.0
        self.cleared_channels: set[int] = set()

    @property
    def source_count(self) -> int:
        return len(self._sources)

    @property
    def cleared_count(self) -> int:
        return len(self.cleared_channels)

    @property
    def all_cleared(self) -> bool:
        return self.cleared_count == self.source_count

    def _move_to(self, target: Point) -> float:
        if not all(math.isfinite(value) and abs(value) <= 2_000_000 for value in target):
            raise ValueError("coordinates must be finite and at most 2,000,000 in magnitude")
        distance = math.hypot(target[0] - self.position[0], target[1] - self.position[1])
        self.position = target
        return distance / self.speed_mps

    def measure(self, position: Point, channel: int) -> MeasureObservation:
        if not 1 <= channel <= 20:
            raise ValueError("channel must be in 1..20")
        move_time = self._move_to(position)
        switch_time = self.switch_time_s if channel != self.current_channel else 0.0
        self.current_channel = channel
        action_duration = move_time + switch_time + self.measure_time_s
        self.virtual_time_s += action_duration

        source = self._sources.get(channel)
        if source is None or channel in self.cleared_channels or not _source_visible(source, position):
            return MeasureObservation(
                "no_signal", position, channel, self.virtual_time_s, action_duration
            )

        dx = source.position[0] - position[0]
        dy = source.position[1] - position[1]
        if math.hypot(dx, dy) <= self.near_radius_m:
            return MeasureObservation(
                "near", position, channel, self.virtual_time_s, action_duration
            )
        true_bearing = _normalize_degrees(math.degrees(math.atan2(dy, dx)))
        observed = _normalize_degrees(true_bearing + _spatial_error_deg(source, position))
        return MeasureObservation(
            "direction",
            position,
            channel,
            self.virtual_time_s,
            action_duration,
            observed,
        )

    def clear(self, position: Point, channel: int) -> ClearObservation:
        if not 1 <= channel <= 20:
            raise ValueError("channel must be in 1..20")
        move_time = self._move_to(position)
        source = self._sources.get(channel)
        success = (
            source is not None
            and channel not in self.cleared_channels
            and math.hypot(
                source.position[0] - position[0], source.position[1] - position[1]
            )
            <= self.clear_radius_m
        )
        operation_time = self.precise_localization_time_s
        if success:
            self.cleared_channels.add(channel)
            operation_time += self.clearing_time_s
        action_duration = move_time + operation_time
        self.virtual_time_s += action_duration
        return ClearObservation(
            "success" if success else "no_target_in_range",
            position,
            channel,
            self.virtual_time_s,
            action_duration,
        )
