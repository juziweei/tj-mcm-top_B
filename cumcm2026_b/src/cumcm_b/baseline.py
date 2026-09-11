"""Deterministic geometry-first baseline for the all-directional-source task."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Sequence

from .geometry import (
    Circle,
    Point,
    localization_polygon,
    minimum_enclosing_circle,
)
from .planning import sample_convex_polygon, select_second_measurement_points
from .simulator import InterferenceEnvironment, MeasureObservation


def guaranteed_discovery_waypoints(
    *,
    target_radius: float = 1800.0,
    receive_radius: float = 1000.0,
    ring_count: int = 6,
    safety_margin: float = 20.0,
) -> list[Point]:
    """Cover the target disk with one central and ``ring_count`` sensing disks.

    For six ring points, the radius is chosen from the exact outer-boundary
    coverage equation.  A small positive margin moves the ring radius into the
    feasible interval instead of placing it exactly at a tangency condition.
    """

    if ring_count < 3:
        raise ValueError("ring_count must be at least three")
    if not 0.0 < receive_radius < target_radius:
        raise ValueError("receive_radius must lie in (0, target_radius)")
    half_sector = math.pi / ring_count
    radial_term = receive_radius ** 2 - (
        target_radius * math.sin(half_sector)
    ) ** 2
    if radial_term < 0.0:
        raise ValueError("the requested ring count cannot cover the disk boundary")
    lower_ring_radius = (
        target_radius * math.cos(half_sector) - math.sqrt(radial_term)
    )
    ring_radius = lower_ring_radius + safety_margin
    if ring_radius - receive_radius > receive_radius:
        raise ValueError("central and ring sensing disks do not overlap")
    return [(0.0, 0.0)] + [
        (
            ring_radius * math.cos(2.0 * math.pi * index / ring_count),
            ring_radius * math.sin(2.0 * math.pi * index / ring_count),
        )
        for index in range(ring_count)
    ]


@dataclass
class ChannelTrack:
    channel: int
    measurements: list[tuple[Point, float]] = field(default_factory=list)
    polygon: list[Point] = field(default_factory=list)
    detected: bool = False
    cleared: bool = False
    no_signal_positions: list[Point] = field(default_factory=list)
    attempted_measurement_positions: list[Point] = field(default_factory=list)

    def update(self, observation: MeasureObservation) -> None:
        self.attempted_measurement_positions.append(observation.position)
        if observation.result == "direction":
            assert observation.bearing_deg is not None
            self.detected = True
            self.measurements.append((observation.position, observation.bearing_deg))
            self.polygon = localization_polygon(
                self.measurements,
                error_deg=1.0,
                target_radius=1800.0,
                disk_vertices=720,
            )
        elif observation.result == "near":
            self.detected = True
        else:
            self.no_signal_positions.append(observation.position)

    @property
    def enclosing_circle(self) -> Circle | None:
        if not self.polygon:
            return None
        return minimum_enclosing_circle(self.polygon)


@dataclass(frozen=True)
class OmniBaselineResult:
    source_count: int
    cleared_count: int
    virtual_time_s: float
    measure_actions: int
    clear_actions: int
    discovery_waypoints: int

    @property
    def cleared_fraction(self) -> float:
        return self.cleared_count / self.source_count if self.source_count else 1.0

    @property
    def mean_clear_time_s(self) -> float:
        return self.virtual_time_s / self.cleared_count if self.cleared_count else math.inf


class OmniGeometryBaseline:
    """Coverage discovery followed by adaptive bearing-only localization."""

    def __init__(
        self,
        *,
        clear_radius: float = 19.5,
        maximum_localization_steps: int = 8,
    ):
        if not 0.0 < clear_radius <= 20.0:
            raise ValueError("clear_radius must be in (0, 20]")
        self.clear_radius = clear_radius
        self.maximum_localization_steps = maximum_localization_steps

    @staticmethod
    def _candidate_measurement_points(
        polygon: Sequence[Point], current_position: Point, attempted: Sequence[Point]
    ) -> list[Point]:
        circle = minimum_enclosing_circle(polygon)
        points: list[Point] = [current_position, circle.center]
        for radius in (250.0, 500.0, 750.0, 1000.0):
            for angle_index in range(12):
                angle = 2.0 * math.pi * angle_index / 12
                point = (
                    circle.center[0] + radius * math.cos(angle),
                    circle.center[1] + radius * math.sin(angle),
                )
                if point[0] ** 2 + point[1] ** 2 <= 1800.0 ** 2:
                    points.append(point)
        return [
            point
            for point in points
            if all(
                math.hypot(point[0] - old[0], point[1] - old[1]) > 1e-6
                for old in attempted
            )
        ]

    def _localize_and_clear(
        self,
        environment: InterferenceEnvironment,
        track: ChannelTrack,
    ) -> tuple[int, int]:
        measurements = 0
        clears = 0
        for _ in range(self.maximum_localization_steps):
            circle = track.enclosing_circle
            if circle is not None and circle.radius <= self.clear_radius:
                result = environment.clear(circle.center, track.channel)
                clears += 1
                if result.result == "success":
                    track.cleared = True
                    return measurements, clears

            if not track.polygon:
                return measurements, clears
            hypotheses = sample_convex_polygon(
                track.polygon, spacing=75.0, max_points=128
            )
            hypotheses = [
                hypothesis
                for hypothesis in hypotheses
                if all(
                    math.hypot(
                        hypothesis[0] - no_signal[0],
                        hypothesis[1] - no_signal[1],
                    )
                    > 1000.0
                    for no_signal in track.no_signal_positions
                )
            ]
            if not hypotheses:
                return measurements, clears
            candidates = self._candidate_measurement_points(
                track.polygon,
                environment.position,
                track.attempted_measurement_positions,
            )
            ranked = select_second_measurement_points(
                track.polygon,
                environment.position,
                candidates,
                source_hypotheses=hypotheses,
                top_n=1,
                guaranteed_receive_radius=1000.0,
            )
            if not ranked:
                return measurements, clears
            observation = environment.measure(ranked[0].point, track.channel)
            measurements += 1
            track.update(observation)
            if observation.result == "near":
                result = environment.clear(observation.position, track.channel)
                clears += 1
                if result.result == "success":
                    track.cleared = True
                    return measurements, clears
        return measurements, clears

    def run(self, environment: InterferenceEnvironment) -> OmniBaselineResult:
        tracks = {channel: ChannelTrack(channel) for channel in range(1, 21)}
        measure_actions = 0
        clear_actions = 0
        waypoints = guaranteed_discovery_waypoints()

        # Unknown channels are scanned at every coverage point.  Once a channel
        # has been detected, later coverage points are also useful triangulation
        # measurements until its enclosing circle is small enough to clear.
        for waypoint in waypoints:
            for channel, track in tracks.items():
                circle = track.enclosing_circle
                if track.cleared or (circle is not None and circle.radius <= self.clear_radius):
                    continue
                observation = environment.measure(waypoint, channel)
                measure_actions += 1
                track.update(observation)
                if observation.result == "near":
                    clearing = environment.clear(waypoint, channel)
                    clear_actions += 1
                    if clearing.result == "success":
                        track.cleared = True

        detected = [track for track in tracks.values() if track.detected]
        while detected:
            uncleared = [track for track in detected if not track.cleared]
            if not uncleared:
                break

            def estimated_distance(track: ChannelTrack) -> float:
                circle = track.enclosing_circle
                if circle is None:
                    return math.inf
                return math.hypot(
                    circle.center[0] - environment.position[0],
                    circle.center[1] - environment.position[1],
                )

            track = min(uncleared, key=estimated_distance)
            new_measures, new_clears = self._localize_and_clear(environment, track)
            measure_actions += new_measures
            clear_actions += new_clears
            if not track.cleared:
                # Avoid a non-progressing loop.  The result exposes the miss and
                # lets later policies improve on it rather than hiding failure.
                detected.remove(track)

        return OmniBaselineResult(
            source_count=environment.source_count,
            cleared_count=environment.cleared_count,
            virtual_time_s=environment.virtual_time_s,
            measure_actions=measure_actions,
            clear_actions=clear_actions,
            discovery_waypoints=len(waypoints),
        )
