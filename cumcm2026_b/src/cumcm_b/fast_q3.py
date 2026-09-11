"""Interleaved discovery, localization, and routing for question 3."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

from .baseline import ChannelTrack, OmniGeometryBaseline
from .geometry import Point, minimum_enclosing_circle
from .planning import grid_points_in_disk, sample_convex_polygon, select_second_measurement_points


@dataclass(frozen=True)
class FastQ3Result:
    cleared_count: int
    virtual_time_s: float
    measure_actions: int
    clear_actions: int
    travel_distance_m: float
    discovery_scan_positions: int
    estimated_remaining_sources: float
    stopped_by_probability: bool
    detected_count: int
    unresolved_channels: tuple[int, ...]
    decision_rounds: int

    @property
    def mean_clear_time_s(self) -> float:
        if not self.cleared_count:
            return math.inf
        return self.virtual_time_s / self.cleared_count


class InterleavedOmniSearch:
    """Use localization/clear movements as discovery measurements.

    Every new position is shared across two purposes: obtaining another bearing
    for one known channel and scanning channels that have not yet been found.
    This removes the long fixed coverage tour used by the conservative baseline.
    """

    def __init__(
        self,
        *,
        clear_radius_m: float = 19.0,
        miss_budget_sources: float = 0.015,
        maximum_localization_measurements: int = 8,
        maximum_decision_rounds: int = 160,
        discovery_grid_spacing_m: float = 120.0,
        candidate_grid_spacing_m: float = 300.0,
    ):
        self.clear_radius_m = clear_radius_m
        self.miss_budget_sources = miss_budget_sources
        self.maximum_localization_measurements = maximum_localization_measurements
        self.maximum_decision_rounds = maximum_decision_rounds
        self.discovery_grid = grid_points_in_disk(spacing=discovery_grid_spacing_m)
        self.exploration_candidates = grid_points_in_disk(spacing=candidate_grid_spacing_m)

    @staticmethod
    def _distance(first: Point, second: Point) -> float:
        return math.hypot(first[0] - second[0], first[1] - second[1])

    @staticmethod
    def _undetected(tracks: dict[int, ChannelTrack]) -> list[ChannelTrack]:
        return [track for track in tracks.values() if not track.detected and not track.cleared]

    @staticmethod
    def _unresolved(tracks: dict[int, ChannelTrack]) -> list[ChannelTrack]:
        return [track for track in tracks.values() if track.detected and not track.cleared]

    def _miss_probability(self, scan_positions: Sequence[Point]) -> float:
        """Expected omni-source miss probability under the published priors."""

        if not scan_positions:
            return 1.0
        total = 0.0
        for point in self.discovery_grid:
            nearest = min(self._distance(point, scan) for scan in scan_positions)
            # Receive radius is uniformly distributed in [1000, 1500].
            total += min(1.0, max(0.0, (nearest - 1000.0) / 500.0))
        return total / len(self.discovery_grid)

    def _expected_remaining_sources(
        self, cleared_count: int, scan_positions: Sequence[Point]
    ) -> float:
        # At most 16 sources exist.  This conservative multiplier prevents the
        # policy from assuming the unknown true count returned only after exit.
        return max(0, 16 - cleared_count) * self._miss_probability(scan_positions)

    def _next_exploration_point(
        self,
        current: Point,
        scan_positions: Sequence[Point],
        unknown_channel_count: int,
    ) -> Point:
        before = self._miss_probability(scan_positions)
        best_point = current
        best_key = (-math.inf, -math.inf)
        operation_s = max(1, unknown_channel_count) * 6.0
        for candidate in self.exploration_candidates:
            if any(self._distance(candidate, old) < 80.0 for old in scan_positions):
                continue
            after = self._miss_probability([*scan_positions, candidate])
            reduction = before - after
            travel_s = self._distance(current, candidate) / 5.0
            utility = reduction / max(1.0, travel_s + operation_s)
            key = (utility, -travel_s)
            if key > best_key:
                best_key = key
                best_point = candidate
        return best_point

    def _next_localization_point(
        self, track: ChannelTrack, current: Point
    ) -> Point | None:
        if not track.polygon:
            return None
        hypotheses = sample_convex_polygon(track.polygon, spacing=60.0, max_points=160)
        hypotheses = [
            point
            for point in hypotheses
            if all(self._distance(point, miss) > 1000.0 for miss in track.no_signal_positions)
        ]
        if not hypotheses:
            return None
        candidates = OmniGeometryBaseline._candidate_measurement_points(
            track.polygon, current, track.attempted_measurement_positions
        )
        if not candidates:
            return None
        ranked = select_second_measurement_points(
            track.polygon,
            current,
            candidates,
            source_hypotheses=hypotheses,
            top_n=1,
            guaranteed_receive_radius=1000.0,
        )
        return ranked[0].point if ranked else None

    def run(self, environment) -> FastQ3Result:
        tracks = {channel: ChannelTrack(channel) for channel in range(1, 21)}
        measure_actions = 0
        clear_actions = 0
        travel_distance_m = 0.0
        scan_positions: list[Point] = []
        stopped_by_probability = False

        def move_distance(target: Point) -> float:
            return self._distance(environment.position, target)

        def clear_track(track: ChannelTrack, point: Point) -> bool:
            nonlocal clear_actions, travel_distance_m
            travel_distance_m += move_distance(point)
            result = environment.clear(point, track.channel)
            clear_actions += 1
            if result.result == "success":
                track.cleared = True
                return True
            return False

        def measure_track(track: ChannelTrack, point: Point) -> None:
            nonlocal measure_actions, clear_actions, travel_distance_m
            travel_distance_m += move_distance(point)
            observation = environment.measure(point, track.channel)
            measure_actions += 1
            track.update(observation)
            if observation.result == "near":
                clear_track(track, point)

        def scan_active(point: Point, primary_channel: int | None = None) -> None:
            unknown_before = self._undetected(tracks)
            active: list[ChannelTrack] = []
            for track in tracks.values():
                if track.cleared:
                    continue
                circle = track.enclosing_circle
                if track.detected and circle is not None and circle.radius <= self.clear_radius_m:
                    continue
                if any(self._distance(point, old) < 1e-8 for old in track.attempted_measurement_positions):
                    continue
                active.append(track)
            if not active:
                return
            active.sort(
                key=lambda track: (
                    track.channel != primary_channel,
                    track.channel != environment.current_channel,
                    track.channel,
                )
            )
            for track in active:
                measure_track(track, point)
            if unknown_before:
                scan_positions.append(point)

        scan_active((0.0, 0.0))

        rounds_completed = 0
        for rounds_completed in range(1, self.maximum_decision_rounds + 1):
            ready: list[tuple[float, ChannelTrack, Point]] = []
            for track in self._unresolved(tracks):
                circle = track.enclosing_circle
                if circle is not None and circle.radius <= self.clear_radius_m:
                    ready.append((self._distance(environment.position, circle.center), track, circle.center))
            if ready:
                _, track, point = min(ready, key=lambda item: item[0])
                clear_track(track, point)
                scan_active(point)
                continue

            unresolved = self._unresolved(tracks)
            measurable = [
                track
                for track in unresolved
                if len(track.attempted_measurement_positions)
                < self.maximum_localization_measurements
            ]
            measurable.sort(
                key=lambda track: self._distance(
                    environment.position, minimum_enclosing_circle(track.polygon).center
                )
            )
            selected_track: ChannelTrack | None = None
            selected_point: Point | None = None
            for track in measurable:
                point = self._next_localization_point(track, environment.position)
                if point is None:
                    continue
                selected_track = track
                selected_point = point
                break
            if selected_track is not None and selected_point is not None:
                scan_active(selected_point, primary_channel=selected_track.channel)
                continue

            remaining = self._expected_remaining_sources(
                environment.cleared_count, scan_positions
            )
            if (
                environment.cleared_count >= 10
                and not unresolved
                and remaining <= self.miss_budget_sources
            ):
                stopped_by_probability = True
                break
            if environment.cleared_count >= 16:
                stopped_by_probability = True
                break

            unknown = self._undetected(tracks)
            if not unknown:
                break
            next_point = self._next_exploration_point(
                environment.position, scan_positions, len(unknown)
            )
            if any(self._distance(next_point, old) < 1e-8 for old in scan_positions):
                break
            scan_active(next_point)

        remaining = self._expected_remaining_sources(
            environment.cleared_count, scan_positions
        )
        return FastQ3Result(
            cleared_count=environment.cleared_count,
            virtual_time_s=environment.virtual_time_s,
            measure_actions=measure_actions,
            clear_actions=clear_actions,
            travel_distance_m=travel_distance_m,
            discovery_scan_positions=len(scan_positions),
            estimated_remaining_sources=remaining,
            stopped_by_probability=stopped_by_probability,
            detected_count=sum(track.detected for track in tracks.values()),
            unresolved_channels=tuple(
                track.channel for track in self._unresolved(tracks)
            ),
            decision_rounds=rounds_completed,
        )
