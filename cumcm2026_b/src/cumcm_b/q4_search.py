"""Guaranteed sparse discovery and bearing pursuit for question 4."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import math
from typing import Callable, Literal, Sequence

import numpy as np

from .baseline import ChannelTrack
from .geometry import Point
from .route_oracle import exact_open_route, exact_open_route_costs, fast_open_route


@dataclass(frozen=True)
class Q4SearchResult:
    cleared_count: int
    virtual_time_s: float
    measure_actions: int
    discovery_measure_actions: int
    undetected_discovery_measure_actions: int
    detected_discovery_measure_actions: int
    pursuit_measure_actions: int
    clear_actions: int
    travel_distance_m: float
    discovery_travel_distance_m: float
    pursuit_travel_distance_m: float
    discovery_positions: int
    discovery_scan_positions: tuple[Point, ...]
    exhausted_discovery_lattice: bool
    unresolved_channels: tuple[int, ...]
    posterior_all_sources_detected: float
    posterior_expected_remaining_sources: float
    posterior_expected_missed_source_fraction: float
    stopped_by_probability: bool
    stopped_by_source_risk: bool

    @property
    def mean_clear_time_s(self) -> float:
        return (
            self.virtual_time_s / self.cleared_count
            if self.cleared_count
            else math.inf
        )


Q4DecisionKind = Literal["discovery", "pursuit"]
Q4DecisionChoice = Point | int | None
Q4Profile = Literal[
    "conservative",
    "balanced",
    "aggressive",
    "source-reliable",
    "source-efficient",
    "source-98",
]


def q4_profile_options(profile: Q4Profile) -> dict[str, object]:
    """Return explicit, auditable reliability/speed settings for deployment."""

    if profile == "conservative":
        return {
            "stop_probability": 0.998,
            "centroid_clear_max_radius_m": 75.0,
            "use_enclosing_circle_target": True,
        }
    if profile == "balanced":
        return {
            "stop_probability": 0.97,
            "centroid_clear_max_radius_m": 75.0,
            "use_enclosing_circle_target": True,
        }
    if profile == "aggressive":
        return {
            "stop_probability": 0.95,
            "centroid_clear_max_radius_m": 75.0,
            "use_enclosing_circle_target": True,
        }
    if profile == "source-reliable":
        return {
            "source_miss_risk_budget": 0.03,
            "any_source_remaining_probability_budget": 0.05,
            "centroid_clear_max_radius_m": 75.0,
            "use_enclosing_circle_target": True,
        }
    if profile == "source-efficient":
        return {
            "source_miss_risk_budget": 0.04,
            "any_source_remaining_probability_budget": 0.10,
            "centroid_clear_max_radius_m": 75.0,
            "use_enclosing_circle_target": True,
        }
    if profile == "source-98":
        return {
            "source_miss_risk_budget": 0.04,
            "any_source_remaining_probability_budget": 0.25,
            "centroid_clear_max_radius_m": 75.0,
            "use_enclosing_circle_target": True,
        }
    raise ValueError(f"unknown Q4 profile: {profile!r}")


@dataclass(frozen=True)
class Q4DecisionContext:
    """Observable high-level choice exposed only for policy evaluation.

    A hook may replace one discovery point or the first channel in a pursuit
    route.  Hidden source state is deliberately absent; offline evaluators can
    retain it separately to construct privileged training targets.
    """

    decision_index: int
    kind: Q4DecisionKind
    position: Point
    virtual_time_s: float
    detected_channels: tuple[int, ...]
    cleared_channels: tuple[int, ...]
    candidate_points: tuple[Point, ...] = ()
    candidate_channels: tuple[int, ...] = ()
    default_point: Point | None = None
    default_channel: int | None = None


def directional_discovery_lattice(
    *, spacing_m: float = 700.0, search_radius_m: float = 2800.0
) -> list[Point]:
    """Return a lattice covering every source/180-degree-sector pair.

    The farthest corner of a square cell is ``spacing*sqrt(2)`` away.  At
    700 m this is below the minimum 1000 m receive radius.  Every closed
    half-plane through a source contains at least one of the cell's four
    corners.  Extending the lattice to 2800 m also covers boundary sources.
    """

    if spacing_m * math.sqrt(2.0) > 1000.0:
        raise ValueError("spacing does not guarantee minimum-radius coverage")
    bound = math.ceil(search_radius_m / spacing_m)
    points: list[Point] = []
    for ix in range(-bound, bound + 1):
        for iy in range(-bound, bound + 1):
            point = (ix * spacing_m, iy * spacing_m)
            if math.hypot(*point) <= search_radius_m + 1e-9:
                points.append(point)
    if (0.0, 0.0) not in points:
        points.append((0.0, 0.0))
    return points


def directional_discovery_candidates(
    *,
    base_spacing_m: float = 700.0,
    base_search_radius_m: float = 2800.0,
    refinement_spacing_m: float = 600.0,
    refinement_radius_m: float = 2700.0,
    refinement_phase_divisions: int = 1,
    include_polar_refinement: bool = False,
) -> list[Point]:
    """Return a richer optimization pool containing the guaranteed lattice.

    The 700 m lattice remains a subset, so the exhaustive fallback keeps its
    original guarantee.  Additional 600 m grid points let the set-cover route
    form shorter combinations instead of being locked to one coarse phase.
    """

    if refinement_spacing_m <= 0.0 or refinement_radius_m <= 0.0:
        raise ValueError("refinement spacing and radius must be positive")
    if refinement_phase_divisions < 1:
        raise ValueError("refinement phase divisions must be positive")
    points = set(
        directional_discovery_lattice(
            spacing_m=base_spacing_m, search_radius_m=base_search_radius_m
        )
    )
    bound = math.ceil(refinement_radius_m / refinement_spacing_m) + 1
    phases = [
        refinement_spacing_m * index / refinement_phase_divisions
        for index in range(refinement_phase_divisions)
    ]
    for phase_x in phases:
        for phase_y in phases:
            for ix in range(-bound, bound + 1):
                for iy in range(-bound, bound + 1):
                    point = (
                        ix * refinement_spacing_m + phase_x,
                        iy * refinement_spacing_m + phase_y,
                    )
                    if math.hypot(*point) <= refinement_radius_m + 1e-9:
                        points.add(point)
    if include_polar_refinement:
        for ring_index, radius in enumerate((600.0, 1200.0, 1800.0, 2400.0)):
            count = math.ceil(2.0 * math.pi * radius / refinement_spacing_m)
            phase = (ring_index % 2) * math.pi / count
            for angle_index in range(count):
                angle = phase + 2.0 * math.pi * angle_index / count
                points.add((radius * math.cos(angle), radius * math.sin(angle)))
    return sorted(points)


class SparseDirectionalSearch:
    def __init__(
        self,
        *,
        spacing_m: float = 700.0,
        search_radius_m: float = 2800.0,
        initial_pursuit_step_m: float = 125.0,
        single_bearing_pursuit_step_m: float = 350.0,
        cross_bearing_fraction: float = 0.0,
        clear_radius_m: float = 19.0,
        maximum_pursuit_measurements: int = 64,
        stop_probability: float = 0.998,
        source_miss_risk_budget: float | None = None,
        any_source_remaining_probability_budget: float | None = None,
        belief_particle_count: int = 8192,
        belief_directional_probability: float = 0.5,
        belief_range_margin_m: float = 10.0,
        belief_angular_margin_deg: float = 12.0,
        stopping_range_margin_m: float | None = None,
        stopping_angular_margin_deg: float | None = None,
        analytic_validation_position_count: int = 0,
        shared_bearing_target: int = 3,
        centroid_clear_after_bearings: int = 2,
        centroid_clear_max_radius_m: float = math.inf,
        pursuit_deferral_positions: int = 0,
        route_aware_discovery_selection: bool = True,
        terminal_route_commitment_steps: int = 1,
        discovery_travel_weight: float = 1.0,
        scan_after_clear: bool = True,
        opportunistic_scan_rate_ratio: float = 0.5,
        target_posterior_particle_count: int = 0,
        use_enclosing_circle_target: bool = False,
        refinement_phase_divisions: int = 1,
        include_polar_refinement: bool = False,
        discovery_points: Sequence[Point] | None = None,
        decision_override: (
            Callable[[Q4DecisionContext], Q4DecisionChoice] | None
        ) = None,
        target_estimate_override: Callable[[int], Point | None] | None = None,
    ):
        self.discovery_points = (
            list(discovery_points)
            if discovery_points is not None
            else directional_discovery_candidates(
                base_spacing_m=spacing_m,
                base_search_radius_m=search_radius_m,
                refinement_phase_divisions=refinement_phase_divisions,
                include_polar_refinement=include_polar_refinement,
            )
        )
        if not self.discovery_points:
            raise ValueError("at least one discovery point is required")
        self.initial_pursuit_step_m = initial_pursuit_step_m
        if single_bearing_pursuit_step_m <= 0.0:
            raise ValueError("single_bearing_pursuit_step_m must be positive")
        self.single_bearing_pursuit_step_m = single_bearing_pursuit_step_m
        if not 0.0 <= cross_bearing_fraction <= 0.8:
            raise ValueError("cross_bearing_fraction must lie in [0, 0.8]")
        self.cross_bearing_fraction = cross_bearing_fraction
        self.clear_radius_m = clear_radius_m
        self.maximum_pursuit_measurements = maximum_pursuit_measurements
        if not 0.0 < stop_probability <= 1.0:
            raise ValueError("stop_probability must lie in (0, 1]")
        if source_miss_risk_budget is not None and not (
            0.0 <= source_miss_risk_budget < 1.0
        ):
            raise ValueError("source miss risk budget must lie in [0, 1)")
        if any_source_remaining_probability_budget is not None and not (
            0.0 <= any_source_remaining_probability_budget < 1.0
        ):
            raise ValueError(
                "any-source remaining probability budget must lie in [0, 1)"
            )
        if belief_particle_count < 128:
            raise ValueError("belief_particle_count must be at least 128")
        if not 0.0 <= belief_directional_probability <= 1.0:
            raise ValueError("belief_directional_probability must lie in [0, 1]")
        self.stop_probability = stop_probability
        self.source_miss_risk_budget = source_miss_risk_budget
        self.any_source_remaining_probability_budget = (
            any_source_remaining_probability_budget
        )
        self.belief_directional_probability = belief_directional_probability
        if not 0.0 <= belief_range_margin_m < 1000.0:
            raise ValueError("belief_range_margin_m must lie in [0, 1000)")
        if not 0.0 <= belief_angular_margin_deg < 90.0:
            raise ValueError("belief_angular_margin_deg must lie in [0, 90)")
        self.belief_range_margin_m = belief_range_margin_m
        self.belief_angular_margin_deg = belief_angular_margin_deg
        if stopping_range_margin_m is None:
            stopping_range_margin_m = belief_range_margin_m
        if stopping_angular_margin_deg is None:
            stopping_angular_margin_deg = belief_angular_margin_deg
        if not 0.0 <= stopping_range_margin_m < 1000.0:
            raise ValueError("stopping range margin must lie in [0, 1000)")
        if not 0.0 <= stopping_angular_margin_deg < 90.0:
            raise ValueError("stopping angular margin must lie in [0, 90)")
        self.stopping_range_margin_m = stopping_range_margin_m
        self.stopping_angular_margin_deg = stopping_angular_margin_deg
        if analytic_validation_position_count != 0 and analytic_validation_position_count < 512:
            raise ValueError("analytic validation needs either zero or at least 512 positions")
        if pursuit_deferral_positions < 0:
            raise ValueError("pursuit deferral positions must be non-negative")
        self.pursuit_deferral_positions = pursuit_deferral_positions
        self.route_aware_discovery_selection = route_aware_discovery_selection
        if terminal_route_commitment_steps < 1:
            raise ValueError("terminal route commitment must be at least one step")
        self.terminal_route_commitment_steps = terminal_route_commitment_steps
        if discovery_travel_weight <= 0.0:
            raise ValueError("discovery travel weight must be positive")
        self.discovery_travel_weight = discovery_travel_weight
        self.scan_after_clear = scan_after_clear
        if opportunistic_scan_rate_ratio < 0.0:
            raise ValueError("opportunistic scan rate ratio must be non-negative")
        self.opportunistic_scan_rate_ratio = opportunistic_scan_rate_ratio
        if (
            target_posterior_particle_count != 0
            and target_posterior_particle_count < 8192
        ):
            raise ValueError(
                "target posterior particles need either zero or at least 8192 states"
            )
        self.target_posterior_particle_count = target_posterior_particle_count
        self.use_enclosing_circle_target = use_enclosing_circle_target
        self._target_particle_bank = (
            self._make_target_particle_bank(
                target_posterior_particle_count,
                belief_directional_probability,
                500_009,
            )
            if target_posterior_particle_count
            else None
        )
        if shared_bearing_target < 1:
            raise ValueError("shared_bearing_target must be positive")
        self.shared_bearing_target = shared_bearing_target
        if centroid_clear_after_bearings < 0:
            raise ValueError("centroid_clear_after_bearings must be non-negative")
        self.centroid_clear_after_bearings = centroid_clear_after_bearings
        if centroid_clear_max_radius_m <= 0.0:
            raise ValueError("centroid clear radius bound must be positive")
        self.centroid_clear_max_radius_m = centroid_clear_max_radius_m
        self.decision_override = decision_override
        self.target_estimate_override = target_estimate_override
        self._belief_particles = self._make_belief_particles(
            belief_particle_count, belief_directional_probability, 1
        )
        self._validation_particles = self._make_belief_particles(
            belief_particle_count, belief_directional_probability, 100_003
        )
        self._analytic_validation_states = (
            self._make_spatial_validation_states(
                analytic_validation_position_count, 300_007
            )
            if analytic_validation_position_count
            else ()
        )

    @staticmethod
    def _distance(first: Point, second: Point) -> float:
        return math.hypot(first[0] - second[0], first[1] - second[1])

    @staticmethod
    def _unit(angle_deg: float) -> Point:
        angle = math.radians(angle_deg)
        return math.cos(angle), math.sin(angle)

    @staticmethod
    def _radical_inverse(index: int, base: int) -> float:
        inverse = 0.0
        factor = 1.0 / base
        while index:
            index, digit = divmod(index, base)
            inverse += digit * factor
            factor /= base
        return inverse

    @classmethod
    @lru_cache(maxsize=8)
    def _make_belief_particles(
        cls, count: int, directional_probability: float, start_index: int
    ) -> tuple[tuple[float, float, float, bool, float], ...]:
        """Deterministic low-discrepancy prior samples for an unseen source."""

        particles = []
        for index in range(start_index, start_index + count):
            radius = 1800.0 * math.sqrt(cls._radical_inverse(index, 2))
            angle = 2.0 * math.pi * cls._radical_inverse(index, 3)
            particles.append(
                (
                    radius * math.cos(angle),
                    radius * math.sin(angle),
                    1000.0 + 500.0 * cls._radical_inverse(index, 5),
                    cls._radical_inverse(index, 11) < directional_probability,
                    360.0 * cls._radical_inverse(index, 7),
                )
            )
        return tuple(particles)

    @classmethod
    @lru_cache(maxsize=8)
    def _make_spatial_validation_states(
        cls, count: int, start_index: int
    ) -> tuple[tuple[float, float, float], ...]:
        """Low-discrepancy positions/radii for analytic heading integration."""

        states = []
        for index in range(start_index, start_index + count):
            radius = 1800.0 * math.sqrt(cls._radical_inverse(index, 2))
            angle = 2.0 * math.pi * cls._radical_inverse(index, 3)
            receive_radius = 1000.0 + 500.0 * cls._radical_inverse(index, 5)
            states.append(
                (radius * math.cos(angle), radius * math.sin(angle), receive_radius)
            )
        return tuple(states)

    @classmethod
    @lru_cache(maxsize=4)
    def _make_target_particle_bank(
        cls, count: int, directional_probability: float, start_index: int
    ) -> np.ndarray:
        """Dense latent-state bank used only after a channel is detected."""

        rows = np.empty((count, 5), dtype=np.float64)
        for row, index in enumerate(range(start_index, start_index + count)):
            radius = 1800.0 * math.sqrt(cls._radical_inverse(index, 2))
            angle = 2.0 * math.pi * cls._radical_inverse(index, 3)
            rows[row] = (
                radius * math.cos(angle),
                radius * math.sin(angle),
                1000.0 + 500.0 * cls._radical_inverse(index, 5),
                float(cls._radical_inverse(index, 11) < directional_probability),
                360.0 * cls._radical_inverse(index, 7),
            )
        rows.flags.writeable = False
        return rows

    @staticmethod
    def _heading_miss_fraction(
        observer_angles_deg: Sequence[float], half_width_deg: float
    ) -> float:
        """Exactly integrate missed source headings on the angle circle."""

        if not observer_angles_deg or half_width_deg <= 0.0:
            return 1.0
        if half_width_deg >= 180.0:
            return 0.0
        intervals: list[tuple[float, float]] = []
        for angle in observer_angles_deg:
            lower = (angle - half_width_deg) % 360.0
            upper = (angle + half_width_deg) % 360.0
            if lower <= upper:
                intervals.append((lower, upper))
            else:
                intervals.append((0.0, upper))
                intervals.append((lower, 360.0))
        intervals.sort()
        covered = 0.0
        start, end = intervals[0]
        for next_start, next_end in intervals[1:]:
            if next_start <= end:
                end = max(end, next_end)
            else:
                covered += end - start
                start, end = next_start, next_end
        covered += end - start
        return max(0.0, 1.0 - covered / 360.0)

    def _analytic_miss_probability(
        self,
        scan_positions: Sequence[Point],
        *,
        states: Sequence[tuple[float, float, float]] | None = None,
    ) -> float:
        """Integrate missed headings over the position/radius distribution."""

        if not scan_positions:
            return 1.0
        validation_states = (
            self._analytic_validation_states if states is None else states
        )
        half_width = 90.0 - self.stopping_angular_margin_deg
        miss_sum = 0.0
        for source_x, source_y, receive_radius in validation_states:
            planning_radius = max(
                0.0, receive_radius - self.stopping_range_margin_m
            )
            radius_sq = planning_radius * planning_radius
            observer_angles = []
            for observer_x, observer_y in scan_positions:
                dx = observer_x - source_x
                dy = observer_y - source_y
                if dx * dx + dy * dy <= radius_sq:
                    observer_angles.append(math.degrees(math.atan2(dy, dx)))
            omnidirectional_miss = float(not observer_angles)
            directional_miss = self._heading_miss_fraction(
                observer_angles, half_width
            )
            miss_sum += (
                (1.0 - self.belief_directional_probability)
                * omnidirectional_miss
                + self.belief_directional_probability * directional_miss
            )
        return miss_sum / len(validation_states)

    def _single_source_miss_probability(
        self, scan_positions: Sequence[Point]
    ) -> float:
        if not scan_positions:
            return 1.0
        covered = 0
        for point in scan_positions:
            covered |= self._visible_particle_mask(point)
        return 1.0 - covered.bit_count() / len(self._belief_particles)

    def _visible_particle_mask(self, point: Point) -> int:
        """Encode prior source states visible from a point as a Python bitset."""

        return self._visible_particle_mask_for(
            point,
            self._belief_particles,
            range_margin_m=self.belief_range_margin_m,
            angular_margin_deg=self.belief_angular_margin_deg,
        )

    @staticmethod
    def _visible_particle_mask_for(
        point: Point,
        particles: Sequence[tuple[float, float, float, bool, float]],
        *,
        range_margin_m: float = 0.0,
        angular_margin_deg: float = 0.0,
    ) -> int:
        """Encode the supplied latent source states visible from a point."""

        observer_x, observer_y = point
        mask = 0
        for index, (
            source_x,
            source_y,
            receive_radius,
            directional,
            heading,
        ) in enumerate(particles):
            dx = observer_x - source_x
            dy = observer_y - source_y
            planning_radius = max(0.0, receive_radius - range_margin_m)
            if dx * dx + dy * dy > planning_radius * planning_radius:
                continue
            if directional:
                observer_angle = math.degrees(math.atan2(dy, dx)) % 360.0
                difference = (observer_angle - heading + 180.0) % 360.0 - 180.0
                if abs(difference) > 90.0 - angular_margin_deg:
                    continue
            mask |= 1 << index
        return mask

    @staticmethod
    def _source_count_posterior_summary(
        detected_count: int, miss_probability: float
    ) -> tuple[float, float, float]:
        """Return P(no source remains), E[remaining], and expected miss share.

        The source count prior is uniform on 10..16 and occupied channels are
        uniform without replacement among the 20 channels. ``miss_probability``
        is the probability that one still-unseen source escaped the accumulated
        no-signal observations.  The final quantity targets source-level loss,
        unlike the first quantity which targets the stricter whole-case loss.
        """

        miss_probability = min(1.0, max(0.0, miss_probability))
        if detected_count >= 16:
            return 1.0, 0.0, 0.0
        weights = []
        for source_count in range(max(10, detected_count), 17):
            remaining = source_count - detected_count
            occupancy = (
                math.comb(20 - detected_count, remaining)
                / math.comb(20, source_count)
            )
            weights.append((source_count, occupancy * miss_probability**remaining))
        denominator = sum(weight for _, weight in weights)
        if denominator <= 0.0:
            # With fewer than ten detections and exactly zero modeled miss
            # probability the observation has zero likelihood under every
            # allowed source count.  Use the q->0 limiting state (N=10)
            # instead of incorrectly declaring completion.
            remaining = max(0, 10 - detected_count)
            return 0.0, float(remaining), remaining / 10.0
        complete_weight = next(
            (weight for source_count, weight in weights if source_count == detected_count),
            0.0,
        )
        expected_remaining = sum(
            (source_count - detected_count) * weight
            for source_count, weight in weights
        ) / denominator
        expected_total = detected_count + expected_remaining
        expected_missed_fraction = (
            expected_remaining / expected_total if expected_total else 0.0
        )
        return (
            complete_weight / denominator,
            expected_remaining,
            expected_missed_fraction,
        )

    @classmethod
    def _completion_probability_from_miss(
        cls, detected_count: int, miss_probability: float
    ) -> float:
        return cls._source_count_posterior_summary(
            detected_count, miss_probability
        )[0]

    def _stopping_objective_satisfied(
        self, detected_count: int, miss_probability: float
    ) -> bool:
        complete, _, missed_fraction = self._source_count_posterior_summary(
            detected_count, miss_probability
        )
        if self.source_miss_risk_budget is not None:
            if missed_fraction > self.source_miss_risk_budget:
                return False
            if self.any_source_remaining_probability_budget is not None:
                return (
                    1.0 - complete
                    <= self.any_source_remaining_probability_budget
                )
            return True
        return complete >= self.stop_probability

    def _posterior_all_detected(
        self, detected_count: int, scan_positions: Sequence[Point]
    ) -> float:
        """Probability that no present source remains on an unseen channel.

        The source-count prior is uniform on 10..16 and occupied channels are
        sampled without replacement.  For an observed set of ``detected_count``
        channels, each candidate count is weighted by the probability that its
        remaining occupied channels all generated the accumulated no-signal
        history.  A deliberately conservative all-directional belief is the
        mixture prior is used because Q4 contains both directional and
        omnidirectional sources.
        """

        miss_probability = self._single_source_miss_probability(scan_positions)
        return self._completion_probability_from_miss(
            detected_count, miss_probability
        )

    @staticmethod
    def _polygon_centroid(polygon: Sequence[Point]) -> Point:
        """Return an area centroid, falling back to the vertex mean."""

        if not polygon:
            raise ValueError("centroid is undefined for an empty polygon")
        area_twice = 0.0
        x_sum = 0.0
        y_sum = 0.0
        for index, first in enumerate(polygon):
            second = polygon[(index + 1) % len(polygon)]
            cross = first[0] * second[1] - second[0] * first[1]
            area_twice += cross
            x_sum += (first[0] + second[0]) * cross
            y_sum += (first[1] + second[1]) * cross
        if abs(area_twice) <= 1e-12:
            return (
                sum(point[0] for point in polygon) / len(polygon),
                sum(point[1] for point in polygon) / len(polygon),
            )
        return x_sum / (3.0 * area_twice), y_sum / (3.0 * area_twice)

    def _estimated_target(self, track: ChannelTrack) -> Point:
        """Posterior mean proxy used only to organize travel between tracks.

        With one bearing, source area in a thin angular wedge is proportional
        to range.  It is additionally weighted by the probability that a
        uniformly distributed receive radius in [1000, 1500] reaches that
        range.  Multiple bearings use the centroid of their feasible polygon.
        """

        if len(track.measurements) >= 2 and track.polygon:
            return self._polygon_centroid(track.polygon)
        point, bearing_deg = track.measurements[-1]
        ux, uy = self._unit(bearing_deg)
        projection = point[0] * ux + point[1] * uy
        radial_slack = max(
            0.0,
            projection * projection + 1800.0**2 - point[0] ** 2 - point[1] ** 2,
        )
        range_limit = min(1500.0, -projection + math.sqrt(radial_slack))
        if range_limit <= 5.0:
            expected_range = max(0.0, range_limit)
        else:
            # A compact deterministic quadrature is more transparent here
            # than assuming an arbitrary fixed pursuit distance.
            weighted_range = 0.0
            weight_sum = 0.0
            bins = 48
            for index in range(bins):
                distance = 5.0 + (range_limit - 5.0) * (index + 0.5) / bins
                receive_probability = (
                    1.0
                    if distance <= 1000.0
                    else max(0.0, (1500.0 - distance) / 500.0)
                )
                weight = distance * receive_probability
                weight_sum += weight
                weighted_range += weight * distance
            expected_range = weighted_range / weight_sum
        return point[0] + expected_range * ux, point[1] + expected_range * uy

    def _posterior_particle_target(self, track: ChannelTrack) -> Point | None:
        """Condition source position on positive and negative observations."""

        bank = self._target_particle_bank
        if bank is None or not track.measurements:
            return None
        source_x = bank[:, 0]
        source_y = bank[:, 1]
        receive_radius = bank[:, 2]
        directional = bank[:, 3] > 0.5
        heading = bank[:, 4]
        feasible = np.ones(len(bank), dtype=np.bool_)

        def visible(observer: Point) -> np.ndarray:
            observer_dx = observer[0] - source_x
            observer_dy = observer[1] - source_y
            in_range = (
                observer_dx * observer_dx + observer_dy * observer_dy
                <= receive_radius * receive_radius
            )
            observer_angle = np.degrees(np.arctan2(observer_dy, observer_dx))
            heading_error = (observer_angle - heading + 180.0) % 360.0 - 180.0
            in_sector = (~directional) | (np.abs(heading_error) <= 90.0)
            return in_range & in_sector

        for observer, bearing in track.measurements:
            source_angle = np.degrees(
                np.arctan2(source_y - observer[1], source_x - observer[0])
            )
            bearing_error = (source_angle - bearing + 180.0) % 360.0 - 180.0
            feasible &= visible(observer) & (np.abs(bearing_error) <= 1.1)
        for observer in track.no_signal_positions:
            feasible &= ~visible(observer)
        count = int(np.count_nonzero(feasible))
        if count < 16:
            return None
        return (
            float(np.mean(source_x[feasible])),
            float(np.mean(source_y[feasible])),
        )

    def _estimated_visibility_probability(
        self, track: ChannelTrack, observer: Point, target: Point
    ) -> float:
        """Approximate P(signal at observer | previous signal bearings)."""

        if any(self._distance(observer, point) <= 1e-8 for point, _ in track.measurements):
            return 1.0
        observer_range = self._distance(observer, target)
        if observer_range > 1500.0:
            return 0.0
        previous_ranges = [
            self._distance(point, target) for point, _ in track.measurements
        ]
        minimum_radius = max(1000.0, *previous_ranges)
        if minimum_radius >= 1500.0:
            radial_probability = float(observer_range <= 1500.0)
        elif observer_range <= minimum_radius:
            radial_probability = 1.0
        else:
            radial_probability = max(
                0.0, (1500.0 - observer_range) / (1500.0 - minimum_radius)
            )

        def direction_visible(point: Point, heading: float) -> bool:
            angle = math.degrees(
                math.atan2(point[1] - target[1], point[0] - target[0])
            )
            difference = (angle - heading + 180.0) % 360.0 - 180.0
            return abs(difference) <= 90.0

        feasible_headings = [
            heading
            for heading_index in range(180)
            if all(
                direction_visible(point, heading := heading_index * 2.0 + 1.0)
                for point, _ in track.measurements
            )
        ]
        directional_likelihood = len(feasible_headings) / 180.0
        prior_directional = self.belief_directional_probability
        evidence = (1.0 - prior_directional) + prior_directional * directional_likelihood
        if evidence <= 1e-12:
            directional_posterior = 0.0
        else:
            directional_posterior = (
                prior_directional * directional_likelihood / evidence
            )
        if feasible_headings:
            conditional_directional_visibility = sum(
                direction_visible(observer, heading) for heading in feasible_headings
            ) / len(feasible_headings)
        else:
            conditional_directional_visibility = 0.0
        angular_probability = (
            1.0
            - directional_posterior
            + directional_posterior * conditional_directional_visibility
        )
        return radial_probability * angular_probability

    def _expected_service_distance(
        self, origin: Point, track: ChannelTrack, target: Point
    ) -> float:
        """Expected movement to service a track from an arbitrary route node."""

        visible_probability = self._estimated_visibility_probability(
            track, origin, target
        )
        signal_point, bearing = track.measurements[-1]
        ux, uy = self._unit(bearing)
        fallback_step = (
            self.single_bearing_pursuit_step_m
            if len(track.measurements) == 1
            else self.initial_pursuit_step_m
        )
        fallback = (
            signal_point[0] + fallback_step * ux,
            signal_point[1] + fallback_step * uy,
        )
        direct = self._distance(origin, target)
        reset = self._distance(origin, fallback) + self._distance(fallback, target)
        return visible_probability * direct + (1.0 - visible_probability) * reset

    def run(self, environment) -> Q4SearchResult:
        tracks = {channel: ChannelTrack(channel) for channel in range(1, 21)}
        measure_actions = 0
        discovery_measure_actions = 0
        undetected_discovery_measure_actions = 0
        detected_discovery_measure_actions = 0
        pursuit_measure_actions = 0
        clear_actions = 0
        travel_distance_m = 0.0
        discovery_travel_distance_m = 0.0
        pursuit_travel_distance_m = 0.0
        discovery_positions = 0
        unvisited = set(self.discovery_points)
        scan_positions: list[Point] = []
        stopped_by_probability = False
        belief_covered_mask = 0
        belief_full_mask = (1 << len(self._belief_particles)) - 1
        validation_covered_mask = 0
        validation_full_mask = (1 << len(self._validation_particles)) - 1
        discovery_masks: dict[Point, int] = {}
        validation_masks: dict[Point, int] = {}
        failed_home_measurement_count: dict[int, int] = {}
        positions_since_pursuit = 0
        target_estimate_cache: dict[int, tuple[tuple[int, int], Point]] = {}
        committed_discovery_route: list[Point] = []
        decision_index = 0

        def decision_context(
            *,
            kind: Q4DecisionKind,
            candidate_points: tuple[Point, ...] = (),
            candidate_channels: tuple[int, ...] = (),
            default_point: Point | None = None,
            default_channel: int | None = None,
        ) -> Q4DecisionContext:
            return Q4DecisionContext(
                decision_index=decision_index,
                kind=kind,
                position=environment.position,
                virtual_time_s=environment.virtual_time_s,
                detected_channels=tuple(
                    track.channel for track in tracks.values() if track.detected
                ),
                cleared_channels=tuple(sorted(environment.cleared_channels)),
                candidate_points=candidate_points,
                candidate_channels=candidate_channels,
                default_point=default_point,
                default_channel=default_channel,
            )

        def choose_discovery_point(default: Point) -> Point:
            nonlocal decision_index
            if self.decision_override is None:
                return default
            candidates = tuple(sorted(unvisited))
            context = decision_context(
                kind="discovery",
                candidate_points=candidates,
                default_point=default,
            )
            choice = self.decision_override(context)
            decision_index += 1
            if choice is None:
                return default
            if not isinstance(choice, tuple) or choice not in unvisited:
                raise ValueError("discovery override must be an unvisited candidate point")
            return choice

        def choose_pursuit_order(
            pending: Sequence[ChannelTrack], order: Sequence[int]
        ) -> list[int]:
            nonlocal decision_index
            ordered = list(order)
            if self.decision_override is None or not ordered:
                return ordered
            channels = tuple(track.channel for track in pending)
            default_channel = pending[ordered[0]].channel
            context = decision_context(
                kind="pursuit",
                candidate_channels=channels,
                default_channel=default_channel,
            )
            choice = self.decision_override(context)
            decision_index += 1
            if choice is None:
                return ordered
            if not isinstance(choice, int) or choice not in channels:
                raise ValueError("pursuit override must be a pending channel")
            selected = channels.index(choice)
            return [selected, *(index for index in ordered if index != selected)]

        def estimated_target(track: ChannelTrack) -> Point:
            if self.target_estimate_override is not None:
                override = self.target_estimate_override(track.channel)
                if override is not None:
                    return override
            state = (len(track.measurements), len(track.no_signal_positions))
            cached = target_estimate_cache.get(track.channel)
            if cached is not None and cached[0] == state:
                return cached[1]
            estimate = self._posterior_particle_target(track)
            if estimate is None:
                estimate = self._estimated_target(track)
            target_estimate_cache[track.channel] = (state, estimate)
            return estimate

        def route_target(track: ChannelTrack) -> Point:
            circle = track.enclosing_circle
            if (
                self.use_enclosing_circle_target
                and len(track.measurements) >= 2
                and circle is not None
            ):
                return circle.center
            return estimated_target(track)

        def measure(track: ChannelTrack, point: Point, *, discovery: bool = False):
            nonlocal measure_actions, travel_distance_m
            nonlocal discovery_measure_actions, pursuit_measure_actions
            nonlocal undetected_discovery_measure_actions
            nonlocal detected_discovery_measure_actions
            nonlocal discovery_travel_distance_m, pursuit_travel_distance_m
            movement = self._distance(environment.position, point)
            travel_distance_m += movement
            if discovery:
                discovery_travel_distance_m += movement
            else:
                pursuit_travel_distance_m += movement
            observation = environment.measure(point, track.channel)
            measure_actions += 1
            if discovery:
                discovery_measure_actions += 1
                if track.detected:
                    detected_discovery_measure_actions += 1
                else:
                    undetected_discovery_measure_actions += 1
            else:
                pursuit_measure_actions += 1
            track.update(observation)
            return observation

        def clear(track: ChannelTrack, point: Point) -> bool:
            nonlocal clear_actions, travel_distance_m, pursuit_travel_distance_m
            movement = self._distance(environment.position, point)
            travel_distance_m += movement
            pursuit_travel_distance_m += movement
            observation = environment.clear(point, track.channel)
            clear_actions += 1
            if observation.result == "success":
                track.cleared = True
                committed_discovery_route.clear()
                return True
            return False

        def clear_bracketed_ray(
            track: ChannelTrack,
            start: Point,
            bearing_deg: float,
            length_m: float,
        ) -> bool:
            """Cover a just-traversed bearing segment with 20 m clear disks.

            A directional source can sit close to a sector boundary.  In that
            case a forward step may leave the receive half-plane even though
            the source lies on the measured ray between ``start`` and the
            no-signal endpoint.  Clearing is independent of source heading, so
            36 m spacing covers the segment despite up to one degree of bearing
            error (18 m longitudinal plus <7 m lateral error at 400 m).
            """

            ux, uy = self._unit(bearing_deg)
            if length_m <= 36.0:
                offsets = [length_m / 2.0]
            else:
                offsets = []
                offset = length_m - 18.0
                while offset >= 18.0:
                    offsets.append(offset)
                    offset -= 36.0
                if offsets[-1] > 22.0:
                    offsets.append(18.0)
            for offset in offsets:
                candidate = (
                    start[0] + offset * ux,
                    start[1] + offset * uy,
                )
                if clear(track, candidate):
                    return True
            return False

        def home(track: ChannelTrack) -> bool:
            step = (
                self.single_bearing_pursuit_step_m
                if len(track.measurements) == 1
                else self.initial_pursuit_step_m
            )
            last_signal_point: Point | None = None
            last_bearing: float | None = None
            enclosing_circle = track.enclosing_circle
            if (
                self.centroid_clear_after_bearings
                and len(track.measurements) >= self.centroid_clear_after_bearings
                and track.polygon
                and enclosing_circle is not None
                and enclosing_circle.radius <= self.centroid_clear_max_radius_m
            ):
                posterior_center = estimated_target(track)
                if clear(track, posterior_center):
                    return True
                center_observation = measure(track, posterior_center)
                if center_observation.result == "near":
                    return clear(track, posterior_center)
            if len(track.measurements) == 1 and self.cross_bearing_fraction > 0.0:
                origin, bearing = track.measurements[-1]
                ux, uy = self._unit(bearing)
                estimate = estimated_target(track)
                expected_range = self._distance(origin, estimate)
                forward = min(650.0, 0.55 * expected_range)
                lateral = min(380.0, self.cross_bearing_fraction * expected_range)
                cross_candidates = [
                    (
                        origin[0] + forward * ux - sign * lateral * uy,
                        origin[1] + forward * uy + sign * lateral * ux,
                    )
                    for sign in (-1.0, 1.0)
                ]
                cross_candidates.sort(
                    key=lambda point: self._distance(environment.position, point)
                )
                for cross_candidate in cross_candidates:
                    cross_observation = measure(track, cross_candidate)
                    if cross_observation.result == "near":
                        return clear(track, cross_candidate)
                    if cross_observation.result != "direction":
                        continue
                    # The lateral baseline creates a well-conditioned
                    # intersection.  Only now is the polygon center a useful
                    # direct clear hypothesis.
                    if track.polygon:
                        posterior_center = estimated_target(track)
                        if clear(track, posterior_center):
                            return True
                        center_observation = measure(track, posterior_center)
                        if center_observation.result == "near":
                            return clear(track, posterior_center)
                    break
            for _ in range(self.maximum_pursuit_measurements):
                circle = track.enclosing_circle
                if circle is not None and circle.radius <= self.clear_radius_m:
                    if clear(track, circle.center):
                        return True

                if track.measurements:
                    last_signal_point, last_bearing = track.measurements[-1]
                if last_signal_point is None or last_bearing is None:
                    return False

                ux, uy = self._unit(last_bearing)
                candidate = (
                    last_signal_point[0] + step * ux,
                    last_signal_point[1] + step * uy,
                )
                before = last_signal_point
                observation = measure(track, candidate)
                if observation.result == "near":
                    return clear(track, candidate)
                if observation.result == "no_signal":
                    # A source may be observed exactly along the boundary of its
                    # 180-degree sector.  Pure forward pursuit can then converge
                    # to that boundary rather than to the source.  Symmetric
                    # angled probes identify the interior side without knowing
                    # the hidden transmitter heading.
                    # First exploit the information in the failed forward
                    # step: if the source was crossed (or the sector boundary
                    # was crossed very near it), one of these heading-agnostic
                    # clear attempts succeeds without requiring another signal.
                    if clear_bracketed_ray(track, before, last_bearing, step):
                        return True

                    recovered = False
                    probe_step = max(50.0, min(200.0, step))
                    for offset_deg in (25.0, -25.0, 55.0, -55.0):
                        px, py = self._unit(last_bearing + offset_deg)
                        probe = (
                            before[0] + probe_step * px,
                            before[1] + probe_step * py,
                        )
                        probe_observation = measure(track, probe)
                        if probe_observation.result == "near":
                            return clear(track, probe)
                        if probe_observation.result == "direction":
                            step = max(50.0, step * 0.75)
                            recovered = True
                            break
                    if recovered:
                        continue
                    step = max(12.5, step * 0.5)
                    continue

                assert observation.bearing_deg is not None
                if len(track.measurements) >= 2:
                    step = min(step, self.initial_pursuit_step_m)
                vx, vy = self._unit(observation.bearing_deg)
                # If the new bearing points back against the last movement, the
                # source has been bracketed; halve subsequent pursuit distance.
                if ux * vx + uy * vy < -0.25:
                    step = max(8.0, step * 0.5)
            return False

        def scan_undetected(
            point: Point, *, opportunistic: bool = False
        ) -> list[ChannelTrack]:
            nonlocal discovery_positions, belief_covered_mask, validation_covered_mask
            if sum(track.detected for track in tracks.values()) >= 16:
                return []
            channels = [
                track
                for track in tracks.values()
                if not track.cleared
                and (
                    not track.detected
                    or failed_home_measurement_count.get(track.channel)
                    == len(track.measurements)
                    or len(track.measurements) < self.shared_bearing_target
                )
            ]
            if not channels:
                return []
            if opportunistic and self.opportunistic_scan_rate_ratio > 0.0:
                missed_mask = belief_full_mask ^ belief_covered_mask
                validation_missed_mask = (
                    validation_full_mask ^ validation_covered_mask
                )
                point_mask = self._visible_particle_mask(point)
                point_validation_mask = self._visible_particle_mask_for(
                    point,
                    self._validation_particles,
                    range_margin_m=self.stopping_range_margin_m,
                    angular_margin_deg=self.stopping_angular_margin_deg,
                )
                point_gain = (
                    (point_mask & missed_mask).bit_count()
                    + (point_validation_mask & validation_missed_mask).bit_count()
                )
                point_rate = point_gain / max(1.0, len(channels) * 6.0)
                best_alternative_rate = 0.0
                for candidate in unvisited:
                    candidate_mask = discovery_masks.get(candidate)
                    if candidate_mask is None:
                        candidate_mask = self._visible_particle_mask(candidate)
                        discovery_masks[candidate] = candidate_mask
                    candidate_validation_mask = validation_masks.get(candidate)
                    if candidate_validation_mask is None:
                        candidate_validation_mask = self._visible_particle_mask_for(
                            candidate,
                            self._validation_particles,
                            range_margin_m=self.stopping_range_margin_m,
                            angular_margin_deg=self.stopping_angular_margin_deg,
                        )
                        validation_masks[candidate] = candidate_validation_mask
                    candidate_gain = (
                        (candidate_mask & missed_mask).bit_count()
                        + (
                            candidate_validation_mask & validation_missed_mask
                        ).bit_count()
                    )
                    candidate_cost = (
                        self._distance(environment.position, candidate) / 5.0
                        + len(channels) * 6.0
                    )
                    best_alternative_rate = max(
                        best_alternative_rate,
                        candidate_gain / max(1.0, candidate_cost),
                    )
                if (
                    point_rate
                    < self.opportunistic_scan_rate_ratio * best_alternative_rate
                ):
                    return []
            channels.sort(
                key=lambda track: (
                    track.channel != environment.current_channel,
                    track.channel,
                )
            )
            discovery_positions += 1
            scan_positions.append(point)
            belief_covered_mask |= self._visible_particle_mask(point)
            validation_covered_mask |= self._visible_particle_mask_for(
                point,
                self._validation_particles,
                range_margin_m=self.stopping_range_margin_m,
                angular_margin_deg=self.stopping_angular_margin_deg,
            )
            newly_detected: list[ChannelTrack] = []
            for track in channels:
                observation = measure(track, point, discovery=True)
                if observation.result == "near":
                    clear(track, point)
                elif observation.result == "direction":
                    newly_detected.append(track)
                    committed_discovery_route.clear()
            return newly_detected

        def service_track(track: ChannelTrack) -> tuple[bool, bool]:
            """Attempt one known source and report (cleared, new discovery)."""

            if all(
                self._distance(environment.position, old) > 1e-8
                for old in track.attempted_measurement_positions
            ):
                handoff = measure(track, environment.position)
                if handoff.result == "near" and clear(track, environment.position):
                    newly_detected = (
                        self.scan_after_clear
                        and environment.cleared_count < 16
                        and bool(
                            scan_undetected(
                                environment.position,
                                opportunistic=True,
                            )
                        )
                    )
                    return True, newly_detected
            if not home(track):
                failed_home_measurement_count[track.channel] = len(
                    track.measurements
                )
                return False, False
            newly_detected = (
                self.scan_after_clear
                and environment.cleared_count < 16
                and bool(
                    scan_undetected(
                        environment.position,
                        opportunistic=True,
                    )
                )
            )
            return True, newly_detected

        def clear_detected_tracks() -> None:
            """Jointly route the currently detected sources without resets.

            The previous controller called ``home`` inside the channel scan.
            Every subsequent channel measurement therefore pulled the robot
            back to the old discovery point.  Here all co-located observations
            finish first, then an exact open route through posterior target
            proxies supplies the pursuit order.  New clear positions are
            reused for discovery and the route is replanned when new tracks
            appear.
            """

            attempted: set[int] = set()
            while True:
                pending = [
                    track
                    for track in tracks.values()
                    if track.detected
                    and not track.cleared
                    and track.channel not in attempted
                    and len(track.measurements)
                    > failed_home_measurement_count.get(track.channel, -1)
                ]
                if not pending:
                    return
                estimates = [route_target(track) for track in pending]
                start_costs = [
                    self._expected_service_distance(
                        environment.position, track, estimates[index]
                    )
                    for index, track in enumerate(pending)
                ]
                transition_costs = [
                    [
                        0.0
                        if first == second
                        else self._expected_service_distance(
                            estimates[first], pending[second], estimates[second]
                        )
                        for second in range(len(pending))
                    ]
                    for first in range(len(pending))
                ]
                route = exact_open_route_costs(start_costs, transition_costs)
                pursuit_order = choose_pursuit_order(pending, route.order)
                discovered_during_route = False
                for index in pursuit_order:
                    track = pending[index]
                    if track.cleared:
                        continue
                    cleared, discovered = service_track(track)
                    if not cleared:
                        attempted.add(track.channel)
                    if discovered:
                        discovered_during_route = True
                        break
                if not discovered_during_route:
                    return

        def plan_discovery_points(origin: Point) -> list[Point]:
            """Return an observable coverage batch ordered from ``origin``."""

            if not unvisited:
                return []
            missed_mask = belief_full_mask ^ belief_covered_mask
            validation_missed_mask = validation_full_mask ^ validation_covered_mask
            unknown_channels = sum(not track.detected for track in tracks.values())
            detected_count = sum(track.detected for track in tracks.values())

            # Once the minimum possible source count has been reached, first
            # select a compact set that would satisfy the stopping posterior,
            # then solve its open route exactly.  This prevents myopic
            # information-gain choices from alternating across the arena.
            if detected_count >= 10:
                projected_covered = belief_covered_mask
                projected_validation_covered = validation_covered_mask
                remaining_candidates = set(unvisited)
                selected: list[Point] = []
                while remaining_candidates and len(selected) < 16:
                    for candidate in remaining_candidates:
                        if candidate not in discovery_masks:
                            discovery_masks[candidate] = self._visible_particle_mask(
                                candidate
                            )
                        if candidate not in validation_masks:
                            validation_masks[candidate] = self._visible_particle_mask_for(
                                candidate,
                                self._validation_particles,
                                range_margin_m=self.stopping_range_margin_m,
                                angular_margin_deg=self.stopping_angular_margin_deg,
                            )
                    projected_missed = belief_full_mask ^ projected_covered
                    projected_validation_missed = (
                        validation_full_mask ^ projected_validation_covered
                    )
                    def selection_key(candidate: Point) -> tuple[float, ...]:
                        gain = (
                            discovery_masks[candidate] & projected_missed
                        ).bit_count()
                        validation_gain = (
                            validation_masks[candidate]
                            & projected_validation_missed
                        ).bit_count()
                        if not self.route_aware_discovery_selection:
                            return (
                                float(gain),
                                float(validation_gain),
                                -self._distance(origin, candidate),
                            )
                        attachment_points = [origin, *selected]
                        insertion_distance = min(
                            self._distance(anchor, candidate)
                            for anchor in attachment_points
                        )
                        incremental_time = (
                            self.discovery_travel_weight
                            * insertion_distance
                            / 5.0
                            + max(1, unknown_channels) * 6.0
                        )
                        combined_gain = gain + validation_gain
                        return (
                            combined_gain / max(1.0, incremental_time),
                            float(combined_gain),
                            -insertion_distance,
                        )

                    best = max(remaining_candidates, key=selection_key)
                    gain_mask = discovery_masks[best] & (
                        belief_full_mask ^ projected_covered
                    )
                    validation_gain_mask = validation_masks[best] & (
                        validation_full_mask ^ projected_validation_covered
                    )
                    if not gain_mask and not validation_gain_mask:
                        break
                    selected.append(best)
                    remaining_candidates.remove(best)
                    projected_covered |= discovery_masks[best]
                    projected_validation_covered |= validation_masks[best]
                    projected_miss = 1.0 - (
                        projected_validation_covered.bit_count()
                        / len(self._validation_particles)
                    )
                    if self._stopping_objective_satisfied(
                        detected_count, projected_miss
                    ):
                        break
                if selected:
                    route = (
                        exact_open_route(selected, start=origin)
                        if len(selected) <= 10
                        else fast_open_route(selected, start=origin)
                    )
                    return [selected[index] for index in route.order]

            # Before ten detections there is no meaningful completion
            # posterior.  Maximize newly covered latent states per full action
            # second to accelerate initial discovery.
            best_point: Point | None = None
            best_key = (-math.inf, -math.inf, -math.inf)
            for candidate in unvisited:
                visible_mask = discovery_masks.get(candidate)
                if visible_mask is None:
                    visible_mask = self._visible_particle_mask(candidate)
                    discovery_masks[candidate] = visible_mask
                gain = (visible_mask & missed_mask).bit_count()
                validation_visible_mask = validation_masks.get(candidate)
                if validation_visible_mask is None:
                    validation_visible_mask = self._visible_particle_mask_for(
                        candidate,
                        self._validation_particles,
                        range_margin_m=self.stopping_range_margin_m,
                        angular_margin_deg=self.stopping_angular_margin_deg,
                    )
                    validation_masks[candidate] = validation_visible_mask
                validation_gain = (
                    validation_visible_mask & validation_missed_mask
                ).bit_count()
                travel_s = self._distance(origin, candidate) / 5.0
                action_s = travel_s + max(1, unknown_channels) * 6.0
                active_gain = gain if missed_mask else validation_gain
                key = (
                    active_gain / max(1.0, action_s),
                    active_gain,
                    validation_gain,
                    -travel_s,
                )
                if key > best_key:
                    best_key = key
                    best_point = candidate
            assert best_point is not None
            return [best_point]

        def next_discovery_point() -> Point:
            """Plan a coverage batch, then take its exact shortest first leg."""

            while committed_discovery_route:
                point = committed_discovery_route.pop(0)
                if point in unvisited:
                    return point
            ordered = plan_discovery_points(environment.position)
            if not ordered:
                raise RuntimeError("no useful discovery point remains")
            commit_until = min(len(ordered), self.terminal_route_commitment_steps)
            committed_discovery_route.extend(ordered[1:commit_until])
            return choose_discovery_point(ordered[0])

        while unvisited:
            detected_count = sum(track.detected for track in tracks.values())
            particle_miss_probability = 1.0 - (
                validation_covered_mask.bit_count() / len(self._validation_particles)
            )
            if particle_miss_probability == 0.0:
                # Finite particles cannot prove a truly zero-probability miss.
                particle_miss_probability = 1.0 / (
                    len(self._validation_particles) + 1.0
                )
            miss_probability = particle_miss_probability
            if self._analytic_validation_states:
                analytic_miss_probability = self._analytic_miss_probability(
                    scan_positions
                )
                if analytic_miss_probability == 0.0:
                    analytic_miss_probability = 1.0 / (
                        len(self._analytic_validation_states) + 1.0
                    )
                miss_probability = max(
                    particle_miss_probability, analytic_miss_probability
                )
            (
                posterior_complete,
                posterior_expected_remaining,
                posterior_missed_fraction,
            ) = self._source_count_posterior_summary(
                detected_count, miss_probability
            )
            if (
                environment.cleared_count >= 10
                and not any(track.detected and not track.cleared for track in tracks.values())
                and self._stopping_objective_satisfied(
                    detected_count, miss_probability
                )
            ):
                stopped_by_probability = True
                break
            if all(track.cleared or track.detected for track in tracks.values()):
                break
            point = next_discovery_point()
            unvisited.remove(point)
            scan_undetected(point)
            pending_count = sum(
                track.detected and not track.cleared for track in tracks.values()
            )
            if pending_count:
                positions_since_pursuit += 1
            if (
                pending_count
                and positions_since_pursuit > self.pursuit_deferral_positions
            ):
                clear_detected_tracks()
                positions_since_pursuit = 0

        clear_detected_tracks()

        unresolved = tuple(
            track.channel for track in tracks.values() if track.detected and not track.cleared
        )
        particle_miss_probability = 1.0 - (
            validation_covered_mask.bit_count() / len(self._validation_particles)
        )
        if particle_miss_probability == 0.0:
            particle_miss_probability = 1.0 / (
                len(self._validation_particles) + 1.0
            )
        miss_probability = particle_miss_probability
        if self._analytic_validation_states:
            analytic_miss_probability = self._analytic_miss_probability(scan_positions)
            if analytic_miss_probability == 0.0:
                analytic_miss_probability = 1.0 / (
                    len(self._analytic_validation_states) + 1.0
                )
            miss_probability = max(
                particle_miss_probability, analytic_miss_probability
            )
        (
            posterior_complete,
            posterior_expected_remaining,
            posterior_missed_fraction,
        ) = self._source_count_posterior_summary(
            sum(track.detected for track in tracks.values()), miss_probability
        )
        return Q4SearchResult(
            cleared_count=environment.cleared_count,
            virtual_time_s=environment.virtual_time_s,
            measure_actions=measure_actions,
            discovery_measure_actions=discovery_measure_actions,
            undetected_discovery_measure_actions=undetected_discovery_measure_actions,
            detected_discovery_measure_actions=detected_discovery_measure_actions,
            pursuit_measure_actions=pursuit_measure_actions,
            clear_actions=clear_actions,
            travel_distance_m=travel_distance_m,
            discovery_travel_distance_m=discovery_travel_distance_m,
            pursuit_travel_distance_m=pursuit_travel_distance_m,
            discovery_positions=discovery_positions,
            discovery_scan_positions=tuple(scan_positions),
            exhausted_discovery_lattice=not unvisited,
            unresolved_channels=unresolved,
            posterior_all_sources_detected=posterior_complete,
            posterior_expected_remaining_sources=posterior_expected_remaining,
            posterior_expected_missed_source_fraction=posterior_missed_fraction,
            stopped_by_probability=stopped_by_probability,
            stopped_by_source_risk=(
                stopped_by_probability and self.source_miss_risk_budget is not None
            ),
        )
