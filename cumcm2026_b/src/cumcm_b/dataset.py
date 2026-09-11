"""Trajectory dataset construction for learning a B-problem action policy.

The policy inputs in this module are observable summaries only.  Hidden source
state is used by the teacher to score the *same* candidate set the policy sees;
it is never copied into an input feature.  This yields a practical imitation
learning dataset without teaching the network an impossible shortcut.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import random
from typing import Iterable, Literal, Sequence

import numpy as np

from .baseline import ChannelTrack, guaranteed_discovery_waypoints
from .geometry import Point, convex_diameter
from .q4_search import SparseDirectionalSearch, directional_discovery_lattice
from .simulator import (
    InterferenceEnvironment,
    MeasureObservation,
    Source,
    _source_visible,
    generate_sources,
)


ActionKind = Literal["measure", "clear"]


FEATURE_NAMES = (
    # Global observable state.
    "global_x", "global_y", "global_channel", "global_time",
    "global_cleared_fraction", "global_detected_fraction", "global_step",
    "is_q4", "directional_prior",
    # Candidate channel belief summary.
    "channel_id", "detected", "cleared", "measure_count", "direction_count",
    "no_signal_count", "near_count", "belief_center_x", "belief_center_y",
    "belief_radius", "belief_diameter", "last_measure_x", "last_measure_y",
    "last_bearing_sin", "last_bearing_cos", "last_no_signal",
    "last_direction", "last_near", "attempt_progress", "detection_age",
    # Candidate action.
    "action_measure", "action_clear", "target_x", "target_y",
    "travel_distance", "travel_time", "switch_time", "operation_time",
    "same_channel", "target_in_disk", "target_center_distance",
    # Cross-channel context.
    "remaining_detected_fraction", "uncertain_channel_fraction",
    "minimum_belief_radius", "mean_belief_radius", "time_per_clear",
    "candidate_rank_within_channel", "candidate_count_norm",
)

OBSERVATION_CODES = {
    "none": 0,
    "no_signal": 1,
    "direction": 2,
    "near": 3,
    "clear_success": 4,
    "clear_failure": 5,
}


@dataclass(frozen=True)
class CandidateAction:
    kind: ActionKind
    channel: int
    point: Point
    within_channel_index: int


@dataclass
class EpisodeArrays:
    features: np.ndarray
    decision_id: np.ndarray
    episode_id: np.ndarray
    step: np.ndarray
    candidate_index: np.ndarray
    chosen: np.ndarray
    target_clear_gain: np.ndarray
    target_information_gain: np.ndarray
    target_duration_s: np.ndarray
    q_variant: np.ndarray
    observations: np.ndarray
    summary: dict[str, float | int]


def _clip_to_disk(point: Point, radius: float = 1800.0) -> Point:
    norm = math.hypot(*point)
    if norm <= radius or norm <= 1e-12:
        return point
    scale = radius / norm
    return (point[0] * scale, point[1] * scale)


def directional_scan_waypoints() -> list[Point]:
    """Order the proven 700 m Q4 discovery lattice into a short route.

    A square-cell corner is at most ``700*sqrt(2) < 1000`` m from a source,
    and every closed 180-degree emission half-plane contains a cell corner.
    Radius 2800 m covers the full 1800 m source disk plus that corner distance.
    """

    unvisited = set(directional_discovery_lattice())
    current = (0.0, 0.0)
    ordered: list[Point] = []
    while unvisited:
        point = min(
            unvisited,
            key=lambda item: (
                math.hypot(item[0] - current[0], item[1] - current[1]),
                item[0],
                item[1],
            ),
        )
        unvisited.remove(point)
        ordered.append(point)
        current = point
    return ordered


def _track_last_result(track: ChannelTrack) -> str:
    value = getattr(track, "last_result", "none")
    return value if isinstance(value, str) else "none"


def _track_last_bearing(track: ChannelTrack) -> float | None:
    value = getattr(track, "last_bearing_deg", None)
    return float(value) if value is not None else None


def _remember_observation(track: ChannelTrack, observation: MeasureObservation) -> None:
    track.update(observation)
    track.last_result = observation.result
    track.last_bearing_deg = observation.bearing_deg
    track.near_count = int(getattr(track, "near_count", 0)) + int(
        observation.result == "near"
    )
    if observation.result == "direction" and not hasattr(track, "first_detection_step"):
        track.first_detection_step = len(track.attempted_measurement_positions)


def _candidate_points_for_track(
    track: ChannelTrack,
    *,
    q_variant: int,
) -> list[Point]:
    attempts = len(track.attempted_measurement_positions)
    scan = (
        guaranteed_discovery_waypoints()
        if q_variant == 3
        else directional_scan_waypoints()
    )
    if not track.detected or not track.polygon:
        if q_variant == 4:
            return [scan[attempts % len(scan)]]
        stride = 2
        return [
            scan[attempts % len(scan)],
            scan[(attempts + stride) % len(scan)],
        ]

    circle = track.enclosing_circle
    assert circle is not None
    points = [circle.center]
    last_bearing = _track_last_bearing(track)
    base = math.radians(last_bearing if last_bearing is not None else 0.0)
    if q_variant == 4 and track.measurements:
        # The last direction observation proves that its observer position lies
        # inside the unknown emission wedge.  A forward point plus short
        # tangential baselines therefore explores locally without throwing away
        # that hard-won visibility information.
        last_visible = track.measurements[-1][0]
        for forward in (50.0, 100.0, 200.0, 400.0):
            angular_offsets = (-1.0, 0.0, 1.0) if forward == 50.0 else (0.0,)
            for angular_offset_deg in angular_offsets:
                forward_angle = base + math.radians(angular_offset_deg)
                points.append(
                    _clip_to_disk(
                        (
                            last_visible[0] + forward * math.cos(forward_angle),
                            last_visible[1] + forward * math.sin(forward_angle),
                        ),
                        radius=2800.0,
                    )
                )
        for offset in (180.0, 400.0):
            for sign in (-1.0, 1.0):
                angle = base + sign * math.pi / 2.0
                points.append(
                    _clip_to_disk(
                        (
                            last_visible[0] + offset * math.cos(angle),
                            last_visible[1] + offset * math.sin(angle),
                        ),
                        radius=2450.0,
                    )
                )
    else:
        if track.measurements and last_bearing is not None:
            last_visible = track.measurements[-1][0]
            for forward in (25.0, 50.0, 100.0, 200.0, 400.0):
                points.append(
                    _clip_to_disk(
                        (
                            last_visible[0] + forward * math.cos(base),
                            last_visible[1] + forward * math.sin(base),
                        )
                    )
                )
        offset = min(800.0, max(300.0, circle.radius + 150.0))
        for sign in (-1.0, 1.0):
            angle = base + sign * math.pi / 2.0
            points.append(
                _clip_to_disk(
                    (
                        circle.center[0] + offset * math.cos(angle),
                        circle.center[1] + offset * math.sin(angle),
                    )
                )
            )
    unique: list[Point] = []
    for point in points:
        if all(math.hypot(point[0] - old[0], point[1] - old[1]) > 1e-6 for old in unique):
            unique.append(point)
    return unique


def build_candidates(
    tracks: dict[int, ChannelTrack],
    *,
    q_variant: int,
) -> list[CandidateAction]:
    candidates: list[CandidateAction] = []
    for channel in range(1, 21):
        track = tracks[channel]
        if track.cleared:
            continue
        if (
            _track_last_result(track) == "near"
            and track.attempted_measurement_positions
        ):
            candidates.append(
                CandidateAction(
                    "clear",
                    channel,
                    track.attempted_measurement_positions[-1],
                    0,
                )
            )
        circle = track.enclosing_circle
        if circle is not None:
            candidates.append(CandidateAction("clear", channel, circle.center, 0))
        if (
            q_variant == 4
            and track.detected
            and track.measurements
            and _track_last_result(track) == "no_signal"
        ):
            last_visible, last_bearing = track.measurements[-1]
            angle = math.radians(last_bearing)
            no_signal_point = track.attempted_measurement_positions[-1]
            traversed = min(
                400.0,
                math.hypot(
                    no_signal_point[0] - last_visible[0],
                    no_signal_point[1] - last_visible[1],
                ),
            )
            probe_index = 1
            offset = 18.0
            while offset <= max(18.0, traversed):
                candidates.append(
                    CandidateAction(
                        "clear",
                        channel,
                        (
                            last_visible[0] + offset * math.cos(angle),
                            last_visible[1] + offset * math.sin(angle),
                        ),
                        probe_index,
                    )
                )
                probe_index += 1
                offset += 36.0
        for index, point in enumerate(
            _candidate_points_for_track(track, q_variant=q_variant), start=1
        ):
            candidates.append(CandidateAction("measure", channel, point, index))
    return candidates


def _predicted_observation(
    source: Source | None,
    point: Point,
    *,
    already_cleared: bool,
) -> str:
    if source is None or already_cleared or not _source_visible(source, point):
        return "no_signal"
    distance = math.hypot(source.position[0] - point[0], source.position[1] - point[1])
    return "near" if distance <= 5.0 else "direction"


def score_candidate(
    candidate: CandidateAction,
    *,
    environment: InterferenceEnvironment,
    track: ChannelTrack,
) -> tuple[float, float, float]:
    """Return teacher targets (clear gain, information gain, duration)."""

    source = environment._sources.get(candidate.channel)
    distance = math.hypot(
        candidate.point[0] - environment.position[0],
        candidate.point[1] - environment.position[1],
    )
    switch = 1.0 if (
        candidate.kind == "measure" and candidate.channel != environment.current_channel
    ) else 0.0
    operation = 5.0 if candidate.kind == "measure" else 3.0
    duration = distance / 5.0 + switch + operation

    if candidate.kind == "clear":
        success = float(
            source is not None
            and candidate.channel not in environment.cleared_channels
            and math.hypot(
                candidate.point[0] - source.position[0],
                candidate.point[1] - source.position[1],
            ) <= 20.0
        )
        return success, 0.0, duration + 2.0 * success

    result = _predicted_observation(
        source,
        candidate.point,
        already_cleared=candidate.channel in environment.cleared_channels,
    )
    attempts = len(track.attempted_measurement_positions)
    repeated = any(
        math.hypot(candidate.point[0] - old[0], candidate.point[1] - old[1]) < 1.0
        for old in track.attempted_measurement_positions
    )
    novelty = 0.05 if repeated else 0.35
    if not track.detected and result == "no_signal":
        # Before the first signal, source existence and directional visibility
        # cannot be separated.  Give absent and temporarily invisible channels
        # the same observable score so the rollout performs a deployable,
        # balanced sweep instead of exploiting hidden existence labels.
        information = 1.0 / (1.0 + 0.02 * attempts) + novelty
    elif result == "near":
        information = 4.0 + novelty
    elif result == "no_signal":
        information = (0.65 if not source.directional else 0.25) + novelty
    elif not track.detected:
        information = 3.0 + novelty
    else:
        true_bearing = math.atan2(
            source.position[1] - candidate.point[1],
            source.position[0] - candidate.point[0],
        )
        if track.measurements:
            old_point, _ = track.measurements[-1]
            old_bearing = math.atan2(
                source.position[1] - old_point[1],
                source.position[0] - old_point[0],
            )
            crossing = abs(math.sin(true_bearing - old_bearing))
        else:
            crossing = 0.0
        information = 1.0 + 2.0 * crossing + novelty
        if track.measurements and source is not None:
            old_point = track.measurements[-1][0]
            old_distance = math.hypot(
                old_point[0] - source.position[0], old_point[1] - source.position[1]
            )
            new_distance = math.hypot(
                candidate.point[0] - source.position[0],
                candidate.point[1] - source.position[1],
            )
            # Potential-based shaping for the hitting-time objective.  Signed
            # progress prevents high-crossing-angle measurements from cycling
            # between informative-looking points that move away from the source.
            normalized_progress = max(
                -4.0,
                min(8.0, (old_distance - new_distance) / 50.0),
            )
            information += 2.0 * normalized_progress
    # Tiny, deterministic travel preference avoids unstable ties but cannot
    # override a meaningful information difference.
    information -= min(duration, 1000.0) * 1e-5
    return 0.0, information, duration


def _belief_stats(tracks: dict[int, ChannelTrack]) -> tuple[float, float, float, float]:
    radii = [
        track.enclosing_circle.radius
        for track in tracks.values()
        if track.enclosing_circle is not None and not track.cleared
    ]
    detected_remaining = sum(t.detected and not t.cleared for t in tracks.values()) / 20.0
    uncertain = sum(not t.detected and not t.cleared for t in tracks.values()) / 20.0
    if not radii:
        return detected_remaining, uncertain, 1.0, 1.0
    return (
        detected_remaining,
        uncertain,
        min(radii) / 1800.0,
        sum(radii) / len(radii) / 1800.0,
    )


def feature_vector(
    candidate: CandidateAction,
    *,
    environment: InterferenceEnvironment,
    tracks: dict[int, ChannelTrack],
    step: int,
    max_steps: int,
    q_variant: int,
    directional_prior: float,
    candidate_count: int,
) -> list[float]:
    track = tracks[candidate.channel]
    circle = track.enclosing_circle
    center = circle.center if circle is not None else (0.0, 0.0)
    radius = circle.radius if circle is not None else 1800.0
    diameter = (
        convex_diameter(track.polygon).distance
        if len(track.polygon) >= 2
        else 3600.0
    )
    last_position = (
        track.attempted_measurement_positions[-1]
        if track.attempted_measurement_positions
        else (0.0, 0.0)
    )
    bearing = _track_last_bearing(track)
    last_result = _track_last_result(track)
    measures = len(track.attempted_measurement_positions)
    directions = len(track.measurements)
    near_count = int(getattr(track, "near_count", 0))
    first_detection = int(getattr(track, "first_detection_step", measures))
    travel_distance = math.hypot(
        candidate.point[0] - environment.position[0],
        candidate.point[1] - environment.position[1],
    )
    switch_time = float(
        candidate.kind == "measure" and candidate.channel != environment.current_channel
    )
    operation = 5.0 if candidate.kind == "measure" else 3.0
    remaining_detected, uncertain, min_radius, mean_radius = _belief_stats(tracks)
    # The official API never reveals the true source count before exit.  Divide
    # by the published maximum instead of leaking hidden episode state.
    cleared_fraction = environment.cleared_count / 16.0
    time_per_clear = environment.virtual_time_s / max(environment.cleared_count, 1) / 1000.0
    return [
        environment.position[0] / 1800.0,
        environment.position[1] / 1800.0,
        (environment.current_channel - 1) / 19.0,
        environment.virtual_time_s / 36000.0,
        cleared_fraction,
        sum(t.detected for t in tracks.values()) / 20.0,
        step / max(max_steps, 1),
        float(q_variant == 4),
        directional_prior,
        (candidate.channel - 1) / 19.0,
        float(track.detected),
        float(track.cleared),
        min(measures, 50) / 50.0,
        min(directions, 20) / 20.0,
        min(len(track.no_signal_positions), 50) / 50.0,
        min(near_count, 5) / 5.0,
        center[0] / 1800.0,
        center[1] / 1800.0,
        min(radius, 3600.0) / 1800.0,
        min(diameter, 7200.0) / 3600.0,
        last_position[0] / 1800.0,
        last_position[1] / 1800.0,
        math.sin(math.radians(bearing)) if bearing is not None else 0.0,
        math.cos(math.radians(bearing)) if bearing is not None else 0.0,
        float(last_result == "no_signal"),
        float(last_result == "direction"),
        float(last_result == "near"),
        min(measures, 40) / 40.0,
        min(max(measures - first_detection, 0), 20) / 20.0,
        float(candidate.kind == "measure"),
        float(candidate.kind == "clear"),
        candidate.point[0] / 1800.0,
        candidate.point[1] / 1800.0,
        min(travel_distance, 5000.0) / 5000.0,
        min(travel_distance / 5.0, 1000.0) / 1000.0,
        switch_time,
        operation / 5.0,
        float(candidate.channel == environment.current_channel),
        float(candidate.point[0] ** 2 + candidate.point[1] ** 2 <= 1800.0 ** 2 + 1e-6),
        math.hypot(*candidate.point) / 1800.0,
        remaining_detected,
        uncertain,
        min_radius,
        mean_radius,
        min(time_per_clear, 10.0) / 10.0,
        min(candidate.within_channel_index, 8) / 8.0,
        min(candidate_count, 160) / 160.0,
    ]


def _teacher_choice(scores: Sequence[tuple[float, float, float]]) -> int:
    # True lexicographic objective: clearance, information, then elapsed time.
    return max(range(len(scores)), key=lambda i: (scores[i][0], scores[i][1], -scores[i][2]))


def generate_greedy_episode_legacy(
    *,
    seed: int,
    episode_id: int,
    decision_id_start: int,
    max_steps: int = 180,
    q_variant_override: int | None = None,
    record_arrays: bool = True,
) -> EpisodeArrays:
    rng = random.Random(seed)
    sampled_q_variant = 3 if rng.random() < 0.5 else 4
    q_variant = sampled_q_variant if q_variant_override is None else q_variant_override
    if q_variant not in (3, 4):
        raise ValueError("q_variant_override must be 3 or 4")
    directional_prior = 0.0 if q_variant == 3 else rng.uniform(0.2, 0.85)
    sources = generate_sources(
        seed,
        directional_probability=directional_prior,
        error_modes=rng.randint(2, 6),
    )
    environment = InterferenceEnvironment(sources)
    tracks = {channel: ChannelTrack(channel) for channel in range(1, 21)}

    feature_rows: list[list[float]] = []
    decision_ids: list[int] = []
    episode_ids: list[int] = []
    steps: list[int] = []
    candidate_indices: list[int] = []
    chosen_rows: list[int] = []
    clear_targets: list[float] = []
    information_targets: list[float] = []
    duration_targets: list[float] = []
    observations: list[list[float]] = []
    total_decisions = 0
    total_candidate_rows = 0

    for step_index in range(max_steps):
        if environment.all_cleared:
            break
        candidates = build_candidates(tracks, q_variant=q_variant)
        if not candidates:
            break
        scores = [
            score_candidate(action, environment=environment, track=tracks[action.channel])
            for action in candidates
        ]
        expert_index = _teacher_choice(scores)
        decision_id = decision_id_start + total_decisions
        total_candidate_rows += len(candidates)
        if record_arrays:
            for candidate_index, (candidate, score) in enumerate(zip(candidates, scores)):
                feature_rows.append(
                    feature_vector(
                        candidate,
                        environment=environment,
                        tracks=tracks,
                        step=step_index,
                        max_steps=max_steps,
                        q_variant=q_variant,
                        directional_prior=directional_prior,
                        candidate_count=len(candidates),
                    )
                )
                decision_ids.append(decision_id)
                episode_ids.append(episode_id)
                steps.append(step_index)
                candidate_indices.append(candidate_index)
                chosen_rows.append(int(candidate_index == expert_index))
                clear_targets.append(score[0])
                information_targets.append(score[1])
                duration_targets.append(score[2])

        action = candidates[expert_index]
        before_cleared = environment.cleared_count
        if action.kind == "measure":
            result = environment.measure(action.point, action.channel)
            _remember_observation(tracks[action.channel], result)
            observation_code = OBSERVATION_CODES[result.result]
            bearing = result.bearing_deg if result.bearing_deg is not None else math.nan
            action_duration = result.action_duration_s
        else:
            result = environment.clear(action.point, action.channel)
            if result.result == "success":
                tracks[action.channel].cleared = True
                observation_code = OBSERVATION_CODES["clear_success"]
            else:
                observation_code = OBSERVATION_CODES["clear_failure"]
            bearing = math.nan
            action_duration = result.action_duration_s
        observations.append(
            [
                float(decision_id), float(episode_id), float(step_index),
                float(action.channel), float(action.kind == "clear"),
                action.point[0], action.point[1], float(observation_code), bearing,
                action_duration, environment.virtual_time_s,
                float(environment.cleared_count - before_cleared),
            ]
        )
        total_decisions += 1

    features = np.asarray(feature_rows, dtype=np.float32)
    if features.size == 0:
        features = np.empty((0, len(FEATURE_NAMES)), dtype=np.float32)
    return EpisodeArrays(
        features=features,
        decision_id=np.asarray(decision_ids, dtype=np.int64),
        episode_id=np.asarray(episode_ids, dtype=np.int64),
        step=np.asarray(steps, dtype=np.int16),
        candidate_index=np.asarray(candidate_indices, dtype=np.int16),
        chosen=np.asarray(chosen_rows, dtype=np.uint8),
        target_clear_gain=np.asarray(clear_targets, dtype=np.float32),
        target_information_gain=np.asarray(information_targets, dtype=np.float32),
        target_duration_s=np.asarray(duration_targets, dtype=np.float32),
        q_variant=np.full(len(feature_rows), q_variant, dtype=np.uint8),
        observations=np.asarray(observations, dtype=np.float64).reshape(-1, 12),
        summary={
            "episode_id": episode_id,
            "seed": seed,
            "q_variant": q_variant,
            "source_count": environment.source_count,
            "cleared_count": environment.cleared_count,
            "decisions": total_decisions,
            "candidate_rows": total_candidate_rows,
            "virtual_time_s": environment.virtual_time_s,
        },
    )


class _ControllerRecordingEnvironment:
    """Record a reliable controller trajectory as grouped imitation samples."""

    def __init__(
        self,
        environment: InterferenceEnvironment,
        *,
        episode_id: int,
        decision_id_start: int,
        q_variant: int,
        directional_model_prior: float,
        max_steps: int,
        record_arrays: bool,
    ):
        self._environment = environment
        self._sources = environment._sources
        self.episode_id = episode_id
        self.decision_id_start = decision_id_start
        self.q_variant = q_variant
        self.directional_model_prior = directional_model_prior
        self.max_steps = max_steps
        self.record_arrays = record_arrays
        self.tracks = {channel: ChannelTrack(channel) for channel in range(1, 21)}
        self.feature_rows: list[list[float]] = []
        self.decision_ids: list[int] = []
        self.episode_ids: list[int] = []
        self.steps: list[int] = []
        self.candidate_indices: list[int] = []
        self.chosen_rows: list[int] = []
        self.clear_targets: list[float] = []
        self.information_targets: list[float] = []
        self.duration_targets: list[float] = []
        self.observations: list[list[float]] = []
        self.total_candidate_rows = 0

    @property
    def position(self) -> Point:
        return self._environment.position

    @property
    def current_channel(self) -> int:
        return self._environment.current_channel

    @property
    def virtual_time_s(self) -> float:
        return self._environment.virtual_time_s

    @property
    def cleared_count(self) -> int:
        return self._environment.cleared_count

    @property
    def source_count(self) -> int:
        return self._environment.source_count

    @property
    def cleared_channels(self) -> set[int]:
        return self._environment.cleared_channels

    def _record_candidate_group(self, selected: CandidateAction) -> int:
        decision_id = self.decision_id_start + len(self.observations)
        if not self.record_arrays:
            return decision_id
        candidates = build_candidates(self.tracks, q_variant=self.q_variant)
        selected_index = next(
            (
                index
                for index, candidate in enumerate(candidates)
                if candidate.kind == selected.kind
                and candidate.channel == selected.channel
                and math.hypot(
                    candidate.point[0] - selected.point[0],
                    candidate.point[1] - selected.point[1],
                )
                <= 1e-6
            ),
            None,
        )
        if selected_index is None:
            candidates.append(selected)
            selected_index = len(candidates) - 1
        scores = [
            score_candidate(
                candidate,
                environment=self._environment,
                track=self.tracks[candidate.channel],
            )
            for candidate in candidates
        ]
        step = len(self.observations)
        for candidate_index, (candidate, score) in enumerate(zip(candidates, scores)):
            self.feature_rows.append(
                feature_vector(
                    candidate,
                    environment=self,
                    tracks=self.tracks,
                    step=step,
                    max_steps=self.max_steps,
                    q_variant=self.q_variant,
                    directional_prior=self.directional_model_prior,
                    candidate_count=len(candidates),
                )
            )
            self.decision_ids.append(decision_id)
            self.episode_ids.append(self.episode_id)
            self.steps.append(step)
            self.candidate_indices.append(candidate_index)
            self.chosen_rows.append(int(candidate_index == selected_index))
            self.clear_targets.append(score[0])
            self.information_targets.append(score[1])
            self.duration_targets.append(score[2])
        self.total_candidate_rows += len(candidates)
        return decision_id

    def measure(self, position: Point, channel: int) -> MeasureObservation:
        selected = CandidateAction("measure", channel, position, 999)
        decision_id = self._record_candidate_group(selected)
        before_cleared = self.cleared_count
        result = self._environment.measure(position, channel)
        _remember_observation(self.tracks[channel], result)
        self.observations.append(
            [
                float(decision_id), float(self.episode_id), float(len(self.observations)),
                float(channel), 0.0, position[0], position[1],
                float(OBSERVATION_CODES[result.result]),
                result.bearing_deg if result.bearing_deg is not None else math.nan,
                result.action_duration_s, result.virtual_time_s,
                float(self.cleared_count - before_cleared),
            ]
        )
        return result

    def clear(self, position: Point, channel: int):
        selected = CandidateAction("clear", channel, position, 999)
        decision_id = self._record_candidate_group(selected)
        before_cleared = self.cleared_count
        result = self._environment.clear(position, channel)
        if result.result == "success":
            self.tracks[channel].cleared = True
            result_code = OBSERVATION_CODES["clear_success"]
        else:
            result_code = OBSERVATION_CODES["clear_failure"]
        self.observations.append(
            [
                float(decision_id), float(self.episode_id), float(len(self.observations)),
                float(channel), 1.0, position[0], position[1], float(result_code),
                math.nan, result.action_duration_s, result.virtual_time_s,
                float(self.cleared_count - before_cleared),
            ]
        )
        return result


def generate_episode(
    *,
    seed: int,
    episode_id: int,
    decision_id_start: int,
    max_steps: int = 3_000,
    q_variant_override: int | None = None,
    record_arrays: bool = True,
) -> EpisodeArrays:
    """Generate one reliable controller rollout plus counterfactual labels."""

    rng = random.Random(seed)
    sampled_q_variant = 3 if rng.random() < 0.5 else 4
    q_variant = sampled_q_variant if q_variant_override is None else q_variant_override
    if q_variant not in (3, 4):
        raise ValueError("q_variant_override must be 3 or 4")
    generation_directional_probability = (
        0.0 if q_variant == 3 else rng.uniform(0.2, 0.85)
    )
    directional_model_prior = 0.0 if q_variant == 3 else 0.5
    sources = generate_sources(
        seed,
        directional_probability=generation_directional_probability,
        error_modes=rng.randint(2, 6),
    )
    base_environment = InterferenceEnvironment(sources)
    environment = _ControllerRecordingEnvironment(
        base_environment,
        episode_id=episode_id,
        decision_id_start=decision_id_start,
        q_variant=q_variant,
        directional_model_prior=directional_model_prior,
        max_steps=max_steps,
        record_arrays=record_arrays,
    )
    controller = (
        SparseDirectionalSearch(discovery_points=guaranteed_discovery_waypoints())
        if q_variant == 3
        else SparseDirectionalSearch()
    )
    result = controller.run(environment)
    features = np.asarray(environment.feature_rows, dtype=np.float32)
    if features.size == 0:
        features = np.empty((0, len(FEATURE_NAMES)), dtype=np.float32)
    return EpisodeArrays(
        features=features,
        decision_id=np.asarray(environment.decision_ids, dtype=np.int64),
        episode_id=np.asarray(environment.episode_ids, dtype=np.int64),
        step=np.asarray(environment.steps, dtype=np.int16),
        candidate_index=np.asarray(environment.candidate_indices, dtype=np.int16),
        chosen=np.asarray(environment.chosen_rows, dtype=np.uint8),
        target_clear_gain=np.asarray(environment.clear_targets, dtype=np.float32),
        target_information_gain=np.asarray(environment.information_targets, dtype=np.float32),
        target_duration_s=np.asarray(environment.duration_targets, dtype=np.float32),
        q_variant=np.full(len(environment.feature_rows), q_variant, dtype=np.uint8),
        observations=np.asarray(environment.observations, dtype=np.float64).reshape(-1, 12),
        summary={
            "episode_id": episode_id,
            "seed": seed,
            "q_variant": q_variant,
            "generation_directional_probability": generation_directional_probability,
            "directional_model_prior": directional_model_prior,
            "source_count": base_environment.source_count,
            "cleared_count": base_environment.cleared_count,
            "decisions": len(environment.observations),
            "candidate_rows": environment.total_candidate_rows,
            "virtual_time_s": base_environment.virtual_time_s,
            "unresolved_channels": result.unresolved_channels,
        },
    )


def concatenate_episodes(episodes: Iterable[EpisodeArrays]) -> EpisodeArrays:
    items = list(episodes)
    if not items:
        raise ValueError("at least one episode is required")
    array_fields = (
        "features", "decision_id", "episode_id", "step", "candidate_index",
        "chosen", "target_clear_gain", "target_information_gain",
        "target_duration_s", "q_variant", "observations",
    )
    merged = {name: np.concatenate([getattr(item, name) for item in items]) for name in array_fields}
    summaries = [item.summary for item in items]
    return EpisodeArrays(
        **merged,
        summary={
            "episodes": len(items),
            "decisions": int(sum(int(s["decisions"]) for s in summaries)),
            "candidate_rows": int(sum(int(s["candidate_rows"]) for s in summaries)),
            "sources": int(sum(int(s["source_count"]) for s in summaries)),
            "cleared": int(sum(int(s["cleared_count"]) for s in summaries)),
            "q3_episodes": int(sum(int(s["q_variant"] == 3) for s in summaries)),
            "q4_episodes": int(sum(int(s["q_variant"] == 4) for s in summaries)),
            "virtual_time_s": float(sum(float(s["virtual_time_s"]) for s in summaries)),
            "episode_summaries": summaries,
        },
    )
