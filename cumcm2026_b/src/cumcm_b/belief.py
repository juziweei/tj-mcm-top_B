"""Particle beliefs for one possibly occupied interference channel."""

from __future__ import annotations

from dataclasses import dataclass
import math
import random
from typing import Literal

from .geometry import Point


@dataclass(frozen=True)
class SourceParticle:
    exists: bool
    position: Point
    receive_radius: float
    directional: bool
    heading_deg: float
    weight: float


@dataclass(frozen=True)
class BeliefSummary:
    existence_probability: float
    directional_probability: float
    mean_position: Point
    covariance_xx: float
    covariance_xy: float
    covariance_yy: float
    spatial_spread: float
    cleared: bool


def _signed_angle_difference(first: float, second: float) -> float:
    return (first - second + 180.0) % 360.0 - 180.0


def _visible(particle: SourceParticle, observer: Point) -> bool:
    if not particle.exists:
        return False
    dx = observer[0] - particle.position[0]
    dy = observer[1] - particle.position[1]
    if math.hypot(dx, dy) > particle.receive_radius:
        return False
    if not particle.directional:
        return True
    source_to_observer = math.degrees(math.atan2(dy, dx)) % 360.0
    # The statement defines 90 degrees on each side of the heading: a
    # 180-degree closed sector, not a 90-degree total beam.
    return abs(_signed_angle_difference(source_to_observer, particle.heading_deg)) <= 90.0


class ChannelBelief:
    """Sequential Bayesian approximation for a single channel.

    The observation model deliberately uses bounded bearing residuals because the
    official statement guarantees an error interval but does not specify a
    Gaussian distribution.
    """

    def __init__(
        self,
        particles: list[SourceParticle],
        *,
        seed: int = 2026,
        bearing_error_deg: float = 1.0,
        likelihood_floor: float = 1e-6,
        resample_ratio: float = 0.45,
    ):
        if not particles:
            raise ValueError("particles must be non-empty")
        if bearing_error_deg <= 0:
            raise ValueError("bearing_error_deg must be positive")
        if not 0.0 < likelihood_floor < 1.0:
            raise ValueError("likelihood_floor must lie in (0, 1)")
        if not 0.0 < resample_ratio <= 1.0:
            raise ValueError("resample_ratio must lie in (0, 1]")
        self._rng = random.Random(seed)
        self.bearing_error_deg = bearing_error_deg
        self.likelihood_floor = likelihood_floor
        self.resample_ratio = resample_ratio
        self.cleared = False
        self.particles = self._normalize(particles)

    @classmethod
    def prior(
        cls,
        count: int,
        *,
        seed: int,
        existence_probability: float = 0.65,
        directional_probability: float = 0.5,
        target_radius: float = 1800.0,
    ) -> "ChannelBelief":
        if count < 1:
            raise ValueError("count must be positive")
        if not 0.0 <= existence_probability <= 1.0:
            raise ValueError("existence_probability must lie in [0, 1]")
        if not 0.0 <= directional_probability <= 1.0:
            raise ValueError("directional_probability must lie in [0, 1]")
        rng = random.Random(seed)
        particles: list[SourceParticle] = []
        for _ in range(count):
            exists = rng.random() < existence_probability
            radius = target_radius * math.sqrt(rng.random())
            angle = rng.uniform(0.0, 2.0 * math.pi)
            particles.append(
                SourceParticle(
                    exists=exists,
                    position=(radius * math.cos(angle), radius * math.sin(angle)),
                    receive_radius=rng.uniform(1000.0, 1500.0),
                    directional=rng.random() < directional_probability,
                    heading_deg=rng.uniform(0.0, 360.0),
                    weight=1.0 / count,
                )
            )
        return cls(particles, seed=seed)

    @staticmethod
    def _normalize(particles: list[SourceParticle]) -> list[SourceParticle]:
        total = sum(max(0.0, particle.weight) for particle in particles)
        if total <= 0.0:
            uniform = 1.0 / len(particles)
            return [
                SourceParticle(
                    particle.exists,
                    particle.position,
                    particle.receive_radius,
                    particle.directional,
                    particle.heading_deg,
                    uniform,
                )
                for particle in particles
            ]
        return [
            SourceParticle(
                particle.exists,
                particle.position,
                particle.receive_radius,
                particle.directional,
                particle.heading_deg,
                max(0.0, particle.weight) / total,
            )
            for particle in particles
        ]

    def _apply_likelihoods(self, likelihoods: list[float]) -> None:
        updated = [
            SourceParticle(
                particle.exists,
                particle.position,
                particle.receive_radius,
                particle.directional,
                particle.heading_deg,
                particle.weight * max(self.likelihood_floor, likelihood),
            )
            for particle, likelihood in zip(self.particles, likelihoods, strict=True)
        ]
        self.particles = self._normalize(updated)
        effective_count = 1.0 / sum(particle.weight ** 2 for particle in self.particles)
        if effective_count < self.resample_ratio * len(self.particles):
            self._systematic_resample()

    def _systematic_resample(self) -> None:
        count = len(self.particles)
        start = self._rng.random() / count
        thresholds = [start + index / count for index in range(count)]
        cumulative = self.particles[0].weight
        source_index = 0
        selected: list[SourceParticle] = []
        for threshold in thresholds:
            while threshold > cumulative and source_index < count - 1:
                source_index += 1
                cumulative += self.particles[source_index].weight
            particle = self.particles[source_index]
            selected.append(
                SourceParticle(
                    particle.exists,
                    particle.position,
                    particle.receive_radius,
                    particle.directional,
                    particle.heading_deg,
                    1.0 / count,
                )
            )
        self.particles = selected

    def update_measurement(
        self,
        position: Point,
        result: Literal["no_signal", "near", "direction"],
        bearing_deg: float | None = None,
    ) -> None:
        if self.cleared:
            return
        if result == "direction" and bearing_deg is None:
            raise ValueError("direction observations require bearing_deg")
        likelihoods: list[float] = []
        for particle in self.particles:
            visible = _visible(particle, position)
            distance = math.hypot(
                particle.position[0] - position[0],
                particle.position[1] - position[1],
            )
            if result == "no_signal":
                likelihoods.append(1.0 if not visible else self.likelihood_floor)
            elif result == "near":
                likelihoods.append(1.0 if visible and distance <= 5.0 else self.likelihood_floor)
            else:
                predicted = math.degrees(
                    math.atan2(
                        particle.position[1] - position[1],
                        particle.position[0] - position[0],
                    )
                )
                residual = abs(_signed_angle_difference(bearing_deg or 0.0, predicted))
                valid = visible and distance > 5.0 and residual <= self.bearing_error_deg + 1e-10
                likelihoods.append(1.0 if valid else self.likelihood_floor)
        self._apply_likelihoods(likelihoods)

    def update_clear(
        self,
        position: Point,
        result: Literal["success", "no_target_in_range"],
    ) -> None:
        if result == "success":
            self.cleared = True
        likelihoods: list[float] = []
        for particle in self.particles:
            in_range = particle.exists and math.hypot(
                particle.position[0] - position[0],
                particle.position[1] - position[1],
            ) <= 20.0
            likelihoods.append(
                1.0
                if (result == "success") == in_range
                else self.likelihood_floor
            )
        self._apply_likelihoods(likelihoods)

    def clear_probability(self, position: Point) -> float:
        if self.cleared:
            return 0.0
        return sum(
            particle.weight
            for particle in self.particles
            if particle.exists
            and math.hypot(
                particle.position[0] - position[0],
                particle.position[1] - position[1],
            ) <= 20.0
        )

    def summary(self) -> BeliefSummary:
        existence = sum(
            particle.weight for particle in self.particles if particle.exists
        )
        if existence <= 1e-12:
            return BeliefSummary(0.0, 0.0, (0.0, 0.0), 0.0, 0.0, 0.0, 0.0, self.cleared)
        existing = [particle for particle in self.particles if particle.exists]
        mean_x = sum(particle.weight * particle.position[0] for particle in existing) / existence
        mean_y = sum(particle.weight * particle.position[1] for particle in existing) / existence
        covariance_xx = sum(
            particle.weight * (particle.position[0] - mean_x) ** 2
            for particle in existing
        ) / existence
        covariance_xy = sum(
            particle.weight
            * (particle.position[0] - mean_x)
            * (particle.position[1] - mean_y)
            for particle in existing
        ) / existence
        covariance_yy = sum(
            particle.weight * (particle.position[1] - mean_y) ** 2
            for particle in existing
        ) / existence
        directional = sum(
            particle.weight
            for particle in existing
            if particle.directional
        ) / existence
        return BeliefSummary(
            existence_probability=existence,
            directional_probability=directional,
            mean_position=(mean_x, mean_y),
            covariance_xx=covariance_xx,
            covariance_xy=covariance_xy,
            covariance_yy=covariance_yy,
            spatial_spread=math.sqrt(max(0.0, covariance_xx + covariance_yy)),
            cleared=self.cleared,
        )
