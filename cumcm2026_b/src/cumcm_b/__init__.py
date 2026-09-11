"""Core algorithms for CUMCM 2026 problem B."""

from .geometry import (
    Circle,
    DiameterResult,
    Point,
    bearing_feasible_polygon,
    clip_polygon_halfplane,
    convex_diameter,
    disk_polygon,
    localization_polygon,
    minimum_enclosing_circle,
)
from .planning import (
    SecondPointEvaluation,
    evaluate_second_point,
    grid_points_in_disk,
    sample_convex_polygon,
    select_second_measurement_points,
)
from .simulator import (
    ClearObservation,
    InterferenceEnvironment,
    MeasureObservation,
    Source,
    generate_sources,
)
from .belief import ChannelBelief, BeliefSummary, SourceParticle
from .baseline import (
    ChannelTrack,
    OmniBaselineResult,
    OmniGeometryBaseline,
    guaranteed_discovery_waypoints,
)
from .client import SimulatorClient, SimulatorProtocolError

__all__ = [
    "Circle",
    "DiameterResult",
    "Point",
    "bearing_feasible_polygon",
    "clip_polygon_halfplane",
    "convex_diameter",
    "disk_polygon",
    "localization_polygon",
    "minimum_enclosing_circle",
    "SecondPointEvaluation",
    "evaluate_second_point",
    "grid_points_in_disk",
    "sample_convex_polygon",
    "select_second_measurement_points",
    "ClearObservation",
    "InterferenceEnvironment",
    "MeasureObservation",
    "Source",
    "generate_sources",
    "ChannelBelief",
    "BeliefSummary",
    "SourceParticle",
    "ChannelTrack",
    "OmniBaselineResult",
    "OmniGeometryBaseline",
    "guaranteed_discovery_waypoints",
    "SimulatorClient",
    "SimulatorProtocolError",
]
