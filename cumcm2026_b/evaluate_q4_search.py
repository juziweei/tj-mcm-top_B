"""Evaluate the deployable Q4 search on independent surrogate cases."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from cumcm_b.q4_search import (  # noqa: E402
    SparseDirectionalSearch,
    directional_discovery_lattice,
)
from cumcm_b.route_oracle import fast_open_route  # noqa: E402
from cumcm_b.simulator import InterferenceEnvironment, generate_sources  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed-start", type=int, default=90_000_000)
    parser.add_argument("--source-count", type=int)
    parser.add_argument("--initial-pursuit-step-m", type=float, default=125.0)
    parser.add_argument("--single-bearing-pursuit-step-m", type=float, default=350.0)
    parser.add_argument("--stop-probability", type=float, default=0.998)
    parser.add_argument("--belief-directional-probability", type=float, default=0.5)
    parser.add_argument(
        "--environment-directional-probability", type=float, default=0.5
    )
    parser.add_argument("--belief-particle-count", type=int, default=8192)
    parser.add_argument("--belief-range-margin-m", type=float, default=10.0)
    parser.add_argument("--belief-angular-margin-deg", type=float, default=12.0)
    parser.add_argument("--stopping-range-margin-m", type=float)
    parser.add_argument("--stopping-angular-margin-deg", type=float)
    parser.add_argument("--analytic-validation-position-count", type=int, default=0)
    parser.add_argument("--shared-bearing-target", type=int, default=3)
    parser.add_argument("--cross-bearing-fraction", type=float, default=0.0)
    parser.add_argument("--centroid-clear-after-bearings", type=int, default=2)
    parser.add_argument("--pursuit-deferral-positions", type=int, default=0)
    parser.add_argument(
        "--disable-route-aware-discovery-selection", action="store_true"
    )
    parser.add_argument("--terminal-route-commitment-steps", type=int, default=1)
    parser.add_argument("--discovery-travel-weight", type=float, default=1.0)
    parser.add_argument("--disable-scan-after-clear", action="store_true")
    parser.add_argument("--opportunistic-scan-rate-ratio", type=float, default=0.5)
    parser.add_argument("--target-posterior-particle-count", type=int, default=0)
    parser.add_argument("--refinement-phase-divisions", type=int, default=1)
    parser.add_argument("--polar-refinement", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--base-lattice-only", action="store_true")
    args = parser.parse_args()
    rows = []
    wall_started = time.perf_counter()
    for offset in range(args.episodes):
        episode_wall_started = time.perf_counter()
        sources = generate_sources(
            args.seed_start + offset,
            count=args.source_count,
            directional_probability=args.environment_directional_probability,
        )
        environment = InterferenceEnvironment(sources)
        result = SparseDirectionalSearch(
            initial_pursuit_step_m=args.initial_pursuit_step_m,
            single_bearing_pursuit_step_m=args.single_bearing_pursuit_step_m,
            stop_probability=args.stop_probability,
            belief_directional_probability=args.belief_directional_probability,
            belief_particle_count=args.belief_particle_count,
            belief_range_margin_m=args.belief_range_margin_m,
            belief_angular_margin_deg=args.belief_angular_margin_deg,
            stopping_range_margin_m=args.stopping_range_margin_m,
            stopping_angular_margin_deg=args.stopping_angular_margin_deg,
            analytic_validation_position_count=args.analytic_validation_position_count,
            shared_bearing_target=args.shared_bearing_target,
            cross_bearing_fraction=args.cross_bearing_fraction,
            centroid_clear_after_bearings=args.centroid_clear_after_bearings,
            pursuit_deferral_positions=args.pursuit_deferral_positions,
            route_aware_discovery_selection=(
                not args.disable_route_aware_discovery_selection
            ),
            terminal_route_commitment_steps=args.terminal_route_commitment_steps,
            discovery_travel_weight=args.discovery_travel_weight,
            scan_after_clear=not args.disable_scan_after_clear,
            opportunistic_scan_rate_ratio=args.opportunistic_scan_rate_ratio,
            target_posterior_particle_count=args.target_posterior_particle_count,
            refinement_phase_divisions=args.refinement_phase_divisions,
            include_polar_refinement=args.polar_refinement,
            discovery_points=(
                directional_discovery_lattice()
                if args.base_lattice_only
                else None
            ),
        ).run(environment)
        approximate_oracle_time_s = (
            fast_open_route([source.position for source in sources]).distance_m / 5.0
            + 5.0 * len(sources)
        )
        rows.append(
            {
                "seed": args.seed_start + offset,
                "sources": len(sources),
                "cleared": result.cleared_count,
                "virtual_time_s": result.virtual_time_s,
                "mean_time_s": result.mean_clear_time_s,
                "measures": result.measure_actions,
                "clears": result.clear_actions,
                "travel_m": result.travel_distance_m,
                "discovery_travel_m": result.discovery_travel_distance_m,
                "pursuit_travel_m": result.pursuit_travel_distance_m,
                "discovery_positions": result.discovery_positions,
                "unresolved": result.unresolved_channels,
                "posterior_complete": result.posterior_all_sources_detected,
                "probability_stop": result.stopped_by_probability,
                "approximate_oracle_time_s": approximate_oracle_time_s,
                "competitive_ratio": result.virtual_time_s / approximate_oracle_time_s,
                "wall_time_s": time.perf_counter() - episode_wall_started,
            }
        )
    source_count = sum(row["sources"] for row in rows)
    cleared_count = sum(row["cleared"] for row in rows)
    mean_times = [row["mean_time_s"] for row in rows]
    report = {
        "episodes": len(rows),
        "fixed_source_count": args.source_count,
        "environment_directional_probability": (
            args.environment_directional_probability
        ),
        "belief_directional_probability": args.belief_directional_probability,
        "source_clear_rate": cleared_count / source_count,
        "complete_case_rate": sum(row["sources"] == row["cleared"] for row in rows) / len(rows),
        "mean_episode_mean_time_s": statistics.fmean(mean_times),
        "p95_episode_mean_time_s": sorted(mean_times)[int(0.95 * (len(mean_times) - 1))],
        "mean_measure_actions": statistics.fmean(row["measures"] for row in rows),
        "mean_clear_actions": statistics.fmean(row["clears"] for row in rows),
        "mean_discovery_positions": statistics.fmean(
            row["discovery_positions"] for row in rows
        ),
        "mean_travel_m": statistics.fmean(row["travel_m"] for row in rows),
        "mean_discovery_travel_m": statistics.fmean(
            row["discovery_travel_m"] for row in rows
        ),
        "mean_pursuit_travel_m": statistics.fmean(
            row["pursuit_travel_m"] for row in rows
        ),
        "mean_movement_time_s_per_source": statistics.fmean(
            row["travel_m"] / 5.0 / row["cleared"] for row in rows
        ),
        "mean_nonmovement_time_s_per_source": statistics.fmean(
            (row["virtual_time_s"] - row["travel_m"] / 5.0) / row["cleared"]
            for row in rows
        ),
        "mean_approximate_oracle_s_per_source": statistics.fmean(
            row["approximate_oracle_time_s"] / row["sources"] for row in rows
        ),
        "mean_competitive_ratio": statistics.fmean(
            row["competitive_ratio"] for row in rows
        ),
        "probability_stop_rate": sum(row["probability_stop"] for row in rows) / len(rows),
        "mean_episode_wall_time_s": statistics.fmean(
            row["wall_time_s"] for row in rows
        ),
        "p95_episode_wall_time_s": sorted(row["wall_time_s"] for row in rows)[
            int(0.95 * (len(rows) - 1))
        ],
        "max_episode_wall_time_s": max(row["wall_time_s"] for row in rows),
        "failures": [row for row in rows if row["sources"] != row["cleared"]],
        "wall_time_s": time.perf_counter() - wall_started,
    }
    encoded = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
    print(encoded)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
