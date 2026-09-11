"""Diagnose one incomplete surrogate trajectory; not part of policy inputs."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import random
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from cumcm_b.baseline import ChannelTrack
from cumcm_b.dataset import _remember_observation, generate_episode
from cumcm_b.simulator import MeasureObservation, generate_sources


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", nargs="?")
    parser.add_argument("--split", default="test")
    parser.add_argument("--episode", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--q-variant", type=int, choices=(3, 4), default=4)
    parser.add_argument("--only-unresolved", action="store_true")
    parser.add_argument("--max-direction-observations", type=int, default=20)
    parser.add_argument("--max-steps", type=int, default=12_000)
    args = parser.parse_args()
    if args.seed is not None:
        episode_id = args.seed
        generated = generate_episode(
            seed=episode_id,
            episode_id=episode_id,
            decision_id_start=episode_id * 20_000,
            max_steps=args.max_steps,
            q_variant_override=args.q_variant,
            record_arrays=False,
        )
        observations = generated.observations
        forced_variant = args.q_variant
    else:
        if args.dataset is None:
            raise SystemExit("dataset or --seed is required")
        root = Path(args.dataset)
        with np.load(root / args.split / "episodes.npz", allow_pickle=False) as data:
            index = data["values"]
        if args.episode is None:
            incomplete_q4 = index[(index[:, 2] == 4) & (index[:, 9] == 0)]
            episode_id = int(incomplete_q4[0, 0])
        else:
            episode_id = args.episode
        observations = None
        for shard in sorted((root / args.split).glob("part-*.npz")):
            with np.load(shard, allow_pickle=False) as data:
                obs = data["observations"]
                selected = obs[obs[:, 1].astype(np.int64) == episode_id]
                if len(selected):
                    observations = selected.copy()
                    break
        if observations is None:
            raise SystemExit(f"episode {episode_id} not found")
        forced_variant = None

    rng = random.Random(episode_id)
    sampled_q_variant = 3 if rng.random() < 0.5 else 4
    q_variant = forced_variant if forced_variant is not None else sampled_q_variant
    directional_prior = 0.0 if q_variant == 3 else rng.uniform(0.2, 0.85)
    error_modes = rng.randint(2, 6)
    sources = generate_sources(
        episode_id,
        directional_probability=directional_prior,
        error_modes=error_modes,
    )
    tracks = {channel: ChannelTrack(channel) for channel in range(1, 21)}
    cleared: set[int] = set()
    for row in observations:
        channel = int(row[3])
        code = int(row[7])
        if code in (1, 2, 3):
            result = {1: "no_signal", 2: "direction", 3: "near"}[code]
            bearing = None if math.isnan(row[8]) else float(row[8])
            _remember_observation(
                tracks[channel],
                MeasureObservation(
                    result, (float(row[5]), float(row[6])), channel,
                    float(row[10]), float(row[9]), bearing,
                ),
            )
        elif code == 4:
            tracks[channel].cleared = True
            cleared.add(channel)

    source_rows = []
    for source in sorted(sources, key=lambda item: item.channel):
        track = tracks[source.channel]
        circle = track.enclosing_circle
        source_rows.append(
            {
                "channel": source.channel,
                "directional": source.directional,
                "source_position": source.position,
                "receive_radius": source.receive_radius,
                "heading_deg": source.heading_deg,
                "cleared": source.channel in cleared,
                "measurements": len(track.attempted_measurement_positions),
                "directions": len(track.measurements),
                "no_signals": len(track.no_signal_positions),
                "final_belief_radius": None if circle is None else circle.radius,
                "minimum_observer_distance": min(
                    (
                        math.hypot(point[0] - source.position[0], point[1] - source.position[1])
                        for point in track.attempted_measurement_positions
                    ),
                    default=None,
                ),
                "direction_observations": [
                    {"point": point, "bearing_deg": bearing}
                    for point, bearing in track.measurements[-args.max_direction_observations :]
                ],
                "last_result": getattr(track, "last_result", "none"),
            }
        )
    result = {
        "episode_id": episode_id,
        "q_variant": q_variant,
        "directional_prior": directional_prior,
        "steps": len(observations),
        "source_count": len(sources),
        "cleared_count": len(cleared),
        "sources": (
            [row for row in source_rows if not row["cleared"]]
            if args.only_unresolved
            else source_rows
        ),
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
