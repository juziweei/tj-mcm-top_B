"""Generate observable histories with privileged count/time labels."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import random
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from cumcm_b.neural_belief import TOKEN_FEATURE_NAMES, observation_token  # noqa: E402
from cumcm_b.q4_search import SparseDirectionalSearch  # noqa: E402
from cumcm_b.simulator import InterferenceEnvironment, generate_sources  # noqa: E402


class RecordingEnvironment(InterferenceEnvironment):
    def __init__(self, sources, q_variant: int):
        super().__init__(sources)
        self.q_variant = q_variant
        self.tokens: list[np.ndarray] = []
        self.times: list[float] = []
        self.cleared_after: list[int] = []
        self.known_unresolved_after: list[int] = []
        self.new_detection: list[bool] = []
        self._detected: set[int] = set()

    def measure(self, position, channel):
        previous = self.position
        switched = channel != self.current_channel
        result = super().measure(position, channel)
        newly_detected = result.result in {"direction", "near"} and channel not in self._detected
        if newly_detected:
            self._detected.add(channel)
        self._record(
            previous, position, channel, True, switched, result.result,
            result.bearing_deg, result.action_duration_s, newly_detected,
        )
        return result

    def clear(self, position, channel):
        previous = self.position
        result = super().clear(position, channel)
        self._record(
            previous, position, channel, False, False,
            "clear_success" if result.result == "success" else "clear_failure",
            None, result.action_duration_s, False,
        )
        return result

    def _record(self, previous, target, channel, is_measure, switched, result,
                bearing, duration, newly_detected):
        self.tokens.append(observation_token(
            previous_position=previous, target=target, channel=channel,
            is_measure=is_measure, switched=switched, result=result,
            bearing_deg=bearing, action_duration_s=duration,
            virtual_time_s=self.virtual_time_s, cleared_count=self.cleared_count,
            q_variant=self.q_variant,
        ))
        self.times.append(self.virtual_time_s)
        self.cleared_after.append(self.cleared_count)
        self.known_unresolved_after.append(len(self._detected) - self.cleared_count)
        self.new_detection.append(newly_detected)


def generate_one(args):
    seed, q_variant = args
    rng = random.Random(seed ^ 0x5A17)
    directional_probability = 0.0 if q_variant == 3 else rng.uniform(0.2, 0.85)
    sources = generate_sources(seed, directional_probability=directional_probability)
    environment = RecordingEnvironment(sources, q_variant)
    policy = SparseDirectionalSearch(
        stop_probability=0.998,
        belief_directional_probability=directional_probability,
        centroid_clear_max_radius_m=75.0,
        use_enclosing_circle_target=True,
    )
    result = policy.run(environment)
    times = np.asarray(environment.times, dtype=np.float32)
    cleared = np.asarray(environment.cleared_after, dtype=np.int16)
    known_unresolved = np.asarray(environment.known_unresolved_after, dtype=np.int16)
    remaining = np.asarray(len(sources) - cleared, dtype=np.int16)
    final_time = float(result.virtual_time_s)
    next_time = np.empty(len(times), dtype=np.float32)
    event = np.zeros(len(times), dtype=np.uint8)
    future_detection = None
    for index in range(len(times) - 1, -1, -1):
        next_time[index] = (
            final_time - times[index]
            if future_detection is None
            else future_detection - times[index]
        )
        event[index] = future_detection is not None
        if environment.new_detection[index]:
            future_detection = float(times[index])
    return {
        "tokens": np.stack(environment.tokens),
        "virtual_time_s": times,
        "cleared_count_after": cleared,
        "known_unresolved_after": known_unresolved,
        "remaining_count": remaining,
        "next_detection_s": np.maximum(next_time, 0.0),
        "next_detection_event": event,
        "remaining_time_s": np.maximum(final_time - times, 0.0).astype(np.float32),
        "source_count": len(sources),
        "cleared_count": result.cleared_count,
        "q_variant": q_variant,
        "seed": seed,
    }


def write_split(root: Path, split: str, seeds, workers: int):
    started = time.perf_counter()
    work = [(seed, 3 if index % 3 == 0 else 4) for index, seed in enumerate(seeds)]
    episodes = []
    report_every = max(1, len(work) // 20)
    chunksize = max(1, len(work) // max(1, workers * 8))
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for index, episode in enumerate(
            pool.map(generate_one, work, chunksize=chunksize), start=1
        ):
            episodes.append(episode)
            if index % report_every == 0 or index == len(work):
                print(
                    f"{split}: {index}/{len(work)} episodes "
                    f"({time.perf_counter() - started:.1f}s)",
                    flush=True,
                )
    offsets = np.concatenate(([0], np.cumsum([len(x["tokens"]) for x in episodes]))).astype(np.int64)
    np.savez_compressed(
        root / f"{split}.npz",
        tokens=np.concatenate([x["tokens"] for x in episodes]),
        virtual_time_s=np.concatenate([x["virtual_time_s"] for x in episodes]),
        cleared_count_after=np.concatenate([x["cleared_count_after"] for x in episodes]),
        known_unresolved_after=np.concatenate([x["known_unresolved_after"] for x in episodes]),
        remaining_count=np.concatenate([x["remaining_count"] for x in episodes]),
        next_detection_s=np.concatenate([x["next_detection_s"] for x in episodes]),
        next_detection_event=np.concatenate([x["next_detection_event"] for x in episodes]),
        remaining_time_s=np.concatenate([x["remaining_time_s"] for x in episodes]),
        offsets=offsets,
        source_count=np.asarray([x["source_count"] for x in episodes], dtype=np.int16),
        cleared_count=np.asarray([x["cleared_count"] for x in episodes], dtype=np.int16),
        q_variant=np.asarray([x["q_variant"] for x in episodes], dtype=np.int8),
        seed=np.asarray([x["seed"] for x in episodes], dtype=np.int64),
    )
    return {"episodes": len(episodes), "tokens": int(offsets[-1]),
            "complete_rate": float(np.mean([x["source_count"] == x["cleared_count"] for x in episodes])),
            "wall_time_s": time.perf_counter() - started}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--train-episodes", type=int, default=2000)
    parser.add_argument("--val-episodes", type=int, default=300)
    parser.add_argument("--test-episodes", type=int, default=300)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    counts = {"train": args.train_episodes, "val": args.val_episodes, "test": args.test_episodes}
    bases = {"train": 110_000_000, "val": 120_000_000, "test": 130_000_000}
    summary = {split: write_split(args.output, split, range(bases[split], bases[split] + count), args.workers)
               for split, count in counts.items()}
    config = vars(args).copy()
    config["output"] = str(args.output)
    manifest = {"schema": "belief-history-v1", "feature_names": list(TOKEN_FEATURE_NAMES),
                "hidden_state_is_input": False, "splits": summary, "config": config}
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
