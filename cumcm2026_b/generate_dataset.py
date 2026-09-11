"""Parallel, shard-oriented dataset generator."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from cumcm_b.dataset import FEATURE_NAMES, concatenate_episodes, generate_episode


SPLIT_BASE_SEEDS = {"train": 10_000_000, "val": 20_000_000, "test": 30_000_000}


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def generate_shard(
    output_root: str,
    split: str,
    shard_index: int,
    target_rows: int,
    max_steps: int,
) -> dict[str, object]:
    start = time.perf_counter()
    split_dir = Path(output_root) / split
    split_dir.mkdir(parents=True, exist_ok=True)
    episodes = []
    rows = 0
    episode_offset = 0
    decision_offset = 0
    episode_namespace = SPLIT_BASE_SEEDS[split] + shard_index * 100_000
    while rows < target_rows:
        seed = episode_namespace + episode_offset
        episode_id = episode_namespace + episode_offset
        episode = generate_episode(
            seed=seed,
            episode_id=episode_id,
            decision_id_start=episode_namespace * 1_000 + decision_offset,
            max_steps=max_steps,
        )
        episodes.append(episode)
        rows += len(episode.features)
        decision_offset += int(episode.summary["decisions"])
        episode_offset += 1
    data = concatenate_episodes(episodes)
    shard_path = split_dir / f"part-{shard_index:05d}.npz"
    np.savez(
        shard_path,
        features=data.features,
        decision_id=data.decision_id,
        episode_id=data.episode_id,
        step=data.step,
        candidate_index=data.candidate_index,
        chosen=data.chosen,
        target_clear_gain=data.target_clear_gain,
        target_information_gain=data.target_information_gain,
        target_duration_s=data.target_duration_s,
        q_variant=data.q_variant,
        observations=data.observations,
    )
    elapsed = time.perf_counter() - start
    summary = dict(data.summary)
    summary.pop("episode_summaries", None)
    summary.update(
        {
            "split": split,
            "shard_index": shard_index,
            "path": str(shard_path),
            "elapsed_s": elapsed,
            "rows_per_s": len(data.features) / max(elapsed, 1e-9),
            "bytes": shard_path.stat().st_size,
            "seed_namespace_start": episode_namespace,
            "seed_namespace_end": episode_namespace + episode_offset - 1,
        }
    )
    _write_json(shard_path.with_suffix(".summary.json"), summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--train-shards", type=int, default=40)
    parser.add_argument("--val-shards", type=int, default=5)
    parser.add_argument("--test-shards", type=int, default=5)
    parser.add_argument("--rows-per-shard", type=int, default=50_000)
    parser.add_argument("--max-steps", type=int, default=180)
    parser.add_argument("--workers", type=int, default=min(32, os.cpu_count() or 1))
    args = parser.parse_args()

    root = Path(args.output).resolve()
    root.mkdir(parents=True, exist_ok=True)
    schema = {
        "format": "uncompressed NumPy NPZ shards",
        "dataset_semantics_version": "surrogate-v5-180deg",
        "candidate_feature_dtype": "float32",
        "feature_names": list(FEATURE_NAMES),
        "observation_columns": [
            "decision_id", "episode_id", "step", "channel", "is_clear",
            "target_x_m", "target_y_m", "result_code", "bearing_deg_or_nan",
            "action_duration_s", "virtual_time_s", "clear_delta",
        ],
        "observation_result_codes": {
            "0": "none", "1": "no_signal", "2": "direction", "3": "near",
            "4": "clear_success", "5": "clear_failure",
        },
        "targets": {
            "chosen": "one expert action per decision group",
            "target_clear_gain": "first lexicographic target",
            "target_information_gain": "second lexicographic target",
            "target_duration_s": "third target, minimized",
        },
        "leakage_contract": (
            "features contain observable history summaries only; hidden source position, "
            "radius, directionality and heading are used only to compute teacher targets"
        ),
    }
    _write_json(root / "schema.json", schema)

    tasks = []
    counts = {"train": args.train_shards, "val": args.val_shards, "test": args.test_shards}
    started = time.perf_counter()
    summaries: list[dict[str, object]] = []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        for split, count in counts.items():
            for shard_index in range(count):
                tasks.append(
                    executor.submit(
                        generate_shard,
                        str(root), split, shard_index, args.rows_per_shard, args.max_steps,
                    )
                )
        for future in as_completed(tasks):
            summary = future.result()
            summaries.append(summary)
            print(json.dumps(summary, ensure_ascii=False), flush=True)

    elapsed = time.perf_counter() - started
    summaries.sort(key=lambda s: (str(s["split"]), int(s["shard_index"])))
    totals = {
        split: {
            "shards": sum(s["split"] == split for s in summaries),
            "candidate_rows": sum(int(s["candidate_rows"]) for s in summaries if s["split"] == split),
            "decisions": sum(int(s["decisions"]) for s in summaries if s["split"] == split),
            "episodes": sum(int(s["episodes"]) for s in summaries if s["split"] == split),
            "bytes": sum(int(s["bytes"]) for s in summaries if s["split"] == split),
        }
        for split in counts
    }
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "generator": "cumcm_b surrogate domain-randomized coverage-pursuit teacher v5",
        "official_simulator_data": False,
        "calibration_status": "Requires recalibration against official simulator when available.",
        "physics_contract": {
            "source_count": {"minimum": 10, "maximum": 16},
            "channel_count": 20,
            "maximum_sources_per_channel": 1,
            "source_position": "uniform by area in the disk of radius 1800 m",
            "receive_radius_m": {"minimum": 1000.0, "maximum": 1500.0},
            "bearing_error_deg": {"minimum": -1.0, "maximum": 1.0},
            "q3_directional_probability": 0.0,
            "q4_directional_probability": "uniform episode prior in [0.2, 0.85]",
            "directional_total_coverage_deg": 180.0,
            "directional_boundary_inclusive": True,
            "near_radius_m": 5.0,
            "clear_radius_m": 20.0,
            "speed_mps": 5.0,
            "measure_time_s": 5.0,
            "channel_switch_time_s": 1.0,
            "successful_clear_time_s": 5.0,
            "failed_clear_time_s": 3.0,
        },
        "config": vars(args),
        "seed_namespaces": SPLIT_BASE_SEEDS,
        "split_policy": "disjoint episode/seed namespaces; an episode appears in exactly one split",
        "elapsed_s": elapsed,
        "totals": totals,
        "total_candidate_rows": sum(v["candidate_rows"] for v in totals.values()),
        "total_decisions": sum(v["decisions"] for v in totals.values()),
        "total_episodes": sum(v["episodes"] for v in totals.values()),
        "total_bytes": sum(v["bytes"] for v in totals.values()),
        "shards": summaries,
    }
    _write_json(root / "manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
