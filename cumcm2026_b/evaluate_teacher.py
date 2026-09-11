"""Out-of-sample stress evaluation without writing candidate feature rows."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from cumcm_b.dataset import generate_episode


def _evaluate_one(arguments: tuple[int, int, int]) -> dict[str, float | int]:
    seed, q_variant, max_steps = arguments
    episode = generate_episode(
        seed=seed,
        episode_id=seed,
        decision_id_start=seed * 20_000,
        max_steps=max_steps,
        q_variant_override=q_variant,
        record_arrays=False,
    )
    return episode.summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes-per-variant", type=int, default=200)
    parser.add_argument("--base-seed", type=int, default=40_000_000)
    parser.add_argument("--max-steps", type=int, default=12_000)
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    jobs = [
        (args.base_seed + variant * 1_000_000 + i, variant, args.max_steps)
        for variant in (3, 4)
        for i in range(args.episodes_per_variant)
    ]
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        summaries = list(executor.map(_evaluate_one, jobs, chunksize=1))

    by_variant = {}
    for variant in (3, 4):
        items = [item for item in summaries if int(item["q_variant"]) == variant]
        sources = sum(int(item["source_count"]) for item in items)
        cleared = sum(int(item["cleared_count"]) for item in items)
        completed = sum(item["source_count"] == item["cleared_count"] for item in items)
        decisions = np.asarray([item["decisions"] for item in items], dtype=np.float64)
        virtual_time = np.asarray([item["virtual_time_s"] for item in items], dtype=np.float64)
        by_variant[str(variant)] = {
            "episodes": len(items),
            "completed_episodes": completed,
            "episode_success_rate": completed / max(len(items), 1),
            "sources": sources,
            "cleared_sources": cleared,
            "source_clear_rate": cleared / max(sources, 1),
            "decisions_mean": float(np.mean(decisions)),
            "decisions_p95": float(np.quantile(decisions, 0.95)),
            "decisions_max": int(np.max(decisions)),
            "virtual_time_mean_s": float(np.mean(virtual_time)),
            "virtual_time_p95_s": float(np.quantile(virtual_time, 0.95)),
            "virtual_time_max_s": float(np.max(virtual_time)),
            "within_100h_rate": float(np.mean(virtual_time <= 360_000.0)),
            "failure_seeds": [
                int(item["seed"])
                for item in items
                if item["source_count"] != item["cleared_count"]
            ],
        }
    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "official_simulator_data": False,
        "seed_policy": "held-out namespace, not used for training, validation, test, or tuning",
        "config": vars(args),
        "by_q_variant": by_variant,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
