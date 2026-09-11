"""Create a compact statistical profile of a generated dataset."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset")
    args = parser.parse_args()
    root = Path(args.dataset).resolve()
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))

    total_sources = sum(int(s["sources"]) for s in manifest["shards"])
    total_cleared = sum(int(s["cleared"]) for s in manifest["shards"])
    q_rows = Counter()
    result_counts = Counter()
    action_rows = Counter()
    expert_action_rows = Counter()
    episode_decisions: Counter[int] = Counter()
    episode_rows: list[np.ndarray] = []
    target_samples: dict[str, list[np.ndarray]] = {
        "clear_gain": [], "information_gain": [], "duration_s": []
    }

    for split in ("train", "val", "test"):
        with np.load(root / split / "episodes.npz", allow_pickle=False) as episode_data:
            episode_rows.append(episode_data["values"])
        for path in sorted((root / split).glob("part-*.npz")):
            with np.load(path, allow_pickle=False) as data:
                q_values, counts = np.unique(data["q_variant"], return_counts=True)
                q_rows.update({int(k): int(v) for k, v in zip(q_values, counts)})
                is_clear = data["features"][:, 30] > 0.5
                action_rows["clear"] += int(np.sum(is_clear))
                action_rows["measure"] += int(np.sum(~is_clear))
                selected = data["chosen"] > 0
                expert_action_rows["clear"] += int(np.sum(is_clear & selected))
                expert_action_rows["measure"] += int(np.sum((~is_clear) & selected))
                obs = data["observations"]
                codes, code_counts = np.unique(obs[:, 7].astype(np.int64), return_counts=True)
                result_counts.update({int(k): int(v) for k, v in zip(codes, code_counts)})
                episodes, decision_counts = np.unique(obs[:, 1].astype(np.int64), return_counts=True)
                episode_decisions.update({int(k): int(v) for k, v in zip(episodes, decision_counts)})
                # A deterministic stride keeps the profile fast while retaining
                # points from every shard and split.
                stride = max(1, len(selected) // 10_000)
                target_samples["clear_gain"].append(data["target_clear_gain"][::stride])
                target_samples["information_gain"].append(data["target_information_gain"][::stride])
                target_samples["duration_s"].append(data["target_duration_s"][::stride])

    quantiles = {}
    for name, arrays in target_samples.items():
        values = np.concatenate(arrays).astype(np.float64)
        quantiles[name] = {
            "mean": float(np.mean(values)),
            "p05": float(np.quantile(values, 0.05)),
            "p50": float(np.quantile(values, 0.50)),
            "p95": float(np.quantile(values, 0.95)),
            "max": float(np.max(values)),
        }
    lengths = np.asarray(list(episode_decisions.values()), dtype=np.float64)
    episode_values = np.concatenate(episode_rows)
    by_variant = {}
    for variant in (3, 4):
        subset = episode_values[episode_values[:, 2] == variant]
        by_variant[str(variant)] = {
            "episodes": len(subset),
            "sources": int(np.sum(subset[:, 4])),
            "cleared_sources": int(np.sum(subset[:, 5])),
            "aggregate_cleared_fraction": float(np.sum(subset[:, 5]) / np.sum(subset[:, 4])),
            "complete_episodes": int(np.sum(subset[:, 9])),
            "episode_completion_rate": float(np.mean(subset[:, 9])),
            "mean_decisions": float(np.mean(subset[:, 6])),
            "max_step_limited_episodes": int(
                np.sum(subset[:, 6] >= int(manifest["config"]["max_steps"]))
            ),
        }
    total_rows = int(manifest["total_candidate_rows"])
    total_decisions = int(manifest["total_decisions"])
    stats = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": str(root),
        "official_simulator_data": False,
        "candidate_rows": total_rows,
        "decisions": total_decisions,
        "episodes": int(manifest["total_episodes"]),
        "candidate_rows_per_decision": total_rows / max(total_decisions, 1),
        "sources": total_sources,
        "cleared_sources": total_cleared,
        "aggregate_cleared_fraction": total_cleared / max(total_sources, 1),
        "by_q_variant": by_variant,
        "q_variant_candidate_rows": dict(sorted(q_rows.items())),
        "candidate_action_rows": dict(action_rows),
        "expert_action_decisions": dict(expert_action_rows),
        "observation_result_counts": dict(sorted(result_counts.items())),
        "episode_decision_length": {
            "mean": float(np.mean(lengths)),
            "p50": float(np.quantile(lengths, 0.50)),
            "p90": float(np.quantile(lengths, 0.90)),
            "max": int(np.max(lengths)),
        },
        "target_sample_quantiles": quantiles,
        "interpretation_notes": [
            "Q3/Q4 row balance follows controller trajectory length and candidate-set size, not a row-level resampling target.",
            "Failed clear actions are intentional Q4 boundary-recovery probes from the deployable controller.",
            "Statistics describe surrogate rule-based simulation and must be recalibrated with official simulator logs.",
        ],
    }
    output = root / "statistics.json"
    output.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
