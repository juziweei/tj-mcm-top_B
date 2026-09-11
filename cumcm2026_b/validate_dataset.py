"""Independent structural and leakage-oriented validation for generated shards."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset")
    args = parser.parse_args()
    root = Path(args.dataset).resolve()
    schema = json.loads((root / "schema.json").read_text(encoding="utf-8"))
    feature_count = len(schema["feature_names"])

    errors: list[str] = []
    split_episodes: dict[str, set[int]] = {"train": set(), "val": set(), "test": set()}
    total_rows = 0
    total_decisions = 0
    q_counts = {3: 0, 4: 0}
    result_counts = {i: 0 for i in range(6)}
    per_split: dict[str, dict[str, int]] = {}
    episode_index_rows: dict[str, int] = {}
    completed_episode_counts: dict[str, int] = {}

    for split in ("train", "val", "test"):
        rows = decisions = observations = 0
        for path in sorted((root / split).glob("part-*.npz")):
            with np.load(path, allow_pickle=False) as data:
                n = len(data["features"])
                if data["features"].shape != (n, feature_count):
                    errors.append(f"{path}: bad feature shape {data['features'].shape}")
                for name in (
                    "decision_id", "episode_id", "step", "candidate_index", "chosen",
                    "target_clear_gain", "target_information_gain", "target_duration_s", "q_variant",
                ):
                    if len(data[name]) != n:
                        errors.append(f"{path}: {name} length mismatch")
                if not np.isfinite(data["features"]).all():
                    errors.append(f"{path}: non-finite feature")
                if not np.isfinite(data["target_information_gain"]).all():
                    errors.append(f"{path}: non-finite information target")
                if not np.isfinite(data["target_duration_s"]).all():
                    errors.append(f"{path}: non-finite duration target")
                cleared_progress = data["features"][:, 4]
                if np.any(cleared_progress < 0.0) or np.any(cleared_progress > 1.0):
                    errors.append(f"{path}: invalid observable cleared progress")
                if np.any(np.abs(cleared_progress * 16.0 - np.rint(cleared_progress * 16.0)) > 1e-5):
                    errors.append(f"{path}: cleared progress leaks a non-public denominator")
                q_values_for_rows = data["q_variant"]
                directional_feature = data["features"][:, 8]
                if np.any(np.abs(directional_feature[q_values_for_rows == 3]) > 1e-6):
                    errors.append(f"{path}: Q3 directional model prior must be zero")
                if np.any(np.abs(directional_feature[q_values_for_rows == 4] - 0.5) > 1e-6):
                    errors.append(f"{path}: Q4 input leaks episode-specific directionality")
                decision_ids = data["decision_id"]
                unique_decisions, first = np.unique(decision_ids, return_index=True)
                chosen_sum = np.add.reduceat(data["chosen"].astype(np.int64), first)
                if np.any(chosen_sum != 1):
                    errors.append(f"{path}: {int(np.sum(chosen_sum != 1))} decision groups do not have one expert action")
                episodes_here = set(map(int, np.unique(data["episode_id"])))
                split_episodes[split].update(episodes_here)
                for variant in (3, 4):
                    q_counts[variant] += int(np.sum(data["q_variant"] == variant))
                obs = data["observations"]
                if obs.ndim != 2 or obs.shape[1] != 12:
                    errors.append(f"{path}: bad observation shape {obs.shape}")
                else:
                    for code in range(6):
                        result_counts[code] += int(np.sum(obs[:, 7] == code))
                rows += n
                decisions += len(unique_decisions)
                observations += len(obs)
        per_split[split] = {"candidate_rows": rows, "decisions": decisions, "observations": observations, "episodes": len(split_episodes[split])}
        total_rows += rows
        total_decisions += decisions
        episode_index_path = root / split / "episodes.npz"
        if not episode_index_path.exists():
            errors.append(f"{split}: missing episodes.npz")
            episode_index_rows[split] = 0
            completed_episode_counts[split] = 0
        else:
            with np.load(episode_index_path, allow_pickle=False) as episode_data:
                values = episode_data["values"]
                episode_index_rows[split] = len(values)
                if values.ndim != 2 or values.shape[1] != 10:
                    errors.append(f"{split}: bad episode index shape {values.shape}")
                    completed_episode_counts[split] = 0
                else:
                    completed_episode_counts[split] = int(np.sum(values[:, 9]))
                    index_ids = values[:, 0].astype(np.int64)
                    if len(np.unique(index_ids)) != len(index_ids):
                        errors.append(f"{split}: duplicate episode ids in episode index")
                    if set(map(int, index_ids)) != split_episodes[split]:
                        errors.append(f"{split}: episode index does not match shards")
                    if np.any(values[:, 4] < 10) or np.any(values[:, 4] > 16):
                        errors.append(f"{split}: source count outside [10,16]")
                    if np.any(values[:, 5] < 0) or np.any(values[:, 5] > values[:, 4]):
                        errors.append(f"{split}: invalid cleared count")
                    if np.any(values[:, 6] <= 0) or np.any(values[:, 7] <= 0):
                        errors.append(f"{split}: empty trajectory in episode index")

    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        overlap = split_episodes[left] & split_episodes[right]
        if overlap:
            errors.append(f"episode leakage between {left} and {right}: {len(overlap)}")
    manifest_path = root / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        config = manifest["config"]
        contract = manifest.get("physics_contract", {})
        if contract.get("directional_total_coverage_deg") != 180.0:
            errors.append("physics contract must use 180-degree total directional coverage")
        if contract.get("directional_boundary_inclusive") is not True:
            errors.append("physics contract must include directional sector boundaries")
        expected_minimum = int(config["rows_per_shard"]) * (
            int(config["train_shards"]) + int(config["val_shards"]) + int(config["test_shards"])
        )
    else:
        expected_minimum = 2_000_000
    if total_rows < expected_minimum:
        errors.append(f"candidate row target missed: {total_rows} < {expected_minimum}")
    if q_counts[3] == 0 or q_counts[4] == 0:
        errors.append("both Q3 and Q4 must be represented")
    incomplete_episodes = sum(
        episode_index_rows.get(split, 0) - completed_episode_counts.get(split, 0)
        for split in ("train", "val", "test")
    )
    if incomplete_episodes:
        errors.append(f"dataset contains {incomplete_episodes} incomplete controller episodes")

    report = {
        "validated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS" if not errors else "FAIL",
        "errors": errors,
        "total_candidate_rows": total_rows,
        "total_decisions": total_decisions,
        "feature_count": feature_count,
        "expected_minimum_candidate_rows": expected_minimum,
        "per_split": per_split,
        "episode_index_rows": episode_index_rows,
        "incomplete_episodes": incomplete_episodes,
        "q_variant_candidate_rows": q_counts,
        "observation_result_counts": result_counts,
        "split_episode_overlap": {
            "train_val": len(split_episodes["train"] & split_episodes["val"]),
            "train_test": len(split_episodes["train"] & split_episodes["test"]),
            "val_test": len(split_episodes["val"] & split_episodes["test"]),
        },
    }
    (root / "validation_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
