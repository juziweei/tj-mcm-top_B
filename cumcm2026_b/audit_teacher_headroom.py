"""Measure one-step privileged-teacher headroom in recorded Q4 decisions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def group_offsets(decision_ids: np.ndarray) -> np.ndarray:
    changes = np.flatnonzero(decision_ids[1:] != decision_ids[:-1]) + 1
    return np.concatenate(([0], changes, [len(decision_ids)])).astype(np.int64)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--split", default="test", choices=("train", "val", "test"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    decisions = 0
    agreement = 0
    missed_immediate_clears = 0
    chosen_clear_failures_with_available_success = 0
    equal_clear_decisions = 0
    information_regret = []
    chosen_duration = []
    teacher_duration = []
    candidate_counts = []

    for path in sorted((args.dataset / args.split).glob("part-*.npz")):
        with np.load(path, allow_pickle=False) as shard:
            decision_id = shard["decision_id"]
            chosen = shard["chosen"]
            clear_gain = shard["target_clear_gain"]
            information_gain = shard["target_information_gain"]
            duration = shard["target_duration_s"]
            q_variant = shard["q_variant"]
        for start, end in zip(
            group_offsets(decision_id)[:-1], group_offsets(decision_id)[1:]
        ):
            if int(q_variant[start]) != 4:
                continue
            local_chosen = np.flatnonzero(chosen[start:end])
            if len(local_chosen) != 1:
                raise ValueError(f"{path}: decision group has {len(local_chosen)} choices")
            chosen_index = int(local_chosen[0])
            candidates = range(end - start)
            teacher_index = max(
                candidates,
                key=lambda index: (
                    float(clear_gain[start + index]),
                    float(information_gain[start + index]),
                    -float(duration[start + index]),
                ),
            )
            decisions += 1
            agreement += chosen_index == teacher_index
            candidate_counts.append(end - start)
            chosen_clear = float(clear_gain[start + chosen_index])
            teacher_clear = float(clear_gain[start + teacher_index])
            if teacher_clear > chosen_clear:
                missed_immediate_clears += 1
                chosen_clear_failures_with_available_success += int(
                    chosen_clear == 0.0
                    and float(duration[start + chosen_index]) <= 5.0
                )
            if teacher_clear == chosen_clear:
                equal_clear_decisions += 1
                information_regret.append(
                    float(information_gain[start + teacher_index])
                    - float(information_gain[start + chosen_index])
                )
            chosen_duration.append(float(duration[start + chosen_index]))
            teacher_duration.append(float(duration[start + teacher_index]))

    if decisions == 0:
        raise RuntimeError("no Q4 decisions found")
    positive_information_regret = [value for value in information_regret if value > 1e-6]
    report = {
        "dataset": str(args.dataset.resolve()),
        "split": args.split,
        "q4_decisions": decisions,
        "teacher_action_agreement": agreement / decisions,
        "missed_immediate_clear_rate": missed_immediate_clears / decisions,
        "missed_immediate_clears": missed_immediate_clears,
        "short_chosen_action_when_clear_was_available": (
            chosen_clear_failures_with_available_success
        ),
        "equal_clear_decisions": equal_clear_decisions,
        "positive_information_regret_rate_with_equal_clear": (
            len(positive_information_regret) / max(1, equal_clear_decisions)
        ),
        "mean_positive_information_regret": (
            float(np.mean(positive_information_regret))
            if positive_information_regret
            else 0.0
        ),
        "mean_chosen_action_duration_s": float(np.mean(chosen_duration)),
        "mean_teacher_action_duration_s": float(np.mean(teacher_duration)),
        "mean_candidates_per_decision": float(np.mean(candidate_counts)),
        "interpretation_limit": (
            "One-step privileged labels establish available headroom, not "
            "closed-loop attainability from observable features."
        ),
    }
    encoded = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
    print(encoded)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
