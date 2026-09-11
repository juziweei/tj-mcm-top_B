"""Decision-group-preserving readers for CUMCM B candidate datasets."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np


@dataclass(frozen=True)
class DecisionBatch:
    """A ragged batch represented by concatenated rows and group offsets."""

    features: np.ndarray
    chosen: np.ndarray
    target_clear_gain: np.ndarray
    target_information_gain: np.ndarray
    target_duration_s: np.ndarray
    q_variant: np.ndarray
    decision_id: np.ndarray
    group_offsets: np.ndarray

    @property
    def group_count(self) -> int:
        return len(self.group_offsets) - 1


def _group_boundaries(decision_ids: np.ndarray) -> np.ndarray:
    if decision_ids.ndim != 1 or len(decision_ids) == 0:
        raise ValueError("decision_id must be a non-empty one-dimensional array")
    changes = np.flatnonzero(decision_ids[1:] != decision_ids[:-1]) + 1
    offsets = np.concatenate(
        (np.asarray([0], dtype=np.int64), changes, np.asarray([len(decision_ids)], dtype=np.int64))
    )
    if len(np.unique(decision_ids)) != len(offsets) - 1:
        raise ValueError("decision groups are not contiguous within the shard")
    return offsets


def iter_decision_batches(
    dataset_root: str | Path,
    *,
    split: str = "train",
    decisions_per_batch: int = 256,
    shuffle: bool = True,
    seed: int = 2026,
) -> Iterator[DecisionBatch]:
    """Yield complete candidate groups without splitting a decision across batches."""

    if split not in {"train", "val", "test"}:
        raise ValueError("split must be train, val, or test")
    if decisions_per_batch <= 0:
        raise ValueError("decisions_per_batch must be positive")
    paths = sorted((Path(dataset_root) / split).glob("part-*.npz"))
    if not paths:
        raise FileNotFoundError(f"no shards found for split {split}")
    rng = np.random.default_rng(seed)
    if shuffle:
        rng.shuffle(paths)

    field_names = (
        "features",
        "chosen",
        "target_clear_gain",
        "target_information_gain",
        "target_duration_s",
        "q_variant",
        "decision_id",
    )
    for path in paths:
        with np.load(path, allow_pickle=False) as shard:
            arrays = {name: shard[name] for name in field_names}
        offsets = _group_boundaries(arrays["decision_id"])
        group_indices = np.arange(len(offsets) - 1)
        if shuffle:
            rng.shuffle(group_indices)
        for batch_start in range(0, len(group_indices), decisions_per_batch):
            selected_groups = group_indices[batch_start : batch_start + decisions_per_batch]
            row_indices = np.concatenate(
                [np.arange(offsets[index], offsets[index + 1]) for index in selected_groups]
            )
            lengths = np.asarray(
                [offsets[index + 1] - offsets[index] for index in selected_groups],
                dtype=np.int64,
            )
            batch_offsets = np.concatenate(
                (np.asarray([0], dtype=np.int64), np.cumsum(lengths))
            )
            batch = DecisionBatch(
                **{name: arrays[name][row_indices] for name in field_names},
                group_offsets=batch_offsets,
            )
            chosen_per_group = np.add.reduceat(
                batch.chosen.astype(np.int64), batch.group_offsets[:-1]
            )
            if np.any(chosen_per_group != 1):
                raise ValueError(f"{path}: a batch contains a non-unique chosen action")
            yield batch

