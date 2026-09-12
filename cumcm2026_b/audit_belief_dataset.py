"""Fail-fast structural and label audit for belief-history datasets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def audit_split(path: Path) -> dict:
    loaded = np.load(path, allow_pickle=False)
    data = {name: loaded[name] for name in loaded.files}
    required = {
        "tokens", "virtual_time_s", "cleared_count_after",
        "known_unresolved_after", "remaining_count", "next_detection_s",
        "next_detection_event", "remaining_time_s", "offsets",
        "source_count", "cleared_count", "q_variant", "seed",
    }
    if missing := required.difference(data):
        raise AssertionError(f"{path.name}: missing arrays {sorted(missing)}")
    offsets = data["offsets"]
    rows = len(data["tokens"])
    episodes = len(offsets) - 1
    assert data["tokens"].shape == (rows, 22)
    assert offsets[0] == 0 and offsets[-1] == rows
    assert (np.diff(offsets) > 0).all()
    for name in (
        "virtual_time_s", "cleared_count_after", "known_unresolved_after",
        "remaining_count", "next_detection_s", "next_detection_event",
        "remaining_time_s",
    ):
        assert len(data[name]) == rows, (path.name, name)
    for name in ("source_count", "cleared_count", "q_variant", "seed"):
        assert len(data[name]) == episodes, (path.name, name)
    assert np.isfinite(data["tokens"]).all()
    assert np.isfinite(data["virtual_time_s"]).all()
    assert np.isfinite(data["next_detection_s"]).all()
    assert np.isfinite(data["remaining_time_s"]).all()
    assert (data["known_unresolved_after"] >= 0).all()
    assert (data["remaining_count"] >= 0).all()
    assert np.isin(data["q_variant"], (3, 4)).all()
    assert len(np.unique(data["seed"])) == episodes
    for episode, (start, end) in enumerate(zip(offsets[:-1], offsets[1:])):
        assert (np.diff(data["virtual_time_s"][start:end]) >= 0).all()
        expected_remaining = (
            data["source_count"][episode] - data["cleared_count_after"][start:end]
        )
        np.testing.assert_array_equal(data["remaining_count"][start:end], expected_remaining)
        assert data["remaining_count"][end - 1] == (
            data["source_count"][episode] - data["cleared_count"][episode]
        )
    unique, counts = np.unique(data["remaining_count"], return_counts=True)
    return {
        "episodes": episodes,
        "tokens": rows,
        "complete_rate": float(np.mean(data["source_count"] == data["cleared_count"])),
        "q3_episodes": int(np.sum(data["q_variant"] == 3)),
        "q4_episodes": int(np.sum(data["q_variant"] == 4)),
        "zero_label_fraction": float(np.mean(data["remaining_count"] == 0)),
        "remaining_count_histogram": dict(zip(map(str, unique.tolist()), counts.tolist())),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = {
        split: audit_split(args.data / f"{split}.npz")
        for split in ("train", "val", "test")
    }
    payload = json.dumps(report, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    print(payload)


if __name__ == "__main__":
    main()
