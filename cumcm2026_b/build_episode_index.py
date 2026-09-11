"""Build compact per-trajectory metadata from immutable candidate shards."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random

import numpy as np


EPISODE_COLUMNS = (
    "episode_id", "seed", "q_variant", "directional_prior", "source_count",
    "cleared_count", "decisions", "candidate_rows", "virtual_time_s", "completed",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset")
    args = parser.parse_args()
    root = Path(args.dataset).resolve()
    all_counts = {}
    for split in ("train", "val", "test"):
        rows: list[list[float]] = []
        for path in sorted((root / split).glob("part-*.npz")):
            with np.load(path, allow_pickle=False) as data:
                ids, first, candidate_counts = np.unique(
                    data["episode_id"], return_index=True, return_counts=True
                )
                obs = data["observations"]
                obs_ids = obs[:, 1].astype(np.int64)
                for episode_id, row_index, candidate_rows in zip(ids, first, candidate_counts):
                    episode_id = int(episode_id)
                    episode_obs = obs[obs_ids == episode_id]
                    source_count = random.Random(episode_id).randint(10, 16)
                    cleared_count = int(np.sum(episode_obs[:, 11]))
                    rows.append(
                        [
                            episode_id,
                            episode_id,
                            int(data["q_variant"][row_index]),
                            float(data["features"][row_index, 8]),
                            source_count,
                            cleared_count,
                            len(episode_obs),
                            int(candidate_rows),
                            float(episode_obs[-1, 10]),
                            int(cleared_count == source_count),
                        ]
                    )
        values = np.asarray(rows, dtype=np.float64).reshape(-1, len(EPISODE_COLUMNS))
        np.savez(root / split / "episodes.npz", values=values)
        all_counts[split] = len(values)

    schema_path = root / "schema.json"
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    schema["episode_index"] = {
        "path": "<split>/episodes.npz",
        "array": "values",
        "columns": list(EPISODE_COLUMNS),
    }
    schema_path.write_text(json.dumps(schema, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"status": "PASS", "episode_index_rows": all_counts}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
