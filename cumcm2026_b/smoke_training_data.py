"""Smoke-test decision-group batching on a generated dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from cumcm_b.training_data import iter_decision_batches  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset")
    parser.add_argument("--split", choices=("train", "val", "test"), default="train")
    parser.add_argument("--batches", type=int, default=10)
    parser.add_argument("--decisions-per-batch", type=int, default=256)
    args = parser.parse_args()
    groups = rows = 0
    q_variants: set[int] = set()
    maximum_rows = 0
    for batch_index, batch in enumerate(
        iter_decision_batches(
            args.dataset,
            split=args.split,
            decisions_per_batch=args.decisions_per_batch,
        )
    ):
        if batch_index >= args.batches:
            break
        if not np.isfinite(batch.features).all():
            raise RuntimeError("non-finite training feature encountered")
        groups += batch.group_count
        rows += len(batch.features)
        maximum_rows = max(maximum_rows, len(batch.features))
        q_variants.update(map(int, np.unique(batch.q_variant)))
    report = {
        "status": "PASS",
        "split": args.split,
        "batches": min(args.batches, batch_index + 1),
        "decision_groups": groups,
        "candidate_rows": rows,
        "feature_count": int(batch.features.shape[1]),
        "q_variants_seen": sorted(q_variants),
        "maximum_candidate_rows_in_batch": maximum_rows,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

