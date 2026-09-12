"""Measure cached online GRU inference without simulator or HTTP latency."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from cumcm_b.neural_belief import build_torch_model  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    device = torch.device("cpu")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = build_torch_model(int(checkpoint["hidden_size"])).eval()
    loaded = np.load(args.data / "test.npz", allow_pickle=False)
    tokens = loaded["tokens"]
    offsets = loaded["offsets"][: args.episodes + 1]

    warmup = torch.from_numpy(tokens[:32])
    hidden = None
    with torch.inference_mode():
        for token in warmup:
            *_, hidden = model.step(token, hidden)

    elapsed = 0.0
    measured_tokens = 0
    with torch.inference_mode():
        for start, end in zip(offsets[:-1], offsets[1:]):
            hidden = None
            begin = time.perf_counter()
            for token in torch.from_numpy(tokens[start:end]):
                *_, hidden = model.step(token, hidden)
            elapsed += time.perf_counter() - begin
            measured_tokens += int(end - start)
    report = {
        "device": "cpu_single_thread",
        "episodes": len(offsets) - 1,
        "tokens": measured_tokens,
        "wall_time_s": elapsed,
        "mean_ms_per_token": 1000.0 * elapsed / max(1, measured_tokens),
        "mean_ms_per_episode": 1000.0 * elapsed / max(1, len(offsets) - 1),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
