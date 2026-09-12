"""Evaluate calibration and counterfactual absence-stopping performance."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from cumcm_b.neural_belief import build_torch_model, time_bin_index  # noqa: E402


def calibration(y_true: np.ndarray, probability: np.ndarray, bins: int = 10):
    rows = []
    edges = np.linspace(0.0, 1.0, bins + 1)
    for index in range(bins):
        selected = (probability >= edges[index]) & (
            probability <= edges[index + 1]
            if index == bins - 1
            else probability < edges[index + 1]
        )
        if selected.any():
            rows.append({
                "lower": float(edges[index]),
                "upper": float(edges[index + 1]),
                "samples": int(selected.sum()),
                "mean_probability": float(probability[selected].mean()),
                "empirical_complete": float(y_true[selected].mean()),
            })
    return rows


@torch.inference_mode()
def infer_episodes(model, tokens: np.ndarray, offsets: np.ndarray, device):
    completion_outputs = []
    hazard_outputs = []
    quantile_outputs = []
    elapsed = 0.0
    for start, end in zip(offsets[:-1], offsets[1:]):
        sequence = torch.from_numpy(tokens[start:end]).unsqueeze(0).to(device)
        begin = time.perf_counter()
        count_logits, hazards, quantiles = model(sequence)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed += time.perf_counter() - begin
        completion_outputs.append(
            torch.softmax(count_logits[0], dim=-1)[:, 0].cpu().numpy()
        )
        hazard_outputs.append(hazards[0].cpu().numpy())
        quantile_outputs.append(quantiles[0].cpu().numpy())
    return completion_outputs, hazard_outputs, quantile_outputs, elapsed


def time_head_metrics(data, hazards: np.ndarray, quantiles: np.ndarray):
    bins = time_bin_index(data["next_detection_s"])
    indices = np.arange(hazards.shape[1])[None, :]
    before = indices < bins[:, None]
    at_event = (
        (indices == bins[:, None])
        & data["next_detection_event"].astype(bool)[:, None]
    )
    censored_at_bin = (
        (indices == bins[:, None])
        & ~data["next_detection_event"].astype(bool)[:, None]
    )
    no_event_loss = np.logaddexp(0.0, hazards)
    event_loss = np.logaddexp(0.0, -hazards)
    hazard_nll = (no_event_loss * (before | censored_at_bin) + event_loss * at_event).sum(1)
    predicted_seconds = np.expm1(np.clip(quantiles, 0.0, 20.0))
    target = data["remaining_time_s"]
    return {
        "next_detection_discrete_nll": float(hazard_nll.mean()),
        "remaining_time_median_mae_s": float(
            np.mean(np.abs(predicted_seconds[:, 1] - target))
        ),
        "remaining_time_10_90_coverage": float(np.mean(
            (target >= predicted_seconds[:, 0]) & (target <= predicted_seconds[:, 2])
        )),
    }


def stopping_metrics(data, probabilities, threshold: float, q_variant: int | None):
    total_sources = 0
    total_cleared = 0
    total_time = 0.0
    baseline_time = 0.0
    complete = 0
    selected_episodes = 0
    learned_stops = 0
    for episode, (start, end) in enumerate(zip(data["offsets"][:-1], data["offsets"][1:])):
        if q_variant is not None and int(data["q_variant"][episode]) != q_variant:
            continue
        selected_episodes += 1
        source_count = int(data["source_count"][episode])
        cleared = data["cleared_count_after"][start:end]
        unresolved = data["known_unresolved_after"][start:end]
        times = data["virtual_time_s"][start:end]
        admissible = np.flatnonzero(
            (cleared >= 10) & (unresolved == 0) & (probabilities[episode] >= threshold)
        )
        stop = int(admissible[0]) if len(admissible) else len(times) - 1
        learned_stops += int(len(admissible) > 0)
        cleared_at_stop = int(cleared[stop])
        total_sources += source_count
        total_cleared += cleared_at_stop
        total_time += float(times[stop])
        baseline_time += float(times[-1])
        complete += int(cleared_at_stop == source_count)
    return {
        "episodes": selected_episodes,
        "threshold": threshold,
        "complete_case_rate": complete / max(1, selected_episodes),
        "source_clear_rate": total_cleared / max(1, total_sources),
        "mean_time_s_per_source": total_time / max(1, total_sources),
        "baseline_mean_time_s_per_source": baseline_time / max(1, total_sources),
        "relative_time_reduction": (baseline_time - total_time) / max(1.0, baseline_time),
        "learned_stop_rate": learned_stops / max(1, selected_episodes),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "val", "test"), default="test")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument(
        "--thresholds", type=float, nargs="+",
        default=(0.5, 0.8, 0.9, 0.95, 0.97, 0.99, 0.995, 0.998),
    )
    args = parser.parse_args()
    device = torch.device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = build_torch_model(int(checkpoint["hidden_size"])).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    loaded = np.load(args.data / f"{args.split}.npz", allow_pickle=False)
    data = {name: loaded[name] for name in loaded.files}
    probabilities, hazards, quantiles, inference_s = infer_episodes(
        model, data["tokens"], data["offsets"], device
    )
    flat_probability = np.concatenate(probabilities)
    flat_hazards = np.concatenate(hazards)
    flat_quantiles = np.concatenate(quantiles)
    y_true = data["remaining_count"] == 0
    report = {
        "device": str(device),
        "split": args.split,
        "episodes": len(probabilities),
        "tokens": int(len(flat_probability)),
        "inference_wall_s": inference_s,
        "mean_inference_ms_per_episode": 1000.0 * inference_s / max(1, len(probabilities)),
        "completion_brier": float(np.mean((flat_probability - y_true) ** 2)),
        "completion_calibration": calibration(y_true, flat_probability),
        "time_heads": time_head_metrics(data, flat_hazards, flat_quantiles),
        "frontier": {
            str(q): [stopping_metrics(data, probabilities, threshold, q) for threshold in args.thresholds]
            for q in (3, 4)
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
