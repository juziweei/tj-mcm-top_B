"""Train the history-conditioned count, survival, and remaining-time model."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import random
import sys

import numpy as np
import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from cumcm_b.neural_belief import HAZARD_BIN_EDGES_S, build_torch_model, time_bin_index  # noqa: E402


class Episodes(Dataset):
    def __init__(self, path: Path):
        data = np.load(path, allow_pickle=False)
        self.data = {name: data[name] for name in data.files}
    def __len__(self): return len(self.data["offsets"]) - 1
    def __getitem__(self, index):
        a, b = self.data["offsets"][index:index + 2]
        names = ("tokens", "remaining_count", "next_detection_s", "next_detection_event", "remaining_time_s")
        return tuple(torch.from_numpy(self.data[name][a:b]) for name in names)


def collate(rows):
    lengths = torch.tensor([len(row[0]) for row in rows])
    values = [pad_sequence([row[i] for row in rows], batch_first=True) for i in range(5)]
    mask = torch.arange(values[0].shape[1])[None, :] < lengths[:, None]
    return (*values, mask)


def hazard_loss(logits, durations, events, mask):
    bins = torch.from_numpy(time_bin_index(durations.detach().cpu().numpy())).to(logits.device)
    indices = torch.arange(logits.shape[-1], device=logits.device)
    before = indices < bins[..., None]
    at_event = (indices == bins[..., None]) & events.bool()[..., None]
    observed_no_event = before | ((indices == bins[..., None]) & ~events.bool()[..., None])
    losses = nn.functional.softplus(logits) * observed_no_event + nn.functional.softplus(-logits) * at_event
    return losses[mask].sum() / mask.sum().clamp_min(1)


def quantile_loss(prediction, target, mask):
    quantiles = torch.tensor([0.1, 0.5, 0.9], device=prediction.device)
    error = torch.log1p(target)[..., None] - prediction
    loss = torch.maximum(quantiles * error, (quantiles - 1.0) * error)
    return loss[mask].mean()


def run_epoch(model, loader, optimizer, device):
    training = optimizer is not None
    model.train(training)
    totals = {
        "loss": 0.0,
        "count_mae": 0.0,
        "completion_accuracy": 0.0,
        "completion_brier": 0.0,
        "completion_true_rate": 0.0,
        "batches": 0,
    }
    for tokens, count, duration, event, remaining_time, mask in loader:
        tokens, count, duration, event, remaining_time, mask = [x.to(device) for x in (tokens, count, duration, event, remaining_time, mask)]
        with torch.set_grad_enabled(training):
            count_logits, hazards, quantiles = model(tokens)
            count_loss = nn.functional.cross_entropy(count_logits[mask], count.long()[mask])
            loss = count_loss + 0.35 * hazard_loss(hazards, duration, event, mask) + 0.25 * quantile_loss(quantiles, remaining_time, mask)
            if training:
                optimizer.zero_grad(set_to_none=True); loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 2.0); optimizer.step()
        predicted = count_logits.argmax(-1)
        probability_complete = torch.softmax(count_logits, dim=-1)[..., 0]
        completion = count == 0
        totals["loss"] += loss.detach().item()
        totals["count_mae"] += (predicted[mask] - count[mask]).abs().float().mean().item()
        totals["completion_accuracy"] += (
            (probability_complete[mask] >= 0.5) == completion[mask]
        ).float().mean().item()
        totals["completion_brier"] += (
            probability_complete[mask] - completion[mask].float()
        ).square().mean().item()
        totals["completion_true_rate"] += completion[mask].float().mean().item()
        totals["batches"] += 1
    n = totals.pop("batches")
    return {key: value / max(1, n) for key, value in totals.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=20260911)
    parser.add_argument("--loader-workers", type=int, default=4)
    args = parser.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    generator = torch.Generator().manual_seed(args.seed)
    train = DataLoader(
        Episodes(args.data / "train.npz"), batch_size=args.batch_size,
        shuffle=True, collate_fn=collate, num_workers=args.loader_workers,
        pin_memory=device.type == "cuda", generator=generator,
    )
    val = DataLoader(
        Episodes(args.data / "val.npz"), batch_size=args.batch_size,
        collate_fn=collate, num_workers=max(0, args.loader_workers // 2),
        pin_memory=device.type == "cuda",
    )
    model = build_torch_model(args.hidden_size).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    args.output.mkdir(parents=True, exist_ok=True)
    history = []
    best = math.inf
    for epoch in range(1, args.epochs + 1):
        row = {"epoch": epoch, "train": run_epoch(model, train, optimizer, device), "val": run_epoch(model, val, None, device)}
        history.append(row); print(json.dumps(row), flush=True)
        if row["val"]["completion_brier"] < best:
            best = row["val"]["completion_brier"]
            torch.save({"model": model.state_dict(), "hidden_size": args.hidden_size,
                        "hazard_edges_s": HAZARD_BIN_EDGES_S.tolist(), "metrics": row,
                        "training_config": vars(args)}, args.output / "best.pt")
    (args.output / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
