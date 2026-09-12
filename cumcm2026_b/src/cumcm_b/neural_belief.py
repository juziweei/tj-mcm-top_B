"""History-conditioned belief and time-to-event model for Q3/Q4.

Hidden simulator state is permitted only in labels.  Every input token is
constructed from an action and its returned observation, so the same encoder
can be used against the official HTTP interface.
"""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np


RESULT_NAMES = (
    "no_signal",
    "direction",
    "near",
    "clear_success",
    "clear_failure",
)

TOKEN_FEATURE_NAMES = (
    "target_x", "target_y", "delta_x", "delta_y",
    "channel_norm", "channel_sin", "channel_cos",
    "is_measure", "is_clear", "channel_switched",
    *(f"result_{name}" for name in RESULT_NAMES),
    "bearing_present", "bearing_sin", "bearing_cos",
    "log_action_duration", "log_virtual_time", "cleared_fraction",
    "is_q4",
)

HAZARD_BIN_EDGES_S = np.asarray(
    [30.0, 60.0, 120.0, 240.0, 480.0, 960.0, 1920.0, 3840.0, 7680.0],
    dtype=np.float32,
)


def observation_token(
    *,
    previous_position: tuple[float, float],
    target: tuple[float, float],
    channel: int,
    is_measure: bool,
    switched: bool,
    result: str,
    bearing_deg: float | None,
    action_duration_s: float,
    virtual_time_s: float,
    cleared_count: int,
    q_variant: int,
) -> np.ndarray:
    """Encode one deployable action/observation pair as a numeric token."""

    if result not in RESULT_NAMES:
        raise ValueError(f"unknown observation result: {result!r}")
    if not 1 <= channel <= 20:
        raise ValueError("channel must lie in 1..20")
    angle = 2.0 * math.pi * (channel - 1) / 20.0
    bearing = math.radians(bearing_deg or 0.0)
    result_one_hot = [float(result == name) for name in RESULT_NAMES]
    values = [
        target[0] / 2800.0,
        target[1] / 2800.0,
        (target[0] - previous_position[0]) / 2800.0,
        (target[1] - previous_position[1]) / 2800.0,
        (channel - 1) / 19.0,
        math.sin(angle),
        math.cos(angle),
        float(is_measure),
        float(not is_measure),
        float(switched),
        *result_one_hot,
        float(bearing_deg is not None),
        math.sin(bearing) if bearing_deg is not None else 0.0,
        math.cos(bearing) if bearing_deg is not None else 0.0,
        math.log1p(max(0.0, action_duration_s)) / math.log(1001.0),
        math.log1p(max(0.0, virtual_time_s)) / math.log(100001.0),
        cleared_count / 16.0,
        float(q_variant == 4),
    ]
    token = np.asarray(values, dtype=np.float32)
    if token.shape != (len(TOKEN_FEATURE_NAMES),):
        raise AssertionError("token schema and encoder diverged")
    return token


def time_bin_index(duration_s: np.ndarray | Sequence[float]) -> np.ndarray:
    """Map durations to discrete hazard bins, including one overflow bin."""

    return np.searchsorted(
        HAZARD_BIN_EDGES_S, np.asarray(duration_s, dtype=np.float32), side="right"
    ).astype(np.int64)


def build_torch_model(hidden_size: int = 128):
    """Construct the PyTorch model lazily so geometry-only clients need no torch."""

    import torch
    from torch import nn

    class BeliefValueGRU(nn.Module):
        def __init__(self):
            super().__init__()
            self.input = nn.Sequential(
                nn.Linear(len(TOKEN_FEATURE_NAMES), hidden_size),
                nn.SiLU(),
                nn.LayerNorm(hidden_size),
            )
            self.memory = nn.GRU(hidden_size, hidden_size, batch_first=True)
            self.norm = nn.LayerNorm(hidden_size)
            self.remaining_count = nn.Linear(hidden_size, 17)
            self.next_detection_hazard = nn.Linear(
                hidden_size, len(HAZARD_BIN_EDGES_S) + 1
            )
            self.remaining_time_quantiles = nn.Linear(hidden_size, 3)

        def forward(self, tokens, lengths=None):
            encoded = self.input(tokens)
            memory, _ = self.memory(encoded)
            memory = self.norm(memory)
            return (
                self.remaining_count(memory),
                self.next_detection_hazard(memory),
                self.remaining_time_quantiles(memory),
            )

        def step(self, token, hidden=None):
            """Advance one online observation while retaining recurrent memory."""

            if token.ndim == 1:
                token = token[None, None, :]
            elif token.ndim == 2:
                token = token[:, None, :]
            encoded = self.input(token)
            memory, hidden = self.memory(encoded, hidden)
            memory = self.norm(memory)
            return (
                self.remaining_count(memory[:, -1]),
                self.next_detection_hazard(memory[:, -1]),
                self.remaining_time_quantiles(memory[:, -1]),
                hidden,
            )

    return BeliefValueGRU()
