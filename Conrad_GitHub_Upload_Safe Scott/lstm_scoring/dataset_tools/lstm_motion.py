#!/usr/bin/env python3
"""Shared feature and model code for LSTM rehabilitation-motion comparison."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn


ANGLE_COLUMNS = (
    "left_hip_angle_deg",
    "left_knee_angle_deg",
    "left_thigh_tilt_deg",
    "left_shin_tilt_deg",
    "right_hip_angle_deg",
    "right_knee_angle_deg",
    "right_thigh_tilt_deg",
    "right_shin_tilt_deg",
)
INPUT_SIZE = len(ANGLE_COLUMNS) * 2  # eight angles plus eight visibility masks


def exercise_group(exercise: str) -> str:
    value = exercise.lower()
    if "sit_to_stand" in value:
        return "sit_to_stand"
    if "squat" in value:
        return "squat"
    if "heel_raise" in value:
        return "heel_raise"
    if "bridge" in value:
        return "bridge"
    if "lunge" in value:
        return "lunge"
    if "balance" in value:
        return "balance"
    if "quadriceps" in value:
        return value
    return value


def _fill_missing(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    filled = np.zeros_like(values, dtype=np.float32)
    mask = np.isfinite(values).astype(np.float32)
    indexes = np.arange(len(values), dtype=np.float32)
    for column in range(values.shape[1]):
        valid = np.isfinite(values[:, column])
        if valid.sum() >= 2:
            filled[:, column] = np.interp(indexes, indexes[valid], values[valid, column])
        elif valid.sum() == 1:
            filled[:, column] = float(values[valid, column][0])
    return filled, mask


def resample_features(angle_values: np.ndarray, sequence_length: int = 96) -> np.ndarray:
    if angle_values.ndim != 2 or angle_values.shape[1] != len(ANGLE_COLUMNS):
        raise ValueError(f"Expected N x {len(ANGLE_COLUMNS)} angle matrix")
    if len(angle_values) < 2:
        raise ValueError("At least two frames are required")
    filled, mask = _fill_missing(angle_values)
    filled = np.clip(filled / 180.0, 0.0, 1.0)
    source = np.linspace(0.0, 1.0, len(filled))
    target = np.linspace(0.0, 1.0, sequence_length)
    angle_output = np.column_stack([
        np.interp(target, source, filled[:, column]) for column in range(filled.shape[1])
    ])
    mask_output = np.column_stack([
        np.interp(target, source, mask[:, column]) for column in range(mask.shape[1])
    ])
    return np.concatenate([angle_output, mask_output], axis=1).astype(np.float32)


def load_angle_sequence(path: Path, sequence_length: int = 96) -> np.ndarray:
    rows: list[list[float]] = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            rows.append([
                float(row[column]) if row.get(column, "") not in {"", None} else np.nan
                for column in ANGLE_COLUMNS
            ])
    return resample_features(np.asarray(rows, dtype=np.float32), sequence_length)


def sequence_from_angle_data(data: dict[str, np.ndarray], sequence_length: int = 96) -> np.ndarray:
    matrix = np.column_stack([
        data["left_hip_angle"], data["left_knee_angle"],
        data["left_thigh_tilt"], data["left_shin_tilt"],
        data["right_hip_angle"], data["right_knee_angle"],
        data["right_thigh_tilt"], data["right_shin_tilt"],
    ])
    return resample_features(matrix, sequence_length)


def mirror_sequence(sequence: np.ndarray) -> np.ndarray:
    mirrored = sequence.copy()
    mirrored[:, 0:4] = sequence[:, 4:8]
    mirrored[:, 4:8] = sequence[:, 0:4]
    mirrored[:, 8:12] = sequence[:, 12:16]
    mirrored[:, 12:16] = sequence[:, 8:12]
    return mirrored


def augment_sequence(sequence: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    length = len(sequence)
    exponent = float(rng.uniform(0.82, 1.22))
    target = np.linspace(0.0, 1.0, length)
    warped = np.power(target, exponent)
    source = np.linspace(0.0, 1.0, length)
    output = np.column_stack([
        np.interp(warped, source, sequence[:, column]) for column in range(sequence.shape[1])
    ]).astype(np.float32)
    output[:, :8] += rng.normal(0.0, 1.8 / 180.0, size=(length, 8)).astype(np.float32)
    output[:, :8] = np.clip(output[:, :8], 0.0, 1.0)
    if rng.random() < 0.5:
        output = mirror_sequence(output)
    if rng.random() < 0.7:
        start = int(rng.integers(0, max(1, length - 8)))
        width = int(rng.integers(2, min(9, length - start + 1)))
        side = int(rng.integers(0, 2))
        angle_slice = slice(side * 4, side * 4 + 4)
        mask_slice = slice(8 + side * 4, 8 + side * 4 + 4)
        output[start:start + width, angle_slice] = 0.0
        output[start:start + width, mask_slice] = 0.0
    return output


class SiameseMotionLSTM(nn.Module):
    def __init__(
        self,
        input_size: int = INPUT_SIZE,
        hidden_size: int = 64,
        layers: int = 2,
        embedding_size: int = 32,
        dropout: float = 0.20,
    ) -> None:
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.layers = layers
        self.embedding_size = embedding_size
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=layers,
            batch_first=True,
            dropout=dropout if layers > 1 else 0.0,
            bidirectional=True,
        )
        self.projection = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, embedding_size),
        )

    def encode(self, sequence: torch.Tensor) -> torch.Tensor:
        _, (hidden, _) = self.lstm(sequence)
        forward = hidden[-2]
        backward = hidden[-1]
        embedding = self.projection(torch.cat([forward, backward], dim=1))
        return nn.functional.normalize(embedding, dim=1)

    def forward(self, first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
        return nn.functional.cosine_similarity(self.encode(first), self.encode(second))


def load_lstm_checkpoint(path: Path, device: str | torch.device = "cpu") -> tuple[SiameseMotionLSTM, dict[str, Any]]:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    config = checkpoint["model_config"]
    model = SiameseMotionLSTM(**config)
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()
    return model, checkpoint


@torch.no_grad()
def lstm_similarity(
    model: SiameseMotionLSTM,
    standard: np.ndarray,
    user: np.ndarray,
    calibration: dict[str, float],
    device: str | torch.device = "cpu",
) -> dict[str, float | bool]:
    standard_tensor = torch.from_numpy(standard).unsqueeze(0).to(device)
    user_variants = [user, mirror_sequence(user)]
    cosine_values: list[float] = []
    for variant in user_variants:
        variant_tensor = torch.from_numpy(variant).unsqueeze(0).to(device)
        cosine_values.append(float(model(standard_tensor, variant_tensor).item()))
    best_index = int(np.argmax(cosine_values))
    cosine = cosine_values[best_index]
    negative = float(calibration.get("negative_mean", 0.0))
    positive = float(calibration.get("positive_mean", 1.0))
    denominator = max(0.05, positive - negative)
    score = float(np.clip((cosine - negative) / denominator * 100.0, 0.0, 100.0))
    return {
        "score": score,
        "cosine_similarity": cosine,
        "mirrored_user_selected": bool(best_index == 1),
    }
