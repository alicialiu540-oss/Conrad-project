#!/usr/bin/env python3
"""Train a temporal convolutional autoencoder on a reference angle sequence."""

from __future__ import annotations

import argparse
import csv
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from lstm_movement_quality import make_windows, prepare_features, read_angle_csv
from train_landmark_tcn_classifier import TemporalBlock


@dataclass
class Config:
    sequence_length: int = 64
    stride: int = 16
    train_fraction: float = 0.70
    min_coverage: float = 0.60
    channels: tuple[int, ...] = (32, 32, 32)
    kernel_size: int = 3
    dropout: float = 0.15
    epochs: int = 120
    patience: int = 15
    batch_size: int = 16
    learning_rate: float = 1e-3
    threshold_percentile: float = 95.0
    seed: int = 42


class TCNAutoencoder(nn.Module):
    def __init__(
        self,
        input_features: int,
        channels: tuple[int, ...],
        kernel_size: int,
        dropout: float,
    ) -> None:
        super().__init__()
        blocks = []
        current = input_features
        for level, channel_count in enumerate(channels):
            blocks.append(
                TemporalBlock(current, channel_count, kernel_size, 2**level, dropout)
            )
            current = channel_count
        self.temporal = nn.Sequential(*blocks)
        self.output = nn.Conv1d(current, input_features, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        encoded = self.temporal(x.transpose(1, 2))
        return self.output(encoded).transpose(1, 2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference_csv", type=Path)
    parser.add_argument("movement_csv", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("results/tcn-reference-deviation"))
    parser.add_argument("--sequence-length", type=int, default=64)
    parser.add_argument("--stride", type=int, default=16)
    parser.add_argument("--channels", default="32,32,32")
    parser.add_argument("--epochs", type=int, default=120)
    return parser.parse_args()


def evaluate(model: nn.Module, windows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    with torch.no_grad():
        source = torch.from_numpy(windows.astype(np.float32))
        reconstructed = model(source).cpu().numpy()
    errors = np.mean((reconstructed - windows) ** 2, axis=(1, 2))
    return errors, reconstructed


def main() -> None:
    args = parse_args()
    channels = tuple(int(value) for value in args.channels.split(",") if value.strip())
    config = Config(
        sequence_length=args.sequence_length,
        stride=args.stride,
        channels=channels,
        epochs=args.epochs,
    )
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.set_num_threads(min(4, max(1, torch.get_num_threads())))
    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)

    ref_time, ref_columns = read_angle_csv(args.reference_csv.expanduser().resolve())
    movement_time, movement_columns = read_angle_csv(args.movement_csv.expanduser().resolve())
    common = sorted(set(ref_columns) & set(movement_columns))
    coverage = {
        name: {
            "reference": float(np.isfinite(ref_columns[name]).mean()),
            "movement": float(np.isfinite(movement_columns[name]).mean()),
        }
        for name in common
    }
    features = [
        name
        for name in common
        if coverage[name]["reference"] >= config.min_coverage
        and coverage[name]["movement"] >= config.min_coverage
    ]
    if len(features) < 3:
        raise ValueError("Too few common well-observed angle features")
    ref_raw, _ = prepare_features(ref_time, ref_columns, features)
    movement_raw, movement_angles = prepare_features(
        movement_time, movement_columns, features
    )
    split = int(len(ref_raw) * config.train_fraction)
    if min(split, len(ref_raw) - split) < config.sequence_length:
        raise ValueError("Reference sequence is too short for the temporal split")
    mean = ref_raw[:split].mean(axis=0)
    std = ref_raw[:split].std(axis=0)
    std[std < 1e-6] = 1.0
    ref_scaled = (ref_raw - mean) / std
    movement_scaled = (movement_raw - mean) / std
    train_windows, _, _ = make_windows(
        ref_scaled[:split], ref_time[:split], config.sequence_length, config.stride
    )
    validation_windows, _, _ = make_windows(
        ref_scaled[split:], ref_time[split:], config.sequence_length, config.stride
    )
    movement_windows, starts, ends = make_windows(
        movement_scaled, movement_time, config.sequence_length, config.stride
    )

    model = TCNAutoencoder(
        train_windows.shape[2], config.channels, config.kernel_size, config.dropout
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    loss_fn = nn.MSELoss()
    loader = DataLoader(
        TensorDataset(torch.from_numpy(train_windows.astype(np.float32))),
        batch_size=min(config.batch_size, len(train_windows)),
        shuffle=True,
    )
    history = []
    best_loss = float("inf")
    best_state = None
    best_epoch = 0
    stale = 0
    for epoch in range(1, config.epochs + 1):
        model.train()
        total = 0.0
        for (batch,) in loader:
            optimizer.zero_grad()
            loss = loss_fn(model(batch), batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += float(loss) * len(batch)
        validation_loss = float(np.mean(evaluate(model, validation_windows)[0]))
        history.append(
            {
                "epoch": epoch,
                "train_loss": total / len(train_windows),
                "validation_loss": validation_loss,
            }
        )
        if validation_loss < best_loss - 1e-7:
            best_loss = validation_loss
            best_epoch = epoch
            best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= config.patience:
                break
    if best_state is None:
        raise RuntimeError("TCN training did not produce a checkpoint")
    model.load_state_dict(best_state)

    validation_errors, _ = evaluate(model, validation_windows)
    movement_errors, reconstruction = evaluate(model, movement_windows)
    threshold = float(np.percentile(validation_errors, config.threshold_percentile))
    percentiles = np.asarray(
        [100 * np.mean(validation_errors <= error) for error in movement_errors]
    )
    rows = []
    for index, (start, end, error, percentile) in enumerate(
        zip(starts, ends, movement_errors, percentiles), start=1
    ):
        rows.append(
            {
                "window": index,
                "start_time_s": float(start),
                "end_time_s": float(end),
                "center_time_s": float((start + end) / 2),
                "reconstruction_error": float(error),
                "reference_percentile": float(percentile),
                "above_threshold": bool(error > threshold),
            }
        )
    with (output / "window_scores.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with (output / "training_history.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)

    angle_count = len(features)
    reconstructed_angles = reconstruction[:, :, :angle_count] * std[:angle_count] + mean[:angle_count]
    actual_windows = np.stack(
        [
            movement_angles[index : index + config.sequence_length]
            for index in range(0, len(movement_angles) - config.sequence_length + 1, config.stride)
        ]
    )
    feature_rows = []
    for index, name in enumerate(features):
        difference = reconstructed_angles[:, :, index] - actual_windows[:, :, index]
        feature_rows.append(
            {
                "angle": name.removesuffix("_smooth_deg"),
                "mae_deg": float(np.mean(np.abs(difference))),
                "rmse_deg": float(np.sqrt(np.mean(difference**2))),
                "reference_coverage": coverage[name]["reference"],
                "movement_coverage": coverage[name]["movement"],
            }
        )
    feature_rows.sort(key=lambda row: row["mae_deg"], reverse=True)
    with (output / "feature_deviation.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(feature_rows[0]))
        writer.writeheader()
        writer.writerows(feature_rows)

    centers = (starts + ends) / 2
    fig, axis = plt.subplots(figsize=(10, 4.8))
    axis.plot(centers, movement_errors, color="#2455a4", linewidth=2, label="IMG_6086")
    axis.axhline(threshold, color="#c0392b", linestyle="--", label="Reference 95th percentile")
    axis.fill_between(centers, threshold, movement_errors, where=movement_errors > threshold, color="#e74c3c", alpha=0.25)
    axis.set(
        xlabel="Time (seconds)",
        ylabel="Standardized TCN reconstruction error",
        title="TCN temporal deviation from quadruped reference",
    )
    axis.grid(alpha=0.25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(output / "tcn_deviation_timeline.png", dpi=160)
    plt.close(fig)

    receptive_field = 1 + 2 * (config.kernel_size - 1) * sum(
        2**level for level in range(len(config.channels))
    )
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "config": asdict(config),
        "features": features,
        "mean": mean,
        "std": std,
        "threshold": threshold,
        "validation_errors": validation_errors,
    }
    torch.save(checkpoint, output / "tcn_reference_autoencoder.pt")
    summary = {
        "status": "prototype_complete",
        "reference_csv": str(args.reference_csv.expanduser().resolve()),
        "movement_csv": str(args.movement_csv.expanduser().resolve()),
        "angle_features": features,
        "excluded_low_coverage_features": [name for name in common if name not in features],
        "training_windows": len(train_windows),
        "validation_windows": len(validation_windows),
        "movement_windows": len(movement_windows),
        "best_epoch": best_epoch,
        "receptive_field_frames": receptive_field,
        "threshold": threshold,
        "movement_mean_error": float(np.mean(movement_errors)),
        "windows_above_threshold": int(np.sum(movement_errors > threshold)),
        "percent_windows_above_threshold": float(100 * np.mean(movement_errors > threshold)),
        "largest_angle_deviations": feature_rows[:3],
        "limitations": [
            "The TCN learned from windows of one reference recording.",
            "The result measures deviation from that recording, not a clinical class or diagnosis.",
            "Training and validation windows are from different time segments of the same reference video.",
        ],
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
