#!/usr/bin/env python3
"""Train a one-class LSTM prototype and score movement-angle CSV files.

This is an engineering prototype. Its output measures reconstruction error
relative to reference recordings; it is not a diagnosis or clinical score.
"""

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


DISCLAIMER = (
    "Engineering prototype only. Scores indicate similarity to the supplied "
    "reference recording and are not validated measures of rehabilitation quality."
)


@dataclass
class Config:
    sequence_length: int = 64
    stride: int = 16
    train_fraction: float = 0.70
    min_coverage: float = 0.60
    hidden_size: int = 32
    epochs: int = 120
    patience: int = 15
    batch_size: int = 16
    learning_rate: float = 1e-3
    threshold_percentile: float = 95.0
    seed: int = 42


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train an LSTM autoencoder on reference angles and score a movement."
    )
    parser.add_argument("reference_csv", type=Path)
    parser.add_argument("movement_csv", type=Path)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("results/lstm-prototype")
    )
    parser.add_argument("--sequence-length", type=int, default=64)
    parser.add_argument("--stride", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--hidden-size", type=int, default=32)
    parser.add_argument("--min-coverage", type=float, default=0.60)
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(min(4, max(1, torch.get_num_threads())))


def read_angle_csv(path: Path) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    if not path.is_file():
        raise FileNotFoundError(f"CSV not found: {path}")
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or "time_s" not in reader.fieldnames:
            raise ValueError(f"{path} must contain a time_s column")
        angle_names = [c for c in reader.fieldnames if c.endswith("_smooth_deg")]
        if not angle_names:
            raise ValueError(f"{path} has no *_smooth_deg columns")
        times: list[float] = []
        values = {name: [] for name in angle_names}
        for row in reader:
            try:
                times.append(float(row["time_s"]))
            except (TypeError, ValueError):
                continue
            for name in angle_names:
                try:
                    values[name].append(float(row[name]))
                except (TypeError, ValueError):
                    values[name].append(float("nan"))
    t = np.asarray(times, dtype=np.float64)
    if len(t) < 2 or np.any(np.diff(t) <= 0):
        raise ValueError(f"{path} needs at least two rows with increasing time_s")
    return t, {k: np.asarray(v, dtype=np.float64) for k, v in values.items()}


def interpolate(values: np.ndarray) -> np.ndarray:
    output = values.copy()
    x = np.arange(len(output))
    for column in range(output.shape[1]):
        valid = np.isfinite(output[:, column])
        if not valid.any():
            raise ValueError("A selected feature has no finite values")
        output[:, column] = np.interp(x, x[valid], output[valid, column])
    return output


def prepare_features(
    times: np.ndarray, columns: dict[str, np.ndarray], feature_names: list[str]
) -> tuple[np.ndarray, np.ndarray]:
    angles = np.column_stack([columns[name] for name in feature_names])
    angles = interpolate(angles)
    velocity = np.gradient(angles, times, axis=0)
    return np.concatenate([angles, velocity], axis=1), angles


def make_windows(
    values: np.ndarray, times: np.ndarray, length: int, stride: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    starts = np.arange(0, len(values) - length + 1, stride, dtype=int)
    if not len(starts):
        raise ValueError(
            f"Only {len(values)} frames are available, fewer than sequence length {length}"
        )
    windows = np.stack([values[start : start + length] for start in starts])
    return windows, times[starts], times[starts + length - 1]


class LSTMAutoencoder(nn.Module):
    def __init__(self, input_size: int, hidden_size: int) -> None:
        super().__init__()
        self.encoder = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.decoder = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        self.output = nn.Linear(hidden_size, input_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, (hidden, _) = self.encoder(x)
        repeated = hidden[-1].unsqueeze(1).expand(-1, x.shape[1], -1)
        decoded, _ = self.decoder(repeated)
        return self.output(decoded)


def evaluate(model: nn.Module, windows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    with torch.no_grad():
        source = torch.from_numpy(windows.astype(np.float32))
        reconstructed = model(source).cpu().numpy()
    errors = np.mean((reconstructed - windows) ** 2, axis=(1, 2))
    return errors, reconstructed


def train_model(
    train_windows: np.ndarray, validation_windows: np.ndarray, config: Config
) -> tuple[LSTMAutoencoder, list[dict[str, float]], int]:
    model = LSTMAutoencoder(train_windows.shape[2], config.hidden_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    loss_fn = nn.MSELoss()
    loader = DataLoader(
        TensorDataset(torch.from_numpy(train_windows.astype(np.float32))),
        batch_size=min(config.batch_size, len(train_windows)),
        shuffle=True,
    )
    history: list[dict[str, float]] = []
    best_state: dict[str, torch.Tensor] | None = None
    best_loss = float("inf")
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
        train_loss = total / len(train_windows)
        validation_loss = float(np.mean(evaluate(model, validation_windows)[0]))
        history.append(
            {"epoch": epoch, "train_loss": train_loss, "validation_loss": validation_loss}
        )
        if validation_loss < best_loss - 1e-7:
            best_loss = validation_loss
            best_epoch = epoch
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= config.patience:
                break
    if best_state is None:
        raise RuntimeError("Training did not produce a model")
    model.load_state_dict(best_state)
    return model, history, best_epoch


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_plots(
    output_dir: Path,
    history: list[dict[str, float]],
    centers: np.ndarray,
    scores: np.ndarray,
    threshold: float,
) -> None:
    epochs = [row["epoch"] for row in history]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(epochs, [row["train_loss"] for row in history], label="Training")
    ax.plot(epochs, [row["validation_loss"] for row in history], label="Validation")
    ax.set(xlabel="Epoch", ylabel="Mean squared reconstruction error", title="LSTM training history")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "training_history.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(centers, scores, color="#2455a4", linewidth=1.8, label="Movement window")
    ax.axhline(threshold, color="#c0392b", linestyle="--", label="Reference validation 95th percentile")
    ax.fill_between(centers, threshold, scores, where=scores > threshold, color="#e74c3c", alpha=0.25)
    ax.set(xlabel="Time (s)", ylabel="Standardized reconstruction error", title="Deviation from the reference model")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "deviation_timeline.png", dpi=160)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    config = Config(
        sequence_length=args.sequence_length,
        stride=args.stride,
        epochs=args.epochs,
        hidden_size=args.hidden_size,
        min_coverage=args.min_coverage,
    )
    seed_everything(config.seed)
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    ref_time, ref_columns = read_angle_csv(args.reference_csv.expanduser().resolve())
    move_time, move_columns = read_angle_csv(args.movement_csv.expanduser().resolve())
    common = sorted(set(ref_columns) & set(move_columns))
    coverage = {
        name: {
            "reference": float(np.isfinite(ref_columns[name]).mean()),
            "movement": float(np.isfinite(move_columns[name]).mean()),
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
        raise ValueError(
            f"Only {len(features)} common angle features meet {config.min_coverage:.0%} coverage"
        )

    ref_raw, _ = prepare_features(ref_time, ref_columns, features)
    move_raw, move_angles = prepare_features(move_time, move_columns, features)
    split = int(len(ref_raw) * config.train_fraction)
    if min(split, len(ref_raw) - split) < config.sequence_length:
        raise ValueError("Reference is too short for the requested temporal split and sequence length")
    mean = ref_raw[:split].mean(axis=0)
    std = ref_raw[:split].std(axis=0)
    std[std < 1e-6] = 1.0
    ref_scaled = (ref_raw - mean) / std
    move_scaled = (move_raw - mean) / std

    train_windows, _, _ = make_windows(
        ref_scaled[:split], ref_time[:split], config.sequence_length, config.stride
    )
    validation_windows, _, _ = make_windows(
        ref_scaled[split:], ref_time[split:], config.sequence_length, config.stride
    )
    movement_windows, starts, ends = make_windows(
        move_scaled, move_time, config.sequence_length, config.stride
    )

    model, history, best_epoch = train_model(train_windows, validation_windows, config)
    validation_errors, _ = evaluate(model, validation_windows)
    movement_errors, reconstructed = evaluate(model, movement_windows)
    threshold = float(np.percentile(validation_errors, config.threshold_percentile))
    percentiles = np.asarray(
        [100.0 * np.mean(validation_errors <= score) for score in movement_errors]
    )

    window_rows = []
    for index, (start, end, score, percentile) in enumerate(
        zip(starts, ends, movement_errors, percentiles), start=1
    ):
        window_rows.append(
            {
                "window": index,
                "start_time_s": round(float(start), 6),
                "end_time_s": round(float(end), 6),
                "center_time_s": round(float((start + end) / 2), 6),
                "reconstruction_error": float(score),
                "reference_percentile": float(percentile),
                "above_threshold": bool(score > threshold),
            }
        )
    write_csv(output_dir / "window_scores.csv", window_rows, list(window_rows[0]))
    write_csv(output_dir / "training_history.csv", history, list(history[0]))

    angle_count = len(features)
    reconstructed_angles = reconstructed[:, :, :angle_count] * std[:angle_count] + mean[:angle_count]
    actual_angle_windows = np.stack(
        [move_angles[i : i + config.sequence_length] for i in range(0, len(move_angles) - config.sequence_length + 1, config.stride)]
    )
    feature_rows = []
    for index, name in enumerate(features):
        difference = reconstructed_angles[:, :, index] - actual_angle_windows[:, :, index]
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
    write_csv(output_dir / "feature_deviation.csv", feature_rows, list(feature_rows[0]))

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "config": asdict(config),
        "angle_features": features,
        "input_feature_names": features + [f"{name}_velocity_deg_s" for name in features],
        "scaler_mean": mean,
        "scaler_std": std,
        "threshold": threshold,
        "validation_errors": validation_errors,
    }
    torch.save(checkpoint, output_dir / "lstm_autoencoder.pt")

    summary = {
        "status": "prototype_complete",
        "disclaimer": DISCLAIMER,
        "reference_csv": str(args.reference_csv.expanduser().resolve()),
        "movement_csv": str(args.movement_csv.expanduser().resolve()),
        "angle_features": features,
        "excluded_or_low_coverage_features": [name for name in common if name not in features],
        "coverage": coverage,
        "reference_frames": len(ref_time),
        "movement_frames": len(move_time),
        "training_windows": len(train_windows),
        "validation_windows": len(validation_windows),
        "movement_windows": len(movement_windows),
        "best_epoch": best_epoch,
        "threshold": threshold,
        "movement_mean_error": float(np.mean(movement_errors)),
        "movement_median_reference_percentile": float(np.median(percentiles)),
        "windows_above_threshold": int(np.sum(movement_errors > threshold)),
        "percent_windows_above_threshold": float(100 * np.mean(movement_errors > threshold)),
        "largest_angle_deviations": feature_rows[:3],
        "limitations": [
            "The model was trained on temporal windows from one reference recording.",
            "Training and validation segments come from the same reference video and are not participant-independent.",
            "MediaPipe angles are 2D estimates affected by camera view, occlusion, and landmark error.",
            "A higher reconstruction error means less similarity to this reference, not unsafe or irrational movement.",
        ],
        "config": asdict(config),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    centers = (starts + ends) / 2
    save_plots(output_dir, history, centers, movement_errors, threshold)

    print(f"Prototype saved to: {output_dir}")
    print(f"Angles used: {len(features)}")
    print(f"Best epoch: {best_epoch}")
    print(f"Reference threshold: {threshold:.6f}")
    print(f"Movement windows above threshold: {summary['percent_windows_above_threshold']:.1f}%")
    print(DISCLAIMER)


if __name__ == "__main__":
    main()
