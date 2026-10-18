#!/usr/bin/env python3
"""Score a joint-angle CSV with a saved LSTM movement prototype."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from lstm_movement_quality import (
    DISCLAIMER,
    LSTMAutoencoder,
    evaluate,
    make_windows,
    prepare_features,
    read_angle_csv,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Score a movement using a trained LSTM checkpoint.")
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("movement_csv", type=Path)
    parser.add_argument("--output-csv", type=Path, default=Path("lstm_window_scores.csv"))
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint.expanduser().resolve(), map_location="cpu")
    features = checkpoint["angle_features"]
    config = checkpoint["config"]
    mean = np.asarray(checkpoint["scaler_mean"], dtype=np.float64)
    std = np.asarray(checkpoint["scaler_std"], dtype=np.float64)
    threshold = float(checkpoint["threshold"])
    validation_errors = np.asarray(checkpoint["validation_errors"], dtype=np.float64)

    times, columns = read_angle_csv(args.movement_csv.expanduser().resolve())
    missing = [name for name in features if name not in columns]
    if missing:
        raise ValueError(f"Movement CSV is missing model features: {', '.join(missing)}")
    raw, _ = prepare_features(times, columns, features)
    scaled = (raw - mean) / std
    windows, starts, ends = make_windows(
        scaled, times, int(config["sequence_length"]), int(config["stride"])
    )

    model = LSTMAutoencoder(windows.shape[2], int(config["hidden_size"]))
    model.load_state_dict(checkpoint["model_state_dict"])
    errors, _ = evaluate(model, windows)
    output = args.output_csv.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        fields = [
            "window",
            "start_time_s",
            "end_time_s",
            "center_time_s",
            "reconstruction_error",
            "reference_percentile",
            "above_threshold",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for index, (start, end, error) in enumerate(zip(starts, ends, errors), start=1):
            writer.writerow(
                {
                    "window": index,
                    "start_time_s": float(start),
                    "end_time_s": float(end),
                    "center_time_s": float((start + end) / 2),
                    "reconstruction_error": float(error),
                    "reference_percentile": float(100 * np.mean(validation_errors <= error)),
                    "above_threshold": bool(error > threshold),
                }
            )
    result = {
        "output_csv": str(output),
        "windows": len(errors),
        "mean_error": float(np.mean(errors)),
        "percent_above_threshold": float(100 * np.mean(errors > threshold)),
        "threshold": threshold,
        "disclaimer": DISCLAIMER,
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
