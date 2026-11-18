#!/usr/bin/env python3
"""Run a trained landmark TCN over a video or MediaPipe landmark CSV."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from prepare_landmark_lstm_dataset import fixed_length, read_landmarks
from train_landmark_tcn_classifier import TCNClassifier


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Video file or *_landmarks.csv")
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("results/tcn-temporal-analysis"))
    parser.add_argument("--window-frames", type=int, default=128)
    parser.add_argument("--stride-frames", type=int, default=16)
    parser.add_argument("--fps", type=float, help="Required only if a CSV lacks usable time information")
    parser.add_argument("--critical-landmarks", default="left_knee,right_knee")
    parser.add_argument("--min-visibility", type=float, default=0.5)
    parser.add_argument("--max-critical-low-visibility", type=float, default=0.20)
    return parser.parse_args()


def extract_landmarks(video: Path, output_dir: Path) -> tuple[Path, Path]:
    project = Path(__file__).resolve().parent
    python = project / ".venv-mediapipe" / "bin" / "python"
    analyzer = project / "analyze_body.py"
    if not python.is_file():
        raise FileNotFoundError(f"MediaPipe Python environment not found: {python}")
    media_dir = output_dir / "mediapipe"
    subprocess.run(
        [str(python), str(analyzer), str(video), "--output-dir", str(media_dir)],
        check=True,
    )
    landmark_csv = media_dir / f"{video.stem}_landmarks.csv"
    annotated_video = media_dir / f"{video.stem}_body_annotated.mp4"
    return landmark_csv, annotated_video


def read_times(path: Path) -> np.ndarray:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    try:
        times = np.asarray([float(row["time_s"]) for row in rows], dtype=np.float64)
    except (KeyError, TypeError, ValueError):
        return np.arange(len(rows), dtype=np.float64)
    return times


def main() -> None:
    args = parse_args()
    input_path = args.input.expanduser().resolve()
    checkpoint_path = args.checkpoint.expanduser().resolve()
    if not input_path.is_file():
        raise FileNotFoundError(f"Input not found: {input_path}")
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)

    annotated_video = None
    if input_path.suffix.lower() == ".csv":
        landmark_csv = input_path
    else:
        landmark_csv, annotated_video = extract_landmarks(input_path, output)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    expected_features = checkpoint["feature_names"]
    includes_velocity = len(expected_features) > 0 and expected_features[-1].endswith("_velocity")
    critical = [item.strip().lower() for item in args.critical_landmarks.split(",") if item.strip()]
    positions, position_names, quality = read_landmarks(
        landmark_csv,
        args.min_visibility,
        critical,
        args.max_critical_low_visibility,
    )
    if positions is None:
        raise ValueError(f"Sequence rejected by visibility filtering: {quality['reason']}")
    base_expected = expected_features[: len(position_names)]
    if list(position_names) != list(base_expected):
        raise ValueError("Landmark feature order does not match the trained TCN")
    times = read_times(landmark_csv)
    if args.fps and (len(times) < 2 or np.any(np.diff(times) <= 0)):
        times = np.arange(len(positions)) / args.fps

    window_size = max(2, args.window_frames)
    starts = list(range(0, max(1, len(positions) - window_size + 1), args.stride_frames))
    if len(positions) <= window_size:
        starts = [0]
    elif starts[-1] != len(positions) - window_size:
        starts.append(len(positions) - window_size)
    model_steps = int(checkpoint["time_steps"])
    tensors = []
    lengths = []
    temporal_rows = []
    for start in starts:
        stop = min(start + window_size, len(positions))
        tensor, valid_length = fixed_length(positions[start:stop], model_steps, "padding", None)
        if includes_velocity:
            velocity = np.zeros_like(tensor)
            if valid_length > 1:
                velocity[:valid_length] = np.gradient(tensor[:valid_length], axis=0)
            tensor = np.concatenate([tensor, velocity], axis=1)
        if tensor.shape[1] != len(expected_features):
            raise ValueError(
                f"Model expects {len(expected_features)} features, but analysis produced {tensor.shape[1]}"
            )
        tensors.append(tensor)
        lengths.append(valid_length)
        temporal_rows.append(
            {
                "start_frame": start,
                "end_frame": stop - 1,
                "start_time_s": float(times[start]),
                "end_time_s": float(times[stop - 1]),
                "center_time_s": float((times[start] + times[stop - 1]) / 2),
            }
        )

    X = np.asarray(tensors, dtype=np.float32)
    lengths_array = np.asarray(lengths, dtype=np.int64)
    mean = np.asarray(checkpoint["mean"], dtype=np.float32)
    std = np.asarray(checkpoint["std"], dtype=np.float32)
    for index in range(len(X)):
        X[index, : lengths_array[index]] = (X[index, : lengths_array[index]] - mean) / std
        X[index, lengths_array[index] :] = 0.0
    labels = checkpoint["labels"]
    model = TCNClassifier(
        len(expected_features),
        list(checkpoint["channels"]),
        len(labels),
        int(checkpoint["kernel_size"]),
        float(checkpoint["dropout"]),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    with torch.no_grad():
        logits = model(torch.from_numpy(X), torch.from_numpy(lengths_array))
        probabilities = torch.softmax(logits, dim=1).numpy()
    predictions = probabilities.argmax(axis=1)

    fields = list(temporal_rows[0]) + ["predicted_label", "confidence"] + [
        f"probability_{label}" for label in labels
    ]
    for index, row in enumerate(temporal_rows):
        row["predicted_label"] = labels[predictions[index]]
        row["confidence"] = float(probabilities[index, predictions[index]])
        row.update(
            {f"probability_{label}": float(probabilities[index, class_index]) for class_index, label in enumerate(labels)}
        )
    scores_path = output / "tcn_temporal_scores.csv"
    with scores_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(temporal_rows)

    centers = np.asarray([row["center_time_s"] for row in temporal_rows])
    fig, axis = plt.subplots(figsize=(11, 5.5))
    for class_index, label in enumerate(labels):
        axis.plot(centers, probabilities[:, class_index], marker="o", markersize=3, label=label)
    axis.set(
        xlabel="Video time (seconds)",
        ylabel="TCN class probability",
        ylim=(0, 1),
        title="Temporal movement-pattern analysis",
    )
    axis.grid(alpha=0.25)
    axis.legend()
    fig.tight_layout()
    graph_path = output / "tcn_temporal_probabilities.png"
    fig.savefig(graph_path, dpi=160)
    plt.close(fig)

    majority = int(np.bincount(predictions, minlength=len(labels)).argmax())
    summary = {
        "input": str(input_path),
        "landmarks_csv": str(landmark_csv),
        "annotated_video": str(annotated_video) if annotated_video else None,
        "windows": len(temporal_rows),
        "window_frames": window_size,
        "stride_frames": args.stride_frames,
        "dominant_predicted_label": labels[majority],
        "mean_probabilities": {
            label: float(probabilities[:, index].mean()) for index, label in enumerate(labels)
        },
        "visibility_quality": quality,
        "warning": "Prototype class probabilities require clinician-reviewed labels and are not diagnoses.",
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Temporal scores: {scores_path}")
    print(f"Probability graph: {graph_path}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
