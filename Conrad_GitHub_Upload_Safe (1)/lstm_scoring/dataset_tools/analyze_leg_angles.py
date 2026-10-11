#!/usr/bin/env python3
"""Create left/right leg-angle graphs and CSV data from YOLO Pose keypoints."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent.parent
KEYPOINT_DIR = ROOT / "data" / "processed" / "keypoints"
ANGLE_DATA_DIR = ROOT / "data" / "processed" / "angle_data"
ANGLE_PLOT_DIR = ROOT / "data" / "processed" / "angle_plots"
PROCESS_MANIFEST = ROOT / "data" / "processed" / "processed_videos.json"
DATASET_MANIFEST = ROOT / "data" / "inbox" / "leg_rehab_20" / "LEG_REHAB_20_MANIFEST.csv"
SUMMARY_PATH = ROOT / "data" / "processed" / "leg_rehab_20_angle_summary.csv"

SIDES = ("left", "right")
METRICS = ("hip_angle", "knee_angle", "thigh_tilt", "shin_tilt")
COLORS = {
    "hip_angle": "#7C3AED",
    "knee_angle": "#2563EB",
    "thigh_tilt": "#F97316",
    "shin_tilt": "#16A34A",
}
LABELS = {
    "hip_angle": "Hip angle",
    "knee_angle": "Knee angle",
    "thigh_tilt": "Thigh tilt",
    "shin_tilt": "Shin tilt",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "inputs",
        nargs="*",
        type=Path,
        help="YOLO keypoint CSV files. Defaults to the 20-clip leg-rehab dataset.",
    )
    parser.add_argument("--confidence", type=float, default=0.50)
    parser.add_argument("--smooth-window", type=int, default=9)
    return parser.parse_args()


def clean_label(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def point(frame: dict[str, str], name: str, threshold: float) -> np.ndarray | None:
    confidence = float(frame.get(f"{name}_confidence") or 0)
    if confidence < threshold:
        return None
    return np.array([
        float(frame[f"{name}_x"]),
        float(frame[f"{name}_y"]),
    ])


def joint_angle(first: np.ndarray | None, center: np.ndarray | None, last: np.ndarray | None) -> float:
    if first is None or center is None or last is None:
        return math.nan
    a = first - center
    b = last - center
    denominator = np.linalg.norm(a) * np.linalg.norm(b)
    if denominator < 1e-8:
        return math.nan
    cosine = float(np.clip(np.dot(a, b) / denominator, -1.0, 1.0))
    return math.degrees(math.acos(cosine))


def segment_tilt(start: np.ndarray | None, end: np.ndarray | None) -> float:
    """Return segment angle relative to the downward image direction."""
    if start is None or end is None:
        return math.nan
    vector = end - start
    length = np.linalg.norm(vector)
    if length < 1e-8:
        return math.nan
    cosine = float(np.clip(vector[1] / length, -1.0, 1.0))
    return math.degrees(math.acos(cosine))


def moving_average(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return values.copy()
    output = np.full(values.shape, np.nan, dtype=float)
    radius = window // 2
    for index in range(len(values)):
        start = max(0, index - radius)
        end = min(len(values), index + radius + 1)
        sample = values[start:end]
        valid = sample[np.isfinite(sample)]
        if len(valid) >= max(2, (end - start) // 3):
            output[index] = float(np.mean(valid))
    return output


def analyze_csv(path: Path, confidence: float) -> dict[str, np.ndarray]:
    data: dict[str, list[float]] = {"frame_index": [], "timestamp_sec": []}
    for side in SIDES:
        for metric in METRICS:
            data[f"{side}_{metric}"] = []

    with path.open(newline="", encoding="utf-8") as handle:
        for frame in csv.DictReader(handle):
            data["frame_index"].append(float(frame["frame_index"]))
            data["timestamp_sec"].append(float(frame["timestamp_sec"]))
            for side in SIDES:
                shoulder = point(frame, f"{side}_shoulder", confidence)
                hip = point(frame, f"{side}_hip", confidence)
                knee = point(frame, f"{side}_knee", confidence)
                ankle = point(frame, f"{side}_ankle", confidence)
                data[f"{side}_hip_angle"].append(joint_angle(shoulder, hip, knee))
                data[f"{side}_knee_angle"].append(joint_angle(hip, knee, ankle))
                data[f"{side}_thigh_tilt"].append(segment_tilt(hip, knee))
                data[f"{side}_shin_tilt"].append(segment_tilt(knee, ankle))
    return {key: np.asarray(values, dtype=float) for key, values in data.items()}


def write_angle_csv(data: dict[str, np.ndarray], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    columns = ["frame_index", "timestamp_sec"] + [
        f"{side}_{metric}_deg" for side in SIDES for metric in METRICS
    ]
    with destination.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(columns)
        for index in range(len(data["timestamp_sec"])):
            row: list[object] = [int(data["frame_index"][index]), round(data["timestamp_sec"][index], 5)]
            for side in SIDES:
                for metric in METRICS:
                    value = data[f"{side}_{metric}"][index]
                    row.append("" if not np.isfinite(value) else round(float(value), 4))
            writer.writerow(row)


def plot_angles(
    data: dict[str, np.ndarray],
    destination: Path,
    title: str,
    subtitle: str,
    smooth_window: int,
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    times = data["timestamp_sec"]
    fig, axes = plt.subplots(2, 1, figsize=(14, 9), sharex=True, sharey=True)
    fig.patch.set_facecolor("#F8FAFC")
    fig.suptitle(title, fontsize=18, fontweight="bold", color="#0F172A", y=0.985)
    fig.text(0.5, 0.952, subtitle, ha="center", va="top", fontsize=10.5, color="#475569")

    for axis, side in zip(axes, SIDES):
        axis.set_facecolor("white")
        valid_any = False
        for metric in METRICS:
            values = data[f"{side}_{metric}"]
            smoothed = moving_average(values, smooth_window)
            coverage = float(np.isfinite(values).mean()) if len(values) else 0.0
            if np.isfinite(smoothed).any():
                valid_any = True
                axis.plot(
                    times,
                    smoothed,
                    color=COLORS[metric],
                    linewidth=2.2,
                    label=f"{LABELS[metric]} ({coverage:.0%} valid)",
                )
        axis.set_title(f"{side.title()} leg", fontsize=13, fontweight="bold", color="#1E293B", pad=10)
        axis.set_ylabel("2D angle (degrees)", color="#334155")
        axis.set_ylim(0, 185)
        axis.set_yticks(np.arange(0, 181, 30))
        axis.grid(True, color="#CBD5E1", alpha=0.55, linewidth=0.8)
        axis.spines[["top", "right"]].set_visible(False)
        axis.spines[["left", "bottom"]].set_color("#94A3B8")
        if valid_any:
            axis.legend(loc="best", ncol=2, frameon=True, framealpha=0.95, fontsize=9)
        else:
            axis.text(0.5, 0.5, "No valid leg angles", transform=axis.transAxes,
                      ha="center", va="center", fontsize=13, color="#64748B")
    axes[-1].set_xlabel("Time (seconds)", color="#334155")
    fig.text(
        0.01,
        0.012,
        "Hip = shoulder–hip–knee; knee = hip–knee–ankle. "
        "Thigh and shin tilt are measured against the downward vertical. "
        "YOLO Pose has no toe/heel points, so a true ankle joint angle is unavailable.",
        fontsize=9,
        color="#475569",
    )
    fig.tight_layout(rect=(0.02, 0.045, 0.99, 0.93))
    fig.savefig(destination, dpi=160, facecolor=fig.get_facecolor())
    plt.close(fig)


def dataset_inputs() -> list[tuple[str, str, Path]]:
    processed = json.loads(PROCESS_MANIFEST.read_text(encoding="utf-8"))["processed"]
    by_filename = {Path(item["source"]).name: item for item in processed.values()}
    with DATASET_MANIFEST.open(newline="", encoding="utf-8") as handle:
        manifest_rows = list(csv.DictReader(handle))
    inputs: list[tuple[str, str, Path]] = []
    for row in manifest_rows:
        item = by_filename.get(row["filename"])
        if not item:
            continue
        inputs.append((row["sample_id"], row["exercise"], ROOT / item["keypoints_csv"]))
    return inputs


def summarize(sample_id: str, exercise: str, data: dict[str, np.ndarray]) -> dict[str, object]:
    row: dict[str, object] = {
        "sample_id": sample_id,
        "exercise": exercise,
        "duration_sec": round(float(data["timestamp_sec"][-1]), 3) if len(data["timestamp_sec"]) else 0,
    }
    for side in SIDES:
        for metric in METRICS:
            values = data[f"{side}_{metric}"]
            valid = values[np.isfinite(values)]
            prefix = f"{side}_{metric}"
            row[f"{prefix}_valid_coverage"] = round(len(valid) / len(values), 4) if len(values) else 0
            row[f"{prefix}_min_deg"] = round(float(np.min(valid)), 3) if len(valid) else ""
            row[f"{prefix}_max_deg"] = round(float(np.max(valid)), 3) if len(valid) else ""
            row[f"{prefix}_range_deg"] = round(float(np.max(valid) - np.min(valid)), 3) if len(valid) else ""
    return row


def main() -> int:
    args = parse_args()
    if args.inputs:
        inputs = [("custom", path.stem, path.expanduser().resolve()) for path in args.inputs]
    else:
        inputs = dataset_inputs()
    if not inputs:
        raise RuntimeError("No keypoint CSV files were found")

    summary_rows: list[dict[str, object]] = []
    for index, (sample_id, exercise, path) in enumerate(inputs, start=1):
        data = analyze_csv(path, args.confidence)
        stem = clean_label(path.stem.removesuffix("_keypoints"))
        angle_csv = ANGLE_DATA_DIR / f"{stem}_leg_angles.csv"
        angle_plot = ANGLE_PLOT_DIR / f"{stem}_leg_angles.png"
        write_angle_csv(data, angle_csv)
        plot_angles(
            data,
            angle_plot,
            "YOLO Pose leg joint angles",
            f"{sample_id} · {exercise.replace('_', ' ')} · confidence ≥ {args.confidence:.2f}",
            max(1, args.smooth_window),
        )
        summary_rows.append(summarize(sample_id, exercise, data))
        print(f"[{index:02d}/{len(inputs):02d}] {exercise}: {angle_plot}")

    if not args.inputs:
        with SUMMARY_PATH.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0]))
            writer.writeheader()
            writer.writerows(summary_rows)
        print(f"Summary: {SUMMARY_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
