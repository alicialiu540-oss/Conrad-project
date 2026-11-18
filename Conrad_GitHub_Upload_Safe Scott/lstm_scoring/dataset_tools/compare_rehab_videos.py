#!/usr/bin/env python3
"""Compare a standard rehabilitation video with a user's video using YOLO Pose."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
from pathlib import Path
from typing import Any

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from ultralytics import YOLO

from analyze_leg_angles import METRICS, analyze_csv, clean_label
from lstm_motion import load_lstm_checkpoint, lstm_similarity, sequence_from_angle_data
from process_videos import (
    DEFAULT_MODEL,
    ROOT,
    choose_device,
    process_video,
    sha256_file,
)


OUTPUT_ROOT = ROOT / "outputs" / "comparisons"
PROCESS_MANIFEST = ROOT / "data" / "processed" / "processed_videos.json"
DEFAULT_LSTM_MODEL = ROOT / "models" / "leg_rehab_siamese_lstm.pt"
METRIC_WEIGHTS = {
    "knee_angle": 0.50,
    "hip_angle": 0.25,
    "thigh_tilt": 0.15,
    "shin_tilt": 0.10,
}
PLOT_COLORS = {"standard": "#2563EB", "user": "#F97316"}
plt.rcParams["font.family"] = ["Heiti TC", "Arial Unicode MS", "Arial", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--standard", type=Path, required=True, help="Standard rehabilitation video")
    parser.add_argument("--user", type=Path, required=True, help="User's own video")
    parser.add_argument("--output-dir", type=Path, help="Output folder")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--lstm-model", type=Path, default=DEFAULT_LSTM_MODEL)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--confidence", type=float, default=0.50, help="Keypoint confidence threshold")
    parser.add_argument("--max-dtw-frames", type=int, default=220)
    return parser.parse_args()


def relative_or_absolute(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path.resolve())


def load_processed_lookup() -> dict[str, dict[str, Any]]:
    if not PROCESS_MANIFEST.exists():
        return {}
    items = json.loads(PROCESS_MANIFEST.read_text(encoding="utf-8")).get("processed", {}).values()
    lookup: dict[str, dict[str, Any]] = {}
    for item in items:
        source = Path(item.get("source", ""))
        if not source.is_absolute():
            source = ROOT / source
        lookup[str(source.resolve())] = item
    return lookup


def existing_pose_files(video: Path, lookup: dict[str, dict[str, Any]]) -> tuple[Path, Path] | None:
    item = lookup.get(str(video.resolve()))
    if not item:
        return None
    keypoints = Path(item["keypoints_csv"])
    annotated = Path(item["annotated_video"])
    if not keypoints.is_absolute():
        keypoints = ROOT / keypoints
    if not annotated.is_absolute():
        annotated = ROOT / annotated
    if keypoints.exists() and annotated.exists():
        return keypoints, annotated
    return None


def ensure_pose_files(
    video: Path,
    role: str,
    lookup: dict[str, dict[str, Any]],
    model_holder: dict[str, YOLO],
    device: str,
    model_path: Path,
) -> tuple[Path, Path]:
    existing = existing_pose_files(video, lookup)
    if existing:
        print(f"Using existing YOLO result for {role}: {video.name}")
        return existing
    video_hash = sha256_file(video)
    stem = f"comparison_{role}_{clean_label(video.stem)}_{video_hash[:8]}"
    keypoints = ROOT / "data" / "processed" / "keypoints" / f"{stem}_keypoints.csv"
    annotated = ROOT / "data" / "processed" / "annotated_videos" / f"{stem}_pose.mp4"
    if keypoints.exists() and annotated.exists():
        print(f"Using cached YOLO result for {role}: {video.name}")
        return keypoints, annotated
    if "model" not in model_holder:
        print(f"Loading {model_path.name} on {device}...")
        model_holder["model"] = YOLO(str(model_path))
    print(f"Running YOLO Pose on {role}: {video.name}")
    result = process_video(
        model_holder["model"], video, stem, device, 640, 0.25, 0.50
    )
    return result["csv_path"], result["annotated_path"]


def interpolate_series(values: np.ndarray) -> np.ndarray | None:
    valid = np.isfinite(values)
    if valid.sum() < 2:
        return None
    indexes = np.arange(len(values), dtype=float)
    filled = np.interp(indexes, indexes[valid], values[valid])
    if len(filled) >= 5:
        window = min(9, len(filled) if len(filled) % 2 == 1 else len(filled) - 1)
        if window >= 3:
            kernel = np.ones(window) / window
            padded = np.pad(filled, (window // 2, window // 2), mode="edge")
            filled = np.convolve(padded, kernel, mode="valid")
    return filled


def side_features(data: dict[str, np.ndarray], side: str, max_frames: int) -> dict[str, Any] | None:
    available: dict[str, np.ndarray] = {}
    coverage: dict[str, float] = {}
    for metric in METRICS:
        raw = data[f"{side}_{metric}"]
        coverage[metric] = float(np.isfinite(raw).mean()) if len(raw) else 0.0
        filled = interpolate_series(raw)
        if filled is not None and coverage[metric] >= 0.30:
            available[metric] = filled
    if "knee_angle" not in available:
        return None
    indexes = np.linspace(0, len(data["timestamp_sec"]) - 1, min(max_frames, len(data["timestamp_sec"]))).round().astype(int)
    metrics = [metric for metric in METRICS if metric in available]
    weights = np.array([METRIC_WEIGHTS[metric] for metric in metrics], dtype=float)
    weights /= weights.sum()
    matrix = np.column_stack([available[metric][indexes] for metric in metrics])
    return {
        "side": side,
        "metrics": metrics,
        "weights": weights,
        "matrix": matrix,
        "indexes": indexes,
        "full": available,
        "coverage": coverage,
    }


def dtw(first: np.ndarray, second: np.ndarray, weights: np.ndarray) -> tuple[float, list[tuple[int, int]]]:
    n, m = len(first), len(second)
    costs = np.full((n + 1, m + 1), np.inf, dtype=float)
    costs[0, 0] = 0.0
    parent = np.zeros((n + 1, m + 1), dtype=np.uint8)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            local = float(np.sum(np.abs(first[i - 1] - second[j - 1]) * weights))
            candidates = (costs[i - 1, j - 1], costs[i - 1, j], costs[i, j - 1])
            direction = int(np.argmin(candidates))
            costs[i, j] = local + candidates[direction]
            parent[i, j] = direction
    path: list[tuple[int, int]] = []
    i, j = n, m
    while i > 0 and j > 0:
        path.append((i - 1, j - 1))
        direction = parent[i, j]
        if direction == 0:
            i -= 1
            j -= 1
        elif direction == 1:
            i -= 1
        else:
            j -= 1
    path.reverse()
    mean_error = costs[n, m] / max(1, len(path))
    return mean_error, path


def best_alignment(
    standard: dict[str, np.ndarray], user: dict[str, np.ndarray], max_frames: int
) -> dict[str, Any]:
    candidates: list[dict[str, Any]] = []
    for standard_side in ("left", "right"):
        standard_features = side_features(standard, standard_side, max_frames)
        if not standard_features:
            continue
        for user_side in ("left", "right"):
            user_features = side_features(user, user_side, max_frames)
            if not user_features:
                continue
            metrics = [
                metric for metric in METRICS
                if metric in standard_features["metrics"] and metric in user_features["metrics"]
            ]
            if "knee_angle" not in metrics:
                continue
            weights = np.array([METRIC_WEIGHTS[metric] for metric in metrics], dtype=float)
            weights /= weights.sum()
            standard_columns = [standard_features["metrics"].index(metric) for metric in metrics]
            user_columns = [user_features["metrics"].index(metric) for metric in metrics]
            mean_error, path = dtw(
                standard_features["matrix"][:, standard_columns],
                user_features["matrix"][:, user_columns],
                weights,
            )
            candidates.append({
                "standard": standard_features,
                "user": user_features,
                "metrics": metrics,
                "weights": weights,
                "mean_error_deg": mean_error,
                "path": path,
            })
    if not candidates:
        raise RuntimeError("The hip, knee, and ankle points were not visible enough to compare these videos.")
    return min(candidates, key=lambda item: item["mean_error_deg"])


def robust_range(values: np.ndarray | None) -> float | None:
    if values is None or not len(values):
        return None
    return float(np.percentile(values, 95) - np.percentile(values, 5))


def build_report(
    standard_video: Path,
    user_video: Path,
    standard_data: dict[str, np.ndarray],
    user_data: dict[str, np.ndarray],
    alignment: dict[str, Any],
    lstm_result: dict[str, float | bool] | None,
    lstm_model_path: Path,
) -> dict[str, Any]:
    standard_side = alignment["standard"]["side"]
    user_side = alignment["user"]["side"]
    mean_error = float(alignment["mean_error_deg"])
    angle_alignment_score = float(np.clip(100.0 - mean_error * 2.0, 0.0, 100.0))
    similarity = float(lstm_result["score"]) if lstm_result else angle_alignment_score
    standard_duration = float(standard_data["timestamp_sec"][-1]) if len(standard_data["timestamp_sec"]) else 0.0
    user_duration = float(user_data["timestamp_sec"][-1]) if len(user_data["timestamp_sec"]) else 0.0
    tempo_ratio = user_duration / standard_duration if standard_duration else math.nan
    standard_knee_rom = robust_range(alignment["standard"]["full"].get("knee_angle"))
    user_knee_rom = robust_range(alignment["user"]["full"].get("knee_angle"))
    standard_hip_rom = robust_range(alignment["standard"]["full"].get("hip_angle"))
    user_hip_rom = robust_range(alignment["user"]["full"].get("hip_angle"))

    if similarity >= 85:
        rating = "Very close to the standard movement"
    elif similarity >= 70:
        rating = "Generally close to the standard movement"
    elif similarity >= 50:
        rating = "Noticeable movement differences"
    else:
        rating = "Large movement differences"

    observations: list[str] = []
    if standard_knee_rom is not None and user_knee_rom is not None:
        difference = user_knee_rom - standard_knee_rom
        if difference < -10:
            observations.append(f"Your knee range of motion is about {abs(difference):.1f}° smaller than the standard.")
        elif difference > 10:
            observations.append(f"Your knee range of motion is about {difference:.1f}° larger than the standard.")
        else:
            observations.append("Your knee range of motion is close to the standard video.")
    if np.isfinite(tempo_ratio):
        if tempo_ratio > 1.20:
            observations.append("Your video is longer; the movement may be slower or contain a different number of repetitions.")
        elif tempo_ratio < 0.80:
            observations.append("Your video is shorter; the movement may be faster or contain a different number of repetitions.")
        else:
            observations.append("The two videos have similar durations.")
    observations.append("The score measures visible pose similarity. It is not a medical diagnosis or a rehabilitation-safety decision.")

    report = {
        "standard_video": str(standard_video),
        "user_video": str(user_video),
        "standard_selected_side": standard_side,
        "user_selected_side": user_side,
        "compared_metrics": alignment["metrics"],
        "similarity_score": round(similarity, 1),
        "similarity_method": "Siamese LSTM" if lstm_result else "angle DTW fallback",
        "rating": rating,
        "angle_alignment_score": round(angle_alignment_score, 1),
        "weighted_mean_angle_error_deg": round(mean_error, 2),
        "standard_duration_sec": round(standard_duration, 3),
        "user_duration_sec": round(user_duration, 3),
        "duration_ratio_user_to_standard": round(tempo_ratio, 3) if np.isfinite(tempo_ratio) else None,
        "standard_knee_range_deg": round(standard_knee_rom, 2) if standard_knee_rom is not None else None,
        "user_knee_range_deg": round(user_knee_rom, 2) if user_knee_rom is not None else None,
        "standard_hip_range_deg": round(standard_hip_rom, 2) if standard_hip_rom is not None else None,
        "user_hip_range_deg": round(user_hip_rom, 2) if user_hip_rom is not None else None,
        "observations": observations,
    }
    if lstm_result:
        report.update({
            "lstm_model": str(lstm_model_path),
            "lstm_cosine_similarity": round(float(lstm_result["cosine_similarity"]), 5),
            "lstm_mirrored_user_selected": bool(lstm_result["mirrored_user_selected"]),
        })
        report["observations"].insert(
            0,
            "The primary similarity score comes from the Siamese LSTM sequence model; angle alignment is retained for explanation.",
        )
    return report


def aligned_rows(alignment: dict[str, Any]) -> list[dict[str, object]]:
    standard = alignment["standard"]
    user = alignment["user"]
    path = alignment["path"]
    rows: list[dict[str, object]] = []
    for position, (standard_index, user_index) in enumerate(path):
        row: dict[str, object] = {
            "aligned_progress_percent": round(100 * position / max(1, len(path) - 1), 3),
            "standard_sample_index": int(standard_index),
            "user_sample_index": int(user_index),
        }
        for metric in alignment["metrics"]:
            standard_value = float(standard["matrix"][standard_index, standard["metrics"].index(metric)])
            user_value = float(user["matrix"][user_index, user["metrics"].index(metric)])
            row[f"standard_{metric}_deg"] = round(standard_value, 3)
            row[f"user_{metric}_deg"] = round(user_value, 3)
            row[f"absolute_{metric}_difference_deg"] = round(abs(standard_value - user_value), 3)
        rows.append(row)
    return rows


def write_comparison_csv(rows: list[dict[str, object]], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_graph(rows: list[dict[str, object]], report: dict[str, Any], path: Path) -> None:
    metrics = [metric for metric in ("knee_angle", "hip_angle", "thigh_tilt", "shin_tilt")
               if f"standard_{metric}_deg" in rows[0]]
    plotted = metrics[:2] if len(metrics) >= 2 else metrics
    fig, axes = plt.subplots(len(plotted), 1, figsize=(14, 4.2 * len(plotted)), sharex=True)
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])
    fig.patch.set_facecolor("#F8FAFC")
    fig.suptitle("Rehabilitation movement angle comparison", fontsize=20, fontweight="normal", color="#0F172A", y=0.985)
    fig.text(
        0.5, 0.895,
        f"LSTM similarity {report['similarity_score']:.1f}/100 · {report['rating']} · "
        f"standard {report['standard_selected_side']} / user {report['user_selected_side']}",
        ha="center", fontsize=11, color="#475569",
    )
    progress = np.array([float(row["aligned_progress_percent"]) for row in rows])
    labels = {"knee_angle": "Knee angle", "hip_angle": "Hip angle",
              "thigh_tilt": "Thigh tilt", "shin_tilt": "Shin tilt"}
    for axis, metric in zip(axes, plotted):
        standard = np.array([float(row[f"standard_{metric}_deg"]) for row in rows])
        user = np.array([float(row[f"user_{metric}_deg"]) for row in rows])
        axis.set_facecolor("white")
        axis.plot(progress, standard, color=PLOT_COLORS["standard"], linewidth=2.5, label="Standard video")
        axis.plot(progress, user, color=PLOT_COLORS["user"], linewidth=2.5, label="Your video")
        axis.fill_between(progress, standard, user, color="#94A3B8", alpha=0.18)
        axis.set_title(labels[metric], fontsize=14, fontweight="normal", color="#1E293B")
        axis.set_ylabel("Angle (degrees)")
        axis.set_ylim(0, 185)
        axis.grid(True, color="#CBD5E1", alpha=0.55)
        axis.legend(loc="best")
        axis.spines[["top", "right"]].set_visible(False)
    axes[-1].set_xlabel("Aligned movement progress (%)")
    fig.text(0.01, 0.012, "The gray area shows the angle difference. This research result does not replace assessment by a physical therapist.",
             fontsize=9, color="#475569")
    fig.tight_layout(rect=(0.02, 0.045, 0.99, 0.84))
    fig.savefig(path, dpi=160, facecolor=fig.get_facecolor())
    plt.close(fig)


def fit_frame(frame: np.ndarray, width: int, height: int) -> np.ndarray:
    source_height, source_width = frame.shape[:2]
    scale = min(width / source_width, height / source_height)
    resized = cv2.resize(frame, (max(1, round(source_width * scale)), max(1, round(source_height * scale))))
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    y = (height - resized.shape[0]) // 2
    x = (width - resized.shape[1]) // 2
    canvas[y:y + resized.shape[0], x:x + resized.shape[1]] = resized
    return canvas


def angle_at_progress(feature: dict[str, Any], metric: str, progress: float) -> float | None:
    values = feature["full"].get(metric)
    if values is None or not len(values):
        return None
    index = int(round(progress * (len(values) - 1)))
    return float(values[index])


def write_side_by_side_video(
    standard_video: Path,
    user_video: Path,
    standard_feature: dict[str, Any],
    user_feature: dict[str, Any],
    report: dict[str, Any],
    destination: Path,
) -> None:
    standard_capture = cv2.VideoCapture(str(standard_video))
    user_capture = cv2.VideoCapture(str(user_video))
    if not standard_capture.isOpened() or not user_capture.isOpened():
        raise RuntimeError("Could not open annotated videos for side-by-side output")
    standard_frames = int(standard_capture.get(cv2.CAP_PROP_FRAME_COUNT) or 1)
    user_frames = int(user_capture.get(cv2.CAP_PROP_FRAME_COUNT) or 1)
    standard_fps = float(standard_capture.get(cv2.CAP_PROP_FPS) or 25)
    user_fps = float(user_capture.get(cv2.CAP_PROP_FPS) or 25)
    duration = max(standard_frames / standard_fps, user_frames / user_fps)
    output_fps = 15.0
    output_frames = max(2, min(450, round(duration * output_fps)))
    panel_width, panel_height, header = 640, 360, 84
    writer = cv2.VideoWriter(
        str(destination), cv2.VideoWriter_fourcc(*"mp4v"), output_fps,
        (panel_width * 2, panel_height + header),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not create {destination}")
    for output_index in range(output_frames):
        progress = output_index / max(1, output_frames - 1)
        panels: list[np.ndarray] = []
        for capture, count in ((standard_capture, standard_frames), (user_capture, user_frames)):
            capture.set(cv2.CAP_PROP_POS_FRAMES, int(round(progress * (count - 1))))
            ok, frame = capture.read()
            if not ok:
                frame = np.zeros((panel_height, panel_width, 3), dtype=np.uint8)
            panels.append(fit_frame(frame, panel_width, panel_height))
        canvas = np.full((panel_height + header, panel_width * 2, 3), (22, 28, 38), dtype=np.uint8)
        canvas[header:, :panel_width] = panels[0]
        canvas[header:, panel_width:] = panels[1]
        cv2.putText(canvas, "STANDARD", (20, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(canvas, "YOUR VIDEO", (panel_width + 20, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        standard_knee = angle_at_progress(standard_feature, "knee_angle", progress)
        user_knee = angle_at_progress(user_feature, "knee_angle", progress)
        standard_text = f"Knee: {standard_knee:.1f} deg" if standard_knee is not None else "Knee: unavailable"
        user_text = f"Knee: {user_knee:.1f} deg" if user_knee is not None else "Knee: unavailable"
        cv2.putText(canvas, standard_text, (20, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 210, 255), 1)
        cv2.putText(canvas, user_text, (panel_width + 20, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (120, 200, 255), 1)
        score_text = f"Similarity: {report['similarity_score']:.1f}/100"
        size = cv2.getTextSize(score_text, cv2.FONT_HERSHEY_SIMPLEX, 0.58, 1)[0]
        cv2.putText(canvas, score_text, (panel_width - size[0] // 2, 77), cv2.FONT_HERSHEY_SIMPLEX, 0.58,
                    (120, 240, 160), 1)
        writer.write(canvas)
    writer.release()
    standard_capture.release()
    user_capture.release()


def write_text_report(report: dict[str, Any], path: Path) -> None:
    lines = [
        "Rehabilitation movement comparison",
        "",
        f"Similarity: {report['similarity_score']:.1f}/100",
        f"Method: {report['similarity_method']}",
        f"Assessment: {report['rating']}",
        f"Angle-alignment score: {report['angle_alignment_score']:.1f}/100",
        f"Mean angle difference: {report['weighted_mean_angle_error_deg']:.2f}°",
        f"Standard video side used: {report['standard_selected_side']}",
        f"User video side used: {report['user_selected_side']}",
        f"Standard knee range of motion: {report['standard_knee_range_deg']}°",
        f"User knee range of motion: {report['user_knee_range_deg']}°",
        "",
        "Observations:",
        *[f"- {item}" for item in report["observations"]],
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def default_output_dir(standard: Path, user: Path) -> Path:
    name = f"{clean_label(standard.stem)}_vs_{clean_label(user.stem)}"
    return OUTPUT_ROOT / name[:150]


def main() -> int:
    args = parse_args()
    standard_video = args.standard.expanduser().resolve()
    user_video = args.user.expanduser().resolve()
    for path in (standard_video, user_video):
        if not path.exists():
            raise FileNotFoundError(path)
    if not args.model.exists():
        raise FileNotFoundError(args.model)
    output_dir = (args.output_dir or default_output_dir(standard_video, user_video)).expanduser().resolve()
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    lookup = load_processed_lookup()
    model_holder: dict[str, YOLO] = {}
    device = choose_device(args.device)
    standard_keypoints, standard_annotated = ensure_pose_files(
        standard_video, "standard", lookup, model_holder, device, args.model
    )
    user_keypoints, user_annotated = ensure_pose_files(
        user_video, "user", lookup, model_holder, device, args.model
    )
    standard_data = analyze_csv(standard_keypoints, args.confidence)
    user_data = analyze_csv(user_keypoints, args.confidence)
    alignment = best_alignment(standard_data, user_data, args.max_dtw_frames)
    lstm_result = None
    if args.lstm_model.exists():
        lstm_model, lstm_checkpoint = load_lstm_checkpoint(args.lstm_model, "cpu")
        sequence_length = int(lstm_checkpoint.get("sequence_length", 96))
        standard_sequence = sequence_from_angle_data(standard_data, sequence_length)
        user_sequence = sequence_from_angle_data(user_data, sequence_length)
        lstm_result = lstm_similarity(
            lstm_model,
            standard_sequence,
            user_sequence,
            lstm_checkpoint.get("calibration", {}),
            "cpu",
        )
    else:
        print(f"LSTM model not found at {args.lstm_model}; using angle-alignment fallback.")
    report = build_report(
        standard_video, user_video, standard_data, user_data, alignment,
        lstm_result, args.lstm_model,
    )
    report.update({
        "standard_keypoints_csv": relative_or_absolute(standard_keypoints),
        "user_keypoints_csv": relative_or_absolute(user_keypoints),
        "standard_annotated_video": relative_or_absolute(standard_annotated),
        "user_annotated_video": relative_or_absolute(user_annotated),
        "method": (
            "YOLO Pose 2D keypoints with Siamese LSTM sequence similarity; "
            "dynamic time warping is used for the explanatory angle graph"
        ),
    })

    rows = aligned_rows(alignment)
    write_comparison_csv(rows, output_dir / "comparison_data.csv")
    write_graph(rows, report, output_dir / "comparison_graph.png")
    write_side_by_side_video(
        standard_annotated, user_annotated,
        alignment["standard"], alignment["user"], report,
        output_dir / "comparison_video.mp4",
    )
    (output_dir / "comparison_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    write_text_report(report, output_dir / "comparison_report.txt")
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    (OUTPUT_ROOT / "LATEST_COMPARISON.txt").write_text(str(output_dir) + "\n", encoding="utf-8")

    print("\nComparison complete")
    print(f"Similarity: {report['similarity_score']:.1f}/100 ({report['rating']})")
    print(f"Output: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
