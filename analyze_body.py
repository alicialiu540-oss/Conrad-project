"""Analyze a full-body video with MediaPipe Pose.

Outputs an annotated video, all 33 pose landmarks, major joint angles, an angle
graph, and a JSON summary. Angles are 2D image-plane estimates.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import mediapipe as mp  # noqa: E402


POSE = mp.solutions.pose

ANGLE_DEFINITIONS = {
    "left_elbow": ("LEFT_SHOULDER", "LEFT_ELBOW", "LEFT_WRIST"),
    "right_elbow": ("RIGHT_SHOULDER", "RIGHT_ELBOW", "RIGHT_WRIST"),
    "left_shoulder": ("LEFT_ELBOW", "LEFT_SHOULDER", "LEFT_HIP"),
    "right_shoulder": ("RIGHT_ELBOW", "RIGHT_SHOULDER", "RIGHT_HIP"),
    "left_hip": ("LEFT_SHOULDER", "LEFT_HIP", "LEFT_KNEE"),
    "right_hip": ("RIGHT_SHOULDER", "RIGHT_HIP", "RIGHT_KNEE"),
    "left_knee": ("LEFT_HIP", "LEFT_KNEE", "LEFT_ANKLE"),
    "right_knee": ("RIGHT_HIP", "RIGHT_KNEE", "RIGHT_ANKLE"),
    "left_ankle": ("LEFT_KNEE", "LEFT_ANKLE", "LEFT_FOOT_INDEX"),
    "right_ankle": ("RIGHT_KNEE", "RIGHT_ANKLE", "RIGHT_FOOT_INDEX"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path, help="Input MP4/MOV/AVI video")
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Output directory (default: results/mediapipe-body/<video-name>)",
    )
    parser.add_argument(
        "--model-complexity",
        type=int,
        choices=(0, 1, 2),
        default=1,
        help="0 is fastest, 2 is most accurate (default: 1)",
    )
    parser.add_argument(
        "--min-visibility",
        type=float,
        default=0.5,
        help="Minimum landmark visibility for angle calculation (default: 0.5)",
    )
    parser.add_argument(
        "--smooth-alpha",
        type=float,
        default=0.25,
        help="EMA angle smoothing strength from 0 to 1 (default: 0.25)",
    )
    parser.add_argument("--min-detection-confidence", type=float, default=0.5)
    parser.add_argument("--min-tracking-confidence", type=float, default=0.5)
    args = parser.parse_args()
    if not args.video.is_file():
        parser.error(f"video does not exist: {args.video}")
    if not 0 <= args.min_visibility <= 1:
        parser.error("--min-visibility must be between 0 and 1")
    if not 0 < args.smooth_alpha <= 1:
        parser.error("--smooth-alpha must be greater than 0 and at most 1")
    return args


def point(landmark, width: int, height: int) -> tuple[float, float]:
    return landmark.x * width, landmark.y * height


def joint_angle(a, b, c, width: int, height: int) -> float | None:
    """Return the smaller 2D angle ABC in degrees."""
    ax, ay = point(a, width, height)
    bx, by = point(b, width, height)
    cx, cy = point(c, width, height)
    ba = (ax - bx, ay - by)
    bc = (cx - bx, cy - by)
    denominator = math.hypot(*ba) * math.hypot(*bc)
    if denominator < 1e-6:
        return None
    cosine = max(-1.0, min(1.0, (ba[0] * bc[0] + ba[1] * bc[1]) / denominator))
    return math.degrees(math.acos(cosine))


def ema(previous: float | None, current: float, alpha: float) -> float:
    return current if previous is None else previous + alpha * (current - previous)


def landmark_by_name(landmarks, name: str):
    return landmarks[getattr(POSE.PoseLandmark, name).value]


def graph_angles(rows: list[dict], path: Path) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    for axis, side in zip(axes, ("left", "right")):
        plotted = False
        for joint in ("shoulder", "elbow", "hip", "knee", "ankle"):
            key = f"{side}_{joint}_smooth_deg"
            points = [row for row in rows if row[key] != ""]
            if not points:
                continue
            axis.plot(
                [row["time_s"] for row in points],
                [row[key] for row in points],
                label=joint.title(),
                linewidth=1.4,
            )
            plotted = True
        axis.set_title(f"{side.title()} side")
        axis.set_ylabel("2D angle (degrees)")
        axis.set_ylim(0, 190)
        axis.grid(alpha=0.25)
        if plotted:
            axis.legend(ncol=5, fontsize=8)
        else:
            axis.text(0.5, 0.5, "No valid angles", ha="center", transform=axis.transAxes)
    axes[-1].set_xlabel("Time (seconds)")
    fig.suptitle("MediaPipe full-body joint angles")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir or Path("results") / "mediapipe-body" / args.video.stem
    output_dir.mkdir(parents=True, exist_ok=True)
    annotated_path = output_dir / f"{args.video.stem}_body_annotated.mp4"
    landmarks_path = output_dir / f"{args.video.stem}_landmarks.csv"
    angles_path = output_dir / f"{args.video.stem}_joint_angles.csv"
    graph_path = output_dir / f"{args.video.stem}_joint_angles.png"
    summary_path = output_dir / f"{args.video.stem}_summary.json"

    capture = cv2.VideoCapture(str(args.video))
    if not capture.isOpened():
        raise SystemExit(f"Could not open video: {args.video}")
    fps = capture.get(cv2.CAP_PROP_FPS)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if fps <= 0 or width <= 0 or height <= 0:
        capture.release()
        raise SystemExit("Video has invalid FPS or dimensions")

    writer = cv2.VideoWriter(
        str(annotated_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )
    if not writer.isOpened():
        capture.release()
        raise SystemExit(f"Could not create output video: {annotated_path}")

    landmark_names = [landmark.name.lower() for landmark in POSE.PoseLandmark]
    landmark_fields = ["frame", "time_s"]
    for name in landmark_names:
        landmark_fields.extend(
            [f"{name}_x", f"{name}_y", f"{name}_z", f"{name}_visibility"]
        )
    angle_fields = ["frame", "time_s"]
    for angle_name in ANGLE_DEFINITIONS:
        angle_fields.extend([f"{angle_name}_raw_deg", f"{angle_name}_smooth_deg"])

    landmark_rows: list[dict] = []
    angle_rows: list[dict] = []
    smoothed: dict[str, float | None] = {name: None for name in ANGLE_DEFINITIONS}
    detected_frames = 0
    drawing = mp.solutions.drawing_utils

    with POSE.Pose(
        static_image_mode=False,
        model_complexity=args.model_complexity,
        smooth_landmarks=True,
        enable_segmentation=False,
        min_detection_confidence=args.min_detection_confidence,
        min_tracking_confidence=args.min_tracking_confidence,
    ) as detector:
        frame_number = 0
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            time_s = round(frame_number / fps, 4)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            result = detector.process(rgb)
            rgb.flags.writeable = True
            landmark_row: dict[str, int | float | str] = {
                "frame": frame_number,
                "time_s": time_s,
            }
            angle_row: dict[str, int | float | str] = {
                "frame": frame_number,
                "time_s": time_s,
            }

            if result.pose_landmarks:
                detected_frames += 1
                landmarks = result.pose_landmarks.landmark
                drawing.draw_landmarks(
                    frame,
                    result.pose_landmarks,
                    POSE.POSE_CONNECTIONS,
                    drawing.DrawingSpec(color=(80, 220, 80), thickness=2, circle_radius=2),
                    drawing.DrawingSpec(color=(40, 130, 255), thickness=2),
                )
                for name, landmark in zip(landmark_names, landmarks):
                    landmark_row[f"{name}_x"] = round(landmark.x, 6)
                    landmark_row[f"{name}_y"] = round(landmark.y, 6)
                    landmark_row[f"{name}_z"] = round(landmark.z, 6)
                    landmark_row[f"{name}_visibility"] = round(landmark.visibility, 6)

                for angle_name, (a_name, b_name, c_name) in ANGLE_DEFINITIONS.items():
                    a = landmark_by_name(landmarks, a_name)
                    b = landmark_by_name(landmarks, b_name)
                    c = landmark_by_name(landmarks, c_name)
                    if min(a.visibility, b.visibility, c.visibility) < args.min_visibility:
                        angle_row[f"{angle_name}_raw_deg"] = ""
                        angle_row[f"{angle_name}_smooth_deg"] = ""
                        continue
                    value = joint_angle(a, b, c, width, height)
                    if value is None:
                        angle_row[f"{angle_name}_raw_deg"] = ""
                        angle_row[f"{angle_name}_smooth_deg"] = ""
                        continue
                    smoothed[angle_name] = ema(
                        smoothed[angle_name], value, args.smooth_alpha
                    )
                    angle_row[f"{angle_name}_raw_deg"] = round(value, 3)
                    angle_row[f"{angle_name}_smooth_deg"] = round(
                        smoothed[angle_name], 3
                    )
                    if angle_name.endswith(("elbow", "knee")):
                        bx, by = point(b, width, height)
                        cv2.putText(
                            frame,
                            f"{smoothed[angle_name]:.0f}",
                            (round(bx) + 6, round(by) - 6),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (0, 255, 255),
                            2,
                            cv2.LINE_AA,
                        )
            else:
                for name in landmark_names:
                    for suffix in ("x", "y", "z", "visibility"):
                        landmark_row[f"{name}_{suffix}"] = ""
                for angle_name in ANGLE_DEFINITIONS:
                    angle_row[f"{angle_name}_raw_deg"] = ""
                    angle_row[f"{angle_name}_smooth_deg"] = ""

            # Ensure columns exist when pose was found but an angle was unavailable.
            for angle_name in ANGLE_DEFINITIONS:
                angle_row.setdefault(f"{angle_name}_raw_deg", "")
                angle_row.setdefault(f"{angle_name}_smooth_deg", "")

            landmark_rows.append(landmark_row)
            angle_rows.append(angle_row)
            cv2.putText(
                frame,
                "2D MediaPipe estimate - validate before clinical use",
                (16, height - 18),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 220, 255),
                2,
                cv2.LINE_AA,
            )
            writer.write(frame)
            frame_number += 1
            if total_frames and frame_number % 100 == 0:
                print(f"Processed {frame_number}/{total_frames} frames", flush=True)

    capture.release()
    writer.release()

    with landmarks_path.open("w", newline="", encoding="utf-8") as handle:
        csv_writer = csv.DictWriter(handle, fieldnames=landmark_fields)
        csv_writer.writeheader()
        csv_writer.writerows(landmark_rows)
    with angles_path.open("w", newline="", encoding="utf-8") as handle:
        csv_writer = csv.DictWriter(handle, fieldnames=angle_fields)
        csv_writer.writeheader()
        csv_writer.writerows(angle_rows)

    graph_angles(angle_rows, graph_path)
    summary: dict[str, object] = {
        "input_video": str(args.video.resolve()),
        "frames_processed": len(angle_rows),
        "pose_detected_frames": detected_frames,
        "pose_coverage_percent": round(
            100 * detected_frames / max(1, len(angle_rows)), 1
        ),
        "fps": fps,
        "angles": {},
        "warning": "Angles are 2D image-plane estimates, not clinical measurements.",
    }
    for angle_name in ANGLE_DEFINITIONS:
        key = f"{angle_name}_smooth_deg"
        values = [float(row[key]) for row in angle_rows if row[key] != ""]
        item: dict[str, int | float] = {
            "valid_frames": len(values),
            "coverage_percent": round(100 * len(values) / max(1, len(angle_rows)), 1),
        }
        if values:
            item.update(
                {
                    "minimum_deg": round(min(values), 2),
                    "maximum_deg": round(max(values), 2),
                    "range_deg": round(max(values) - min(values), 2),
                }
            )
        summary["angles"][angle_name] = item
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    print("Analysis complete")
    print(f"Annotated video: {annotated_path.resolve()}")
    print(f"Landmarks CSV: {landmarks_path.resolve()}")
    print(f"Joint angles CSV: {angles_path.resolve()}")
    print(f"Angle graph: {graph_path.resolve()}")
    print(f"Summary: {summary_path.resolve()}")


if __name__ == "__main__":
    main()
