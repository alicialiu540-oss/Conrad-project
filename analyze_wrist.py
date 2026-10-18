"""Analyze wrist motion in a video with MediaPipe Pose and Hands.

The reported angle is the signed 2D angle from the forearm direction
(elbow -> wrist) to the hand direction (wrist -> middle-finger MCP). A neutral,
straight wrist is near zero degrees. The sign depends on camera orientation, so
do not interpret it as flexion versus extension without validating the view.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import mediapipe as mp  # noqa: E402


@dataclass
class WristMeasurement:
    angle: float
    hand_confidence: float
    pose_visibility: float
    elbow: tuple[int, int]
    wrist: tuple[int, int]
    middle_mcp: tuple[int, int]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path, help="Input MP4/MOV/AVI video")
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Output directory (default: results/mediapipe/<video-name>)",
    )
    parser.add_argument(
        "--side",
        choices=("both", "left", "right"),
        default="both",
        help="Which anatomical side to report (default: both)",
    )
    parser.add_argument(
        "--smooth-alpha",
        type=float,
        default=0.25,
        help="EMA smoothing strength from 0 to 1 (default: 0.25)",
    )
    parser.add_argument(
        "--min-detection-confidence", type=float, default=0.5
    )
    parser.add_argument("--min-tracking-confidence", type=float, default=0.5)
    args = parser.parse_args()
    if not args.video.is_file():
        parser.error(f"video does not exist: {args.video}")
    if not 0 < args.smooth_alpha <= 1:
        parser.error("--smooth-alpha must be greater than 0 and at most 1")
    return args


def pixel(landmark, width: int, height: int) -> tuple[int, int]:
    return round(landmark.x * width), round(landmark.y * height)


def signed_angle(
    elbow: tuple[int, int], wrist: tuple[int, int], middle_mcp: tuple[int, int]
) -> float | None:
    forearm = (wrist[0] - elbow[0], wrist[1] - elbow[1])
    hand = (middle_mcp[0] - wrist[0], middle_mcp[1] - wrist[1])
    if math.hypot(*forearm) < 5 or math.hypot(*hand) < 5:
        return None
    cross = forearm[0] * hand[1] - forearm[1] * hand[0]
    dot = forearm[0] * hand[0] + forearm[1] * hand[1]
    return math.degrees(math.atan2(cross, dot))


def ema(previous: float | None, current: float, alpha: float) -> float:
    if previous is None:
        return current
    # Smooth across the -180/180 boundary correctly.
    delta = (current - previous + 180) % 360 - 180
    return previous + alpha * delta


def detect_measurements(
    pose_result,
    hands_result,
    width: int,
    height: int,
) -> dict[str, WristMeasurement]:
    if not pose_result.pose_landmarks or not hands_result.multi_hand_landmarks:
        return {}

    pose = pose_result.pose_landmarks.landmark
    pose_indices = {
        "left": (
            mp.solutions.pose.PoseLandmark.LEFT_ELBOW.value,
            mp.solutions.pose.PoseLandmark.LEFT_WRIST.value,
        ),
        "right": (
            mp.solutions.pose.PoseLandmark.RIGHT_ELBOW.value,
            mp.solutions.pose.PoseLandmark.RIGHT_WRIST.value,
        ),
    }

    candidates: list[tuple[float, str, WristMeasurement]] = []
    for hand_number, hand_landmarks in enumerate(hands_result.multi_hand_landmarks):
        hand = hand_landmarks.landmark
        hand_wrist = pixel(hand[mp.solutions.hands.HandLandmark.WRIST], width, height)
        middle_mcp = pixel(
            hand[mp.solutions.hands.HandLandmark.MIDDLE_FINGER_MCP], width, height
        )
        hand_confidence = 0.0
        if hands_result.multi_handedness and hand_number < len(
            hands_result.multi_handedness
        ):
            hand_confidence = hands_result.multi_handedness[hand_number].classification[
                0
            ].score

        for side, (elbow_index, wrist_index) in pose_indices.items():
            pose_wrist_lm = pose[wrist_index]
            pose_elbow_lm = pose[elbow_index]
            pose_wrist = pixel(pose_wrist_lm, width, height)
            elbow = pixel(pose_elbow_lm, width, height)
            distance = math.dist(hand_wrist, pose_wrist) / math.hypot(width, height)
            angle = signed_angle(elbow, hand_wrist, middle_mcp)
            if angle is None:
                continue
            measurement = WristMeasurement(
                angle=angle,
                hand_confidence=hand_confidence,
                pose_visibility=min(pose_elbow_lm.visibility, pose_wrist_lm.visibility),
                elbow=elbow,
                wrist=hand_wrist,
                middle_mcp=middle_mcp,
            )
            candidates.append((distance, side, measurement))

    # Greedily match hands to the nearest pose wrist. Reject distant matches.
    matched: dict[str, WristMeasurement] = {}
    used_wrist_points: set[tuple[int, int]] = set()
    for distance, side, measurement in sorted(candidates, key=lambda item: item[0]):
        if distance > 0.22 or side in matched or measurement.wrist in used_wrist_points:
            continue
        matched[side] = measurement
        used_wrist_points.add(measurement.wrist)
    return matched


def draw_measurement(
    frame,
    side: str,
    measurement: WristMeasurement,
    smoothed_angle: float,
) -> None:
    color = (40, 220, 40) if side == "left" else (40, 170, 255)
    cv2.line(frame, measurement.elbow, measurement.wrist, color, 3)
    cv2.line(frame, measurement.wrist, measurement.middle_mcp, color, 3)
    for point in (measurement.elbow, measurement.wrist, measurement.middle_mcp):
        cv2.circle(frame, point, 6, color, -1)
        cv2.circle(frame, point, 8, (255, 255, 255), 2)
    label = f"{side}: {smoothed_angle:+.1f} deg"
    label_position = (measurement.wrist[0] + 12, measurement.wrist[1] - 12)
    cv2.putText(
        frame,
        label,
        label_position,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 0, 0),
        4,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        label,
        label_position,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        color,
        2,
        cv2.LINE_AA,
    )


def save_graph(rows: list[dict], graph_path: Path, sides: list[str]) -> None:
    plt.figure(figsize=(11, 5))
    plotted = False
    for side in sides:
        points = [row for row in rows if row[f"{side}_angle_smooth_deg"] != ""]
        if not points:
            continue
        plt.plot(
            [row["time_s"] for row in points],
            [row[f"{side}_angle_smooth_deg"] for row in points],
            label=f"{side.title()} wrist",
            linewidth=1.8,
        )
        plotted = True
    plt.axhline(0, color="gray", linewidth=1, linestyle="--")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Signed 2D wrist angle (degrees)")
    plt.title("MediaPipe wrist motion")
    plt.grid(alpha=0.25)
    if plotted:
        plt.legend()
    else:
        plt.text(0.5, 0.5, "No valid wrist measurements", ha="center")
    plt.tight_layout()
    plt.savefig(graph_path, dpi=160)
    plt.close()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir or Path("results") / "mediapipe" / args.video.stem
    output_dir.mkdir(parents=True, exist_ok=True)
    annotated_path = output_dir / f"{args.video.stem}_wrist_annotated.mp4"
    csv_path = output_dir / f"{args.video.stem}_wrist_angles.csv"
    graph_path = output_dir / f"{args.video.stem}_wrist_angles.png"
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
        str(annotated_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        capture.release()
        raise SystemExit(f"Could not create output video: {annotated_path}")

    selected_sides = ["left", "right"] if args.side == "both" else [args.side]
    smoothed: dict[str, float | None] = {"left": None, "right": None}
    rows: list[dict] = []
    drawing = mp.solutions.drawing_utils

    with mp.solutions.pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        smooth_landmarks=True,
        min_detection_confidence=args.min_detection_confidence,
        min_tracking_confidence=args.min_tracking_confidence,
    ) as pose_detector, mp.solutions.hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        model_complexity=1,
        min_detection_confidence=args.min_detection_confidence,
        min_tracking_confidence=args.min_tracking_confidence,
    ) as hand_detector:
        frame_number = 0
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            pose_result = pose_detector.process(rgb)
            hands_result = hand_detector.process(rgb)
            rgb.flags.writeable = True

            if hands_result.multi_hand_landmarks:
                for hand_landmarks in hands_result.multi_hand_landmarks:
                    drawing.draw_landmarks(
                        frame,
                        hand_landmarks,
                        mp.solutions.hands.HAND_CONNECTIONS,
                    )

            measurements = detect_measurements(
                pose_result, hands_result, width, height
            )
            row: dict[str, float | int | str] = {
                "frame": frame_number,
                "time_s": round(frame_number / fps, 4),
            }
            for side in ("left", "right"):
                measurement = measurements.get(side)
                if measurement and side in selected_sides:
                    smoothed[side] = ema(
                        smoothed[side], measurement.angle, args.smooth_alpha
                    )
                    row[f"{side}_angle_raw_deg"] = round(measurement.angle, 3)
                    row[f"{side}_angle_smooth_deg"] = round(smoothed[side], 3)
                    row[f"{side}_hand_confidence"] = round(
                        measurement.hand_confidence, 4
                    )
                    row[f"{side}_pose_visibility"] = round(
                        measurement.pose_visibility, 4
                    )
                    draw_measurement(frame, side, measurement, smoothed[side])
                else:
                    row[f"{side}_angle_raw_deg"] = ""
                    row[f"{side}_angle_smooth_deg"] = ""
                    row[f"{side}_hand_confidence"] = ""
                    row[f"{side}_pose_visibility"] = ""
            rows.append(row)

            cv2.putText(
                frame,
                "2D image-plane estimate - validate before clinical use",
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

    fieldnames = [
        "frame",
        "time_s",
        "left_angle_raw_deg",
        "left_angle_smooth_deg",
        "left_hand_confidence",
        "left_pose_visibility",
        "right_angle_raw_deg",
        "right_angle_smooth_deg",
        "right_hand_confidence",
        "right_pose_visibility",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer_csv = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer_csv.writeheader()
        writer_csv.writerows(rows)

    save_graph(rows, graph_path, selected_sides)
    summary: dict[str, object] = {
        "input_video": str(args.video.resolve()),
        "fps": fps,
        "frames_processed": len(rows),
        "angle_definition": "signed 2D angle: elbow -> wrist -> middle-finger MCP",
        "sides": {},
    }
    for side in selected_sides:
        values = [
            float(row[f"{side}_angle_smooth_deg"])
            for row in rows
            if row[f"{side}_angle_smooth_deg"] != ""
        ]
        side_summary = {
            "valid_frames": len(values),
            "coverage_percent": round(100 * len(values) / max(1, len(rows)), 1),
        }
        if values:
            side_summary.update(
                {
                    "minimum_deg": round(min(values), 2),
                    "maximum_deg": round(max(values), 2),
                    "range_of_motion_deg": round(max(values) - min(values), 2),
                }
            )
        summary["sides"][side] = side_summary
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    print("Analysis complete")
    print(f"Annotated video: {annotated_path.resolve()}")
    print(f"Measurements CSV: {csv_path.resolve()}")
    print(f"Angle graph: {graph_path.resolve()}")
    print(f"Summary: {summary_path.resolve()}")


if __name__ == "__main__":
    main()
