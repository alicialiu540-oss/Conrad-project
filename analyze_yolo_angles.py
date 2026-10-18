"""Run YOLO Pose on a video and graph joint angles over time.

Use the original video whenever possible. A previously rendered/annotated video
does not retain YOLO's numerical keypoints and its overlay can reduce accuracy.
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
from ultralytics import YOLO  # noqa: E402


# COCO 17-keypoint indices used by Ultralytics pose models.
KEYPOINTS = {
    "left_shoulder": 5,
    "right_shoulder": 6,
    "left_elbow": 7,
    "right_elbow": 8,
    "left_wrist": 9,
    "right_wrist": 10,
    "left_hip": 11,
    "right_hip": 12,
    "left_knee": 13,
    "right_knee": 14,
    "left_ankle": 15,
    "right_ankle": 16,
}

ANGLE_DEFINITIONS = {
    "left_elbow": ("left_shoulder", "left_elbow", "left_wrist"),
    "right_elbow": ("right_shoulder", "right_elbow", "right_wrist"),
    "left_shoulder": ("left_elbow", "left_shoulder", "left_hip"),
    "right_shoulder": ("right_elbow", "right_shoulder", "right_hip"),
    "left_hip": ("left_shoulder", "left_hip", "left_knee"),
    "right_hip": ("right_shoulder", "right_hip", "right_knee"),
    "left_knee": ("left_hip", "left_knee", "left_ankle"),
    "right_knee": ("right_hip", "right_knee", "right_ankle"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path, help="Original input MP4/MOV/AVI")
    parser.add_argument(
        "--model",
        default="yolo11n-pose.pt",
        help="Pose model name or .pt path (default: yolo11n-pose.pt)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Output directory (default: results/yolo-angles/<video-name>)",
    )
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument(
        "--keypoint-confidence",
        type=float,
        default=0.35,
        help="Minimum confidence for every keypoint in an angle (default: 0.35)",
    )
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument(
        "--smooth-alpha",
        type=float,
        default=0.25,
        help="EMA smoothing strength from 0 to 1 (default: 0.25)",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        help="Optional frame limit, useful for a quick test",
    )
    args = parser.parse_args()
    if not args.video.is_file():
        parser.error(f"video does not exist: {args.video}")
    if not 0 < args.smooth_alpha <= 1:
        parser.error("--smooth-alpha must be greater than 0 and at most 1")
    if not 0 <= args.keypoint_confidence <= 1:
        parser.error("--keypoint-confidence must be between 0 and 1")
    return args


def angle_2d(a, b, c) -> float | None:
    """Return the smaller angle ABC in degrees."""
    ba = (float(a[0] - b[0]), float(a[1] - b[1]))
    bc = (float(c[0] - b[0]), float(c[1] - b[1]))
    denominator = math.hypot(*ba) * math.hypot(*bc)
    if denominator < 1e-6:
        return None
    cosine = max(-1.0, min(1.0, (ba[0] * bc[0] + ba[1] * bc[1]) / denominator))
    return math.degrees(math.acos(cosine))


def ema(previous: float | None, current: float, alpha: float) -> float:
    return current if previous is None else previous + alpha * (current - previous)


def select_primary_person(result, previous_center, frame_diagonal: float):
    """Choose the largest person, then maintain identity by center proximity."""
    if result.boxes is None or result.keypoints is None or len(result.boxes) == 0:
        return None, None
    boxes = result.boxes.xyxy.cpu().numpy()
    centers = [((b[0] + b[2]) / 2, (b[1] + b[3]) / 2) for b in boxes]
    areas = [(b[2] - b[0]) * (b[3] - b[1]) for b in boxes]
    if previous_center is None:
        index = max(range(len(boxes)), key=lambda i: areas[i])
    else:
        distances = [math.dist(center, previous_center) / frame_diagonal for center in centers]
        nearby = [i for i, distance in enumerate(distances) if distance < 0.35]
        index = min(nearby, key=lambda i: distances[i]) if nearby else max(
            range(len(boxes)), key=lambda i: areas[i]
        )
    return index, centers[index]


def create_graph(rows: list[dict], graph_path: Path) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    for axis, side in zip(axes, ("left", "right")):
        plotted = False
        for joint in ("shoulder", "elbow", "hip", "knee"):
            key = f"{side}_{joint}_smooth_deg"
            points = [row for row in rows if row[key] != ""]
            if not points:
                continue
            axis.plot(
                [row["time_s"] for row in points],
                [row[key] for row in points],
                label=joint.title(),
                linewidth=1.5,
            )
            plotted = True
        axis.set_title(f"{side.title()} side")
        axis.set_ylabel("2D angle (degrees)")
        axis.set_ylim(0, 190)
        axis.grid(alpha=0.25)
        if plotted:
            axis.legend(ncol=4, fontsize=9)
        else:
            axis.text(0.5, 0.5, "No valid angles", ha="center", transform=axis.transAxes)
    axes[-1].set_xlabel("Time (seconds)")
    fig.suptitle("YOLO Pose joint angles")
    fig.tight_layout()
    fig.savefig(graph_path, dpi=160)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir or Path("results") / "yolo-angles" / args.video.stem
    output_dir.mkdir(parents=True, exist_ok=True)
    annotated_path = output_dir / f"{args.video.stem}_yolo_angles.mp4"
    csv_path = output_dir / f"{args.video.stem}_yolo_angles.csv"
    graph_path = output_dir / f"{args.video.stem}_yolo_angles.png"
    summary_path = output_dir / f"{args.video.stem}_yolo_summary.json"

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

    video_writer = cv2.VideoWriter(
        str(annotated_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )
    if not video_writer.isOpened():
        capture.release()
        raise SystemExit(f"Could not create output video: {annotated_path}")

    model = YOLO(args.model)
    rows: list[dict] = []
    smoothed: dict[str, float | None] = {name: None for name in ANGLE_DEFINITIONS}
    previous_center = None
    frame_diagonal = math.hypot(width, height)
    frames_with_person = 0
    frame_number = 0

    while True:
        if args.max_frames is not None and frame_number >= args.max_frames:
            break
        ok, frame = capture.read()
        if not ok:
            break
        result = model.predict(
            frame,
            conf=args.confidence,
            imgsz=args.imgsz,
            device="cpu",
            verbose=False,
        )[0]
        annotated = result.plot()
        person_index, new_center = select_primary_person(
            result, previous_center, frame_diagonal
        )
        if new_center is not None:
            previous_center = new_center

        row: dict[str, int | float | str] = {
            "frame": frame_number,
            "time_s": round(frame_number / fps, 4),
            "person_box_confidence": "",
        }
        for angle_name in ANGLE_DEFINITIONS:
            row[f"{angle_name}_raw_deg"] = ""
            row[f"{angle_name}_smooth_deg"] = ""
            row[f"{angle_name}_keypoint_confidence"] = ""

        if person_index is not None:
            frames_with_person += 1
            row["person_box_confidence"] = round(
                float(result.boxes.conf[person_index].cpu()), 4
            )
            data = result.keypoints.data[person_index].cpu().numpy()
            for angle_name, (a_name, b_name, c_name) in ANGLE_DEFINITIONS.items():
                a = data[KEYPOINTS[a_name]]
                b = data[KEYPOINTS[b_name]]
                c = data[KEYPOINTS[c_name]]
                confidence = float(min(a[2], b[2], c[2]))
                row[f"{angle_name}_keypoint_confidence"] = round(confidence, 4)
                if confidence < args.keypoint_confidence:
                    continue
                value = angle_2d(a, b, c)
                if value is None:
                    continue
                smoothed[angle_name] = ema(smoothed[angle_name], value, args.smooth_alpha)
                row[f"{angle_name}_raw_deg"] = round(value, 3)
                row[f"{angle_name}_smooth_deg"] = round(smoothed[angle_name], 3)
                if angle_name.endswith(("elbow", "knee")):
                    x, y = round(float(b[0])), round(float(b[1]))
                    cv2.putText(
                        annotated,
                        f"{smoothed[angle_name]:.0f}",
                        (x + 6, y - 6),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 255),
                        2,
                        cv2.LINE_AA,
                    )

        cv2.putText(
            annotated,
            "2D YOLO estimate - validate before clinical use",
            (16, height - 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 220, 255),
            2,
            cv2.LINE_AA,
        )
        video_writer.write(annotated)
        rows.append(row)
        frame_number += 1
        if total_frames and frame_number % 100 == 0:
            print(f"Processed {frame_number}/{total_frames} frames", flush=True)

    capture.release()
    video_writer.release()

    fields = ["frame", "time_s", "person_box_confidence"]
    for angle_name in ANGLE_DEFINITIONS:
        fields.extend(
            [
                f"{angle_name}_raw_deg",
                f"{angle_name}_smooth_deg",
                f"{angle_name}_keypoint_confidence",
            ]
        )
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    create_graph(rows, graph_path)
    summary: dict[str, object] = {
        "input_video": str(args.video.resolve()),
        "model": args.model,
        "frames_processed": len(rows),
        "person_detected_frames": frames_with_person,
        "person_coverage_percent": round(
            100 * frames_with_person / max(1, len(rows)), 1
        ),
        "fps": fps,
        "angles": {},
        "warning": "Angles are 2D image-plane estimates, not clinical measurements.",
    }
    for angle_name in ANGLE_DEFINITIONS:
        key = f"{angle_name}_smooth_deg"
        values = [float(row[key]) for row in rows if row[key] != ""]
        item: dict[str, int | float] = {
            "valid_frames": len(values),
            "coverage_percent": round(100 * len(values) / max(1, len(rows)), 1),
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
    print(f"Angles CSV: {csv_path.resolve()}")
    print(f"Angle graph: {graph_path.resolve()}")
    print(f"Summary: {summary_path.resolve()}")


if __name__ == "__main__":
    main()
