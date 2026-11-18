#!/usr/bin/env python3
"""Create a side-by-side reviewer video with an LSTM deviation timeline."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import cv2
import numpy as np


WIDTH, HEIGHT = 1280, 720
VIDEO_Y, VIDEO_H, VIDEO_W = 58, 390, 510
CHART_X, CHART_Y, CHART_W, CHART_H = 70, 515, 1140, 150


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a synchronized reference/movement video with LSTM scores."
    )
    parser.add_argument("reference_video", type=Path)
    parser.add_argument("movement_video", type=Path)
    parser.add_argument("window_scores_csv", type=Path)
    parser.add_argument("summary_json", type=Path)
    parser.add_argument("output_video", type=Path)
    parser.add_argument("--model-label", default="LSTM", help="Label shown in the reviewer display")
    return parser.parse_args()


def fit_frame(frame: np.ndarray, width: int, height: int) -> np.ndarray:
    canvas = np.full((height, width, 3), 25, dtype=np.uint8)
    scale = min(width / frame.shape[1], height / frame.shape[0])
    resized = cv2.resize(
        frame,
        (max(1, int(frame.shape[1] * scale)), max(1, int(frame.shape[0] * scale))),
        interpolation=cv2.INTER_AREA,
    )
    y = (height - resized.shape[0]) // 2
    x = (width - resized.shape[1]) // 2
    canvas[y : y + resized.shape[0], x : x + resized.shape[1]] = resized
    return canvas


def text(
    image: np.ndarray,
    value: str,
    position: tuple[int, int],
    scale: float = 0.58,
    color: tuple[int, int, int] = (235, 235, 235),
    thickness: int = 1,
) -> None:
    cv2.putText(
        image,
        value,
        position,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        thickness,
        cv2.LINE_AA,
    )


def read_scores(path: Path) -> dict[str, np.ndarray]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("Window score CSV is empty")
    return {
        "time": np.asarray([float(row["center_time_s"]) for row in rows]),
        "error": np.asarray([float(row["reconstruction_error"]) for row in rows]),
        "percentile": np.asarray([float(row["reference_percentile"]) for row in rows]),
    }


def draw_chart(
    canvas: np.ndarray,
    times: np.ndarray,
    errors: np.ndarray,
    threshold: float,
    current_time: float,
    model_label: str,
) -> None:
    maximum = max(float(errors.max()), threshold) * 1.08
    x_values = CHART_X + (times / max(times[-1], 1e-6) * CHART_W).astype(int)
    y_values = CHART_Y + CHART_H - (errors / maximum * CHART_H).astype(int)
    cv2.rectangle(canvas, (CHART_X, CHART_Y), (CHART_X + CHART_W, CHART_Y + CHART_H), (90, 90, 90), 1)
    threshold_y = int(CHART_Y + CHART_H - threshold / maximum * CHART_H)
    cv2.line(canvas, (CHART_X, threshold_y), (CHART_X + CHART_W, threshold_y), (70, 100, 230), 2)
    points = np.column_stack([x_values, y_values]).astype(np.int32)
    cv2.polylines(canvas, [points], False, (225, 160, 55), 3, cv2.LINE_AA)
    cursor_x = int(CHART_X + min(max(current_time / max(times[-1], 1e-6), 0), 1) * CHART_W)
    cv2.line(canvas, (cursor_x, CHART_Y), (cursor_x, CHART_Y + CHART_H), (240, 240, 240), 2)
    text(canvas, f"{model_label} reconstruction error over movement time", (CHART_X, CHART_Y - 13), 0.55)
    text(canvas, "reference threshold", (CHART_X + CHART_W - 180, threshold_y - 7), 0.42, (110, 145, 255))
    text(canvas, "0 s", (CHART_X, CHART_Y + CHART_H + 25), 0.42, (180, 180, 180))
    text(canvas, f"{times[-1]:.1f} s", (CHART_X + CHART_W - 45, CHART_Y + CHART_H + 25), 0.42, (180, 180, 180))


def main() -> None:
    args = parse_args()
    paths = [args.reference_video, args.movement_video, args.window_scores_csv, args.summary_json]
    for path in paths:
        if not path.expanduser().is_file():
            raise FileNotFoundError(f"File not found: {path}")
    scores = read_scores(args.window_scores_csv.expanduser().resolve())
    summary = json.loads(args.summary_json.expanduser().read_text(encoding="utf-8"))
    threshold = float(summary["threshold"])
    feature_notes = summary.get("largest_angle_deviations", [])[:3]

    reference = cv2.VideoCapture(str(args.reference_video.expanduser().resolve()))
    movement = cv2.VideoCapture(str(args.movement_video.expanduser().resolve()))
    if not reference.isOpened() or not movement.isOpened():
        raise RuntimeError("Could not open one or both videos")
    ref_count = int(reference.get(cv2.CAP_PROP_FRAME_COUNT))
    move_count = int(movement.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = movement.get(cv2.CAP_PROP_FPS) or 30.0
    output = args.output_video.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output), cv2.VideoWriter_fourcc(*"mp4v"), fps, (WIDTH, HEIGHT)
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not create output video: {output}")

    for frame_number in range(move_count):
        ok_move, move_frame = movement.read()
        if not ok_move:
            break
        progress = frame_number / max(move_count - 1, 1)
        reference.set(cv2.CAP_PROP_POS_FRAMES, round(progress * max(ref_count - 1, 0)))
        ok_ref, ref_frame = reference.read()
        if not ok_ref:
            ref_frame = np.zeros_like(move_frame)
        current_time = frame_number / fps
        error = float(np.interp(current_time, scores["time"], scores["error"]))
        percentile = float(np.interp(current_time, scores["time"], scores["percentile"]))
        similarity = float(np.clip(100 * math.exp(-math.log(2) * error / threshold), 0, 100))
        above = error > threshold

        canvas = np.full((HEIGHT, WIDTH, 3), 18, dtype=np.uint8)
        canvas[VIDEO_Y : VIDEO_Y + VIDEO_H, 20 : 20 + VIDEO_W] = fit_frame(ref_frame, VIDEO_W, VIDEO_H)
        canvas[VIDEO_Y : VIDEO_Y + VIDEO_H, 550 : 550 + VIDEO_W] = fit_frame(move_frame, VIDEO_W, VIDEO_H)
        text(canvas, "REFERENCE (normalized time)", (20, 38), 0.65)
        text(canvas, "MOVEMENT BEING REVIEWED", (550, 38), 0.65)

        status_color = (70, 165, 245) if above else (95, 205, 120)
        text(canvas, f"{args.model_label} similarity", (1080, 82), 0.56)
        text(canvas, f"{similarity:5.1f} / 100", (1080, 123), 0.82, status_color, 2)
        text(canvas, "Prototype index", (1080, 150), 0.45, (175, 175, 175))
        text(canvas, "Deviation error", (1080, 192), 0.50)
        text(canvas, f"{error:.3f}", (1080, 222), 0.65, status_color, 2)
        text(canvas, f"Threshold {threshold:.3f}", (1080, 250), 0.43, (175, 175, 175))
        text(canvas, f"Ref percentile {percentile:.0f}", (1080, 276), 0.43, (175, 175, 175))
        text(canvas, "Higher deviation" if above else "Within ref. range", (1080, 311), 0.50, status_color, 2)
        text(canvas, "Largest angle", (1080, 351), 0.46)
        text(canvas, "differences overall:", (1080, 373), 0.46)
        for offset, item in enumerate(feature_notes):
            label = str(item["angle"]).replace("_", " ")
            text(canvas, f"{label}: {item['mae_deg']:.1f} deg", (1080, 400 + 22 * offset), 0.40, (195, 195, 195))

        draw_chart(canvas, scores["time"], scores["error"], threshold, current_time, args.model_label)
        text(canvas, "Engineering comparison only; not a clinical pass/fail score.", (70, 704), 0.43, (170, 170, 170))
        writer.write(canvas)

    writer.release()
    reference.release()
    movement.release()
    print(f"Reviewer video saved to: {output}")


if __name__ == "__main__":
    main()
