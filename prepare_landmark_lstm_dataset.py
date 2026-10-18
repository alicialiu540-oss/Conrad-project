#!/usr/bin/env python3
"""Convert MediaPipe landmark CSV recordings into an LSTM-ready 3D tensor.

Output X has shape [samples, time_steps, features]. Low-visibility coordinates
are interpolated, while sequences with excessive low visibility at critical
landmarks are rejected. Temporal alignment can use padding or multivariate DTW.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


CLINICAL_LABELS = (
    "Correct Execution",
    "Compensatory Guarding",
    "Reduced Range of Motion",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest_csv", type=Path)
    parser.add_argument("output_npz", type=Path)
    parser.add_argument("--time-steps", type=int, default=128)
    parser.add_argument("--alignment", choices=("padding", "dtw"), default="padding")
    parser.add_argument("--dtw-reference", type=Path)
    parser.add_argument("--min-visibility", type=float, default=0.5)
    parser.add_argument("--max-critical-low-visibility", type=float, default=0.20)
    parser.add_argument(
        "--critical-landmarks",
        default="left_knee,right_knee",
        help="Comma-separated MediaPipe landmark names",
    )
    parser.add_argument("--include-velocity", action="store_true")
    parser.add_argument("--allow-custom-labels", action="store_true")
    parser.add_argument("--report-json", type=Path)
    args = parser.parse_args()
    if args.time_steps < 2:
        parser.error("--time-steps must be at least 2")
    if not 0 <= args.min_visibility <= 1:
        parser.error("--min-visibility must be between 0 and 1")
    if not 0 <= args.max_critical_low_visibility <= 1:
        parser.error("--max-critical-low-visibility must be between 0 and 1")
    return args


def read_manifest(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"Manifest not found: {path}")
    with path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    required = {"sample_id", "landmarks_csv", "label", "subject_id"}
    fields = set(rows[0]) if rows else set()
    missing = sorted(required - fields)
    if missing:
        raise ValueError(f"Manifest is missing columns: {', '.join(missing)}")
    return rows


def landmark_names(fieldnames: list[str]) -> list[str]:
    suffix = "_visibility"
    return [name[: -len(suffix)] for name in fieldnames if name.endswith(suffix)]


def read_landmarks(
    path: Path,
    min_visibility: float,
    critical: list[str],
    max_low_fraction: float,
) -> tuple[np.ndarray | None, list[str], dict]:
    if not path.is_file():
        return None, [], {"status": "rejected", "reason": "file_not_found"}
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            return None, [], {"status": "rejected", "reason": "missing_header"}
        names = landmark_names(reader.fieldnames)
        rows = list(reader)
    if not rows or not names:
        return None, names, {"status": "rejected", "reason": "no_landmark_rows"}
    unknown = sorted(set(critical) - set(names))
    if unknown:
        return None, names, {
            "status": "rejected",
            "reason": f"unknown_critical_landmarks:{','.join(unknown)}",
        }

    frames = len(rows)
    coordinates = np.full((frames, len(names), 3), np.nan, dtype=np.float64)
    visibility = np.full((frames, len(names)), np.nan, dtype=np.float64)
    for frame_index, row in enumerate(rows):
        for landmark_index, name in enumerate(names):
            try:
                visibility[frame_index, landmark_index] = float(row[f"{name}_visibility"])
                coordinates[frame_index, landmark_index] = [
                    float(row[f"{name}_x"]),
                    float(row[f"{name}_y"]),
                    float(row[f"{name}_z"]),
                ]
            except (KeyError, TypeError, ValueError):
                continue

    critical_quality: dict[str, dict[str, float]] = {}
    for name in critical:
        values = visibility[:, names.index(name)]
        low_fraction = float(np.mean(~np.isfinite(values) | (values < min_visibility)))
        critical_quality[name] = {"low_visibility_fraction": low_fraction}
        if low_fraction > max_low_fraction:
            return None, names, {
                "status": "rejected",
                "reason": f"critical_visibility:{name}",
                "critical_quality": critical_quality,
            }

    valid = np.isfinite(visibility) & (visibility >= min_visibility)
    coordinates[~valid] = np.nan
    x = np.arange(frames)
    interpolated_values = 0
    for landmark_index, name in enumerate(names):
        for axis in range(3):
            values = coordinates[:, landmark_index, axis]
            finite = np.isfinite(values)
            if not finite.any():
                return None, names, {
                    "status": "rejected",
                    "reason": f"no_visible_values:{name}",
                    "critical_quality": critical_quality,
                }
            interpolated_values += int((~finite).sum())
            coordinates[:, landmark_index, axis] = np.interp(x, x[finite], values[finite])

    # Center on the midpoint of the hips and scale by mean shoulder/hip width.
    left_hip, right_hip = names.index("left_hip"), names.index("right_hip")
    left_shoulder, right_shoulder = names.index("left_shoulder"), names.index("right_shoulder")
    center = (coordinates[:, left_hip] + coordinates[:, right_hip]) / 2
    shoulder_width = np.linalg.norm(
        coordinates[:, left_shoulder] - coordinates[:, right_shoulder], axis=1
    )
    hip_width = np.linalg.norm(coordinates[:, left_hip] - coordinates[:, right_hip], axis=1)
    scale = (shoulder_width + hip_width) / 2
    positive = scale[np.isfinite(scale) & (scale > 1e-6)]
    if not len(positive):
        return None, names, {"status": "rejected", "reason": "invalid_body_scale"}
    scale[~np.isfinite(scale) | (scale <= 1e-6)] = float(np.median(positive))
    normalized = (coordinates - center[:, None, :]) / scale[:, None, None]
    features = normalized.reshape(frames, -1)
    feature_names = [f"{name}_{axis}" for name in names for axis in ("x", "y", "z")]
    report = {
        "status": "accepted",
        "frames": frames,
        "critical_quality": critical_quality,
        "interpolated_coordinate_values": interpolated_values,
    }
    return features, feature_names, report


def resample(sequence: np.ndarray, length: int) -> np.ndarray:
    if len(sequence) == length:
        return sequence.copy()
    source = np.linspace(0, 1, len(sequence))
    target = np.linspace(0, 1, length)
    return np.column_stack(
        [np.interp(target, source, sequence[:, index]) for index in range(sequence.shape[1])]
    )


def dtw_align(sequence: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Warp sequence onto reference indices using multivariate Euclidean DTW."""
    n, m = len(reference), len(sequence)
    cost = np.full((n + 1, m + 1), np.inf, dtype=np.float64)
    cost[0, 0] = 0.0
    for i in range(1, n + 1):
        distances = np.linalg.norm(sequence - reference[i - 1], axis=1)
        for j in range(1, m + 1):
            cost[i, j] = distances[j - 1] + min(
                cost[i - 1, j], cost[i, j - 1], cost[i - 1, j - 1]
            )
    i, j = n, m
    path: list[tuple[int, int]] = []
    while i > 0 and j > 0:
        path.append((i - 1, j - 1))
        step = int(np.argmin((cost[i - 1, j], cost[i, j - 1], cost[i - 1, j - 1])))
        if step == 0:
            i -= 1
        elif step == 1:
            j -= 1
        else:
            i -= 1
            j -= 1
    path.reverse()
    groups: list[list[np.ndarray]] = [[] for _ in range(n)]
    for ref_index, sequence_index in path:
        groups[ref_index].append(sequence[sequence_index])
    aligned = np.empty_like(reference)
    for index, group in enumerate(groups):
        aligned[index] = np.mean(group, axis=0) if group else sequence[min(index, m - 1)]
    return aligned


def fixed_length(
    sequence: np.ndarray, length: int, alignment: str, reference: np.ndarray | None
) -> tuple[np.ndarray, int]:
    if alignment == "padding":
        if len(sequence) > length:
            return resample(sequence, length), length
        output = np.zeros((length, sequence.shape[1]), dtype=np.float64)
        valid_length = len(sequence)
        output[:valid_length] = sequence
        return output, valid_length
    sequence_resampled = resample(sequence, length)
    if reference is None:
        return sequence_resampled, length
    return dtw_align(sequence_resampled, reference), length


def main() -> None:
    args = parse_args()
    manifest_path = args.manifest_csv.expanduser().resolve()
    rows = read_manifest(manifest_path)
    critical = [item.strip().lower() for item in args.critical_landmarks.split(",") if item.strip()]
    accepted: list[tuple[dict[str, str], np.ndarray]] = []
    audit: list[dict] = []
    feature_names: list[str] | None = None

    for row in rows:
        label = row["label"].strip()
        if not args.allow_custom_labels and label not in CLINICAL_LABELS:
            audit.append({"sample_id": row["sample_id"], "status": "rejected", "reason": "unsupported_label"})
            continue
        path = Path(row["landmarks_csv"]).expanduser()
        if not path.is_absolute():
            path = (manifest_path.parent / path).resolve()
        sequence, names, report = read_landmarks(
            path, args.min_visibility, critical, args.max_critical_low_visibility
        )
        report.update({"sample_id": row["sample_id"], "landmarks_csv": str(path)})
        audit.append(report)
        if sequence is None:
            continue
        current_names = names
        if feature_names is None:
            feature_names = current_names
        elif current_names != feature_names:
            raise ValueError(f"Landmark schema differs for sample {row['sample_id']}")
        accepted.append((row, sequence))

    if not accepted or feature_names is None:
        raise ValueError("No sequences passed the label and visibility requirements")

    dtw_reference: np.ndarray | None = None
    if args.alignment == "dtw":
        if args.dtw_reference:
            ref, ref_names, report = read_landmarks(
                args.dtw_reference.expanduser().resolve(),
                args.min_visibility,
                critical,
                args.max_critical_low_visibility,
            )
            if ref is None:
                raise ValueError(f"DTW reference was rejected: {report['reason']}")
            if ref_names != feature_names:
                raise ValueError("DTW reference landmark schema differs from the dataset")
            dtw_reference = resample(ref, args.time_steps)
        else:
            dtw_reference = resample(accepted[0][1], args.time_steps)

    tensors: list[np.ndarray] = []
    lengths: list[int] = []
    for _, sequence in accepted:
        tensor, valid_length = fixed_length(
            sequence, args.time_steps, args.alignment, dtw_reference
        )
        if args.include_velocity:
            velocity = np.zeros_like(tensor)
            if valid_length > 1:
                velocity[:valid_length] = np.gradient(tensor[:valid_length], axis=0)
            tensor = np.concatenate([tensor, velocity], axis=1)
        tensors.append(tensor.astype(np.float32))
        lengths.append(valid_length)

    if args.include_velocity:
        feature_names = feature_names + [f"{name}_velocity" for name in feature_names]
    labels = sorted({row["label"].strip() for row, _ in accepted})
    label_to_index = {label: index for index, label in enumerate(labels)}
    output = args.output_npz.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        X=np.stack(tensors),
        y=np.asarray([label_to_index[row["label"].strip()] for row, _ in accepted], dtype=np.int64),
        lengths=np.asarray(lengths, dtype=np.int64),
        labels=np.asarray(labels),
        feature_names=np.asarray(feature_names),
        sample_ids=np.asarray([row["sample_id"] for row, _ in accepted]),
        subject_ids=np.asarray([row["subject_id"] for row, _ in accepted]),
        exercise_ids=np.asarray([row.get("exercise_id", "") for row, _ in accepted]),
        splits=np.asarray([row.get("split", "") for row, _ in accepted]),
    )
    report_path = args.report_json or output.with_suffix(".report.json")
    report_path = report_path.expanduser().resolve()
    summary = {
        "tensor_shape": list(np.stack(tensors).shape),
        "axis_meaning": ["samples", "time_steps", "features"],
        "alignment": args.alignment,
        "time_steps": args.time_steps,
        "feature_count": len(feature_names),
        "features": "normalized MediaPipe x/y/z" + (" plus velocity" if args.include_velocity else ""),
        "min_visibility": args.min_visibility,
        "critical_landmarks": critical,
        "max_critical_low_visibility_fraction": args.max_critical_low_visibility,
        "accepted_samples": len(accepted),
        "rejected_samples": len(rows) - len(accepted),
        "labels": labels,
        "audit": audit,
    }
    report_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Saved tensor {tuple(np.stack(tensors).shape)} to {output}")
    print(f"Quality report: {report_path}")


if __name__ == "__main__":
    main()
