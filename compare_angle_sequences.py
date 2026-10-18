"""Compare a target joint-angle sequence with a reference exercise sequence.

The program aligns different-duration movements with multivariate Dynamic Time
Warping (DTW), optionally checks a left/right mirrored reference, and reports
per-joint angular deviation. It is a one-reference comparison, not a validated
clinical normativity score.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


@dataclass
class AngleSequence:
    path: Path
    frames: np.ndarray
    times: np.ndarray
    features: list[str]
    values: np.ndarray


@dataclass
class Alignment:
    path: list[tuple[int, int]]
    mean_cost: float
    mirrored: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("target_csv", type=Path, help="Movement to evaluate")
    parser.add_argument("reference_csv", type=Path, help="Reference movement")
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Output directory (default: results/angle-comparison/<target-name>)",
    )
    parser.add_argument(
        "--mirror",
        choices=("auto", "yes", "no"),
        default="auto",
        help="Swap reference left/right angles; auto tests both (default: auto)",
    )
    parser.add_argument(
        "--band-ratio",
        type=float,
        default=0.20,
        help="DTW search width around scaled diagonal (default: 0.20)",
    )
    parser.add_argument(
        "--minimum-common-angles",
        type=int,
        default=3,
        help="Minimum jointly observed angles for matching frames (default: 3)",
    )
    args = parser.parse_args()
    for path in (args.target_csv, args.reference_csv):
        if not path.is_file():
            parser.error(f"CSV does not exist: {path}")
    if not 0 < args.band_ratio <= 1:
        parser.error("--band-ratio must be greater than 0 and at most 1")
    if args.minimum_common_angles < 1:
        parser.error("--minimum-common-angles must be at least 1")
    return args


def display_name(feature: str) -> str:
    return feature.removesuffix("_smooth_deg").replace("_", " ").title()


def load_sequence(path: Path) -> AngleSequence:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"CSV has no data rows: {path}")
    required = {"frame", "time_s"}
    missing = required.difference(rows[0])
    if missing:
        raise ValueError(f"CSV is missing columns {sorted(missing)}: {path}")
    features = [name for name in rows[0] if name.endswith("_smooth_deg")]
    if not features:
        raise ValueError(f"CSV has no *_smooth_deg columns: {path}")

    values = np.full((len(rows), len(features)), np.nan, dtype=float)
    for row_index, row in enumerate(rows):
        for column_index, feature in enumerate(features):
            value = row.get(feature, "").strip()
            if value:
                values[row_index, column_index] = float(value)
    frames = np.array([int(float(row["frame"])) for row in rows], dtype=int)
    times = np.array([float(row["time_s"]) for row in rows], dtype=float)
    return AngleSequence(path, frames, times, features, values)


def select_common_features(
    target: AngleSequence, reference: AngleSequence
) -> tuple[AngleSequence, AngleSequence]:
    features = [name for name in target.features if name in reference.features]
    if not features:
        raise ValueError("The two CSV files have no matching angle columns")

    def subset(sequence: AngleSequence) -> AngleSequence:
        indices = [sequence.features.index(name) for name in features]
        return AngleSequence(
            sequence.path,
            sequence.frames,
            sequence.times,
            features,
            sequence.values[:, indices],
        )

    return subset(target), subset(reference)


def trim_unusable_edges(sequence: AngleSequence, minimum_valid: int) -> AngleSequence:
    valid_counts = np.isfinite(sequence.values).sum(axis=1)
    usable = np.flatnonzero(valid_counts >= minimum_valid)
    if not len(usable):
        raise ValueError(
            f"No frames in {sequence.path.name} have at least {minimum_valid} angles"
        )
    start, stop = int(usable[0]), int(usable[-1]) + 1
    return AngleSequence(
        sequence.path,
        sequence.frames[start:stop],
        sequence.times[start:stop],
        sequence.features,
        sequence.values[start:stop],
    )


def mirrored_reference(reference: AngleSequence) -> AngleSequence:
    indices: list[int] = []
    for feature in reference.features:
        if feature.startswith("left_"):
            counterpart = "right_" + feature[len("left_") :]
        elif feature.startswith("right_"):
            counterpart = "left_" + feature[len("right_") :]
        else:
            counterpart = feature
        indices.append(reference.features.index(counterpart))
    return AngleSequence(
        reference.path,
        reference.frames,
        reference.times,
        reference.features,
        reference.values[:, indices],
    )


def local_costs(
    target: np.ndarray, reference: np.ndarray, minimum_common: int
) -> np.ndarray:
    """Calculate frame-pair RMSE using only jointly observed angles."""
    costs = np.full((len(target), len(reference)), np.inf, dtype=np.float32)
    reference_finite = np.isfinite(reference)
    for i, target_row in enumerate(target):
        jointly_valid = reference_finite & np.isfinite(target_row)[None, :]
        counts = jointly_valid.sum(axis=1)
        differences = np.where(jointly_valid, reference - target_row, 0.0)
        squared_sum = np.square(differences).sum(axis=1)
        eligible = counts >= minimum_common
        costs[i, eligible] = np.sqrt(squared_sum[eligible] / counts[eligible])
    return costs


def dtw_align(
    target: np.ndarray,
    reference: np.ndarray,
    minimum_common: int,
    band_ratio: float,
    mirrored: bool,
) -> Alignment:
    costs = local_costs(target, reference, minimum_common)
    n, m = costs.shape
    accumulated = np.full((n + 1, m + 1), np.inf, dtype=np.float64)
    back = np.zeros((n + 1, m + 1), dtype=np.int8)
    accumulated[0, 0] = 0.0
    band = max(3, math.ceil(max(n, m) * band_ratio))

    for i in range(1, n + 1):
        expected = 1 if n == 1 else 1 + round((i - 1) * (m - 1) / (n - 1))
        start = max(1, expected - band)
        stop = min(m, expected + band)
        for j in range(start, stop + 1):
            cost = float(costs[i - 1, j - 1])
            if not math.isfinite(cost):
                continue
            diagonal = accumulated[i - 1, j - 1]
            vertical = accumulated[i - 1, j]
            horizontal = accumulated[i, j - 1]
            best = min(diagonal, vertical, horizontal)
            if not math.isfinite(best):
                continue
            accumulated[i, j] = cost + best
            back[i, j] = 1 if best == diagonal else (2 if best == vertical else 3)

    if not math.isfinite(accumulated[n, m]):
        raise ValueError(
            "DTW could not connect the sequences. Increase --band-ratio or lower "
            "--minimum-common-angles."
        )

    path: list[tuple[int, int]] = []
    i, j = n, m
    while i > 0 and j > 0:
        path.append((i - 1, j - 1))
        step = back[i, j]
        if step == 1:
            i, j = i - 1, j - 1
        elif step == 2:
            i -= 1
        elif step == 3:
            j -= 1
        else:
            raise RuntimeError("Broken DTW backtrace")
    path.reverse()
    return Alignment(path, float(accumulated[n, m] / len(path)), mirrored)


def metric_rows(
    target: AngleSequence,
    reference: AngleSequence,
    alignment: Alignment,
) -> list[dict[str, float | int | str]]:
    results: list[dict[str, float | int | str]] = []
    for column, feature in enumerate(target.features):
        target_values = np.array(
            [target.values[i, column] for i, _ in alignment.path], dtype=float
        )
        reference_values = np.array(
            [reference.values[j, column] for _, j in alignment.path], dtype=float
        )
        valid = np.isfinite(target_values) & np.isfinite(reference_values)
        if not valid.any():
            results.append(
                {
                    "angle": display_name(feature),
                    "feature": feature,
                    "aligned_pairs": 0,
                    "coverage_percent": 0.0,
                    "mean_absolute_deviation_deg": "",
                    "root_mean_square_deviation_deg": "",
                    "median_absolute_deviation_deg": "",
                    "p95_absolute_deviation_deg": "",
                    "mean_signed_difference_deg": "",
                    "correlation": "",
                }
            )
            continue
        differences = target_values[valid] - reference_values[valid]
        absolute = np.abs(differences)
        correlation = ""
        if len(differences) >= 3:
            target_std = float(np.std(target_values[valid]))
            reference_std = float(np.std(reference_values[valid]))
            if target_std > 1e-8 and reference_std > 1e-8:
                correlation = round(
                    float(np.corrcoef(target_values[valid], reference_values[valid])[0, 1]),
                    4,
                )
        results.append(
            {
                "angle": display_name(feature),
                "feature": feature,
                "aligned_pairs": int(valid.sum()),
                "coverage_percent": round(100 * float(valid.mean()), 1),
                "mean_absolute_deviation_deg": round(float(absolute.mean()), 3),
                "root_mean_square_deviation_deg": round(
                    float(np.sqrt(np.square(differences).mean())), 3
                ),
                "median_absolute_deviation_deg": round(float(np.median(absolute)), 3),
                "p95_absolute_deviation_deg": round(
                    float(np.percentile(absolute, 95)), 3
                ),
                "mean_signed_difference_deg": round(float(differences.mean()), 3),
                "correlation": correlation,
            }
        )
    return results


def write_aligned_csv(
    path: Path,
    target: AngleSequence,
    reference: AngleSequence,
    alignment: Alignment,
) -> None:
    fields = [
        "alignment_step",
        "movement_frame",
        "movement_time_s",
        "reference_frame",
        "reference_time_s",
    ]
    for feature in target.features:
        base = feature.removesuffix("_smooth_deg")
        fields.extend(
            [f"{base}_movement_deg", f"{base}_reference_deg", f"{base}_difference_deg"]
        )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for step, (i, j) in enumerate(alignment.path):
            row: dict[str, float | int | str] = {
                "alignment_step": step,
                "movement_frame": int(target.frames[i]),
                "movement_time_s": round(float(target.times[i]), 4),
                "reference_frame": int(reference.frames[j]),
                "reference_time_s": round(float(reference.times[j]), 4),
            }
            for column, feature in enumerate(target.features):
                base = feature.removesuffix("_smooth_deg")
                movement_value = target.values[i, column]
                reference_value = reference.values[j, column]
                row[f"{base}_movement_deg"] = (
                    round(float(movement_value), 3) if math.isfinite(movement_value) else ""
                )
                row[f"{base}_reference_deg"] = (
                    round(float(reference_value), 3) if math.isfinite(reference_value) else ""
                )
                row[f"{base}_difference_deg"] = (
                    round(float(movement_value - reference_value), 3)
                    if math.isfinite(movement_value) and math.isfinite(reference_value)
                    else ""
                )
            writer.writerow(row)


def create_plots(
    comparison_path: Path,
    deviation_path: Path,
    target: AngleSequence,
    reference: AngleSequence,
    alignment: Alignment,
    metrics: list[dict],
) -> None:
    columns = 2
    rows = math.ceil(len(target.features) / columns)
    fig, axes = plt.subplots(rows, columns, figsize=(13, 2.8 * rows), sharex=True)
    axes_array = np.atleast_1d(axes).ravel()
    phase = np.linspace(0, 100, len(alignment.path))
    for column, (feature, axis) in enumerate(zip(target.features, axes_array)):
        movement = np.array([target.values[i, column] for i, _ in alignment.path])
        reference_line = np.array([reference.values[j, column] for _, j in alignment.path])
        axis.plot(phase, reference_line, label="Reference", linewidth=1.5)
        axis.plot(phase, movement, label="IMG_6086", linewidth=1.3, alpha=0.9)
        axis.set_title(display_name(feature))
        axis.set_ylabel("Degrees")
        axis.grid(alpha=0.25)
    for axis in axes_array[len(target.features) :]:
        axis.set_visible(False)
    for axis in axes_array[-columns:]:
        axis.set_xlabel("Aligned movement phase (%)")
    axes_array[0].legend()
    fig.suptitle("Angle trajectories after dynamic time warping")
    fig.tight_layout()
    fig.savefig(comparison_path, dpi=160)
    plt.close(fig)

    available = [row for row in metrics if row["mean_absolute_deviation_deg"] != ""]
    available.sort(key=lambda row: float(row["mean_absolute_deviation_deg"]))
    fig, axis = plt.subplots(figsize=(10, 6))
    axis.barh(
        [row["angle"] for row in available],
        [row["mean_absolute_deviation_deg"] for row in available],
        color="#3274a1",
    )
    axis.set_xlabel("Mean absolute deviation (degrees)")
    axis.set_title("Deviation from the reference by joint angle")
    axis.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(deviation_path, dpi=160)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    target, reference = select_common_features(
        load_sequence(args.target_csv), load_sequence(args.reference_csv)
    )
    minimum_valid = min(args.minimum_common_angles, len(target.features))
    target = trim_unusable_edges(target, minimum_valid)
    reference = trim_unusable_edges(reference, minimum_valid)

    alignments: list[tuple[Alignment, AngleSequence]] = []
    if args.mirror in ("auto", "no"):
        alignments.append(
            (
                dtw_align(
                    target.values,
                    reference.values,
                    minimum_valid,
                    args.band_ratio,
                    mirrored=False,
                ),
                reference,
            )
        )
    if args.mirror in ("auto", "yes"):
        mirrored = mirrored_reference(reference)
        alignments.append(
            (
                dtw_align(
                    target.values,
                    mirrored.values,
                    minimum_valid,
                    args.band_ratio,
                    mirrored=True,
                ),
                mirrored,
            )
        )
    alignment, aligned_reference = min(alignments, key=lambda item: item[0].mean_cost)
    metrics = metric_rows(target, aligned_reference, alignment)

    output_dir = args.output_dir or Path("results") / "angle-comparison" / args.target_csv.stem.removesuffix("_joint_angles")
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "joint_deviation_summary.csv"
    aligned_path = output_dir / "aligned_angle_comparison.csv"
    comparison_plot = output_dir / "aligned_angle_trajectories.png"
    deviation_plot = output_dir / "joint_deviation.png"
    summary_path = output_dir / "comparison_summary.json"

    metric_fields = list(metrics[0])
    with metrics_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=metric_fields)
        writer.writeheader()
        writer.writerows(metrics)
    write_aligned_csv(aligned_path, target, aligned_reference, alignment)
    create_plots(
        comparison_plot,
        deviation_plot,
        target,
        aligned_reference,
        alignment,
        metrics,
    )

    weighted_abs_sum = 0.0
    weighted_sq_sum = 0.0
    observation_count = 0
    for row in metrics:
        count = int(row["aligned_pairs"])
        if count and row["mean_absolute_deviation_deg"] != "":
            weighted_abs_sum += float(row["mean_absolute_deviation_deg"]) * count
            weighted_sq_sum += float(row["root_mean_square_deviation_deg"]) ** 2 * count
            observation_count += count
    overall_mae = weighted_abs_sum / observation_count
    overall_rmse = math.sqrt(weighted_sq_sum / observation_count)
    ranked = sorted(
        (row for row in metrics if row["mean_absolute_deviation_deg"] != ""),
        key=lambda row: float(row["mean_absolute_deviation_deg"]),
        reverse=True,
    )
    summary = {
        "movement_csv": str(args.target_csv.resolve()),
        "reference_csv": str(args.reference_csv.resolve()),
        "method": "multivariate dynamic time warping on smoothed 2D joint angles",
        "reference_left_right_swapped": alignment.mirrored,
        "movement_frames_used": len(target.frames),
        "reference_frames_used": len(aligned_reference.frames),
        "aligned_steps": len(alignment.path),
        "overall_mean_absolute_deviation_deg": round(overall_mae, 3),
        "overall_root_mean_square_deviation_deg": round(overall_rmse, 3),
        "angular_deviation_percent_of_180": round(100 * overall_mae / 180, 2),
        "largest_mean_deviations": [
            {
                "angle": row["angle"],
                "mean_absolute_deviation_deg": row["mean_absolute_deviation_deg"],
                "coverage_percent": row["coverage_percent"],
            }
            for row in ranked[:3]
        ],
        "interpretation": (
            "This compares one movement with one reference recording. The percent is "
            "MAE divided by the 180-degree angular range; it is not a clinical "
            "normativity percentile or probability."
        ),
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    print("Comparison complete")
    print(f"Overall mean absolute deviation: {overall_mae:.2f} degrees")
    print(f"Reference mirrored: {'yes' if alignment.mirrored else 'no'}")
    print(f"Summary: {summary_path.resolve()}")
    print(f"Joint metrics: {metrics_path.resolve()}")
    print(f"Aligned data: {aligned_path.resolve()}")
    print(f"Trajectory graph: {comparison_plot.resolve()}")
    print(f"Deviation graph: {deviation_plot.resolve()}")


if __name__ == "__main__":
    main()
