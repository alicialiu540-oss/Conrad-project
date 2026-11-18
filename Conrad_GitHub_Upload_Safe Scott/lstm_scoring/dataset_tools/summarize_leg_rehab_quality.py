#!/usr/bin/env python3
"""Summarize leg-keypoint visibility for the 20-clip rehab starter set."""

from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
DATASET_MANIFEST = ROOT / "data" / "inbox" / "leg_rehab_20" / "LEG_REHAB_20_MANIFEST.csv"
PROCESS_MANIFEST = ROOT / "data" / "processed" / "processed_videos.json"
OUTPUT = ROOT / "data" / "processed" / "leg_rehab_20_quality.csv"
THRESHOLD = 0.50


def ratio(count: int, total: int) -> float:
    return count / total if total else 0.0


def main() -> int:
    with DATASET_MANIFEST.open(newline="", encoding="utf-8") as handle:
        clips = list(csv.DictReader(handle))
    processed = json.loads(PROCESS_MANIFEST.read_text(encoding="utf-8"))["processed"]
    by_filename = {Path(item["source"]).name: item for item in processed.values()}

    rows: list[dict[str, object]] = []
    for clip in clips:
        item = by_filename.get(clip["filename"])
        if not item:
            raise RuntimeError(f"No YOLO result for {clip['filename']}")
        csv_path = ROOT / item["keypoints_csv"]
        total = detected = left_chain = right_chain = either_chain = both_chains = 0
        hip_group = knee_group = ankle_group = all_groups = 0
        with csv_path.open(newline="", encoding="utf-8") as handle:
            for frame in csv.DictReader(handle):
                total += 1
                detected += int(frame["detected"] or 0)
                conf = {
                    name: float(frame.get(f"{name}_confidence") or 0)
                    for name in (
                        "left_hip", "right_hip", "left_knee", "right_knee",
                        "left_ankle", "right_ankle",
                    )
                }
                left = all(conf[name] >= THRESHOLD for name in ("left_hip", "left_knee", "left_ankle"))
                right = all(conf[name] >= THRESHOLD for name in ("right_hip", "right_knee", "right_ankle"))
                hip = max(conf["left_hip"], conf["right_hip"]) >= THRESHOLD
                knee = max(conf["left_knee"], conf["right_knee"]) >= THRESHOLD
                ankle = max(conf["left_ankle"], conf["right_ankle"]) >= THRESHOLD
                left_chain += left
                right_chain += right
                either_chain += left or right
                both_chains += left and right
                hip_group += hip
                knee_group += knee
                ankle_group += ankle
                all_groups += hip and knee and ankle

        leg_coverage = ratio(all_groups, total)
        quality = "good" if leg_coverage >= 0.90 else "usable" if leg_coverage >= 0.70 else "limited"
        rows.append({
            "sample_id": clip["sample_id"],
            "filename": clip["filename"],
            "exercise": clip["exercise"],
            "camera_view": clip["camera_view"],
            "frames": total,
            "person_detection_coverage": round(ratio(detected, total), 4),
            "hip_group_coverage": round(ratio(hip_group, total), 4),
            "knee_group_coverage": round(ratio(knee_group, total), 4),
            "ankle_group_coverage": round(ratio(ankle_group, total), 4),
            "leg_joint_group_coverage": round(leg_coverage, 4),
            "left_leg_chain_coverage": round(ratio(left_chain, total), 4),
            "right_leg_chain_coverage": round(ratio(right_chain, total), 4),
            "either_leg_chain_coverage": round(ratio(either_chain, total), 4),
            "both_leg_chains_coverage": round(ratio(both_chains, total), 4),
            "pose_tracking_quality": quality,
            "keypoints_csv": item["keypoints_csv"],
            "annotated_video": item["annotated_video"],
        })

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows to {OUTPUT}")
    for row in rows:
        print(
            f"{row['sample_id']} {row['exercise']}: person={row['person_detection_coverage']:.1%}, "
            f"leg joints={row['leg_joint_group_coverage']:.1%}, {row['pose_tracking_quality']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
