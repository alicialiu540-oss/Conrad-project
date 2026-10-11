#!/usr/bin/env python3
"""Train a prototype Siamese LSTM on the leg-rehabilitation angle sequences."""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn

from lstm_motion import (
    INPUT_SIZE,
    SiameseMotionLSTM,
    augment_sequence,
    exercise_group,
    load_angle_sequence,
)


ROOT = Path(__file__).resolve().parent.parent
DATASET_MANIFEST = ROOT / "data" / "inbox" / "leg_rehab_20" / "LEG_REHAB_20_MANIFEST.csv"
PROCESS_MANIFEST = ROOT / "data" / "processed" / "processed_videos.json"
ANGLE_DIR = ROOT / "data" / "processed" / "angle_data"
DEFAULT_OUTPUT = ROOT / "models" / "leg_rehab_siamese_lstm.pt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--epochs", type=int, default=70)
    parser.add_argument("--pairs-per-epoch", type=int, default=240)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--sequence-length", type=int, default=96)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["auto", "mps", "cpu"], default="auto")
    return parser.parse_args()


def choose_device(requested: str) -> str:
    if requested != "auto":
        return requested
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_dataset(sequence_length: int) -> list[dict[str, object]]:
    processed = json.loads(PROCESS_MANIFEST.read_text(encoding="utf-8"))["processed"]
    by_filename = {Path(item["source"]).name: item for item in processed.values()}
    with DATASET_MANIFEST.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    dataset: list[dict[str, object]] = []
    for row in rows:
        item = by_filename.get(row["filename"])
        if not item:
            continue
        keypoint_stem = Path(item["keypoints_csv"]).stem.removesuffix("_keypoints")
        angle_path = ANGLE_DIR / f"{keypoint_stem}_leg_angles.csv"
        if not angle_path.exists():
            raise FileNotFoundError(angle_path)
        dataset.append({
            "sample_id": row["sample_id"],
            "exercise": row["exercise"],
            "group": exercise_group(row["exercise"]),
            "sequence": load_angle_sequence(angle_path, sequence_length),
        })
    if len(dataset) < 4:
        raise RuntimeError("Not enough angle sequences to train the LSTM")
    return dataset


def create_pairs(
    dataset: list[dict[str, object]],
    count: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    by_group: dict[str, list[dict[str, object]]] = {}
    for item in dataset:
        by_group.setdefault(str(item["group"]), []).append(item)
    groups = sorted(by_group)
    first: list[np.ndarray] = []
    second: list[np.ndarray] = []
    labels: list[float] = []
    for index in range(count):
        positive = index % 2 == 0
        if positive:
            group = groups[int(rng.integers(0, len(groups)))]
            items = by_group[group]
            item_a = items[int(rng.integers(0, len(items)))]
            item_b = items[int(rng.integers(0, len(items)))]
            labels.append(1.0)
        else:
            group_a, group_b = rng.choice(groups, size=2, replace=False)
            item_a = by_group[str(group_a)][int(rng.integers(0, len(by_group[str(group_a)])))]
            item_b = by_group[str(group_b)][int(rng.integers(0, len(by_group[str(group_b)])))]
            labels.append(0.0)
        first.append(augment_sequence(np.asarray(item_a["sequence"]), rng))
        second.append(augment_sequence(np.asarray(item_b["sequence"]), rng))
    order = rng.permutation(count)
    return np.stack(first)[order], np.stack(second)[order], np.asarray(labels, dtype=np.float32)[order]


def contrastive_loss(similarity: torch.Tensor, labels: torch.Tensor, margin: float = 0.30) -> torch.Tensor:
    positive_loss = labels * (1.0 - similarity).pow(2)
    negative_loss = (1.0 - labels) * torch.relu(similarity - margin).pow(2)
    return (positive_loss + negative_loss).mean()


@torch.no_grad()
def evaluate(
    model: SiameseMotionLSTM,
    dataset: list[dict[str, object]],
    rng: np.random.Generator,
    device: str,
) -> dict[str, float]:
    first, second, labels = create_pairs(dataset, 240, rng)
    first_tensor = torch.from_numpy(first).to(device)
    second_tensor = torch.from_numpy(second).to(device)
    similarities = model(first_tensor, second_tensor).cpu().numpy()
    positive = similarities[labels == 1]
    negative = similarities[labels == 0]
    threshold = float((positive.mean() + negative.mean()) / 2)
    predictions = similarities >= threshold
    accuracy = float((predictions == labels.astype(bool)).mean())
    return {
        "positive_mean": float(positive.mean()),
        "negative_mean": float(negative.mean()),
        "threshold": threshold,
        "augmented_pair_accuracy": accuracy,
    }


def main() -> int:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = choose_device(args.device)
    dataset = load_dataset(args.sequence_length)
    groups = sorted({str(item["group"]) for item in dataset})
    print(f"Training on {len(dataset)} sequences in {len(groups)} movement groups using {device}")

    config = {
        "input_size": INPUT_SIZE,
        "hidden_size": 64,
        "layers": 2,
        "embedding_size": 32,
        "dropout": 0.20,
    }
    model = SiameseMotionLSTM(**config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=1e-5)

    for epoch in range(1, args.epochs + 1):
        model.train()
        first, second, labels = create_pairs(dataset, args.pairs_per_epoch, rng)
        losses: list[float] = []
        for start in range(0, len(labels), args.batch_size):
            end = start + args.batch_size
            first_tensor = torch.from_numpy(first[start:end]).to(device)
            second_tensor = torch.from_numpy(second[start:end]).to(device)
            label_tensor = torch.from_numpy(labels[start:end]).to(device)
            optimizer.zero_grad(set_to_none=True)
            similarities = model(first_tensor, second_tensor)
            loss = contrastive_loss(similarities, label_tensor)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.item()))
        if epoch == 1 or epoch % 10 == 0 or epoch == args.epochs:
            print(f"Epoch {epoch:03d}/{args.epochs}: loss={np.mean(losses):.5f}")

    model.eval()
    calibration = evaluate(model, dataset, np.random.default_rng(args.seed + 1000), device)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "model_state": model.state_dict(),
        "model_config": config,
        "sequence_length": args.sequence_length,
        "movement_groups": groups,
        "calibration": calibration,
        "training_sequences": len(dataset),
        "limitations": (
            "Prototype trained from one source demonstrator with augmented pairs. "
            "Collect participant-level correct/incorrect labels before clinical use."
        ),
    }
    torch.save(checkpoint, args.output)
    print(
        f"Saved {args.output}\n"
        f"Positive similarity={calibration['positive_mean']:.3f}; "
        f"negative similarity={calibration['negative_mean']:.3f}; "
        f"augmented-pair accuracy={calibration['augmented_pair_accuracy']:.1%}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
