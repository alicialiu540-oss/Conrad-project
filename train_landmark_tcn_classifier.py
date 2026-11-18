#!/usr/bin/env python3
"""Train a multiclass Temporal Convolutional Network on landmark tensors."""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from train_landmark_lstm_classifier import automatic_split, metrics


class CausalConv1d(nn.Conv1d):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dilation: int) -> None:
        self.right_trim = (kernel_size - 1) * dilation
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            padding=self.right_trim,
            dilation=dilation,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = super().forward(x)
        return output[:, :, : -self.right_trim] if self.right_trim else output


class TemporalBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.network = nn.Sequential(
            CausalConv1d(in_channels, out_channels, kernel_size, dilation),
            nn.ReLU(),
            nn.Dropout(dropout),
            CausalConv1d(out_channels, out_channels, kernel_size, dilation),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.residual = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
        self.activation = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(self.network(x) + self.residual(x))


class TCNClassifier(nn.Module):
    def __init__(
        self,
        input_features: int,
        channels: list[int],
        classes: int,
        kernel_size: int = 3,
        dropout: float = 0.25,
    ) -> None:
        super().__init__()
        blocks = []
        current = input_features
        for level, channel_count in enumerate(channels):
            blocks.append(
                TemporalBlock(current, channel_count, kernel_size, 2**level, dropout)
            )
            current = channel_count
        self.tcn = nn.Sequential(*blocks)
        self.classifier = nn.Linear(current, classes)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        temporal = self.tcn(x.transpose(1, 2)).transpose(1, 2)
        steps = torch.arange(temporal.shape[1], device=temporal.device)[None, :]
        mask = steps < lengths[:, None].to(temporal.device)
        pooled = (temporal * mask[:, :, None]).sum(dim=1) / lengths.clamp(min=1)[:, None].to(temporal.device)
        return self.classifier(pooled)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_npz", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("results/landmark-tcn-classifier"))
    parser.add_argument("--channels", default="64,64,64,64")
    parser.add_argument("--kernel-size", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.25)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    args.channels = [int(value) for value in args.channels.split(",") if value.strip()]
    if not args.channels or any(value < 1 for value in args.channels):
        parser.error("--channels must contain positive comma-separated integers")
    if args.kernel_size < 2:
        parser.error("--kernel-size must be at least 2")
    return args


def evaluate(
    model: nn.Module,
    x: torch.Tensor,
    lengths: torch.Tensor,
    y: torch.Tensor,
    loss_fn: nn.Module,
) -> tuple[float, np.ndarray]:
    model.eval()
    with torch.no_grad():
        logits = model(x, lengths)
        loss = float(loss_fn(logits, y))
        probabilities = torch.softmax(logits, dim=1).cpu().numpy()
    return loss, probabilities


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(min(4, max(1, torch.get_num_threads())))

    dataset_path = args.dataset_npz.expanduser().resolve()
    data = np.load(dataset_path, allow_pickle=False)
    X = data["X"].astype(np.float32)
    y = data["y"].astype(np.int64)
    lengths = data["lengths"].astype(np.int64)
    labels = data["labels"].astype(str)
    feature_names = data["feature_names"].astype(str)
    sample_ids = data["sample_ids"].astype(str)
    subject_ids = data["subject_ids"].astype(str)
    splits = data["splits"].astype(str)
    if X.ndim != 3:
        raise ValueError(f"Expected [samples,time_steps,features], got {X.shape}")
    if len(labels) < 2:
        raise ValueError("TCN classification requires at least two reviewed classes")

    if np.any(splits == "train") and np.any(splits == "validation"):
        train_indices = np.flatnonzero(splits == "train")
        validation_indices = np.flatnonzero(splits == "validation")
        split_method = "manifest"
    else:
        train_indices, validation_indices = automatic_split(y, args.seed)
        split_method = "automatic_stratified_sample_split"
    test_indices = np.flatnonzero(splits == "test")

    training_values = np.concatenate(
        [X[index, : lengths[index]] for index in train_indices], axis=0
    )
    mean = training_values.mean(axis=0)
    std = training_values.std(axis=0)
    std[std < 1e-6] = 1.0
    scaled = X.copy()
    for index in range(len(scaled)):
        scaled[index, : lengths[index]] = (scaled[index, : lengths[index]] - mean) / std
        scaled[index, lengths[index] :] = 0.0

    class_counts = np.bincount(y[train_indices], minlength=len(labels))
    if np.any(class_counts == 0):
        raise ValueError("Every label must appear in the training split")
    class_weights = len(train_indices) / (len(labels) * class_counts)
    model = TCNClassifier(
        X.shape[2], args.channels, len(labels), args.kernel_size, args.dropout
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(class_weights, dtype=torch.float32))
    loader = DataLoader(
        TensorDataset(
            torch.from_numpy(scaled[train_indices]),
            torch.from_numpy(lengths[train_indices]),
            torch.from_numpy(y[train_indices]),
        ),
        batch_size=min(args.batch_size, len(train_indices)),
        shuffle=True,
    )
    validation_x = torch.from_numpy(scaled[validation_indices])
    validation_lengths = torch.from_numpy(lengths[validation_indices])
    validation_y = torch.from_numpy(y[validation_indices])

    history: list[dict] = []
    best_loss = float("inf")
    best_state = None
    stale = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        for batch_x, batch_lengths, batch_y in loader:
            optimizer.zero_grad()
            loss = loss_fn(model(batch_x, batch_lengths), batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += float(loss) * len(batch_x)
        validation_loss, probabilities = evaluate(
            model, validation_x, validation_lengths, validation_y, loss_fn
        )
        history.append(
            {
                "epoch": epoch,
                "train_loss": total / len(train_indices),
                "validation_loss": validation_loss,
            }
        )
        if validation_loss < best_loss - 1e-6:
            best_loss = validation_loss
            best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= args.patience:
                break
    if best_state is None:
        raise RuntimeError("Training failed to create a checkpoint")
    model.load_state_dict(best_state)
    _, probabilities = evaluate(model, validation_x, validation_lengths, validation_y, loss_fn)
    predictions = probabilities.argmax(axis=1)
    result = metrics(y[validation_indices], predictions, len(labels))

    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    receptive_field = 1 + 2 * (args.kernel_size - 1) * sum(
        2**level for level in range(len(args.channels))
    )
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "labels": labels.tolist(),
            "feature_names": feature_names.tolist(),
            "mean": mean,
            "std": std,
            "channels": args.channels,
            "kernel_size": args.kernel_size,
            "dropout": args.dropout,
            "time_steps": X.shape[1],
            "receptive_field_steps": receptive_field,
        },
        output / "landmark_tcn_classifier.pt",
    )
    with (output / "training_history.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    fields = ["sample_id", "subject_id", "true_label", "predicted_label"] + [
        f"probability_{label}" for label in labels
    ]
    with (output / "validation_predictions.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row_index, sample_index in enumerate(validation_indices):
            row = {
                "sample_id": sample_ids[sample_index],
                "subject_id": subject_ids[sample_index],
                "true_label": labels[y[sample_index]],
                "predicted_label": labels[predictions[row_index]],
            }
            row.update(
                {
                    f"probability_{label}": float(probabilities[row_index, class_index])
                    for class_index, label in enumerate(labels)
                }
            )
            writer.writerow(row)

    fig, axis = plt.subplots(figsize=(7, 6))
    image = axis.imshow(result["confusion_matrix"], cmap="Blues")
    axis.set_xticks(range(len(labels)), labels, rotation=30, ha="right")
    axis.set_yticks(range(len(labels)), labels)
    axis.set(xlabel="Predicted label", ylabel="True label", title="TCN validation confusion matrix")
    for i in range(len(labels)):
        for j in range(len(labels)):
            axis.text(j, i, str(result["confusion_matrix"][i, j]), ha="center", va="center")
    fig.colorbar(image, ax=axis)
    fig.tight_layout()
    fig.savefig(output / "validation_confusion_matrix.png", dpi=160)
    plt.close(fig)

    summary = {
        "status": "prototype_complete",
        "tensor_shape": list(X.shape),
        "labels": labels.tolist(),
        "channels": args.channels,
        "kernel_size": args.kernel_size,
        "receptive_field_steps": receptive_field,
        "split_method": split_method,
        "train_samples": len(train_indices),
        "validation_samples": len(validation_indices),
        "test_samples_reserved": len(test_indices),
        "validation_accuracy": result["accuracy"],
        "validation_balanced_accuracy": result["balanced_accuracy"],
        "warning": "Prototype probabilities are not clinical diagnoses or injury predictions.",
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
