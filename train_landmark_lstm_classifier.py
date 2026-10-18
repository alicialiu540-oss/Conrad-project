#!/usr/bin/env python3
"""Train a multiclass LSTM classifier from a prepared landmark tensor."""

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
from torch.nn.utils.rnn import pack_padded_sequence
from torch.utils.data import DataLoader, TensorDataset


class LandmarkLSTM(nn.Module):
    def __init__(self, features: int, hidden: int, classes: int, layers: int, dropout: float) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            features,
            hidden,
            num_layers=layers,
            batch_first=True,
            dropout=dropout if layers > 1 else 0.0,
            bidirectional=True,
        )
        self.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden * 2, classes))

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, (hidden, _) = self.lstm(packed)
        representation = torch.cat((hidden[-2], hidden[-1]), dim=1)
        return self.classifier(representation)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_npz", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("results/landmark-lstm-classifier"))
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.25)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def automatic_split(y: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    train: list[int] = []
    validation: list[int] = []
    for label in np.unique(y):
        indices = np.flatnonzero(y == label)
        rng.shuffle(indices)
        if len(indices) < 2:
            raise ValueError("Every class needs at least two samples when split is not supplied")
        count = max(1, int(round(len(indices) * 0.2)))
        validation.extend(indices[:count])
        train.extend(indices[count:])
    return np.asarray(train), np.asarray(validation)


def metrics(y_true: np.ndarray, y_pred: np.ndarray, classes: int) -> dict:
    confusion = np.zeros((classes, classes), dtype=int)
    for truth, prediction in zip(y_true, y_pred):
        confusion[truth, prediction] += 1
    recalls = []
    for index in range(classes):
        denominator = confusion[index].sum()
        recalls.append(float(confusion[index, index] / denominator) if denominator else float("nan"))
    return {
        "accuracy": float(np.mean(y_true == y_pred)),
        "balanced_accuracy": float(np.nanmean(recalls)),
        "confusion_matrix": confusion,
    }


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    data = np.load(args.dataset_npz.expanduser().resolve(), allow_pickle=False)
    X = data["X"].astype(np.float32)
    y = data["y"].astype(np.int64)
    lengths = data["lengths"].astype(np.int64)
    labels = data["labels"].astype(str)
    sample_ids = data["sample_ids"].astype(str)
    splits = data["splits"].astype(str)
    if X.ndim != 3:
        raise ValueError(f"Expected X shape [samples,time_steps,features], got {X.shape}")
    if len(labels) < 2:
        raise ValueError("Classifier training needs at least two labeled classes")

    if np.any(splits == "train") and np.any(splits == "validation"):
        train_indices = np.flatnonzero(splits == "train")
        validation_indices = np.flatnonzero(splits == "validation")
        split_method = "manifest"
    else:
        train_indices, validation_indices = automatic_split(y, args.seed)
        split_method = "automatic_stratified_sample_split"
    test_indices = np.flatnonzero(splits == "test")

    valid_training_rows = np.concatenate(
        [X[index, : lengths[index]] for index in train_indices], axis=0
    )
    mean = valid_training_rows.mean(axis=0)
    std = valid_training_rows.std(axis=0)
    std[std < 1e-6] = 1.0
    scaled = X.copy()
    for index in range(len(scaled)):
        scaled[index, : lengths[index]] = (scaled[index, : lengths[index]] - mean) / std
        scaled[index, lengths[index] :] = 0.0

    class_counts = np.bincount(y[train_indices], minlength=len(labels))
    if np.any(class_counts == 0):
        raise ValueError("Every label must appear in the training split")
    weights = len(train_indices) / (len(labels) * class_counts)
    model = LandmarkLSTM(X.shape[2], args.hidden_size, len(labels), args.layers, args.dropout)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32))
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
        model.eval()
        with torch.no_grad():
            validation_logits = model(validation_x, validation_lengths)
            validation_loss = float(loss_fn(validation_logits, validation_y))
        history.append(
            {"epoch": epoch, "train_loss": total / len(train_indices), "validation_loss": validation_loss}
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
        raise RuntimeError("Training did not produce a checkpoint")
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        logits = model(validation_x, validation_lengths)
        probabilities = torch.softmax(logits, dim=1).numpy()
    predictions = probabilities.argmax(axis=1)
    result = metrics(y[validation_indices], predictions, len(labels))

    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "labels": labels.tolist(),
            "feature_names": data["feature_names"].astype(str).tolist(),
            "mean": mean,
            "std": std,
            "hidden_size": args.hidden_size,
            "layers": args.layers,
            "dropout": args.dropout,
            "time_steps": X.shape[1],
        },
        output / "landmark_lstm_classifier.pt",
    )
    with (output / "training_history.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    prediction_fields = ["sample_id", "true_label", "predicted_label"] + [f"probability_{label}" for label in labels]
    with (output / "validation_predictions.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=prediction_fields)
        writer.writeheader()
        for row_index, sample_index in enumerate(validation_indices):
            row = {
                "sample_id": sample_ids[sample_index],
                "true_label": labels[y[sample_index]],
                "predicted_label": labels[predictions[row_index]],
            }
            row.update({f"probability_{label}": float(probabilities[row_index, i]) for i, label in enumerate(labels)})
            writer.writerow(row)

    fig, axis = plt.subplots(figsize=(7, 6))
    image = axis.imshow(result["confusion_matrix"], cmap="Blues")
    axis.set_xticks(range(len(labels)), labels, rotation=30, ha="right")
    axis.set_yticks(range(len(labels)), labels)
    axis.set(xlabel="Predicted label", ylabel="True label", title="Validation confusion matrix")
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
        "split_method": split_method,
        "train_samples": len(train_indices),
        "validation_samples": len(validation_indices),
        "test_samples_reserved": len(test_indices),
        "validation_accuracy": result["accuracy"],
        "validation_balanced_accuracy": result["balanced_accuracy"],
        "limitations": [
            "Labels must be assigned by qualified reviewers using a written rubric.",
            "Use participant-independent train, validation, and test splits.",
            "Class probabilities are not clinical diagnoses or injury predictions.",
        ],
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
