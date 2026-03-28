"""
Train a NumPy logistic baseline for GLD swing-entry classification.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

import numpy as np

from prepare import FEATURE_COLUMNS, HORIZON_DAYS, TARGET_COLUMN, load_splits

SEED = 42
LEARNING_RATE = 0.02
L2_REG = 1e-3
MAX_EPOCHS = 1200
PATIENCE = 120
POS_WEIGHT = 1.0
NEG_WEIGHT = 1.0
THRESHOLD_MIN = 0.30
THRESHOLD_MAX = 0.70
THRESHOLD_STEPS = 401


@dataclass
class Metrics:
    accuracy: float
    precision: float
    recall: float
    f1: float
    balanced_accuracy: float
    positive_rate: float
    avg_realized_return: float


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def sigmoid(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(values, -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def standardize(train_x: np.ndarray, validation_x: np.ndarray, test_x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = train_x.mean(axis=0, keepdims=True)
    std = train_x.std(axis=0, keepdims=True)
    std = np.where(std < 1e-6, 1.0, std)
    return (train_x - mean) / std, (validation_x - mean) / std, (test_x - mean) / std


def add_bias(features: np.ndarray) -> np.ndarray:
    return np.concatenate([features, np.ones((features.shape[0], 1), dtype=features.dtype)], axis=1)


def classification_stats(probabilities: np.ndarray, labels: np.ndarray, threshold: float) -> tuple[float, float, float, float, np.ndarray]:
    predictions = (probabilities >= threshold).astype(np.float32)
    tp = float(((predictions == 1) & (labels == 1)).sum())
    tn = float(((predictions == 0) & (labels == 0)).sum())
    fp = float(((predictions == 1) & (labels == 0)).sum())
    fn = float(((predictions == 0) & (labels == 1)).sum())
    return tp, tn, fp, fn, predictions


def select_threshold(probabilities: np.ndarray, labels: np.ndarray) -> float:
    best_threshold = 0.5
    best_f1 = -1.0
    best_bal_acc = -1.0
    for threshold in np.linspace(THRESHOLD_MIN, THRESHOLD_MAX, THRESHOLD_STEPS):
        tp, tn, fp, fn, _ = classification_stats(probabilities, labels, float(threshold))
        precision = tp / max(tp + fp, 1.0)
        recall = tp / max(tp + fn, 1.0)
        specificity = tn / max(tn + fp, 1.0)
        f1 = 2 * precision * recall / max(precision + recall, 1e-8)
        bal_acc = 0.5 * (recall + specificity)
        if f1 > best_f1 or (abs(f1 - best_f1) < 1e-8 and bal_acc > best_bal_acc):
            best_threshold = float(threshold)
            best_f1 = f1
            best_bal_acc = bal_acc
    return best_threshold


def compute_metrics(logits: np.ndarray, labels: np.ndarray, realized_returns: np.ndarray, threshold: float) -> Metrics:
    probabilities = sigmoid(logits)
    tp, tn, fp, fn, predictions = classification_stats(probabilities, labels, threshold)
    accuracy = float((predictions == labels).mean())
    precision = tp / max(tp + fp, 1.0)
    recall = tp / max(tp + fn, 1.0)
    specificity = tn / max(tn + fp, 1.0)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)
    bal_acc = 0.5 * (recall + specificity)
    selected = realized_returns[predictions == 1]
    avg_realized_return = float(selected.mean()) if len(selected) else 0.0
    return Metrics(
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        f1=f1,
        balanced_accuracy=bal_acc,
        positive_rate=float(predictions.mean()),
        avg_realized_return=avg_realized_return,
    )


def main() -> None:
    set_seed(SEED)
    splits = load_splits()
    train_x = splits["train"].features
    validation_x = splits["validation"].features
    test_x = splits["test"].features
    train_y = splits["train"].labels
    validation_y = splits["validation"].labels
    test_y = splits["test"].labels

    train_x, validation_x, test_x = standardize(train_x, validation_x, test_x)
    train_x = add_bias(train_x)
    validation_x = add_bias(validation_x)
    test_x = add_bias(test_x)

    weights = np.zeros(train_x.shape[1], dtype=np.float32)
    best_weights = weights.copy()
    best_validation_f1 = -math.inf
    best_threshold = 0.5
    best_epoch = -1
    epochs_without_improvement = 0

    for epoch in range(1, MAX_EPOCHS + 1):
        logits = train_x @ weights
        probs = sigmoid(logits)
        sample_weights = np.where(train_y == 1.0, POS_WEIGHT, NEG_WEIGHT).astype(np.float32)
        gradient = train_x.T @ ((probs - train_y) * sample_weights) / train_x.shape[0]
        gradient[:-1] += L2_REG * weights[:-1]
        weights -= LEARNING_RATE * gradient

        validation_logits = validation_x @ weights
        threshold = select_threshold(sigmoid(validation_logits), validation_y)
        validation_metrics = compute_metrics(
            validation_logits,
            validation_y,
            splits["validation"].frame["future_return_60"].to_numpy(dtype=np.float32),
            threshold,
        )
        if validation_metrics.f1 > best_validation_f1:
            best_validation_f1 = validation_metrics.f1
            best_weights = weights.copy()
            best_threshold = threshold
            best_epoch = epoch
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        if epochs_without_improvement >= PATIENCE:
            break

    train_metrics = compute_metrics(
        train_x @ best_weights,
        train_y,
        splits["train"].frame["future_return_60"].to_numpy(dtype=np.float32),
        best_threshold,
    )
    validation_metrics = compute_metrics(
        validation_x @ best_weights,
        validation_y,
        splits["validation"].frame["future_return_60"].to_numpy(dtype=np.float32),
        best_threshold,
    )
    test_metrics = compute_metrics(
        test_x @ best_weights,
        test_y,
        splits["test"].frame["future_return_60"].to_numpy(dtype=np.float32),
        best_threshold,
    )

    print("---")
    print(f"task:                 GLD_{HORIZON_DAYS}d_swing_entry")
    print(f"target_column:        {TARGET_COLUMN}")
    print(f"model:                logistic_regression")
    print(f"features:             {len(FEATURE_COLUMNS)}")
    print(f"decision_threshold:   {best_threshold:.3f}")
    print(f"best_epoch:           {best_epoch}")
    print(f"train_accuracy:       {train_metrics.accuracy:.4f}")
    print(f"validation_accuracy:  {validation_metrics.accuracy:.4f}")
    print(f"validation_f1:        {validation_metrics.f1:.4f}")
    print(f"validation_bal_acc:   {validation_metrics.balanced_accuracy:.4f}")
    print(f"validation_precision: {validation_metrics.precision:.4f}")
    print(f"validation_recall:    {validation_metrics.recall:.4f}")
    print(f"test_accuracy:        {test_metrics.accuracy:.4f}")
    print(f"test_f1:              {test_metrics.f1:.4f}")
    print(f"test_bal_acc:         {test_metrics.balanced_accuracy:.4f}")
    print(f"test_precision:       {test_metrics.precision:.4f}")
    print(f"test_recall:          {test_metrics.recall:.4f}")
    print(f"test_positive_rate:   {test_metrics.positive_rate:.4f}")
    print(f"test_avg_return:      {test_metrics.avg_realized_return:.4%}")


if __name__ == "__main__":
    main()
