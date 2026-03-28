"""
Run the first three-bucket return research round and export compact summaries.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd

import prepare as pr
import research_batch as rb
import train as tr

REPO_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(REPO_DIR, ".cache", "gld-swing-entry")
ROUND_OUTPUT_PATH = os.path.join(CACHE_DIR, "three_bucket_round1.json")
DISTRIBUTION_OUTPUT_PATH = os.path.join(REPO_DIR, "three_bucket_distribution.tsv")
MODEL_OUTPUT_PATH = os.path.join(REPO_DIR, "three_bucket_model_summary.tsv")
RECENT_OUTPUT_PATH = os.path.join(REPO_DIR, "three_bucket_vs_binary.tsv")
SIGNAL_OUTPUT_PATH = os.path.join(REPO_DIR, "three_bucket_signal_summary.tsv")
BINARY_OUTPUT_PATH = os.path.join(REPO_DIR, "round18_binary_summary.tsv")
RULE_OUTPUT_PATH = os.path.join(REPO_DIR, "round18_rule_summary.tsv")

THREE_BUCKET_TARGET = "target_return_bucket_60"
CLASS_NAMES = ("down", "flat", "up")
UP_THRESHOLD = 0.06
DOWN_THRESHOLD = -0.04
RECENT_YEARS = 5
DEFAULT_INTERACTIONS = (("drawdown_20", "volume_vs_20"),)


@dataclass
class ThreeBucketResult:
    name: str
    feature_names: list[str]
    validation_macro_f1: float
    validation_accuracy: float
    validation_bal_acc: float
    test_macro_f1: float
    test_accuracy: float
    test_bal_acc: float
    validation_confusion: list[list[int]]
    test_confusion: list[list[int]]


def ensure_cache_dir() -> None:
    os.makedirs(CACHE_DIR, exist_ok=True)


def round_float(value: float) -> float:
    return round(float(value), 4)


def build_three_bucket_frame() -> pd.DataFrame:
    frame = pr.load_dataset_frame().copy()
    future_returns = frame["future_return_60"].to_numpy(dtype=np.float32)
    labels = np.full(len(frame), 1, dtype=np.int64)
    labels[future_returns <= DOWN_THRESHOLD] = 0
    labels[future_returns >= UP_THRESHOLD] = 2
    frame[THREE_BUCKET_TARGET] = labels
    return frame


def feature_names_for(extra_features: tuple[str, ...]) -> list[str]:
    feature_names = list(pr.FEATURE_COLUMNS)
    for name in extra_features:
        if name not in feature_names:
            feature_names.append(name)
    return feature_names


def split_frame(frame: pd.DataFrame) -> dict[str, pd.DataFrame]:
    train_end, valid_end = pr.split_indices(len(frame))
    return {
        "train": frame.iloc[:train_end].copy().reset_index(drop=True),
        "validation": frame.iloc[train_end:valid_end].copy().reset_index(drop=True),
        "test": frame.iloc[valid_end:].copy().reset_index(drop=True),
    }


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp_values = np.exp(np.clip(shifted, -50.0, 50.0))
    return exp_values / np.maximum(exp_values.sum(axis=1, keepdims=True), 1e-8)


def one_hot(labels: np.ndarray, num_classes: int) -> np.ndarray:
    encoded = np.zeros((len(labels), num_classes), dtype=np.float32)
    encoded[np.arange(len(labels)), labels.astype(int)] = 1.0
    return encoded


def confusion_matrix(labels: np.ndarray, predictions: np.ndarray, num_classes: int) -> np.ndarray:
    matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    for actual, predicted in zip(labels.astype(int), predictions.astype(int)):
        matrix[actual, predicted] += 1
    return matrix


def macro_f1_and_balanced_accuracy(labels: np.ndarray, predictions: np.ndarray, num_classes: int) -> tuple[float, float, np.ndarray]:
    matrix = confusion_matrix(labels, predictions, num_classes)
    recalls: list[float] = []
    f1_scores: list[float] = []
    for idx in range(num_classes):
        tp = float(matrix[idx, idx])
        fn = float(matrix[idx, :].sum() - tp)
        fp = float(matrix[:, idx].sum() - tp)
        recall = tp / max(tp + fn, 1.0)
        precision = tp / max(tp + fp, 1.0)
        f1 = 2.0 * precision * recall / max(precision + recall, 1e-8)
        recalls.append(recall)
        f1_scores.append(f1)
    return float(np.mean(f1_scores)), float(np.mean(recalls)), matrix


def evaluate_multiclass(logits: np.ndarray, labels: np.ndarray) -> tuple[float, float, float, np.ndarray, np.ndarray]:
    probabilities = softmax(logits)
    predictions = probabilities.argmax(axis=1).astype(np.int64)
    accuracy = float((predictions == labels).mean())
    macro_f1, balanced_accuracy, matrix = macro_f1_and_balanced_accuracy(labels, predictions, len(CLASS_NAMES))
    return macro_f1, accuracy, balanced_accuracy, matrix, predictions


def active_interaction_pairs(feature_names: list[str]) -> tuple[tuple[int, int], ...]:
    index = {name: idx for idx, name in enumerate(feature_names)}
    return tuple((index[left], index[right]) for left, right in DEFAULT_INTERACTIONS if left in index and right in index)


def prepare_three_bucket_matrices(
    frame: pd.DataFrame, feature_names: list[str]
) -> tuple[dict[str, pd.DataFrame], dict[str, np.ndarray], np.ndarray, np.ndarray, tuple[tuple[int, int], ...]]:
    splits = split_frame(frame)
    clean_splits: dict[str, pd.DataFrame] = {}
    for split_name, split_frame_df in splits.items():
        clean = split_frame_df.dropna(subset=feature_names + ["future_return_60", THREE_BUCKET_TARGET]).reset_index(drop=True)
        clean_splits[split_name] = clean

    train_x = clean_splits["train"][feature_names].to_numpy(dtype=np.float32)
    validation_x = clean_splits["validation"][feature_names].to_numpy(dtype=np.float32)
    test_x = clean_splits["test"][feature_names].to_numpy(dtype=np.float32)
    train_x, validation_x, test_x, mean, std = rb.standardize_from_train(train_x, validation_x, test_x)
    pair_indices = active_interaction_pairs(feature_names)
    train_x = rb.add_interactions(train_x, pair_indices)
    validation_x = rb.add_interactions(validation_x, pair_indices)
    test_x = rb.add_interactions(test_x, pair_indices)
    matrices = {
        "train": tr.add_bias(train_x),
        "validation": tr.add_bias(validation_x),
        "test": tr.add_bias(test_x),
    }
    return clean_splits, matrices, mean, std, pair_indices


def train_three_bucket_model(frame: pd.DataFrame, name: str, extra_features: tuple[str, ...]) -> tuple[ThreeBucketResult, dict[str, object]]:
    feature_names = feature_names_for(extra_features)
    clean_splits, matrices, mean, std, pair_indices = prepare_three_bucket_matrices(frame, feature_names)
    train_x = matrices["train"]
    validation_x = matrices["validation"]
    test_x = matrices["test"]
    train_y = clean_splits["train"][THREE_BUCKET_TARGET].to_numpy(dtype=np.int64)
    validation_y = clean_splits["validation"][THREE_BUCKET_TARGET].to_numpy(dtype=np.int64)
    test_y = clean_splits["test"][THREE_BUCKET_TARGET].to_numpy(dtype=np.int64)

    train_targets = one_hot(train_y, len(CLASS_NAMES))
    weights = np.zeros((train_x.shape[1], len(CLASS_NAMES)), dtype=np.float32)
    best_weights = weights.copy()
    best_validation_macro_f1 = -np.inf
    epochs_without_improvement = 0

    for _epoch in range(1, tr.MAX_EPOCHS + 1):
        probabilities = softmax(train_x @ weights)
        gradient = train_x.T @ (probabilities - train_targets) / train_x.shape[0]
        gradient[:-1] += tr.L2_REG * weights[:-1]
        weights -= tr.LEARNING_RATE * gradient

        validation_logits = validation_x @ weights
        validation_macro_f1, _validation_accuracy, validation_bal_acc, _validation_matrix, _validation_predictions = evaluate_multiclass(
            validation_logits, validation_y
        )
        if validation_macro_f1 > best_validation_macro_f1:
            best_validation_macro_f1 = validation_macro_f1
            best_weights = weights.copy()
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        if epochs_without_improvement >= tr.PATIENCE:
            break

    validation_logits = validation_x @ best_weights
    test_logits = test_x @ best_weights
    validation_macro_f1, validation_accuracy, validation_bal_acc, validation_matrix, validation_predictions = evaluate_multiclass(
        validation_logits, validation_y
    )
    test_macro_f1, test_accuracy, test_bal_acc, test_matrix, test_predictions = evaluate_multiclass(test_logits, test_y)
    result = ThreeBucketResult(
        name=name,
        feature_names=feature_names,
        validation_macro_f1=validation_macro_f1,
        validation_accuracy=validation_accuracy,
        validation_bal_acc=validation_bal_acc,
        test_macro_f1=test_macro_f1,
        test_accuracy=test_accuracy,
        test_bal_acc=test_bal_acc,
        validation_confusion=validation_matrix.tolist(),
        test_confusion=test_matrix.tolist(),
    )
    artifacts = {
        "feature_names": feature_names,
        "weights": best_weights,
        "train_mean": mean,
        "train_std": std,
        "pair_indices": pair_indices,
        "clean_splits": clean_splits,
        "validation_predictions": validation_predictions,
        "test_predictions": test_predictions,
        "validation_probabilities": softmax(validation_logits),
        "test_probabilities": softmax(test_logits),
    }
    return result, artifacts


def split_distribution_rows(frame: pd.DataFrame) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for split_name, split_frame_df in split_frame(frame).items():
        total = len(split_frame_df)
        counts = split_frame_df[THREE_BUCKET_TARGET].value_counts().to_dict()
        for idx, class_name in enumerate(CLASS_NAMES):
            count = int(counts.get(idx, 0))
            rows.append(
                {
                    "split": split_name,
                    "bucket": class_name,
                    "sample_count": count,
                    "sample_ratio": round_float(count / total if total else 0.0),
                }
            )
    return rows


def selected_return_summary(returns: np.ndarray) -> tuple[float, float, float]:
    if len(returns) == 0:
        return 0.0, 0.0, 0.0
    equity = np.cumprod(1.0 + returns)
    return float((returns > 0).mean()), float(returns.mean()), rb.max_drawdown(equity)


def summarize_three_bucket_predictions(
    model_name: str,
    frame: pd.DataFrame,
    predictions: np.ndarray,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    future_returns = frame["future_return_60"].to_numpy(dtype=np.float64)
    total = len(frame)
    for idx, class_name in enumerate(CLASS_NAMES):
        mask = predictions == idx
        selected = future_returns[mask]
        hit_rate, avg_return, max_drawdown_compound = selected_return_summary(selected)
        rows.append(
            {
                "model_name": model_name,
                "predicted_bucket": class_name,
                "sample_count": int(mask.sum()),
                "sample_ratio": round_float(mask.mean() if total else 0.0),
                "hit_rate": round_float(hit_rate),
                "avg_return": round_float(avg_return),
                "max_drawdown_compound": round_float(max_drawdown_compound),
            }
        )
    return rows


def recent_five_year_frame(frame: pd.DataFrame) -> pd.DataFrame:
    cutoff = frame["date"].max() - pd.DateOffset(years=RECENT_YEARS)
    return frame.loc[frame["date"] >= cutoff].copy().reset_index(drop=True)


def compare_recent_five_years(frame: pd.DataFrame, three_bucket_artifacts: dict[str, object]) -> pd.DataFrame:
    recent = recent_five_year_frame(frame)

    live_result, live_artifacts = rb.train_model(frame, "live_combo", extra_features=("ret_60", "sma_gap_60"))
    live_probabilities = rb.score_frame(
        recent,
        live_artifacts["feature_names"],
        live_artifacts["train_mean"],
        live_artifacts["train_std"],
        live_artifacts["pair_indices"],
        live_artifacts["weights"],
    )
    live_selected = live_probabilities >= float(live_artifacts["threshold"])
    future_returns = recent["future_return_60"].to_numpy(dtype=np.float64)
    live_hit_rate, live_avg_return, live_max_drawdown = selected_return_summary(future_returns[live_selected])
    live_negative_hit_rate, live_negative_avg_return, live_negative_max_drawdown = selected_return_summary(future_returns[~live_selected])

    three_bucket_probabilities = softmax(
        tr.add_bias(
            rb.add_interactions(
                (
                    recent[three_bucket_artifacts["feature_names"]].to_numpy(dtype=np.float32)
                    - three_bucket_artifacts["train_mean"]
                )
                / three_bucket_artifacts["train_std"],
                three_bucket_artifacts["pair_indices"],
            )
        )
        @ three_bucket_artifacts["weights"]
    )
    three_bucket_predictions = three_bucket_probabilities.argmax(axis=1).astype(np.int64)

    rows = [
        {
            "model_name": "binary_live",
            "signal_name": "positive",
            "sample_count": int(live_selected.sum()),
            "sample_ratio": round_float(float(live_selected.mean())),
            "hit_rate": round_float(live_hit_rate),
            "avg_return": round_float(live_avg_return),
            "max_drawdown_compound": round_float(live_max_drawdown),
            "date_start": recent["date"].iloc[0].strftime("%Y-%m-%d"),
            "date_end": recent["date"].iloc[-1].strftime("%Y-%m-%d"),
            "reference_test_f1": round_float(live_result.test_f1),
        },
        {
            "model_name": "binary_live",
            "signal_name": "negative",
            "sample_count": int((~live_selected).sum()),
            "sample_ratio": round_float(float((~live_selected).mean())),
            "hit_rate": round_float(live_negative_hit_rate),
            "avg_return": round_float(live_negative_avg_return),
            "max_drawdown_compound": round_float(live_negative_max_drawdown),
            "date_start": recent["date"].iloc[0].strftime("%Y-%m-%d"),
            "date_end": recent["date"].iloc[-1].strftime("%Y-%m-%d"),
            "reference_test_f1": round_float(live_result.test_f1),
        },
    ]
    rows.extend(summarize_three_bucket_predictions("three_bucket_baseline", recent, three_bucket_predictions))
    for row in rows[2:]:
        row["date_start"] = recent["date"].iloc[0].strftime("%Y-%m-%d")
        row["date_end"] = recent["date"].iloc[-1].strftime("%Y-%m-%d")
        row["reference_test_f1"] = round_float(0.0)
    return pd.DataFrame(rows)


def three_bucket_summary_rows(model_result: ThreeBucketResult) -> dict[str, object]:
    return {
        "model_name": model_result.name,
        "feature_names": ",".join(model_result.feature_names),
        "validation_macro_f1": round_float(model_result.validation_macro_f1),
        "validation_accuracy": round_float(model_result.validation_accuracy),
        "validation_bal_acc": round_float(model_result.validation_bal_acc),
        "test_macro_f1": round_float(model_result.test_macro_f1),
        "test_accuracy": round_float(model_result.test_accuracy),
        "test_bal_acc": round_float(model_result.test_bal_acc),
        "validation_confusion": json.dumps(model_result.validation_confusion),
        "test_confusion": json.dumps(model_result.test_confusion),
    }


def main() -> None:
    ensure_cache_dir()
    dataset = build_three_bucket_frame()

    binary_frame = pr.load_dataset_frame()
    binary_candidates = {
        "ret_60_plus_sma_gap_60_live": ("ret_60", "sma_gap_60"),
        "ret_60_plus_sma_gap_60_plus_atr_pct_20": ("ret_60", "sma_gap_60", "atr_pct_20"),
        "ret_60_plus_sma_gap_60_plus_up_day_ratio_20": ("ret_60", "sma_gap_60", "up_day_ratio_20"),
        "ret_60_plus_sma_gap_60_plus_close_location_20": ("ret_60", "sma_gap_60", "close_location_20"),
    }
    binary_rows: list[dict[str, object]] = []
    for model_name, feature_tuple in binary_candidates.items():
        result, _artifacts = rb.train_model(binary_frame, model_name, extra_features=feature_tuple)
        binary_rows.append(
            {
                "model_name": model_name,
                "validation_f1": round_float(result.validation_f1),
                "validation_accuracy": round_float(result.validation_accuracy),
                "validation_bal_acc": round_float(result.validation_bal_acc),
                "test_f1": round_float(result.test_f1),
                "test_accuracy": round_float(result.test_accuracy),
                "test_bal_acc": round_float(result.test_bal_acc),
                "headline_score": round_float(
                    rb.compute_headline_score(
                        result.validation_f1,
                        result.validation_bal_acc,
                        result.test_f1,
                        result.test_bal_acc,
                    )
                ),
                "promotion_gate": "pass" if rb.passes_promotion_gate(result.validation_bal_acc, result.test_bal_acc) else "fail",
            }
        )
    binary_df = pd.DataFrame(binary_rows).sort_values(["headline_score", "test_f1"], ascending=[False, False]).reset_index(drop=True)
    candidate_rule_df = binary_df.loc[binary_df["model_name"] != "ret_60_plus_sma_gap_60_live"].reset_index(drop=True)
    best_binary_feature_tuple = binary_candidates[candidate_rule_df.iloc[0]["model_name"]]
    rule_rows: list[dict[str, object]] = []
    for rule_name in ("top_15pct", "top_17_5pct", "top_20pct"):
        summary = rb.forward_trade_summary(binary_frame, best_binary_feature_tuple, rule_name)
        rule_rows.append(
            {
                "model_name": candidate_rule_df.iloc[0]["model_name"],
                "rule_name": rule_name,
                "trade_count": int(summary["trade_count"]),
                "hit_rate": round_float(summary["hit_rate"]),
                "avg_return": round_float(summary["avg_return"]),
            }
        )

    three_bucket_models = {
        "three_bucket_baseline": ("ret_60", "sma_gap_60"),
        "three_bucket_plus_atr_pct_20": ("ret_60", "sma_gap_60", "atr_pct_20"),
        "three_bucket_plus_up_day_ratio_20": ("ret_60", "sma_gap_60", "up_day_ratio_20"),
        "three_bucket_plus_close_location_20": ("ret_60", "sma_gap_60", "close_location_20"),
    }
    model_rows: list[dict[str, object]] = []
    model_artifacts: dict[str, dict[str, object]] = {}
    for model_name, features in three_bucket_models.items():
        result, artifacts = train_three_bucket_model(dataset, model_name, features)
        model_rows.append(three_bucket_summary_rows(result))
        model_artifacts[model_name] = artifacts

    distribution_df = pd.DataFrame(split_distribution_rows(dataset))
    model_df = pd.DataFrame(model_rows).sort_values(["test_macro_f1", "test_bal_acc"], ascending=[False, False]).reset_index(drop=True)
    recent_df = compare_recent_five_years(dataset, model_artifacts["three_bucket_baseline"])
    signal_df = pd.DataFrame(
        summarize_three_bucket_predictions(
            "three_bucket_baseline_test",
            model_artifacts["three_bucket_baseline"]["clean_splits"]["test"],
            model_artifacts["three_bucket_baseline"]["test_predictions"],
        )
    )
    binary_rule_df = pd.DataFrame(rule_rows)

    distribution_df.to_csv(DISTRIBUTION_OUTPUT_PATH, sep="\t", index=False)
    model_df.to_csv(MODEL_OUTPUT_PATH, sep="\t", index=False)
    recent_df.to_csv(RECENT_OUTPUT_PATH, sep="\t", index=False)
    signal_df.to_csv(SIGNAL_OUTPUT_PATH, sep="\t", index=False)
    binary_df.to_csv(BINARY_OUTPUT_PATH, sep="\t", index=False)
    binary_rule_df.to_csv(RULE_OUTPUT_PATH, sep="\t", index=False)

    payload = {
        "three_bucket_distribution": distribution_df.to_dict(orient="records"),
        "three_bucket_models": model_df.to_dict(orient="records"),
        "recent_binary_vs_three_bucket": recent_df.to_dict(orient="records"),
        "test_signal_summary": signal_df.to_dict(orient="records"),
        "binary_round18_candidates": binary_df.to_dict(orient="records"),
        "binary_round18_rule_summary": binary_rule_df.to_dict(orient="records"),
    }
    with open(ROUND_OUTPUT_PATH, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)

    print(f"Saved three-bucket distribution to: {DISTRIBUTION_OUTPUT_PATH}")
    print(f"Saved three-bucket model summary to: {MODEL_OUTPUT_PATH}")
    print(f"Saved three-bucket vs binary summary to: {RECENT_OUTPUT_PATH}")
    print(f"Saved three-bucket signal summary to: {SIGNAL_OUTPUT_PATH}")
    print(f"Saved round-18 binary summary to: {BINARY_OUTPUT_PATH}")
    print(f"Saved round-18 rule summary to: {RULE_OUTPUT_PATH}")
    print(f"Saved round json to: {ROUND_OUTPUT_PATH}")
    print()
    print("Binary round-18 candidates:")
    print(binary_df.to_string(index=False))
    print()
    print("Binary rule comparison:")
    print(binary_rule_df.to_string(index=False))
    print()
    print("Three-bucket models:")
    print(model_df.to_string(index=False))


if __name__ == "__main__":
    main()
