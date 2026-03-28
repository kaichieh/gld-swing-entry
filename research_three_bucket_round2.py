"""
Run a final validation round for the three-bucket return setup.
"""

from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd

import prepare as pr
import research_three_bucket as tb

REPO_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(REPO_DIR, ".cache", "gld-swing-entry")
ROUND_OUTPUT_PATH = os.path.join(CACHE_DIR, "three_bucket_round2.json")
VARIANT_OUTPUT_PATH = os.path.join(REPO_DIR, "three_bucket_round2_variants.tsv")
REGIME_OUTPUT_PATH = os.path.join(REPO_DIR, "three_bucket_regime_summary.tsv")


def round_float(value: float) -> float:
    return round(float(value), 4)


def build_bucket_frame(up_threshold: float, down_threshold: float) -> pd.DataFrame:
    frame = pr.load_dataset_frame().copy()
    labels = np.full(len(frame), 1, dtype=np.int64)
    future_returns = frame["future_return_60"].to_numpy(dtype=np.float32)
    labels[future_returns <= down_threshold] = 0
    labels[future_returns >= up_threshold] = 2
    frame[tb.THREE_BUCKET_TARGET] = labels
    return frame


def split_frame(frame: pd.DataFrame) -> dict[str, pd.DataFrame]:
    train_end, valid_end = pr.split_indices(len(frame))
    return {
        "train": frame.iloc[:train_end].copy().reset_index(drop=True),
        "validation": frame.iloc[train_end:valid_end].copy().reset_index(drop=True),
        "test": frame.iloc[valid_end:].copy().reset_index(drop=True),
    }


def distribution_string(frame: pd.DataFrame, split_name: str) -> str:
    split_df = split_frame(frame)[split_name]
    counts = split_df[tb.THREE_BUCKET_TARGET].value_counts().to_dict()
    return f"{counts.get(0, 0)}/{counts.get(1, 0)}/{counts.get(2, 0)}"


def regime_distribution_rows(frame: pd.DataFrame) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    splits = split_frame(frame)
    for split_name, split_df in splits.items():
        for regime_value in (0.0, 1.0):
            regime_df = split_df.loc[split_df["above_200dma_flag"] == regime_value].copy()
            if regime_df.empty:
                continue
            counts = regime_df[tb.THREE_BUCKET_TARGET].value_counts().to_dict()
            for idx, class_name in enumerate(tb.CLASS_NAMES):
                bucket_df = regime_df.loc[regime_df[tb.THREE_BUCKET_TARGET] == idx]
                rows.append(
                    {
                        "split": split_name,
                        "regime": "below_200dma" if regime_value == 0.0 else "above_200dma",
                        "bucket": class_name,
                        "sample_count": int(len(bucket_df)),
                        "sample_ratio": round_float(len(bucket_df) / len(regime_df) if len(regime_df) else 0.0),
                        "avg_forward_return": round_float(bucket_df["future_return_60"].mean() if len(bucket_df) else 0.0),
                    }
                )
    return rows


def regime_filtered_train_predict(
    frame: pd.DataFrame,
    regime_value: float,
    split_name: str,
    feature_names: tuple[str, ...],
) -> tuple[np.ndarray, np.ndarray]:
    splits = split_frame(frame)
    train_df = splits["train"].loc[splits["train"]["above_200dma_flag"] == regime_value].copy().reset_index(drop=True)
    eval_df = splits[split_name].loc[splits[split_name]["above_200dma_flag"] == regime_value].copy().reset_index(drop=True)
    if len(train_df) < 120 or len(eval_df) == 0:
        return np.asarray([], dtype=np.int64), np.asarray([], dtype=np.int64)
    model_result, artifacts = tb.train_three_bucket_model(
        pd.concat([train_df, eval_df, eval_df.iloc[0:0].copy()], ignore_index=True),  # dummy shape holder
        f"regime_{int(regime_value)}_{split_name}",
        feature_names,
    )
    # Rebuild with custom chronological splits: train=train_df, validation=eval_df, test=eval_df.
    clean_splits, matrices, _mean, _std, _pairs = tb.prepare_three_bucket_matrices(
        pd.concat([train_df, eval_df, eval_df], ignore_index=True),
        tb.feature_names_for(feature_names),
    )
    _ = model_result
    _ = artifacts
    return np.asarray([], dtype=np.int64), np.asarray([], dtype=np.int64)


def regime_aware_metrics(frame: pd.DataFrame, feature_names: tuple[str, ...]) -> dict[str, float]:
    splits = split_frame(frame)
    validation_predictions: list[np.ndarray] = []
    validation_labels: list[np.ndarray] = []
    test_predictions: list[np.ndarray] = []
    test_labels: list[np.ndarray] = []

    for regime_value in (0.0, 1.0):
        train_df = splits["train"].loc[splits["train"]["above_200dma_flag"] == regime_value].copy().reset_index(drop=True)
        validation_df = splits["validation"].loc[splits["validation"]["above_200dma_flag"] == regime_value].copy().reset_index(drop=True)
        test_df = splits["test"].loc[splits["test"]["above_200dma_flag"] == regime_value].copy().reset_index(drop=True)
        if len(train_df) < 120 or len(validation_df) == 0 or len(test_df) == 0:
            continue

        feature_list = tb.feature_names_for(feature_names)
        train_clean = train_df.dropna(subset=feature_list + ["future_return_60", tb.THREE_BUCKET_TARGET]).reset_index(drop=True)
        validation_clean = validation_df.dropna(subset=feature_list + ["future_return_60", tb.THREE_BUCKET_TARGET]).reset_index(drop=True)
        test_clean = test_df.dropna(subset=feature_list + ["future_return_60", tb.THREE_BUCKET_TARGET]).reset_index(drop=True)
        if len(train_clean) < 120 or len(validation_clean) == 0 or len(test_clean) == 0:
            continue

        train_x = train_clean[feature_list].to_numpy(dtype=np.float32)
        validation_x = validation_clean[feature_list].to_numpy(dtype=np.float32)
        test_x = test_clean[feature_list].to_numpy(dtype=np.float32)
        train_x, validation_x, test_x, _mean, _std = tb.rb.standardize_from_train(train_x, validation_x, test_x)
        pair_indices = tb.active_interaction_pairs(feature_list)
        train_x = tb.rb.add_interactions(train_x, pair_indices)
        validation_x = tb.rb.add_interactions(validation_x, pair_indices)
        test_x = tb.rb.add_interactions(test_x, pair_indices)
        train_x = tb.tr.add_bias(train_x)
        validation_x = tb.tr.add_bias(validation_x)
        test_x = tb.tr.add_bias(test_x)

        train_y = train_clean[tb.THREE_BUCKET_TARGET].to_numpy(dtype=np.int64)
        validation_y = validation_clean[tb.THREE_BUCKET_TARGET].to_numpy(dtype=np.int64)
        test_y = test_clean[tb.THREE_BUCKET_TARGET].to_numpy(dtype=np.int64)
        train_targets = tb.one_hot(train_y, len(tb.CLASS_NAMES))
        weights = np.zeros((train_x.shape[1], len(tb.CLASS_NAMES)), dtype=np.float32)
        best_weights = weights.copy()
        best_validation_macro_f1 = -np.inf
        epochs_without_improvement = 0

        for _epoch in range(1, tb.tr.MAX_EPOCHS + 1):
            probabilities = tb.softmax(train_x @ weights)
            gradient = train_x.T @ (probabilities - train_targets) / train_x.shape[0]
            gradient[:-1] += tb.tr.L2_REG * weights[:-1]
            weights -= tb.tr.LEARNING_RATE * gradient

            validation_logits = validation_x @ weights
            validation_macro_f1, _validation_accuracy, _validation_bal_acc, _validation_matrix, _validation_predictions = tb.evaluate_multiclass(
                validation_logits, validation_y
            )
            if validation_macro_f1 > best_validation_macro_f1:
                best_validation_macro_f1 = validation_macro_f1
                best_weights = weights.copy()
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
            if epochs_without_improvement >= tb.tr.PATIENCE:
                break

        validation_logits = validation_x @ best_weights
        test_logits = test_x @ best_weights
        _vf1, _vacc, _vbal, _vmat, validation_pred = tb.evaluate_multiclass(validation_logits, validation_y)
        _tf1, _tacc, _tbal, _tmat, test_pred = tb.evaluate_multiclass(test_logits, test_y)
        validation_predictions.append(validation_pred)
        validation_labels.append(validation_y)
        test_predictions.append(test_pred)
        test_labels.append(test_y)

    if not validation_predictions or not test_predictions:
        return {
            "validation_macro_f1": 0.0,
            "validation_accuracy": 0.0,
            "validation_bal_acc": 0.0,
            "test_macro_f1": 0.0,
            "test_accuracy": 0.0,
            "test_bal_acc": 0.0,
        }

    validation_pred_all = np.concatenate(validation_predictions)
    validation_y_all = np.concatenate(validation_labels)
    test_pred_all = np.concatenate(test_predictions)
    test_y_all = np.concatenate(test_labels)
    validation_accuracy = float((validation_pred_all == validation_y_all).mean())
    validation_macro_f1, validation_bal_acc, _ = tb.macro_f1_and_balanced_accuracy(validation_y_all, validation_pred_all, len(tb.CLASS_NAMES))
    test_accuracy = float((test_pred_all == test_y_all).mean())
    test_macro_f1, test_bal_acc, _ = tb.macro_f1_and_balanced_accuracy(test_y_all, test_pred_all, len(tb.CLASS_NAMES))
    return {
        "validation_macro_f1": round_float(validation_macro_f1),
        "validation_accuracy": round_float(validation_accuracy),
        "validation_bal_acc": round_float(validation_bal_acc),
        "test_macro_f1": round_float(test_macro_f1),
        "test_accuracy": round_float(test_accuracy),
        "test_bal_acc": round_float(test_bal_acc),
    }


def main() -> None:
    os.makedirs(CACHE_DIR, exist_ok=True)
    variants = [
        ("up8_down4", 0.08, -0.04),
        ("up8_down6", 0.08, -0.06),
    ]
    variant_rows: list[dict[str, object]] = []
    regime_rows: list[dict[str, object]] = []
    payload: dict[str, object] = {"variants": [], "regime_distribution": [], "regime_models": {}}

    for variant_name, up_threshold, down_threshold in variants:
        frame = build_bucket_frame(up_threshold, down_threshold)
        baseline_result, _artifacts = tb.train_three_bucket_model(frame, variant_name, ("ret_60", "sma_gap_60"))
        variant_row = {
            "variant_name": variant_name,
            "up_threshold": round_float(up_threshold),
            "down_threshold": round_float(down_threshold),
            "train_distribution": distribution_string(frame, "train"),
            "validation_distribution": distribution_string(frame, "validation"),
            "test_distribution": distribution_string(frame, "test"),
            "validation_macro_f1": round_float(baseline_result.validation_macro_f1),
            "validation_accuracy": round_float(baseline_result.validation_accuracy),
            "validation_bal_acc": round_float(baseline_result.validation_bal_acc),
            "test_macro_f1": round_float(baseline_result.test_macro_f1),
            "test_accuracy": round_float(baseline_result.test_accuracy),
            "test_bal_acc": round_float(baseline_result.test_bal_acc),
        }
        variant_rows.append(variant_row)
        payload["variants"].append(variant_row)
        if variant_name == "up8_down4":
            regime_rows = regime_distribution_rows(frame)
            regime_metrics = regime_aware_metrics(frame, ("ret_60", "sma_gap_60"))
            payload["regime_distribution"] = regime_rows
            payload["regime_models"]["above_200dma_split"] = regime_metrics

    variant_df = pd.DataFrame(variant_rows)
    regime_df = pd.DataFrame(regime_rows)
    variant_df.to_csv(VARIANT_OUTPUT_PATH, sep="\t", index=False)
    regime_df.to_csv(REGIME_OUTPUT_PATH, sep="\t", index=False)
    with open(ROUND_OUTPUT_PATH, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)

    print(f"Saved variant summary to: {VARIANT_OUTPUT_PATH}")
    print(f"Saved regime summary to: {REGIME_OUTPUT_PATH}")
    print(f"Saved round json to: {ROUND_OUTPUT_PATH}")
    print()
    print(variant_df.to_string(index=False))
    print()
    if not regime_df.empty:
        print(regime_df.to_string(index=False))
        print()
    if payload["regime_models"]:
        print(json.dumps(payload["regime_models"], indent=2))


if __name__ == "__main__":
    main()
