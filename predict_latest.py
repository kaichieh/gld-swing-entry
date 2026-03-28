"""
Score the latest available GLD daily bar without requiring future labels.

Default live config follows the current strongest practical setup:
- baseline features
- extra features: ret_60, sma_gap_60
- default interaction: drawdown_20:volume_vs_20
"""

from __future__ import annotations

import json

import numpy as np

import train as tr
from prepare import add_price_features, download_gld_prices, load_dataset_frame

DEFAULT_LIVE_EXTRA_FEATURES = ("ret_60", "sma_gap_60")
WEAK_BULLISH_QUANTILE = 0.70
BULLISH_QUANTILE = 0.90
VERY_STRONG_BULLISH_QUANTILE = 0.97


def build_feature_names() -> list[str]:
    feature_names = list(tr.FEATURE_COLUMNS)
    configured = set(tr.get_env_csv("AR_EXTRA_BASE_FEATURES", DEFAULT_LIVE_EXTRA_FEATURES))
    for column in tr.EXPERIMENTAL_FEATURE_COLUMNS:
        if column in configured:
            feature_names.append(column)
    drop_features = set(tr.get_env_csv("AR_DROP_FEATURES"))
    return [name for name in feature_names if name not in drop_features]


def fit_model(splits: dict[str, object], feature_names: list[str]) -> tuple[np.ndarray, float]:
    train_x = splits["train"].frame[feature_names].to_numpy(dtype=np.float32)
    validation_x = splits["validation"].frame[feature_names].to_numpy(dtype=np.float32)
    train_y = splits["train"].labels
    validation_y = splits["validation"].labels

    train_x, validation_x, _ = tr.standardize(train_x, validation_x, validation_x.copy())
    train_x, validation_x, _ = tr.add_interaction_terms(train_x, validation_x, validation_x.copy(), feature_names)
    train_x = tr.add_bias(train_x)
    validation_x = tr.add_bias(validation_x)

    learning_rate = tr.get_env_float("AR_LEARNING_RATE", tr.LEARNING_RATE)
    l2_reg = tr.get_env_float("AR_L2_REG", tr.L2_REG)
    pos_weight = tr.get_env_float("AR_POS_WEIGHT", tr.POS_WEIGHT)
    neg_weight = tr.get_env_float("AR_NEG_WEIGHT", tr.NEG_WEIGHT)
    max_epochs = tr.get_env_int("AR_MAX_EPOCHS", tr.MAX_EPOCHS)
    patience_limit = tr.get_env_int("AR_PATIENCE", tr.PATIENCE)

    weights = np.zeros(train_x.shape[1], dtype=np.float32)
    best_weights = weights.copy()
    best_validation_f1 = -np.inf
    best_threshold = 0.5
    epochs_without_improvement = 0

    validation_returns = splits["validation"].frame["future_return_60"].to_numpy(dtype=np.float32)

    for _epoch in range(1, max_epochs + 1):
        logits = train_x @ weights
        probs = tr.sigmoid(logits)
        sample_weights = np.where(train_y == 1.0, pos_weight, neg_weight).astype(np.float32)
        gradient = train_x.T @ ((probs - train_y) * sample_weights) / train_x.shape[0]
        gradient[:-1] += l2_reg * weights[:-1]
        weights -= learning_rate * gradient

        validation_logits = validation_x @ weights
        threshold = tr.select_threshold(tr.sigmoid(validation_logits), validation_y)
        validation_metrics = tr.compute_metrics(validation_logits, validation_y, validation_returns, threshold)
        if validation_metrics.f1 > best_validation_f1:
            best_validation_f1 = validation_metrics.f1
            best_weights = weights.copy()
            best_threshold = threshold
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        if epochs_without_improvement >= patience_limit:
            break

    return best_weights, best_threshold


def score_latest_row(feature_names: list[str], train_frame, latest_row) -> tuple[float, dict[str, float]]:
    train_x = train_frame[feature_names].to_numpy(dtype=np.float32)
    latest_x = latest_row[feature_names].to_numpy(dtype=np.float32)
    mean = train_x.mean(axis=0, keepdims=True)
    std = train_x.std(axis=0, keepdims=True)
    std = np.where(std < 1e-6, 1.0, std)
    standardized_latest = (latest_x - mean) / std
    _, _, latest_augmented = tr.add_interaction_terms(train_x[:1], train_x[:1], standardized_latest, feature_names)
    latest_augmented = tr.add_bias(latest_augmented)
    raw_snapshot = {name: float(latest_row.iloc[0][name]) for name in feature_names}
    return latest_augmented, raw_snapshot


def classify_signal(
    probability: float, threshold: float, historical_probabilities: np.ndarray
) -> tuple[str, dict[str, float]]:
    confidence_gap = probability - threshold
    historical_gaps = historical_probabilities - threshold
    positive_gaps = historical_gaps[historical_gaps > 0]
    weak_cutoff = float(np.quantile(positive_gaps, WEAK_BULLISH_QUANTILE)) if len(positive_gaps) else 0.0
    strong_cutoff = float(np.quantile(positive_gaps, BULLISH_QUANTILE)) if len(positive_gaps) else 0.0
    very_strong_cutoff = float(np.quantile(positive_gaps, VERY_STRONG_BULLISH_QUANTILE)) if len(positive_gaps) else 0.0

    if confidence_gap <= 0:
        signal = "no_entry"
    elif confidence_gap >= very_strong_cutoff:
        signal = "very_strong_bullish"
    elif confidence_gap >= strong_cutoff:
        signal = "strong_bullish"
    elif confidence_gap >= weak_cutoff:
        signal = "bullish"
    else:
        signal = "weak_bullish"

    return signal, {
        "confidence_gap": round(confidence_gap, 4),
        "weak_bullish_cutoff": round(weak_cutoff, 4),
        "strong_bullish_cutoff": round(strong_cutoff, 4),
        "very_strong_bullish_cutoff": round(very_strong_cutoff, 4),
    }


def main() -> None:
    tr.set_seed(tr.get_env_int("AR_SEED", tr.SEED))
    raw_prices = download_gld_prices()
    live_features = add_price_features(raw_prices)
    splits = tr.load_splits()
    feature_names = build_feature_names()
    weights, threshold = fit_model(splits, feature_names)
    validation_history, test_history = score_latest_row(feature_names, splits["train"].frame, splits["validation"].frame), score_latest_row(
        feature_names, splits["train"].frame, splits["test"].frame
    )

    latest_live = live_features.iloc[[-1]].copy()
    latest_vector, raw_snapshot = score_latest_row(feature_names, splits["train"].frame, latest_live)
    probability = float(tr.sigmoid(latest_vector @ weights)[0])
    predicted_label = int(probability >= threshold)

    validation_probs = tr.sigmoid(validation_history[0] @ weights)
    test_probs = tr.sigmoid(test_history[0] @ weights)
    signal, band_info = classify_signal(probability, float(threshold), np.concatenate([validation_probs, test_probs]))
    bullish = predicted_label == 1
    output = {
        "signal_summary": {
            "signal": signal,
            "verdict": "Model favors a medium-term entry" if bullish else "Model does not favor a medium-term entry",
            "predicted_label": predicted_label,
            "predicted_probability": round(probability, 4),
            "decision_threshold": round(float(threshold), 4),
            **band_info,
        },
        "latest_raw_date": latest_live["date"].iloc[0].strftime("%Y-%m-%d"),
        "latest_open": round(float(latest_live["open"].iloc[0]), 2),
        "latest_high": round(float(latest_live["high"].iloc[0]), 2),
        "latest_low": round(float(latest_live["low"].iloc[0]), 2),
        "latest_close": round(float(latest_live["close"].iloc[0]), 2),
        "trained_until_label_date": splits["test"].frame["date"].iloc[-1].strftime("%Y-%m-%d"),
        "model_summary": {
            "model_family": "logistic_regression",
            "model_extra_features": [name for name in feature_names if name not in tr.FEATURE_COLUMNS],
            "default_interactions": ["drawdown_20:volume_vs_20"],
        },
        "model_extra_features": [name for name in feature_names if name not in tr.FEATURE_COLUMNS],
        "latest_feature_snapshot": {
            key: round(value, 4) for key, value in raw_snapshot.items() if key in {"ret_60", "drawdown_20", "volume_vs_20", "rsi_14"}
        },
    }
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
