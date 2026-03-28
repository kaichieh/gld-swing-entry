"""
Run a batch of formal GLD research comparisons and export compact summaries.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

import predict_latest as live
import prepare as pr
import train as tr

REPO_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(REPO_DIR, ".cache", "gld-swing-entry")
BACKTEST_OUTPUT_PATH = os.path.join(REPO_DIR, "backtest_comparison.tsv")
REGIME_OUTPUT_PATH = os.path.join(REPO_DIR, "regime_summary.tsv")
ROUND_OUTPUT_PATH = os.path.join(CACHE_DIR, "research_batch.json")
FUTURE_RETURN_COLUMN = "future_return_60"
DEFAULT_INTERACTIONS = (("drawdown_20", "volume_vs_20"),)


@dataclass
class ModelResult:
    name: str
    feature_names: list[str]
    threshold: float
    validation_f1: float
    validation_accuracy: float
    validation_bal_acc: float
    validation_positive_rate: float
    test_f1: float
    test_accuracy: float
    test_bal_acc: float
    test_positive_rate: float
    test_avg_return: float
    validation_rows: int
    test_rows: int


@dataclass
class ForwardFoldResult:
    fold_name: str
    validation_f1: float
    validation_bal_acc: float
    test_f1: float
    test_bal_acc: float
    test_positive_rate: float


@dataclass
class BacktestResult:
    model_name: str
    rule_name: str
    selected_count: int
    hit_rate: float
    avg_return: float
    max_drawdown_simple: float
    max_drawdown_compound: float
    longest_win_streak: int
    longest_loss_streak: int
    threshold_or_cutoff: float


def ensure_cache_dir() -> None:
    os.makedirs(CACHE_DIR, exist_ok=True)


def add_regime_features(frame: pd.DataFrame) -> pd.DataFrame:
    df = frame.copy()
    df["year"] = df["date"].dt.year.astype(float)
    df["rolling_return_120"] = df["close"].pct_change(120)
    df["rolling_vol_60"] = df["ret_1"].rolling(60).std()
    df["ret_120"] = df["close"].pct_change(120)
    df["sma_gap_120"] = df["close"] / df["close"].rolling(120).mean() - 1.0
    return df


def build_labeled_frame(
    raw: pd.DataFrame,
    horizon_days: int = 60,
    upper_barrier: float = 0.08,
    lower_barrier: float = -0.04,
    label_mode: str = "drop-neutral",
) -> pd.DataFrame:
    df = pr.add_price_features(raw)
    df = add_regime_features(df)
    labels, realized_returns = pr.build_barrier_labels(df, horizon_days, upper_barrier, lower_barrier)
    if label_mode == "keep-all-binary":
        labels = np.where(np.isnan(labels), 0.0, labels)
    df[pr.TARGET_COLUMN] = labels
    df[FUTURE_RETURN_COLUMN] = realized_returns
    needed = pr.FEATURE_COLUMNS + pr.EXPERIMENTAL_FEATURE_COLUMNS + [FUTURE_RETURN_COLUMN]
    if label_mode != "keep-all-binary":
        needed.append(pr.TARGET_COLUMN)
    df = df.replace([np.inf, -np.inf], np.nan)
    return df.dropna(subset=needed).reset_index(drop=True)


def get_feature_names(extra_features: tuple[str, ...] = (), drop_features: tuple[str, ...] = ()) -> list[str]:
    feature_names = list(pr.FEATURE_COLUMNS)
    for name in extra_features:
        if name not in feature_names:
            feature_names.append(name)
    return [name for name in feature_names if name not in set(drop_features)]


def split_frame(frame: pd.DataFrame) -> dict[str, pd.DataFrame]:
    train_end, valid_end = pr.split_indices(len(frame))
    return {
        "train": frame.iloc[:train_end].copy().reset_index(drop=True),
        "validation": frame.iloc[train_end:valid_end].copy().reset_index(drop=True),
        "test": frame.iloc[valid_end:].copy().reset_index(drop=True),
    }


def active_interaction_pairs(
    feature_names: list[str], extra_interactions: tuple[tuple[str, str], ...] = ()
) -> tuple[tuple[int, int], ...]:
    pairs = list(DEFAULT_INTERACTIONS)
    for pair in extra_interactions:
        if pair not in pairs:
            pairs.append(pair)
    index = {name: idx for idx, name in enumerate(feature_names)}
    return tuple((index[left], index[right]) for left, right in pairs if left in index and right in index)


def standardize_from_train(
    train_x: np.ndarray, validation_x: np.ndarray, test_x: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mean = train_x.mean(axis=0, keepdims=True)
    std = train_x.std(axis=0, keepdims=True)
    std = np.where(std < 1e-6, 1.0, std)
    return (train_x - mean) / std, (validation_x - mean) / std, (test_x - mean) / std, mean, std


def add_interactions(features: np.ndarray, pairs: tuple[tuple[int, int], ...]) -> np.ndarray:
    if not pairs:
        return features
    extras = [features[:, i : i + 1] * features[:, j : j + 1] for i, j in pairs]
    return np.concatenate([features] + extras, axis=1)


def prepare_feature_matrices(
    splits: dict[str, pd.DataFrame], feature_names: list[str], extra_interactions: tuple[tuple[str, str], ...] = ()
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], np.ndarray, np.ndarray, tuple[tuple[int, int], ...]]:
    clean_splits: dict[str, pd.DataFrame] = {}
    for split_name, frame in splits.items():
        clean_splits[split_name] = frame.dropna(subset=feature_names + [pr.TARGET_COLUMN, FUTURE_RETURN_COLUMN]).reset_index(drop=True)
    matrices = {
        split_name: clean_splits[split_name][feature_names].to_numpy(dtype=np.float32)
        for split_name in ("train", "validation", "test")
    }
    train_x, validation_x, test_x, mean, std = standardize_from_train(
        matrices["train"], matrices["validation"], matrices["test"]
    )
    pair_indices = active_interaction_pairs(feature_names, extra_interactions)
    train_x = add_interactions(train_x, pair_indices)
    validation_x = add_interactions(validation_x, pair_indices)
    test_x = add_interactions(test_x, pair_indices)
    return (
        clean_splits,
        {
            "train": tr.add_bias(train_x),
            "validation": tr.add_bias(validation_x),
            "test": tr.add_bias(test_x),
        },
        mean,
        std,
        pair_indices,
    )


def train_model(
    frame: pd.DataFrame,
    name: str,
    extra_features: tuple[str, ...] = (),
    drop_features: tuple[str, ...] = (),
    extra_interactions: tuple[tuple[str, str], ...] = (),
    neg_weight: float | None = None,
    threshold_steps: int | None = None,
) -> tuple[ModelResult, dict[str, object]]:
    feature_names = get_feature_names(extra_features, drop_features)
    splits = split_frame(frame)
    clean_splits, matrices, mean, std, pair_indices = prepare_feature_matrices(splits, feature_names, extra_interactions)
    train_x = matrices["train"]
    validation_x = matrices["validation"]
    test_x = matrices["test"]
    train_y = clean_splits["train"][pr.TARGET_COLUMN].to_numpy(dtype=np.float32)
    validation_y = clean_splits["validation"][pr.TARGET_COLUMN].to_numpy(dtype=np.float32)
    test_y = clean_splits["test"][pr.TARGET_COLUMN].to_numpy(dtype=np.float32)

    weights = np.zeros(train_x.shape[1], dtype=np.float32)
    best_weights = weights.copy()
    best_validation_f1 = -np.inf
    best_threshold = 0.5
    epochs_without_improvement = 0

    neg_weight = tr.NEG_WEIGHT if neg_weight is None else neg_weight
    threshold_steps = tr.THRESHOLD_STEPS if threshold_steps is None else threshold_steps

    for _epoch in range(1, tr.MAX_EPOCHS + 1):
        logits = train_x @ weights
        probs = tr.sigmoid(logits)
        sample_weights = np.where(train_y == 1.0, tr.POS_WEIGHT, neg_weight).astype(np.float32)
        gradient = train_x.T @ ((probs - train_y) * sample_weights) / train_x.shape[0]
        gradient[:-1] += tr.L2_REG * weights[:-1]
        weights -= tr.LEARNING_RATE * gradient

        validation_logits = validation_x @ weights
        threshold = select_threshold_with_steps(tr.sigmoid(validation_logits), validation_y, threshold_steps)
        validation_metrics = tr.compute_metrics(
            validation_logits,
            validation_y,
            clean_splits["validation"][FUTURE_RETURN_COLUMN].to_numpy(dtype=np.float32),
            threshold,
        )
        if validation_metrics.f1 > best_validation_f1:
            best_validation_f1 = validation_metrics.f1
            best_weights = weights.copy()
            best_threshold = threshold
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        if epochs_without_improvement >= tr.PATIENCE:
            break

    validation_logits = validation_x @ best_weights
    test_logits = test_x @ best_weights
    validation_metrics = tr.compute_metrics(
        validation_logits,
        validation_y,
        clean_splits["validation"][FUTURE_RETURN_COLUMN].to_numpy(dtype=np.float32),
        best_threshold,
    )
    test_metrics = tr.compute_metrics(
        test_logits,
        test_y,
        clean_splits["test"][FUTURE_RETURN_COLUMN].to_numpy(dtype=np.float32),
        best_threshold,
    )
    result = ModelResult(
        name=name,
        feature_names=feature_names,
        threshold=best_threshold,
        validation_f1=validation_metrics.f1,
        validation_accuracy=validation_metrics.accuracy,
        validation_bal_acc=validation_metrics.balanced_accuracy,
        validation_positive_rate=validation_metrics.positive_rate,
        test_f1=test_metrics.f1,
        test_accuracy=test_metrics.accuracy,
        test_bal_acc=test_metrics.balanced_accuracy,
        test_positive_rate=test_metrics.positive_rate,
        test_avg_return=test_metrics.avg_realized_return,
        validation_rows=len(clean_splits["validation"]),
        test_rows=len(clean_splits["test"]),
    )
    artifacts = {
        "feature_names": feature_names,
        "pair_indices": pair_indices,
        "weights": best_weights,
        "threshold": best_threshold,
        "train_mean": mean,
        "train_std": std,
        "clean_splits": clean_splits,
        "test_probabilities": tr.sigmoid(test_logits),
        "validation_probabilities": tr.sigmoid(validation_logits),
        "neg_weight": neg_weight,
    }
    return result, artifacts


def select_threshold_with_steps(probabilities: np.ndarray, labels: np.ndarray, threshold_steps: int) -> float:
    best_threshold = 0.5
    best_f1 = -1.0
    best_bal_acc = -1.0
    for threshold in np.linspace(tr.THRESHOLD_MIN, tr.THRESHOLD_MAX, threshold_steps):
        tp, tn, fp, fn, _ = tr.classification_stats(probabilities, labels, float(threshold))
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


def longest_streak(returns: np.ndarray, positive: bool) -> int:
    best = 0
    current = 0
    for value in returns:
        condition = value > 0 if positive else value <= 0
        if condition:
            current += 1
            best = max(best, current)
        else:
            current = 0
    return best


def max_drawdown(equity_curve: np.ndarray) -> float:
    peaks = np.maximum.accumulate(equity_curve)
    drawdowns = equity_curve / np.maximum(peaks, 1e-8) - 1.0
    return float(drawdowns.min()) if len(drawdowns) else 0.0


def run_non_overlap_backtest(
    dates: pd.Series, future_returns: np.ndarray, selected: np.ndarray, horizon_days: int, threshold_or_cutoff: float
) -> BacktestResult:
    chosen_returns: list[float] = []
    idx = 0
    while idx < len(selected):
        if selected[idx]:
            chosen_returns.append(float(future_returns[idx]))
            idx += horizon_days
        else:
            idx += 1
    returns = np.asarray(chosen_returns, dtype=np.float64)
    if len(returns) == 0:
        return BacktestResult("", "", 0, 0.0, 0.0, 0.0, 0.0, 0, 0, threshold_or_cutoff)
    simple_equity = np.cumsum(returns) + 1.0
    compound_equity = np.cumprod(1.0 + returns)
    return BacktestResult(
        model_name="",
        rule_name="",
        selected_count=len(returns),
        hit_rate=float((returns > 0).mean()),
        avg_return=float(returns.mean()),
        max_drawdown_simple=max_drawdown(simple_equity),
        max_drawdown_compound=max_drawdown(compound_equity),
        longest_win_streak=longest_streak(returns, positive=True),
        longest_loss_streak=longest_streak(returns, positive=False),
        threshold_or_cutoff=threshold_or_cutoff,
    )


def backtest_rules(model_name: str, artifacts: dict[str, object]) -> list[BacktestResult]:
    test_frame = artifacts["clean_splits"]["test"]
    probs = np.asarray(artifacts["test_probabilities"], dtype=np.float64)
    threshold = float(artifacts["threshold"])
    future_returns = test_frame[FUTURE_RETURN_COLUMN].to_numpy(dtype=np.float64)
    rows: list[BacktestResult] = []

    selections = {
        "threshold": (probs >= threshold, threshold),
        "top_10pct": (probs >= float(np.quantile(probs, 0.90)), float(np.quantile(probs, 0.90))),
        "top_15pct": (probs >= float(np.quantile(probs, 0.85)), float(np.quantile(probs, 0.85))),
        "top_20pct": (probs >= float(np.quantile(probs, 0.80)), float(np.quantile(probs, 0.80))),
    }
    historical_probs = np.concatenate(
        [np.asarray(artifacts["validation_probabilities"], dtype=np.float64), np.asarray(artifacts["test_probabilities"], dtype=np.float64)]
    )
    bullish_plus = np.array(
        [
            live.classify_signal(float(prob), threshold, historical_probs)[0]
            in {"bullish", "strong_bullish", "very_strong_bullish"}
            for prob in probs
        ],
        dtype=bool,
    )
    selections["bullish_plus"] = (bullish_plus, threshold)
    strong_plus = np.array(
        [
            live.classify_signal(float(prob), threshold, historical_probs)[0]
            in {"strong_bullish", "very_strong_bullish"}
            for prob in probs
        ],
        dtype=bool,
    )
    selections["strong_bullish_plus"] = (strong_plus, threshold)

    for rule_name, (selected, cutoff) in selections.items():
        result = run_non_overlap_backtest(
            test_frame["date"], future_returns, selected.astype(bool), pr.HORIZON_DAYS, float(cutoff)
        )
        result.model_name = model_name
        result.rule_name = rule_name
        rows.append(result)
    return rows


def fixed_threshold_backtests(model_name: str, artifacts: dict[str, object], thresholds: tuple[float, ...]) -> list[BacktestResult]:
    test_frame = artifacts["clean_splits"]["test"]
    probs = np.asarray(artifacts["test_probabilities"], dtype=np.float64)
    future_returns = test_frame[FUTURE_RETURN_COLUMN].to_numpy(dtype=np.float64)
    rows: list[BacktestResult] = []
    for threshold in thresholds:
        result = run_non_overlap_backtest(test_frame["date"], future_returns, probs >= threshold, pr.HORIZON_DAYS, threshold)
        result.model_name = model_name
        result.rule_name = f"fixed_{threshold:.2f}"
        rows.append(result)
    return rows


def precision_recall(probabilities: np.ndarray, labels: np.ndarray, threshold: float) -> dict[str, float]:
    tp, tn, fp, fn, predictions = tr.classification_stats(probabilities, labels, threshold)
    precision = tp / max(tp + fp, 1.0)
    recall = tp / max(tp + fn, 1.0)
    return {
        "predicted_positive_rate": float(predictions.mean()),
        "precision": float(precision),
        "recall": float(recall),
    }


def walk_forward_splits(frame: pd.DataFrame, folds: int = 3) -> list[tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]]:
    total = len(frame)
    fold_size = total // (folds + 2)
    splits: list[tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]] = []
    for fold_idx in range(folds):
        train_end = fold_size * (fold_idx + 2)
        validation_end = train_end + fold_size
        test_end = min(validation_end + fold_size, total)
        if test_end - validation_end < max(30, fold_size // 2):
            break
        train = frame.iloc[:train_end].copy().reset_index(drop=True)
        validation = frame.iloc[train_end:validation_end].copy().reset_index(drop=True)
        test = frame.iloc[validation_end:test_end].copy().reset_index(drop=True)
        splits.append((train, validation, test))
    return splits


def fit_on_custom_splits(
    train_frame: pd.DataFrame,
    validation_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
    feature_names: list[str],
    neg_weight: float = tr.NEG_WEIGHT,
) -> ForwardFoldResult:
    splits = {"train": train_frame, "validation": validation_frame, "test": test_frame}
    clean_splits, matrices, _mean, _std, _pairs = prepare_feature_matrices(splits, feature_names)
    train_x = matrices["train"]
    validation_x = matrices["validation"]
    test_x = matrices["test"]
    train_y = clean_splits["train"][pr.TARGET_COLUMN].to_numpy(dtype=np.float32)
    validation_y = clean_splits["validation"][pr.TARGET_COLUMN].to_numpy(dtype=np.float32)
    test_y = clean_splits["test"][pr.TARGET_COLUMN].to_numpy(dtype=np.float32)
    weights = np.zeros(train_x.shape[1], dtype=np.float32)
    best_weights = weights.copy()
    best_validation_f1 = -np.inf
    best_threshold = 0.5
    epochs_without_improvement = 0

    for _epoch in range(1, tr.MAX_EPOCHS + 1):
        probs = tr.sigmoid(train_x @ weights)
        sample_weights = np.where(train_y == 1.0, tr.POS_WEIGHT, neg_weight).astype(np.float32)
        gradient = train_x.T @ ((probs - train_y) * sample_weights) / train_x.shape[0]
        gradient[:-1] += tr.L2_REG * weights[:-1]
        weights -= tr.LEARNING_RATE * gradient
        validation_logits = validation_x @ weights
        threshold = select_threshold_with_steps(tr.sigmoid(validation_logits), validation_y, tr.THRESHOLD_STEPS)
        validation_metrics = tr.compute_metrics(
            validation_logits,
            validation_y,
            clean_splits["validation"][FUTURE_RETURN_COLUMN].to_numpy(dtype=np.float32),
            threshold,
        )
        if validation_metrics.f1 > best_validation_f1:
            best_validation_f1 = validation_metrics.f1
            best_weights = weights.copy()
            best_threshold = threshold
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        if epochs_without_improvement >= tr.PATIENCE:
            break

    validation_metrics = tr.compute_metrics(
        validation_x @ best_weights,
        validation_y,
        clean_splits["validation"][FUTURE_RETURN_COLUMN].to_numpy(dtype=np.float32),
        best_threshold,
    )
    test_metrics = tr.compute_metrics(
        test_x @ best_weights,
        test_y,
        clean_splits["test"][FUTURE_RETURN_COLUMN].to_numpy(dtype=np.float32),
        best_threshold,
    )
    return ForwardFoldResult(
        fold_name="",
        validation_f1=validation_metrics.f1,
        validation_bal_acc=validation_metrics.balanced_accuracy,
        test_f1=test_metrics.f1,
        test_bal_acc=test_metrics.balanced_accuracy,
        test_positive_rate=test_metrics.positive_rate,
    )


def evaluate_seeds(frame: pd.DataFrame, extra_features: tuple[str, ...], neg_weight: float = tr.NEG_WEIGHT) -> list[ModelResult]:
    feature_names = get_feature_names(extra_features)
    results: list[ModelResult] = []
    for seed in (1, 2, 3):
        np.random.seed(seed)
        result, _ = train_model(frame, f"seed_{seed}", extra_features=extra_features, neg_weight=neg_weight)
        result.name = f"seed_{seed}"
        result.feature_names = feature_names
        results.append(result)
    return results


def evaluate_walk_forward(frame: pd.DataFrame, extra_features: tuple[str, ...], neg_weight: float = tr.NEG_WEIGHT) -> list[ForwardFoldResult]:
    feature_names = get_feature_names(extra_features)
    rows: list[ForwardFoldResult] = []
    for fold_idx, (train_frame, validation_frame, test_frame) in enumerate(walk_forward_splits(frame), start=1):
        result = fit_on_custom_splits(train_frame, validation_frame, test_frame, feature_names, neg_weight=neg_weight)
        result.fold_name = f"fold_{fold_idx}"
        rows.append(result)
    return rows


def score_frame(
    frame: pd.DataFrame, feature_names: list[str], mean: np.ndarray, std: np.ndarray, pair_indices: tuple[tuple[int, int], ...], weights: np.ndarray
) -> np.ndarray:
    matrix = frame[feature_names].to_numpy(dtype=np.float32)
    standardized = (matrix - mean) / std
    standardized = add_interactions(standardized, pair_indices)
    standardized = tr.add_bias(standardized)
    return tr.sigmoid(standardized @ weights)


def regime_summary(frame: pd.DataFrame) -> pd.DataFrame:
    summary_frames: list[pd.DataFrame] = []
    splits = split_frame(frame)
    for split_name in ("validation", "test"):
        split_frame_df = splits[split_name].copy()
        split_frame_df["year_bucket"] = split_frame_df["date"].dt.year.astype(int)
        grouped = (
            split_frame_df.groupby("year_bucket", as_index=False)
            .agg(
                rows=(pr.TARGET_COLUMN, "size"),
                positive_rate=(pr.TARGET_COLUMN, "mean"),
                avg_future_return=(FUTURE_RETURN_COLUMN, "mean"),
            )
            .assign(split=split_name)
        )
        summary_frames.append(grouped[["split", "year_bucket", "rows", "positive_rate", "avg_future_return"]])
    return pd.concat(summary_frames, ignore_index=True)


def stage_positive_rate_summary(full_frame: pd.DataFrame, models: dict[str, dict[str, object]]) -> pd.DataFrame:
    periods = [
        ("2008_2010", "2008-01-01", "2010-12-31"),
        ("2011_2019", "2011-01-01", "2019-12-31"),
        ("2020_2023", "2020-01-01", "2023-12-31"),
        ("2024_plus", "2024-01-01", None),
    ]
    rows: list[dict[str, object]] = []
    for model_name, artifacts in models.items():
        clean_frame = pd.concat(
            [
                artifacts["clean_splits"]["train"],
                artifacts["clean_splits"]["validation"],
                artifacts["clean_splits"]["test"],
            ],
            ignore_index=True,
        )
        probs = score_frame(
            clean_frame,
            artifacts["feature_names"],
            artifacts["train_mean"],
            artifacts["train_std"],
            artifacts["pair_indices"],
            artifacts["weights"],
        )
        clean_frame = clean_frame.assign(predicted_positive=(probs >= float(artifacts["threshold"])).astype(float))
        for label, start, end in periods:
            mask = clean_frame["date"] >= pd.Timestamp(start)
            if end is not None:
                mask &= clean_frame["date"] <= pd.Timestamp(end)
            bucket = clean_frame.loc[mask]
            rows.append(
                {
                    "model_name": model_name,
                    "period": label,
                    "rows": len(bucket),
                    "predicted_positive_rate": float(bucket["predicted_positive"].mean()) if len(bucket) else 0.0,
                }
            )
    return pd.DataFrame(rows)


def round_float(value: float) -> float:
    return round(float(value), 4)


def main() -> None:
    ensure_cache_dir()
    raw = pr.download_gld_prices()
    default_frame = build_labeled_frame(raw)

    model_specs = [
        ("ret_60", ("ret_60",), ()),
        ("sma_gap_60", ("sma_gap_60",), ()),
        ("ret_60_plus_sma_gap_60", ("ret_60", "sma_gap_60"), ()),
        ("ret_60_plus_sma_gap_60_plus_default_interaction", ("ret_60", "sma_gap_60"), ()),
        ("ret_60_replaces_ret_20", ("ret_60",), ("ret_20",)),
        ("ret_60_plus_year", ("ret_60", "year"), ()),
        ("ret_60_plus_rolling_return_120", ("ret_60", "rolling_return_120"), ()),
        ("ret_60_plus_rolling_vol_60", ("ret_60", "rolling_vol_60"), ()),
        ("ret_60_plus_all_regime_features", ("ret_60", "year", "rolling_return_120", "rolling_vol_60"), ()),
        ("ret_60_plus_sma_gap_60_plus_neg_weight_1_1", ("ret_60", "sma_gap_60"), ()),
        ("ret_60_plus_sma_gap_60_plus_rolling_vol_60", ("ret_60", "sma_gap_60", "rolling_vol_60"), ()),
        ("ret_120", ("ret_120",), ()),
        ("sma_gap_120", ("sma_gap_120",), ()),
        ("ret_60_plus_sma_gap_60_interaction", ("ret_60", "sma_gap_60"), ()),
    ]

    model_results: dict[str, ModelResult] = {}
    model_artifacts: dict[str, dict[str, object]] = {}
    for name, extras, drops in model_specs:
        neg_weight = 1.1 if name == "ret_60_plus_sma_gap_60_plus_neg_weight_1_1" else None
        extra_interactions = (("ret_60", "sma_gap_60"),) if name == "ret_60_plus_sma_gap_60_interaction" else ()
        result, artifacts = train_model(
            default_frame,
            name,
            extra_features=extras,
            drop_features=drops,
            extra_interactions=extra_interactions,
            neg_weight=neg_weight,
        )
        model_results[name] = result
        model_artifacts[name] = artifacts
        model_artifacts[name]["feature_names"] = result.feature_names

    backtests: list[BacktestResult] = []
    for model_name in ("ret_60", "sma_gap_60", "ret_60_plus_sma_gap_60"):
        backtests.extend(backtest_rules(model_name, model_artifacts[model_name]))
    backtests.extend(
        fixed_threshold_backtests("ret_60_plus_sma_gap_60", model_artifacts["ret_60_plus_sma_gap_60"], (0.47, 0.49, 0.51))
    )

    backtest_frame = pd.DataFrame(
        [
            {
                "model_name": row.model_name,
                "rule_name": row.rule_name,
                "selected_count": row.selected_count,
                "hit_rate": round_float(row.hit_rate),
                "avg_return": round_float(row.avg_return),
                "max_drawdown_simple": round_float(row.max_drawdown_simple),
                "max_drawdown_compound": round_float(row.max_drawdown_compound),
                "longest_win_streak": row.longest_win_streak,
                "longest_loss_streak": row.longest_loss_streak,
                "threshold_or_cutoff": round_float(row.threshold_or_cutoff),
            }
            for row in backtests
        ]
    )
    backtest_frame.to_csv(BACKTEST_OUTPUT_PATH, sep="\t", index=False)

    regime_frame = regime_summary(default_frame)
    stage_frame = stage_positive_rate_summary(
        default_frame,
        {"ret_60": model_artifacts["ret_60"], "sma_gap_60": model_artifacts["sma_gap_60"]},
    )
    combined_regime = pd.concat(
        [
            regime_frame.assign(summary_type="yearly_barrier"),
            stage_frame.assign(summary_type="stage_model_positive_rate"),
        ],
        ignore_index=True,
        sort=False,
    )
    combined_regime.to_csv(REGIME_OUTPUT_PATH, sep="\t", index=False)

    label_configs = [
        ("80d_8_4_ret_60", 80, 0.08, -0.04),
        ("120d_8_4_ret_60", 120, 0.08, -0.04),
        ("60d_12_6_ret_60", 60, 0.12, -0.06),
    ]
    label_results: dict[str, ModelResult] = {}
    for name, horizon, upper, lower in label_configs:
        frame = build_labeled_frame(raw, horizon_days=horizon, upper_barrier=upper, lower_barrier=lower)
        result, _ = train_model(frame, name, extra_features=("ret_60",))
        label_results[name] = result

    combo_seed_results = evaluate_seeds(default_frame, ("ret_60", "sma_gap_60"))
    combo_walk_forward = evaluate_walk_forward(default_frame, ("ret_60", "sma_gap_60"))
    ret60_walk_forward = evaluate_walk_forward(default_frame, ("ret_60",))

    combo_artifacts = model_artifacts["ret_60_plus_sma_gap_60"]
    precision_summary = {
        "validation": precision_recall(
            np.asarray(combo_artifacts["validation_probabilities"], dtype=np.float64),
            combo_artifacts["clean_splits"]["validation"][pr.TARGET_COLUMN].to_numpy(dtype=np.float32),
            float(combo_artifacts["threshold"]),
        ),
        "test": precision_recall(
            np.asarray(combo_artifacts["test_probabilities"], dtype=np.float64),
            combo_artifacts["clean_splits"]["test"][pr.TARGET_COLUMN].to_numpy(dtype=np.float32),
            float(combo_artifacts["threshold"]),
        ),
    }

    threshold_scan_results: dict[str, ModelResult] = {}
    for steps in (401, 801, 1201):
        result, _ = train_model(
            default_frame,
            f"ret_60_plus_sma_gap_60_threshold_steps_{steps}",
            extra_features=("ret_60", "sma_gap_60"),
            threshold_steps=steps,
        )
        threshold_scan_results[str(steps)] = result

    round_payload = {
        "models": {name: asdict(result) for name, result in model_results.items()},
        "backtests": [asdict(row) for row in backtests],
        "label_sweeps": {name: asdict(result) for name, result in label_results.items()},
        "combo_seed_results": [asdict(row) for row in combo_seed_results],
        "combo_walk_forward": [asdict(row) for row in combo_walk_forward],
        "ret60_walk_forward": [asdict(row) for row in ret60_walk_forward],
        "combo_precision_summary": precision_summary,
        "combo_threshold_scan": {name: asdict(result) for name, result in threshold_scan_results.items()},
    }
    with open(ROUND_OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(round_payload, f, indent=2)
    print(json.dumps(round_payload, indent=2))


if __name__ == "__main__":
    main()
