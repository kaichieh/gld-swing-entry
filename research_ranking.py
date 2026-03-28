"""
Run the first ranking research round on future_return_60.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd

import prepare as pr
import research_batch as rb

REPO_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(REPO_DIR, ".cache", "gld-swing-entry")
ROUND_OUTPUT_PATH = os.path.join(CACHE_DIR, "ranking_round1.json")
DISTRIBUTION_OUTPUT_PATH = os.path.join(REPO_DIR, "ranking_target_summary.tsv")
MODEL_OUTPUT_PATH = os.path.join(REPO_DIR, "ranking_model_summary.tsv")
TEST_RULE_OUTPUT_PATH = os.path.join(REPO_DIR, "ranking_test_rules.tsv")
FORWARD_RULE_OUTPUT_PATH = os.path.join(REPO_DIR, "ranking_forward_rules.tsv")
DECILE_OUTPUT_PATH = os.path.join(REPO_DIR, "ranking_decile_summary.tsv")
ROUND21_OUTPUT_PATH = os.path.join(REPO_DIR, "round21_binary_summary.tsv")
BINARY_RULE_OUTPUT_PATH = os.path.join(REPO_DIR, "binary_live_rule_summary.tsv")
ROUND21_RULE_OUTPUT_PATH = os.path.join(REPO_DIR, "round21_rule_summary.tsv")


@dataclass
class RankingResult:
    model_name: str
    feature_names: list[str]
    threshold: float
    validation_spearman: float
    validation_f1: float
    validation_accuracy: float
    validation_bal_acc: float
    validation_positive_rate: float
    test_spearman: float
    test_f1: float
    test_accuracy: float
    test_bal_acc: float
    test_positive_rate: float


def round_float(value: float) -> float:
    return round(float(value), 4)


def ridge_fit(features: np.ndarray, target: np.ndarray, l2_reg: float) -> np.ndarray:
    identity = np.eye(features.shape[1], dtype=np.float32)
    identity[-1, -1] = 0.0
    lhs = features.T @ features + l2_reg * identity
    rhs = features.T @ target
    return np.linalg.solve(lhs, rhs).astype(np.float32)


def choose_score_threshold(scores: np.ndarray, labels: np.ndarray, steps: int = 401) -> float:
    lower = float(scores.min())
    upper = float(scores.max())
    if abs(upper - lower) < 1e-8:
        return lower
    best_threshold = lower
    best_f1 = -1.0
    best_bal_acc = -1.0
    for threshold in np.linspace(lower, upper, steps):
        predictions = (scores >= threshold).astype(np.float32)
        tp = float(((predictions == 1) & (labels == 1)).sum())
        tn = float(((predictions == 0) & (labels == 0)).sum())
        fp = float(((predictions == 1) & (labels == 0)).sum())
        fn = float(((predictions == 0) & (labels == 1)).sum())
        precision = tp / max(tp + fp, 1.0)
        recall = tp / max(tp + fn, 1.0)
        specificity = tn / max(tn + fp, 1.0)
        f1 = 2.0 * precision * recall / max(precision + recall, 1e-8)
        bal_acc = 0.5 * (recall + specificity)
        if f1 > best_f1 or (abs(f1 - best_f1) < 1e-8 and bal_acc > best_bal_acc):
            best_threshold = float(threshold)
            best_f1 = f1
            best_bal_acc = bal_acc
    return best_threshold


def binary_metrics_from_scores(scores: np.ndarray, labels: np.ndarray, threshold: float) -> tuple[float, float, float, float]:
    predictions = (scores >= threshold).astype(np.float32)
    tp = float(((predictions == 1) & (labels == 1)).sum())
    tn = float(((predictions == 0) & (labels == 0)).sum())
    fp = float(((predictions == 1) & (labels == 0)).sum())
    fn = float(((predictions == 0) & (labels == 1)).sum())
    accuracy = float((predictions == labels).mean())
    precision = tp / max(tp + fp, 1.0)
    recall = tp / max(tp + fn, 1.0)
    specificity = tn / max(tn + fp, 1.0)
    f1 = 2.0 * precision * recall / max(precision + recall, 1e-8)
    bal_acc = 0.5 * (recall + specificity)
    return f1, accuracy, bal_acc, float(predictions.mean())


def spearman_corr(scores: np.ndarray, target: np.ndarray) -> float:
    score_ranks = pd.Series(scores).rank(method="average").to_numpy(dtype=np.float64)
    target_ranks = pd.Series(target).rank(method="average").to_numpy(dtype=np.float64)
    score_std = score_ranks.std()
    target_std = target_ranks.std()
    if score_std < 1e-8 or target_std < 1e-8:
        return 0.0
    corr = np.corrcoef(score_ranks, target_ranks)[0, 1]
    return 0.0 if np.isnan(corr) else float(corr)


def train_ranking_model(frame: pd.DataFrame, model_name: str, extra_features: tuple[str, ...]) -> tuple[RankingResult, dict[str, object]]:
    feature_names = rb.get_feature_names(extra_features)
    splits = rb.split_frame(frame)
    clean_splits, matrices, mean, std, pair_indices = rb.prepare_feature_matrices(splits, feature_names)
    train_x = matrices["train"]
    validation_x = matrices["validation"]
    test_x = matrices["test"]
    train_y_reg = clean_splits["train"][rb.FUTURE_RETURN_COLUMN].to_numpy(dtype=np.float32)
    validation_y_reg = clean_splits["validation"][rb.FUTURE_RETURN_COLUMN].to_numpy(dtype=np.float32)
    test_y_reg = clean_splits["test"][rb.FUTURE_RETURN_COLUMN].to_numpy(dtype=np.float32)
    validation_y_binary = clean_splits["validation"][pr.TARGET_COLUMN].to_numpy(dtype=np.float32)
    test_y_binary = clean_splits["test"][pr.TARGET_COLUMN].to_numpy(dtype=np.float32)

    weights = ridge_fit(train_x, train_y_reg, l2_reg=1e-2)
    validation_scores = validation_x @ weights
    test_scores = test_x @ weights
    threshold = choose_score_threshold(validation_scores, validation_y_binary)
    validation_f1, validation_accuracy, validation_bal_acc, validation_positive_rate = binary_metrics_from_scores(
        validation_scores, validation_y_binary, threshold
    )
    test_f1, test_accuracy, test_bal_acc, test_positive_rate = binary_metrics_from_scores(test_scores, test_y_binary, threshold)

    result = RankingResult(
        model_name=model_name,
        feature_names=feature_names,
        threshold=threshold,
        validation_spearman=spearman_corr(validation_scores, validation_y_reg),
        validation_f1=validation_f1,
        validation_accuracy=validation_accuracy,
        validation_bal_acc=validation_bal_acc,
        validation_positive_rate=validation_positive_rate,
        test_spearman=spearman_corr(test_scores, test_y_reg),
        test_f1=test_f1,
        test_accuracy=test_accuracy,
        test_bal_acc=test_bal_acc,
        test_positive_rate=test_positive_rate,
    )
    artifacts = {
        "feature_names": feature_names,
        "weights": weights,
        "threshold": threshold,
        "train_mean": mean,
        "train_std": std,
        "pair_indices": pair_indices,
        "clean_splits": clean_splits,
        "validation_scores": validation_scores,
        "test_scores": test_scores,
    }
    return result, artifacts


def top_pct_mask(scores: np.ndarray, pct: float) -> tuple[np.ndarray, float]:
    cutoff = float(np.quantile(scores, 1.0 - pct / 100.0))
    return scores >= cutoff, cutoff


def test_rule_rows(model_name: str, artifacts: dict[str, object]) -> list[dict[str, object]]:
    test_frame = artifacts["clean_splits"]["test"]
    scores = np.asarray(artifacts["test_scores"], dtype=np.float64)
    future_returns = test_frame[rb.FUTURE_RETURN_COLUMN].to_numpy(dtype=np.float64)
    rows: list[dict[str, object]] = []
    for pct in (10.0, 15.0, 20.0):
        selected, cutoff = top_pct_mask(scores, pct)
        result = rb.run_non_overlap_backtest(test_frame["date"], future_returns, selected.astype(bool), pr.HORIZON_DAYS, cutoff)
        rows.append(
            {
                "model_name": model_name,
                "rule_name": f"top_{pct:g}pct",
                "threshold_or_cutoff": round_float(cutoff),
                "trade_count": result.selected_count,
                "hit_rate": round_float(result.hit_rate),
                "avg_return": round_float(result.avg_return),
                "max_drawdown_compound": round_float(result.max_drawdown_compound),
            }
        )
    return rows


def score_frame(frame: pd.DataFrame, artifacts: dict[str, object]) -> np.ndarray:
    matrix = frame[artifacts["feature_names"]].to_numpy(dtype=np.float32)
    standardized = (matrix - artifacts["train_mean"]) / artifacts["train_std"]
    standardized = rb.add_interactions(standardized, artifacts["pair_indices"])
    standardized = rb.tr.add_bias(standardized)
    return standardized @ artifacts["weights"]


def ranking_forward_summary(frame: pd.DataFrame, extra_features: tuple[str, ...], pct: float) -> dict[str, object]:
    feature_names = rb.get_feature_names(extra_features)
    trade_returns: list[float] = []
    trade_hits: list[float] = []
    for train_frame, validation_frame, test_frame in rb.walk_forward_splits(frame):
        splits = {"train": train_frame, "validation": validation_frame, "test": test_frame}
        clean_splits, matrices, _mean, _std, _pairs = rb.prepare_feature_matrices(splits, feature_names)
        train_x = matrices["train"]
        test_x = matrices["test"]
        train_y_reg = clean_splits["train"][rb.FUTURE_RETURN_COLUMN].to_numpy(dtype=np.float32)
        weights = ridge_fit(train_x, train_y_reg, l2_reg=1e-2)
        scores = test_x @ weights
        selected, cutoff = top_pct_mask(scores, pct)
        backtest = rb.run_non_overlap_backtest(
            clean_splits["test"]["date"],
            clean_splits["test"][rb.FUTURE_RETURN_COLUMN].to_numpy(dtype=np.float64),
            selected.astype(bool),
            pr.HORIZON_DAYS,
            cutoff,
        )
        if backtest.selected_count:
            idx = 0
            selected_returns = clean_splits["test"][rb.FUTURE_RETURN_COLUMN].to_numpy(dtype=np.float64)
            while idx < len(selected):
                if selected[idx]:
                    trade_returns.append(float(selected_returns[idx]))
                    trade_hits.append(1.0 if selected_returns[idx] > 0 else 0.0)
                    idx += pr.HORIZON_DAYS
                else:
                    idx += 1
    returns = np.asarray(trade_returns, dtype=np.float64)
    return {
        "rule_name": f"top_{pct:g}pct",
        "trade_count": int(len(returns)),
        "hit_rate": round_float(np.mean(trade_hits) if trade_hits else 0.0),
        "avg_return": round_float(returns.mean() if len(returns) else 0.0),
        "max_drawdown_compound": round_float(rb.max_drawdown(np.cumprod(1.0 + returns)) if len(returns) else 0.0),
    }


def decile_rows(model_name: str, artifacts: dict[str, object]) -> list[dict[str, object]]:
    test_frame = artifacts["clean_splits"]["test"].copy()
    test_frame["score"] = np.asarray(artifacts["test_scores"], dtype=np.float64)
    test_frame["score_decile"] = pd.qcut(test_frame["score"].rank(method="first"), 10, labels=False) + 1
    rows: list[dict[str, object]] = []
    for decile in range(1, 11):
        bucket = test_frame.loc[test_frame["score_decile"] == decile]
        rows.append(
            {
                "model_name": model_name,
                "score_decile": decile,
                "sample_count": int(len(bucket)),
                "avg_forward_return": round_float(bucket[rb.FUTURE_RETURN_COLUMN].mean() if len(bucket) else 0.0),
                "hit_rate": round_float((bucket[rb.FUTURE_RETURN_COLUMN] > 0).mean() if len(bucket) else 0.0),
            }
        )
    return rows


def target_summary_rows(frame: pd.DataFrame) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for split_name, split_frame in rb.split_frame(frame).items():
        returns = split_frame[rb.FUTURE_RETURN_COLUMN]
        rows.append(
            {
                "scope": split_name,
                "year": "",
                "rows": int(len(split_frame)),
                "mean_return": round_float(returns.mean()),
                "p10": round_float(returns.quantile(0.10)),
                "p25": round_float(returns.quantile(0.25)),
                "p50": round_float(returns.quantile(0.50)),
                "p75": round_float(returns.quantile(0.75)),
                "p90": round_float(returns.quantile(0.90)),
            }
        )
    yearly = frame.copy()
    yearly["year"] = yearly["date"].dt.year.astype(int)
    for year, bucket in yearly.groupby("year"):
        rows.append(
            {
                "scope": "year",
                "year": int(year),
                "rows": int(len(bucket)),
                "mean_return": round_float(bucket[rb.FUTURE_RETURN_COLUMN].mean()),
                "p10": round_float(bucket[rb.FUTURE_RETURN_COLUMN].quantile(0.10)),
                "p25": round_float(bucket[rb.FUTURE_RETURN_COLUMN].quantile(0.25)),
                "p50": round_float(bucket[rb.FUTURE_RETURN_COLUMN].quantile(0.50)),
                "p75": round_float(bucket[rb.FUTURE_RETURN_COLUMN].quantile(0.75)),
                "p90": round_float(bucket[rb.FUTURE_RETURN_COLUMN].quantile(0.90)),
            }
        )
    return rows


def binary_live_rule_rows(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    live_result, live_artifacts = rb.train_model(frame, "ret_60_plus_sma_gap_60_live", extra_features=("ret_60", "sma_gap_60"))
    candidate_result, candidate_artifacts = rb.train_model(
        frame, "ret_60_plus_sma_gap_60_plus_rolling_vol_60", extra_features=("ret_60", "sma_gap_60", "rolling_vol_60")
    )
    binary_df = pd.DataFrame(
        [
            {
                "model_name": live_result.name,
                "validation_f1": round_float(live_result.validation_f1),
                "validation_accuracy": round_float(live_result.validation_accuracy),
                "validation_bal_acc": round_float(live_result.validation_bal_acc),
                "test_f1": round_float(live_result.test_f1),
                "test_accuracy": round_float(live_result.test_accuracy),
                "test_bal_acc": round_float(live_result.test_bal_acc),
                "headline_score": round_float(
                    rb.compute_headline_score(
                        live_result.validation_f1,
                        live_result.validation_bal_acc,
                        live_result.test_f1,
                        live_result.test_bal_acc,
                    )
                ),
                "promotion_gate": "pass" if rb.passes_promotion_gate(live_result.validation_bal_acc, live_result.test_bal_acc) else "fail",
            },
            {
                "model_name": candidate_result.name,
                "validation_f1": round_float(candidate_result.validation_f1),
                "validation_accuracy": round_float(candidate_result.validation_accuracy),
                "validation_bal_acc": round_float(candidate_result.validation_bal_acc),
                "test_f1": round_float(candidate_result.test_f1),
                "test_accuracy": round_float(candidate_result.test_accuracy),
                "test_bal_acc": round_float(candidate_result.test_bal_acc),
                "headline_score": round_float(
                    rb.compute_headline_score(
                        candidate_result.validation_f1,
                        candidate_result.validation_bal_acc,
                        candidate_result.test_f1,
                        candidate_result.test_bal_acc,
                    )
                ),
                "promotion_gate": "pass" if rb.passes_promotion_gate(candidate_result.validation_bal_acc, candidate_result.test_bal_acc) else "fail",
            },
        ]
    )
    binary_rule_rows = []
    for pct in (10.0, 15.0, 20.0):
        selected, cutoff = top_pct_mask(np.asarray(live_artifacts["test_probabilities"], dtype=np.float64), pct)
        result = rb.run_non_overlap_backtest(
            live_artifacts["clean_splits"]["test"]["date"],
            live_artifacts["clean_splits"]["test"][rb.FUTURE_RETURN_COLUMN].to_numpy(dtype=np.float64),
            selected.astype(bool),
            pr.HORIZON_DAYS,
            cutoff,
        )
        forward = rb.forward_trade_summary(frame, ("ret_60", "sma_gap_60"), f"top_{pct:g}pct")
        binary_rule_rows.append(
            {
                "model_name": "binary_live",
                "rule_name": f"top_{pct:g}pct",
                "test_trade_count": result.selected_count,
                "test_hit_rate": round_float(result.hit_rate),
                "test_avg_return": round_float(result.avg_return),
                "test_max_drawdown_compound": round_float(result.max_drawdown_compound),
                "forward_trade_count": int(forward["trade_count"]),
                "forward_hit_rate": round_float(forward["hit_rate"]),
                "forward_avg_return": round_float(forward["avg_return"]),
            }
        )
    round21_rule_rows = []
    for model_name, artifacts, feature_tuple in (
        ("ret_60_plus_sma_gap_60_live", live_artifacts, ("ret_60", "sma_gap_60")),
        ("ret_60_plus_sma_gap_60_plus_rolling_vol_60", candidate_artifacts, ("ret_60", "sma_gap_60", "rolling_vol_60")),
    ):
        for rule_name in ("top_15pct", "top_17_5pct", "top_20pct"):
            test_probs = np.asarray(artifacts["test_probabilities"], dtype=np.float64)
            pct = float(rule_name[len("top_") : -len("pct")].replace("_", "."))
            selected, cutoff = top_pct_mask(test_probs, pct)
            test_backtest = rb.run_non_overlap_backtest(
                artifacts["clean_splits"]["test"]["date"],
                artifacts["clean_splits"]["test"][rb.FUTURE_RETURN_COLUMN].to_numpy(dtype=np.float64),
                selected.astype(bool),
                pr.HORIZON_DAYS,
                cutoff,
            )
            forward = rb.forward_trade_summary(frame, feature_tuple, rule_name)
            round21_rule_rows.append(
                {
                    "model_name": model_name,
                    "rule_name": rule_name,
                    "test_trade_count": test_backtest.selected_count,
                    "test_hit_rate": round_float(test_backtest.hit_rate),
                    "test_avg_return": round_float(test_backtest.avg_return),
                    "test_max_drawdown_compound": round_float(test_backtest.max_drawdown_compound),
                    "forward_trade_count": int(forward["trade_count"]),
                    "forward_hit_rate": round_float(forward["hit_rate"]),
                    "forward_avg_return": round_float(forward["avg_return"]),
                }
            )
    return binary_df, pd.DataFrame(binary_rule_rows), pd.DataFrame(round21_rule_rows)


def main() -> None:
    os.makedirs(CACHE_DIR, exist_ok=True)
    raw = pr.download_gld_prices()
    frame = rb.build_labeled_frame(raw)

    binary_df, binary_rule_df, round21_rule_df = binary_live_rule_rows(frame)

    ranking_specs = {
        "ranking_baseline": ("ret_60", "sma_gap_60"),
        "ranking_plus_atr_pct_20": ("ret_60", "sma_gap_60", "atr_pct_20"),
        "ranking_plus_rolling_vol_60": ("ret_60", "sma_gap_60", "rolling_vol_60"),
    }
    ranking_rows: list[dict[str, object]] = []
    ranking_artifacts: dict[str, dict[str, object]] = {}
    test_rule_rows_all: list[dict[str, object]] = []
    forward_rule_rows_all: list[dict[str, object]] = []
    decile_rows_all: list[dict[str, object]] = []

    for model_name, features in ranking_specs.items():
        result, artifacts = train_ranking_model(frame, model_name, features)
        ranking_rows.append(
            {
                "model_name": model_name,
                "feature_names": ",".join(result.feature_names),
                "validation_spearman": round_float(result.validation_spearman),
                "validation_f1": round_float(result.validation_f1),
                "validation_accuracy": round_float(result.validation_accuracy),
                "validation_bal_acc": round_float(result.validation_bal_acc),
                "validation_positive_rate": round_float(result.validation_positive_rate),
                "test_spearman": round_float(result.test_spearman),
                "test_f1": round_float(result.test_f1),
                "test_accuracy": round_float(result.test_accuracy),
                "test_bal_acc": round_float(result.test_bal_acc),
                "test_positive_rate": round_float(result.test_positive_rate),
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
        ranking_artifacts[model_name] = artifacts
        test_rule_rows_all.extend(test_rule_rows(model_name, artifacts))
        for pct in (10.0, 15.0, 20.0):
            summary = ranking_forward_summary(frame, features, pct)
            forward_rule_rows_all.append({"model_name": model_name, **summary})
        decile_rows_all.extend(decile_rows(model_name, artifacts))

    distribution_df = pd.DataFrame(target_summary_rows(frame))
    ranking_df = pd.DataFrame(ranking_rows).sort_values(["test_spearman", "headline_score"], ascending=[False, False]).reset_index(drop=True)
    test_rules_df = pd.DataFrame(test_rule_rows_all)
    forward_rules_df = pd.DataFrame(forward_rule_rows_all)
    deciles_df = pd.DataFrame(decile_rows_all)

    distribution_df.to_csv(DISTRIBUTION_OUTPUT_PATH, sep="\t", index=False)
    ranking_df.to_csv(MODEL_OUTPUT_PATH, sep="\t", index=False)
    test_rules_df.to_csv(TEST_RULE_OUTPUT_PATH, sep="\t", index=False)
    forward_rules_df.to_csv(FORWARD_RULE_OUTPUT_PATH, sep="\t", index=False)
    deciles_df.to_csv(DECILE_OUTPUT_PATH, sep="\t", index=False)
    binary_df.to_csv(ROUND21_OUTPUT_PATH, sep="\t", index=False)
    binary_rule_df.to_csv(BINARY_RULE_OUTPUT_PATH, sep="\t", index=False)
    round21_rule_df.to_csv(ROUND21_RULE_OUTPUT_PATH, sep="\t", index=False)

    payload = {
        "target_summary": distribution_df.to_dict(orient="records"),
        "round21_binary_summary": binary_df.to_dict(orient="records"),
        "round21_rule_summary": round21_rule_df.to_dict(orient="records"),
        "ranking_models": ranking_df.to_dict(orient="records"),
        "ranking_test_rules": test_rules_df.to_dict(orient="records"),
        "ranking_forward_rules": forward_rules_df.to_dict(orient="records"),
        "ranking_deciles": deciles_df.to_dict(orient="records"),
        "binary_live_rules": binary_rule_df.to_dict(orient="records"),
    }
    with open(ROUND_OUTPUT_PATH, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)

    print(f"Saved target summary to: {DISTRIBUTION_OUTPUT_PATH}")
    print(f"Saved ranking model summary to: {MODEL_OUTPUT_PATH}")
    print(f"Saved ranking test rules to: {TEST_RULE_OUTPUT_PATH}")
    print(f"Saved ranking forward rules to: {FORWARD_RULE_OUTPUT_PATH}")
    print(f"Saved ranking deciles to: {DECILE_OUTPUT_PATH}")
    print(f"Saved round-21 binary summary to: {ROUND21_OUTPUT_PATH}")
    print(f"Saved binary live rule summary to: {BINARY_RULE_OUTPUT_PATH}")
    print(f"Saved round-21 rule summary to: {ROUND21_RULE_OUTPUT_PATH}")
    print(f"Saved round json to: {ROUND_OUTPUT_PATH}")
    print()
    print("Round 21 binary summary:")
    print(binary_df.to_string(index=False))
    print()
    print("Ranking model summary:")
    print(ranking_df.to_string(index=False))
    print()
    print("Binary live rule summary:")
    print(binary_rule_df.to_string(index=False))
    print()
    print("Round 21 rule summary:")
    print(round21_rule_df.to_string(index=False))


if __name__ == "__main__":
    main()
