"""
Run the binary rule-promotion round: threshold vs top-percentile rule comparison.
"""

from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd

import prepare as pr
import research_batch as rb

REPO_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(REPO_DIR, ".cache", "gld-swing-entry")
ROUND_OUTPUT_PATH = os.path.join(CACHE_DIR, "binary_round3.json")
SUMMARY_OUTPUT_PATH = os.path.join(REPO_DIR, "binary_round3_rule_decision.tsv")


def round_float(value: float) -> float:
    return round(float(value), 4)


def recent_five_year_frame(frame: pd.DataFrame) -> pd.DataFrame:
    cutoff = frame["date"].max() - pd.DateOffset(years=5)
    return frame.loc[frame["date"] >= cutoff].copy().reset_index(drop=True)


def non_overlap_returns(values: np.ndarray, selected: np.ndarray, horizon: int) -> np.ndarray:
    chosen: list[float] = []
    idx = 0
    while idx < len(selected):
        if selected[idx]:
            chosen.append(float(values[idx]))
            idx += horizon
        else:
            idx += 1
    return np.asarray(chosen, dtype=np.float64)


def summarize_returns(returns: np.ndarray) -> tuple[int, float, float, float]:
    if len(returns) == 0:
        return 0, 0.0, 0.0, 0.0
    return (
        int(len(returns)),
        float((returns > 0).mean()),
        float(returns.mean()),
        float(rb.max_drawdown(np.cumprod(1.0 + returns))),
    )


def main() -> None:
    os.makedirs(CACHE_DIR, exist_ok=True)
    raw = pr.download_gld_prices()
    frame = rb.build_labeled_frame(raw)

    model_specs = {
        "live_ret_60_plus_sma_gap_60": ("ret_60", "sma_gap_60"),
        "candidate_ret_60_plus_sma_gap_60_plus_rolling_vol_60": ("ret_60", "sma_gap_60", "rolling_vol_60"),
    }
    rules = ("threshold", "top_20pct")

    recent = recent_five_year_frame(frame)
    rows: list[dict[str, object]] = []
    payload: dict[str, object] = {"models": {}}

    for model_name, features in model_specs.items():
        result, artifacts = rb.train_model(frame, model_name, extra_features=features)
        headline_score = rb.compute_headline_score(
            result.validation_f1, result.validation_bal_acc, result.test_f1, result.test_bal_acc
        )
        feature_names = artifacts["feature_names"]
        test_frame = artifacts["clean_splits"]["test"]
        test_probs = np.asarray(artifacts["test_probabilities"], dtype=np.float64)
        test_returns = test_frame[rb.FUTURE_RETURN_COLUMN].to_numpy(dtype=np.float64)
        threshold = float(artifacts["threshold"])

        recent_probs = rb.score_frame(
            recent,
            feature_names,
            artifacts["train_mean"],
            artifacts["train_std"],
            artifacts["pair_indices"],
            artifacts["weights"],
        )
        recent_returns = recent[rb.FUTURE_RETURN_COLUMN].to_numpy(dtype=np.float64)

        model_payload = {
            "validation_f1": result.validation_f1,
            "validation_bal_acc": result.validation_bal_acc,
            "test_f1": result.test_f1,
            "test_bal_acc": result.test_bal_acc,
            "headline_score": headline_score,
            "threshold": threshold,
            "rules": {},
        }

        for rule_name in rules:
            selected_test, cutoff = rb.classify_probs_by_rule(test_probs, threshold, rule_name)
            test_backtest = rb.run_non_overlap_backtest(
                test_frame["date"],
                test_returns,
                selected_test.astype(bool),
                pr.HORIZON_DAYS,
                float(cutoff),
            )
            forward = rb.forward_trade_summary(frame, features, rule_name)

            selected_recent, recent_cutoff = rb.classify_probs_by_rule(recent_probs, threshold, rule_name)
            recent_trade_returns = non_overlap_returns(recent_returns, selected_recent.astype(bool), pr.HORIZON_DAYS)
            recent_trade_count, recent_hit_rate, recent_avg_return, recent_mdd = summarize_returns(recent_trade_returns)

            row = {
                "model_name": model_name,
                "rule_name": rule_name,
                "validation_f1": round_float(result.validation_f1),
                "validation_bal_acc": round_float(result.validation_bal_acc),
                "test_f1": round_float(result.test_f1),
                "test_bal_acc": round_float(result.test_bal_acc),
                "headline_score": round_float(headline_score),
                "rule_cutoff_test": round_float(cutoff),
                "test_trade_count": int(test_backtest.selected_count),
                "test_hit_rate": round_float(test_backtest.hit_rate),
                "test_avg_return": round_float(test_backtest.avg_return),
                "test_max_drawdown_compound": round_float(test_backtest.max_drawdown_compound),
                "forward_trade_count": int(forward["trade_count"]),
                "forward_hit_rate": round_float(forward["hit_rate"]),
                "forward_avg_return": round_float(forward["avg_return"]),
                "recent_rule_cutoff": round_float(recent_cutoff),
                "recent_trade_count": recent_trade_count,
                "recent_hit_rate": round_float(recent_hit_rate),
                "recent_avg_return": round_float(recent_avg_return),
                "recent_max_drawdown_compound": round_float(recent_mdd),
                "date_start": recent["date"].iloc[0].strftime("%Y-%m-%d"),
                "date_end": recent["date"].iloc[-1].strftime("%Y-%m-%d"),
            }
            rows.append(row)
            model_payload["rules"][rule_name] = row

        payload["models"][model_name] = model_payload

    summary_df = pd.DataFrame(rows)
    summary_df.to_csv(SUMMARY_OUTPUT_PATH, sep="\t", index=False)

    with open(ROUND_OUTPUT_PATH, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)

    print(f"Saved rule decision summary to: {SUMMARY_OUTPUT_PATH}")
    print(f"Saved round json to: {ROUND_OUTPUT_PATH}")
    print()
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
