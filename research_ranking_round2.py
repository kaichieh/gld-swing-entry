"""
Run the second ranking round: denser top-percentile scan and recent-period comparisons.
"""

from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd

import prepare as pr
import research_batch as rb
import research_ranking as rr

REPO_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(REPO_DIR, ".cache", "gld-swing-entry")
ROUND_OUTPUT_PATH = os.path.join(CACHE_DIR, "ranking_round2.json")
DENSITY_OUTPUT_PATH = os.path.join(REPO_DIR, "ranking_round2_density.tsv")
RECENT_OUTPUT_PATH = os.path.join(REPO_DIR, "ranking_vs_binary_recent.tsv")


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


def recent_stats(returns: np.ndarray) -> tuple[float, float, float]:
    if len(returns) == 0:
        return 0.0, 0.0, 0.0
    return (
        float((returns > 0).mean()),
        float(returns.mean()),
        float(rb.max_drawdown(np.cumprod(1.0 + returns))),
    )


def main() -> None:
    os.makedirs(CACHE_DIR, exist_ok=True)
    raw = pr.download_gld_prices()
    frame = rb.build_labeled_frame(raw)

    density_rows: list[dict[str, object]] = []
    recent_rows: list[dict[str, object]] = []

    ranking_specs = {
        "ranking_plus_rolling_vol_60": ("ret_60", "sma_gap_60", "rolling_vol_60"),
        "ranking_plus_atr_pct_20": ("ret_60", "sma_gap_60", "atr_pct_20"),
    }
    density_grid = (12.5, 15.0, 17.5, 20.0)

    for model_name, features in ranking_specs.items():
        for pct in density_grid:
            summary = rr.ranking_forward_summary(frame, features, pct)
            density_rows.append({"model_name": model_name, **summary})

    binary_recent_result, binary_recent_artifacts = rb.train_model(
        frame, "ret_60_plus_sma_gap_60_plus_rolling_vol_60", extra_features=("ret_60", "sma_gap_60", "rolling_vol_60")
    )
    ranking_recent_result, ranking_recent_artifacts = rr.train_ranking_model(
        frame, "ranking_plus_rolling_vol_60", ("ret_60", "sma_gap_60", "rolling_vol_60")
    )

    recent = recent_five_year_frame(frame)
    recent_returns = recent[rb.FUTURE_RETURN_COLUMN].to_numpy(dtype=np.float64)

    binary_scores = rb.score_frame(
        recent,
        binary_recent_artifacts["feature_names"],
        binary_recent_artifacts["train_mean"],
        binary_recent_artifacts["train_std"],
        binary_recent_artifacts["pair_indices"],
        binary_recent_artifacts["weights"],
    )
    ranking_scores = rr.score_frame(recent, ranking_recent_artifacts)

    for pct in density_grid:
        binary_mask, _binary_cutoff = rr.top_pct_mask(binary_scores, pct)
        ranking_mask, _ranking_cutoff = rr.top_pct_mask(ranking_scores, pct)
        binary_trade_returns = non_overlap_returns(recent_returns, binary_mask.astype(bool), pr.HORIZON_DAYS)
        ranking_trade_returns = non_overlap_returns(recent_returns, ranking_mask.astype(bool), pr.HORIZON_DAYS)
        binary_hit_rate, binary_avg_return, binary_mdd = recent_stats(binary_trade_returns)
        ranking_hit_rate, ranking_avg_return, ranking_mdd = recent_stats(ranking_trade_returns)
        recent_rows.extend(
            [
                {
                    "model_name": "binary_plus_rolling_vol_60",
                    "rule_name": f"top_{pct:g}pct",
                    "sample_count": int(binary_mask.sum()),
                    "sample_ratio": round_float(float(binary_mask.mean())),
                    "trade_count": int(len(binary_trade_returns)),
                    "hit_rate": round_float(binary_hit_rate),
                    "avg_return": round_float(binary_avg_return),
                    "max_drawdown_compound": round_float(binary_mdd),
                    "reference_test_f1": round_float(binary_recent_result.test_f1),
                    "reference_test_bal_acc": round_float(binary_recent_result.test_bal_acc),
                    "date_start": recent["date"].iloc[0].strftime("%Y-%m-%d"),
                    "date_end": recent["date"].iloc[-1].strftime("%Y-%m-%d"),
                },
                {
                    "model_name": "ranking_plus_rolling_vol_60",
                    "rule_name": f"top_{pct:g}pct",
                    "sample_count": int(ranking_mask.sum()),
                    "sample_ratio": round_float(float(ranking_mask.mean())),
                    "trade_count": int(len(ranking_trade_returns)),
                    "hit_rate": round_float(ranking_hit_rate),
                    "avg_return": round_float(ranking_avg_return),
                    "max_drawdown_compound": round_float(ranking_mdd),
                    "reference_test_f1": round_float(ranking_recent_result.test_f1),
                    "reference_test_bal_acc": round_float(ranking_recent_result.test_bal_acc),
                    "date_start": recent["date"].iloc[0].strftime("%Y-%m-%d"),
                    "date_end": recent["date"].iloc[-1].strftime("%Y-%m-%d"),
                },
            ]
        )

    density_df = pd.DataFrame(density_rows)
    recent_df = pd.DataFrame(recent_rows)
    density_df.to_csv(DENSITY_OUTPUT_PATH, sep="\t", index=False)
    recent_df.to_csv(RECENT_OUTPUT_PATH, sep="\t", index=False)

    payload = {
        "density_scan": density_df.to_dict(orient="records"),
        "recent_comparison": recent_df.to_dict(orient="records"),
    }
    with open(ROUND_OUTPUT_PATH, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)

    print(f"Saved density scan to: {DENSITY_OUTPUT_PATH}")
    print(f"Saved recent comparison to: {RECENT_OUTPUT_PATH}")
    print(f"Saved round json to: {ROUND_OUTPUT_PATH}")
    print()
    print(density_df.to_string(index=False))
    print()
    print(recent_df.to_string(index=False))


if __name__ == "__main__":
    main()
