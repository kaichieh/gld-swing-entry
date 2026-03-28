"""
Run the second pure-GLD binary candidate validation round.
"""

from __future__ import annotations

import json
import os

import pandas as pd

import prepare as pr
import research_batch as rb

REPO_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(REPO_DIR, ".cache", "gld-swing-entry")
ROUND_OUTPUT_PATH = os.path.join(CACHE_DIR, "binary_round2.json")
RULE_OUTPUT_PATH = os.path.join(REPO_DIR, "binary_round2_rules.tsv")
SEED_OUTPUT_PATH = os.path.join(REPO_DIR, "binary_round2_seeds.tsv")
WALK_OUTPUT_PATH = os.path.join(REPO_DIR, "binary_round2_walkforward.tsv")


def round_float(value: float) -> float:
    return round(float(value), 4)


def main() -> None:
    os.makedirs(CACHE_DIR, exist_ok=True)
    raw = pr.download_gld_prices()
    frame = rb.build_labeled_frame(raw)

    candidates = {
        "ret_60_plus_sma_gap_60_live": ("ret_60", "sma_gap_60"),
        "ret_60_plus_sma_gap_60_plus_rolling_vol_60": ("ret_60", "sma_gap_60", "rolling_vol_60"),
    }

    rule_rows: list[dict[str, object]] = []
    seed_rows: list[dict[str, object]] = []
    walk_rows: list[dict[str, object]] = []

    for model_name, features in candidates.items():
        result, artifacts = rb.train_model(frame, model_name, extra_features=features)
        probs = artifacts["test_probabilities"]
        future_returns = artifacts["clean_splits"]["test"][rb.FUTURE_RETURN_COLUMN].to_numpy(dtype=float)
        for pct in (12.5, 15.0, 17.5, 20.0):
            rule_name = f"top_{str(pct).replace('.', '_')}pct"
            selected, cutoff = rb.classify_probs_by_rule(probs, float(artifacts["threshold"]), rule_name)
            test_backtest = rb.run_non_overlap_backtest(
                artifacts["clean_splits"]["test"]["date"],
                future_returns,
                selected.astype(bool),
                pr.HORIZON_DAYS,
                float(cutoff),
            )
            forward = rb.forward_trade_summary(frame, features, rule_name)
            rule_rows.append(
                {
                    "model_name": model_name,
                    "rule_name": rule_name,
                    "threshold_or_cutoff": round_float(cutoff),
                    "test_trade_count": test_backtest.selected_count,
                    "test_hit_rate": round_float(test_backtest.hit_rate),
                    "test_avg_return": round_float(test_backtest.avg_return),
                    "test_max_drawdown_compound": round_float(test_backtest.max_drawdown_compound),
                    "forward_trade_count": int(forward["trade_count"]),
                    "forward_hit_rate": round_float(forward["hit_rate"]),
                    "forward_avg_return": round_float(forward["avg_return"]),
                }
            )

        for seed_result in rb.evaluate_seeds(frame, features):
            seed_rows.append(
                {
                    "model_name": model_name,
                    "seed_name": seed_result.name,
                    "validation_f1": round_float(seed_result.validation_f1),
                    "validation_bal_acc": round_float(seed_result.validation_bal_acc),
                    "test_f1": round_float(seed_result.test_f1),
                    "test_bal_acc": round_float(seed_result.test_bal_acc),
                    "test_positive_rate": round_float(seed_result.test_positive_rate),
                }
            )

        for fold in rb.evaluate_walk_forward(frame, features):
            walk_rows.append(
                {
                    "model_name": model_name,
                    "fold_name": fold.fold_name,
                    "validation_f1": round_float(fold.validation_f1),
                    "validation_bal_acc": round_float(fold.validation_bal_acc),
                    "test_f1": round_float(fold.test_f1),
                    "test_bal_acc": round_float(fold.test_bal_acc),
                    "test_positive_rate": round_float(fold.test_positive_rate),
                }
            )

    rules_df = pd.DataFrame(rule_rows)
    seeds_df = pd.DataFrame(seed_rows)
    walk_df = pd.DataFrame(walk_rows)
    rules_df.to_csv(RULE_OUTPUT_PATH, sep="\t", index=False)
    seeds_df.to_csv(SEED_OUTPUT_PATH, sep="\t", index=False)
    walk_df.to_csv(WALK_OUTPUT_PATH, sep="\t", index=False)

    payload = {
        "rules": rules_df.to_dict(orient="records"),
        "seeds": seeds_df.to_dict(orient="records"),
        "walkforward": walk_df.to_dict(orient="records"),
    }
    with open(ROUND_OUTPUT_PATH, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)

    print(f"Saved rule summary to: {RULE_OUTPUT_PATH}")
    print(f"Saved seed summary to: {SEED_OUTPUT_PATH}")
    print(f"Saved walk-forward summary to: {WALK_OUTPUT_PATH}")
    print(f"Saved round json to: {ROUND_OUTPUT_PATH}")
    print()
    print(rules_df.to_string(index=False))
    print()
    print(seeds_df.to_string(index=False))
    print()
    print(walk_df.to_string(index=False))


if __name__ == "__main__":
    main()
