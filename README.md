# GLD Swing Entry

This repo predicts medium-term GLD entry quality instead of short-term direction.

## How To Work In This Repo

This repo should be operated primarily by following [program.md](/C:/Users/Jay/OneDrive/文件/codex/gld-swing-entry/program.md).

In practice:

- read `program.md` first
- execute the current round of tasks from `task.md`
- record all formal experiment results in `results.tsv`
- use `ideas.md` to break down the next round of tasks

## Goal

The default target is:

- `1`: within the next `60` trading days, GLD hits `+8%` before `-4%`
- `0`: within the next `60` trading days, GLD hits `-4%` before `+8%`
- `neutral`: neither barrier is hit first, or both are hit on the same day and ordering is ambiguous

Neutral samples are dropped from training by default.

## Why this target

This framing is closer to a real swing-trade entry decision:

- it encodes upside and downside together
- it is longer-horizon than 3-day direction
- it can be extended later into simple backtests and ranking models

## Files

- [prepare.py](/C:/Users/Jay/OneDrive/文件/codex/gld-swing-entry/prepare.py): downloads GLD daily data, builds features, and labels barrier outcomes
- [train.py](/C:/Users/Jay/OneDrive/文件/codex/gld-swing-entry/train.py): trains a NumPy logistic baseline on the processed dataset
- [predict_latest.py](/C:/Users/Jay/OneDrive/文件/codex/gld-swing-entry/predict_latest.py): scores the latest raw GLD bar without waiting for future labels
- [chart_signals.py](/C:/Users/Jay/OneDrive/文件/codex/gld-swing-entry/chart_signals.py): exports an HTML chart of recent closes colored by live signal
- [research_batch.py](/C:/Users/Jay/OneDrive/文件/codex/gld-swing-entry/research_batch.py): runs the current formal research batch and writes compact TSV/JSON summaries
- [score_results.py](/C:/Users/Jay/OneDrive/文件/codex/gld-swing-entry/score_results.py): refreshes `headline_score` and `promotion_gate` in `results.tsv`
- [results.tsv](/C:/Users/Jay/OneDrive/文件/codex/gld-swing-entry/results.tsv): experiment log
- [task.md](/C:/Users/Jay/OneDrive/文件/codex/gld-swing-entry/task.md): next research tasks
- [backtest_comparison.tsv](/C:/Users/Jay/OneDrive/文件/codex/gld-swing-entry/backtest_comparison.tsv): latest non-overlap backtest comparison table
- [regime_summary.tsv](/C:/Users/Jay/OneDrive/文件/codex/gld-swing-entry/regime_summary.tsv): latest yearly barrier and regime-stage summaries

## Default target settings

- horizon: `60` trading days
- upper barrier: `+8%`
- lower barrier: `-4%`
- labeling style: `binary drop-neutral`

## Usage

```powershell
$env:PYTHONPATH='C:\Users\Jay\OneDrive\文件\codex\gld-swing-entry\.packages'
python prepare.py
python train.py
python predict_latest.py
python chart_signals.py
python research_batch.py
```

## Notes

- Barrier ordering uses daily `high` and `low`.
- If both barriers are touched on the same day, the sample is dropped as ambiguous.
- `headline_score = 0.2*validation_f1 + 0.1*validation_bal_acc + 0.4*test_f1 + 0.3*test_bal_acc`.
- `promotion_gate` requires `validation_bal_acc >= 0.52` and `test_bal_acc >= 0.54`.
- This is a baseline research repo, not a production trading system.
