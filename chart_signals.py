"""
Render a local HTML chart of recent GLD closes colored by live model signal.
"""

from __future__ import annotations

import json
import os
from html import escape

import numpy as np

import train as tr
from predict_latest import build_feature_names, classify_signal, fit_model, score_latest_row
from prepare import add_price_features, download_gld_prices, load_splits

OUTPUT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".cache", "gld-swing-entry", "signal_chart.html")
DEFAULT_LOOKBACK_DAYS = 5 * 252
SIGNAL_COLORS = {
    "no_entry": "#9ca3af",
    "weak_bullish": "#fde68a",
    "bullish": "#f59e0b",
    "strong_bullish": "#16a34a",
    "very_strong_bullish": "#065f46",
}


def get_env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    return int(value) if value is not None else default


def build_history_probabilities(weights: np.ndarray, splits: dict[str, object], feature_names: list[str]) -> np.ndarray:
    history_probs: list[np.ndarray] = []
    train_frame = splits["train"].frame
    for split_name in ("validation", "test"):
        split_frame = splits[split_name].frame
        matrix, _ = score_latest_row(feature_names, train_frame, split_frame)
        history_probs.append(tr.sigmoid(matrix @ weights))
    return np.concatenate(history_probs)


def build_chart_rows(lookback_days: int) -> tuple[list[dict[str, object]], dict[str, object]]:
    tr.set_seed(tr.get_env_int("AR_SEED", tr.SEED))
    raw_prices = download_gld_prices()
    live_features = add_price_features(raw_prices)
    splits = load_splits()
    feature_names = build_feature_names()
    weights, threshold = fit_model(splits, feature_names)
    history_probabilities = build_history_probabilities(weights, splits, feature_names)

    train_frame = splits["train"].frame
    scored = live_features.dropna(subset=feature_names).copy()
    scored = scored.tail(lookback_days).reset_index(drop=True)

    rows: list[dict[str, object]] = []
    for idx in range(len(scored)):
        row = scored.iloc[[idx]]
        vector, snapshot = score_latest_row(feature_names, train_frame, row)
        probability = float(tr.sigmoid(vector @ weights)[0])
        signal, band_info = classify_signal(probability, float(threshold), history_probabilities)
        rows.append(
            {
                "date": row["date"].iloc[0].strftime("%Y-%m-%d"),
                "close": round(float(row["close"].iloc[0]), 2),
                "signal": signal,
                "probability": round(probability, 4),
                "threshold": round(float(threshold), 4),
                "confidence_gap": band_info["confidence_gap"],
                "ret_60": round(float(snapshot.get("ret_60", 0.0)), 4),
                "drawdown_20": round(float(snapshot.get("drawdown_20", 0.0)), 4),
                "volume_vs_20": round(float(snapshot.get("volume_vs_20", 0.0)), 4),
                "rsi_14": round(float(snapshot.get("rsi_14", 0.0)), 2),
            }
        )

    meta = {
        "threshold": round(float(threshold), 4),
        "latest_date": rows[-1]["date"] if rows else None,
        "lookback_days": lookback_days,
        "signal_colors": SIGNAL_COLORS,
    }
    return rows, meta


def build_html(rows: list[dict[str, object]], meta: dict[str, object]) -> str:
    title = "GLD Live Signal Chart"
    payload = json.dumps({"rows": rows, "meta": meta}, ensure_ascii=False)
    color_legend = "".join(
        f'<span class="legend-item"><span class="swatch" style="background:{escape(color)}"></span>{escape(name)}</span>'
        for name, color in SIGNAL_COLORS.items()
    )
    recent_rows = rows[-5:]
    bullish_like_count = sum(1 for row in recent_rows if row["signal"] in {"bullish", "strong_bullish"})
    recent_cards = "".join(
        f"""
        <div class="recent-card">
          <div class="recent-date">{escape(str(row["date"]))}</div>
          <div class="recent-signal" style="color:{escape(SIGNAL_COLORS.get(str(row["signal"]), '#1f2937'))}">{escape(str(row["signal"]))}</div>
          <div class="recent-metric">p={escape(f'{row["probability"]:.4f}')}</div>
          <div class="recent-metric">gap={escape(f'{row["confidence_gap"]:.4f}')}</div>
          <div class="recent-metric">close={escape(f'{row["close"]:.2f}')}</div>
        </div>
        """
        for row in recent_rows
    )
    return f"""<!doctype html>
<html lang="zh-Hant">
<head>
  <meta charset="utf-8">
  <title>{escape(title)}</title>
  <style>
    :root {{
      --bg: #f6f3ec;
      --ink: #1f2937;
      --muted: #6b7280;
      --grid: #d6d3d1;
      --panel: #fffdf8;
    }}
    body {{
      margin: 0;
      font-family: "Segoe UI", "Noto Sans TC", sans-serif;
      background: linear-gradient(180deg, #f6f3ec 0%, #ebe5da 100%);
      color: var(--ink);
    }}
    .wrap {{
      max-width: 1400px;
      margin: 0 auto;
      padding: 24px;
    }}
    .card {{
      background: var(--panel);
      border: 1px solid #e7e0d4;
      border-radius: 18px;
      box-shadow: 0 18px 60px rgba(31, 41, 55, 0.08);
      padding: 20px 20px 12px;
    }}
    h1 {{
      margin: 0 0 8px;
      font-size: 28px;
    }}
    .sub {{
      color: var(--muted);
      margin-bottom: 14px;
    }}
    .legend {{
      display: flex;
      flex-wrap: wrap;
      gap: 14px;
      margin-bottom: 18px;
      font-size: 14px;
    }}
    .recent-panel {{
      display: grid;
      gap: 12px;
      margin-bottom: 18px;
    }}
    .recent-summary {{
      font-size: 14px;
      color: var(--muted);
    }}
    .recent-grid {{
      display: grid;
      grid-template-columns: repeat(5, minmax(150px, 1fr));
      gap: 10px;
    }}
    .recent-card {{
      background: #faf6ee;
      border: 1px solid #eadfcb;
      border-radius: 12px;
      padding: 10px 12px;
    }}
    .recent-date {{
      font-size: 12px;
      color: var(--muted);
      margin-bottom: 4px;
    }}
    .recent-signal {{
      font-size: 16px;
      font-weight: 700;
      margin-bottom: 6px;
    }}
    .recent-metric {{
      font-size: 12px;
      color: var(--ink);
      line-height: 1.45;
    }}
    .legend-item {{
      display: inline-flex;
      align-items: center;
      gap: 8px;
    }}
    .swatch {{
      width: 14px;
      height: 14px;
      border-radius: 3px;
      display: inline-block;
    }}
    #chart {{
      width: 100%;
      overflow-x: auto;
      border-top: 1px solid #efe8db;
      padding-top: 10px;
    }}
    svg {{
      display: block;
      height: 560px;
    }}
    .axis-label {{
      fill: var(--muted);
      font-size: 11px;
    }}
    .tooltip {{
      position: fixed;
      pointer-events: none;
      background: rgba(17, 24, 39, 0.94);
      color: #fff;
      padding: 10px 12px;
      border-radius: 10px;
      font-size: 12px;
      line-height: 1.45;
      transform: translate(12px, 12px);
      display: none;
      white-space: pre-line;
      box-shadow: 0 10px 30px rgba(0,0,0,0.2);
    }}
  </style>
</head>
<body>
  <div class="wrap">
    <div class="card">
      <h1>{escape(title)}</h1>
      <div class="sub">上方是收盤價直條圖，顏色代表即時 signal。最新資料日: {escape(str(meta["latest_date"]))}，lookback: {escape(str(meta["lookback_days"]))} bars。</div>
      <div class="recent-panel">
        <div class="recent-summary">最近 5 天中，`bullish` 以上共有 <strong>{bullish_like_count}</strong> 天。建議不要只看最後一天，先看這 5 天 signal 是否連續、是否走強。</div>
        <div class="recent-grid">{recent_cards}</div>
      </div>
      <div class="legend">{color_legend}</div>
      <div id="chart"></div>
    </div>
  </div>
  <div id="tooltip" class="tooltip"></div>
  <script>
    const payload = {payload};
    const rows = payload.rows;
    const colors = payload.meta.signal_colors;
    const chart = document.getElementById('chart');
    const tooltip = document.getElementById('tooltip');
    const width = Math.max(2400, rows.length * 12);
    const height = 560;
    const topPad = 24;
    const priceHeight = 410;
    const axisTop = 450;
    const axisHeight = 80;
    const leftPad = 56;
    const rightPad = 24;
    const innerWidth = width - leftPad - rightPad;
    const barWidth = innerWidth / rows.length;
    const closes = rows.map(r => r.close);
    const minClose = Math.min(...closes);
    const maxClose = Math.max(...closes);
    const closeRange = Math.max(maxClose - minClose, 1);

    const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
    svg.setAttribute('viewBox', `0 0 ${{width}} ${{height}}`);
    svg.setAttribute('width', String(width));
    svg.setAttribute('height', String(height));

    const bg = document.createElementNS('http://www.w3.org/2000/svg', 'rect');
    bg.setAttribute('x', '0');
    bg.setAttribute('y', '0');
    bg.setAttribute('width', String(width));
    bg.setAttribute('height', String(height));
    bg.setAttribute('fill', '#fffdf8');
    svg.appendChild(bg);

    for (let i = 0; i < 5; i += 1) {{
      const y = topPad + (priceHeight / 4) * i;
      const line = document.createElementNS('http://www.w3.org/2000/svg', 'line');
      line.setAttribute('x1', String(leftPad));
      line.setAttribute('x2', String(width - rightPad));
      line.setAttribute('y1', String(y));
      line.setAttribute('y2', String(y));
      line.setAttribute('stroke', '#d6d3d1');
      line.setAttribute('stroke-width', '1');
      line.setAttribute('stroke-dasharray', '3 5');
      svg.appendChild(line);

      const price = maxClose - (closeRange / 4) * i;
      const label = document.createElementNS('http://www.w3.org/2000/svg', 'text');
      label.setAttribute('x', '8');
      label.setAttribute('y', String(y + 4));
      label.setAttribute('class', 'axis-label');
      label.textContent = price.toFixed(0);
      svg.appendChild(label);
    }}

    rows.forEach((row, index) => {{
      const x = leftPad + index * barWidth;
      const normalized = (row.close - minClose) / closeRange;
      const barHeight = Math.max(2, normalized * (priceHeight - 8));
      const y = topPad + priceHeight - barHeight;

      const rect = document.createElementNS('http://www.w3.org/2000/svg', 'rect');
      rect.setAttribute('x', String(x));
      rect.setAttribute('y', String(y));
      rect.setAttribute('width', String(Math.max(1, barWidth - 1)));
      rect.setAttribute('height', String(barHeight));
      rect.setAttribute('fill', colors[row.signal] || '#9ca3af');
      rect.setAttribute('rx', '1.5');
      rect.addEventListener('mousemove', (event) => {{
        tooltip.style.display = 'block';
        tooltip.style.left = `${{event.clientX}}px`;
        tooltip.style.top = `${{event.clientY}}px`;
        tooltip.textContent =
          `${{row.date}}\\nclose=${{row.close}}\\nsignal=${{row.signal}}\\np=${{row.probability}}\\ngap=${{row.confidence_gap}}\\nret_60=${{row.ret_60}}\\ndrawdown_20=${{row.drawdown_20}}\\nrsi_14=${{row.rsi_14}}`;
      }});
      rect.addEventListener('mouseleave', () => {{
        tooltip.style.display = 'none';
      }});
      svg.appendChild(rect);
    }});

    const axisLine = document.createElementNS('http://www.w3.org/2000/svg', 'line');
    axisLine.setAttribute('x1', String(leftPad));
    axisLine.setAttribute('x2', String(width - rightPad));
    axisLine.setAttribute('y1', String(axisTop));
    axisLine.setAttribute('y2', String(axisTop));
    axisLine.setAttribute('stroke', '#6b7280');
    axisLine.setAttribute('stroke-width', '1');
    svg.appendChild(axisLine);

    const tickEvery = Math.max(1, Math.floor(rows.length / 20));
    rows.forEach((row, index) => {{
      if (index % tickEvery !== 0 && index !== rows.length - 1) return;
      const x = leftPad + index * barWidth + barWidth / 2;
      const tick = document.createElementNS('http://www.w3.org/2000/svg', 'line');
      tick.setAttribute('x1', String(x));
      tick.setAttribute('x2', String(x));
      tick.setAttribute('y1', String(axisTop));
      tick.setAttribute('y2', String(axisTop + 8));
      tick.setAttribute('stroke', '#6b7280');
      svg.appendChild(tick);

      const label = document.createElementNS('http://www.w3.org/2000/svg', 'text');
      label.setAttribute('x', String(x));
      label.setAttribute('y', String(axisTop + 24));
      label.setAttribute('text-anchor', 'middle');
      label.setAttribute('class', 'axis-label');
      label.textContent = row.date;
      svg.appendChild(label);
    }});

    chart.appendChild(svg);
  </script>
</body>
</html>
"""


def main() -> None:
    lookback_days = get_env_int("AR_CHART_LOOKBACK_DAYS", DEFAULT_LOOKBACK_DAYS)
    rows, meta = build_chart_rows(lookback_days)
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        f.write(build_html(rows, meta))
    print(f"Saved chart to: {OUTPUT_PATH}")
    print(f"Bars rendered: {len(rows)}")
    print(f"Latest date: {meta['latest_date']}")


if __name__ == "__main__":
    main()
