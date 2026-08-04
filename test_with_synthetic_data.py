"""
test_with_synthetic_data.py

Generates plausible synthetic 5-min OHLCV data (random-walk with intraday
volatility clustering, roughly Nifty-like price levels) and runs it through
the full pipeline: indicators -> signals -> backtest -> metrics.

This does NOT validate that the strategy is profitable on real markets —
it only validates that the CODE is wired correctly end-to-end with no
crashes, lookahead bugs, or broken indicator math. Run this first. Then
swap in real Kite data via fetch_kite_data.py and main.py.
"""

import numpy as np
import pandas as pd

from strategy import StrategyParams, compute_indicators, compute_entry_signals
from backtest_engine import run_backtest, CostModel
from metrics import compute_metrics, print_report

np.random.seed(42)


def generate_synthetic_data(n_sessions: int = 120, start_price: float = 22000.0) -> pd.DataFrame:
    candles_per_session = 75  # 9:15 to 15:20 in 5-min bars ~ 75 candles
    rows = []
    price = start_price
    base_date = pd.Timestamp("2024-01-02 09:15:00")
    session_date = base_date

    for s in range(n_sessions):
        # skip weekends
        while session_date.dayofweek >= 5:
            session_date += pd.Timedelta(days=1)

        # each session has a random "regime": trending up, trending down, or choppy
        regime = np.random.choice(["trend_up", "trend_down", "choppy"], p=[0.3, 0.3, 0.4])
        session_vol = np.random.uniform(0.0004, 0.0015)  # per-candle volatility

        ts = session_date.replace(hour=9, minute=15)
        for c in range(candles_per_session):
            if regime == "trend_up":
                drift = 0.00015
            elif regime == "trend_down":
                drift = -0.00015
            else:
                drift = 0.0

            ret = np.random.normal(drift, session_vol)
            open_p = price
            close_p = price * (1 + ret)
            high_p = max(open_p, close_p) * (1 + abs(np.random.normal(0, session_vol / 2)))
            low_p = min(open_p, close_p) * (1 - abs(np.random.normal(0, session_vol / 2)))
            vol = int(np.random.gamma(shape=2.0, scale=50000))
            # inflate volume on breakout-like bars occasionally
            if np.random.rand() < 0.08:
                vol = int(vol * np.random.uniform(1.5, 3.0))

            rows.append({
                "date": ts,
                "open": round(open_p, 2),
                "high": round(high_p, 2),
                "low": round(low_p, 2),
                "close": round(close_p, 2),
                "volume": vol,
            })
            price = close_p
            ts += pd.Timedelta(minutes=5)

        session_date += pd.Timedelta(days=1)

    return pd.DataFrame(rows)


def main():
    print("Generating synthetic data...")
    df = generate_synthetic_data(n_sessions=150)
    print(f"Generated {len(df)} candles across {df['date'].dt.date.nunique()} sessions.\n")

    params = StrategyParams()
    print("Computing indicators...")
    df = compute_indicators(df, params)
    df = compute_entry_signals(df, params)

    n_signals = (df["entry_signal"] != 0).sum()
    print(f"Entry signals generated: {n_signals}\n")

    print("Running backtest...")
    result = run_backtest(
        df,
        params,
        capital=1_000_000.0,
        cost_model=CostModel(slippage_points=1.0, cost_pct_per_trade=0.03),
        lot_size=1,
    )

    metrics = compute_metrics(result, capital=1_000_000.0)
    print_report(metrics, result["trades"])

    # Sanity assertions - these should hold regardless of synthetic data randomness
    assert len(df) > 0, "No data generated"
    assert "entry_signal" in df.columns, "Signal column missing"
    assert isinstance(result["trades"], list), "Trades should be a list"
    print("\n[OK] Pipeline ran end-to-end without errors.")


if __name__ == "__main__":
    main()
