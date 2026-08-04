"""
main.py

Run the backtest against a real CSV of 5-min OHLCV candles
(produced by fetch_kite_data.py, or any CSV with columns:
date, open, high, low, close, volume).

Usage:
    python main.py --csv data/NIFTY_5minute_2024-01-01_2024-12-31.csv
    python main.py --csv data/BANKNIFTY_5minute_2020-01-01_2024-12-31.csv --capital 500000

Optional: tweak strategy parameters via CLI flags (see --help), or edit
StrategyParams defaults directly in strategy.py for a permanent change.
"""

import argparse

import pandas as pd

from strategy import StrategyParams, compute_indicators, compute_entry_signals
from backtest_engine import run_backtest, CostModel
from metrics import compute_metrics, print_report


def load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"date", "open", "high", "low", "close", "volume"}
    missing = required - set(df.columns.str.lower())
    if missing:
        raise ValueError(f"CSV is missing required columns: {missing}")
    df.columns = [c.lower() for c in df.columns]
    df["date"] = pd.to_datetime(df["date"])
    return df


def main():
    parser = argparse.ArgumentParser(description="Backtest the ORB+VWAP+Supertrend strategy")
    parser.add_argument("--csv", required=True, help="Path to OHLCV CSV file")
    parser.add_argument("--capital", type=float, default=1_000_000.0)
    parser.add_argument("--lot-size", type=int, default=1, help="e.g. 25 for Nifty futures lot, 15 for BankNifty (check current NSE lot size)")
    parser.add_argument("--slippage-points", type=float, default=1.0)
    parser.add_argument("--cost-pct", type=float, default=0.03, help="One-way cost as %% of trade value")
    parser.add_argument("--adx-threshold", type=float, default=20.0)
    parser.add_argument("--or-minutes", type=int, default=30)
    parser.add_argument("--risk-pct", type=float, default=0.75, help="%% of capital risked per trade")
    args = parser.parse_args()

    print(f"Loading {args.csv} ...")
    df = load_csv(args.csv)
    print(f"Loaded {len(df)} candles, {df['date'].dt.date.nunique()} sessions, "
          f"from {df['date'].min()} to {df['date'].max()}\n")

    params = StrategyParams(
        or_minutes=args.or_minutes,
        adx_threshold=args.adx_threshold,
        risk_per_trade_pct=args.risk_pct,
    )

    print("Computing indicators and signals...")
    df = compute_indicators(df, params)
    df = compute_entry_signals(df, params)
    n_signals = (df["entry_signal"] != 0).sum()
    print(f"Entry signals found: {n_signals}\n")

    print("Running backtest...")
    result = run_backtest(
        df,
        params,
        capital=args.capital,
        cost_model=CostModel(slippage_points=args.slippage_points, cost_pct_per_trade=args.cost_pct),
        lot_size=args.lot_size,
    )

    metrics = compute_metrics(result, capital=args.capital)
    print_report(metrics, result["trades"])

    # Save trade log and equity curve for further analysis / plotting
    if result["trades"]:
        trades_df = pd.DataFrame(result["trades"])
        trades_df.to_csv("trade_log.csv", index=False)
        print("\nTrade log saved to trade_log.csv")

    result["equity_curve"].to_csv("equity_curve.csv", header=["equity"])
    print("Equity curve saved to equity_curve.csv")


if __name__ == "__main__":
    main()
