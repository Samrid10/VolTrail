"""
metrics.py

Computes standard performance metrics from a backtest result
(trades list + equity curve) produced by backtest_engine.run_backtest.
"""

import numpy as np
import pandas as pd


def compute_metrics(result: dict, capital: float, trading_days_per_year: int = 252) -> dict:
    trades = result["trades"]
    equity_curve = result["equity_curve"]

    if len(trades) == 0:
        return {"error": "No trades were generated. Check strategy parameters / data range."}

    trades_df = pd.DataFrame(trades)

    # ---- Trade-level stats ----
    wins = trades_df[trades_df["pnl"] > 0]
    losses = trades_df[trades_df["pnl"] <= 0]
    win_rate = len(wins) / len(trades_df) * 100

    gross_profit = wins["pnl"].sum()
    gross_loss = abs(losses["pnl"].sum())
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else np.nan

    avg_win = wins["pnl"].mean() if len(wins) else 0
    avg_loss = losses["pnl"].mean() if len(losses) else 0

    # ---- Equity curve / return-based stats ----
    daily_equity = equity_curve.resample("D").last().dropna()
    daily_returns = daily_equity.pct_change().dropna()

    total_return_pct = (result["final_equity"] / capital - 1) * 100

    n_days = (daily_equity.index[-1] - daily_equity.index[0]).days
    years = max(n_days / 365.25, 1 / 365.25)
    cagr = ((result["final_equity"] / capital) ** (1 / years) - 1) * 100 if years > 0 else np.nan

    if daily_returns.std() > 0:
        sharpe = (daily_returns.mean() / daily_returns.std()) * np.sqrt(trading_days_per_year)
    else:
        sharpe = np.nan

    downside_returns = daily_returns[daily_returns < 0]
    if downside_returns.std() > 0:
        sortino = (daily_returns.mean() / downside_returns.std()) * np.sqrt(trading_days_per_year)
    else:
        sortino = np.nan

    running_max = daily_equity.cummax()
    drawdown = (daily_equity - running_max) / running_max
    max_drawdown_pct = drawdown.min() * 100

    return {
        "total_trades": len(trades_df),
        "win_rate_pct": round(win_rate, 2),
        "profit_factor": round(profit_factor, 3) if not np.isnan(profit_factor) else None,
        "avg_win": round(avg_win, 2),
        "avg_loss": round(avg_loss, 2),
        "total_return_pct": round(total_return_pct, 2),
        "cagr_pct": round(cagr, 2),
        "sharpe_ratio": round(sharpe, 3) if not np.isnan(sharpe) else None,
        "sortino_ratio": round(sortino, 3) if not np.isnan(sortino) else None,
        "max_drawdown_pct": round(max_drawdown_pct, 2),
        "final_equity": round(result["final_equity"], 2),
    }


def print_report(metrics: dict, trades: list):
    print("=" * 50)
    print("BACKTEST PERFORMANCE REPORT")
    print("=" * 50)
    if "error" in metrics:
        print(metrics["error"])
        return
    for k, v in metrics.items():
        print(f"{k:>20}: {v}")
    print("=" * 50)

    if trades:
        reasons = pd.DataFrame(trades)["reason"].value_counts()
        print("\nExit reason breakdown:")
        print(reasons.to_string())
