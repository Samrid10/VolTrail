"""
backtest_engine.py

Bar-by-bar (5-min candle) trade simulator. Consumes the DataFrame produced
by strategy.compute_indicators + compute_entry_signals and:

    - Enters at the NEXT bar's open after a signal (no lookahead)
    - Sizes position by fixed % risk of capital
    - Applies stop-loss, partial take-profit, Supertrend trailing exit,
      and a hard time-stop
    - Deducts slippage (in price points) and per-trade brokerage/cost (in %)
    - Produces a trade log and an equity curve for metrics.py to analyze

This is intentionally a single-position-at-a-time engine (one instrument,
no pyramiding) to match the strategy as specified. Extend `Position` /
`run_backtest` if you want to test portfolio-level variants later.
"""

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from strategy import StrategyParams


@dataclass
class CostModel:
    slippage_points: float = 1.0     # points lost per entry/exit (round-trip = 2x this)
    cost_pct_per_trade: float = 0.03  # brokerage + STT + exchange + GST, as % of trade value, one-way


@dataclass
class Position:
    direction: int          # 1 long, -1 short
    entry_time: pd.Timestamp
    entry_price: float
    stop_price: float
    initial_risk_per_unit: float
    qty: int
    partial_taken: bool = False
    remaining_qty: int = 0

    def __post_init__(self):
        self.remaining_qty = self.qty


def _apply_slippage(price: float, direction: int, is_entry: bool, cost: CostModel) -> float:
    """
    Slippage always works against you: entries fill worse, exits fill worse.
    Long entry: pay slightly more. Long exit: receive slightly less.
    Short entry: receive slightly less (short at a lower fill). Short exit: pay slightly more.
    """
    sign = 1 if (direction == 1) == is_entry else -1
    # For a LONG: entry is_entry=True -> sign=+1 (pay more). exit is_entry=False -> sign=-1 (receive less)... 
    # simpler: just always move price against the trader by slippage_points
    if (direction == 1 and is_entry) or (direction == -1 and not is_entry):
        return price + cost.slippage_points
    else:
        return price - cost.slippage_points


def run_backtest(
    df: pd.DataFrame,
    params: StrategyParams,
    capital: float = 1_000_000.0,
    cost_model: Optional[CostModel] = None,
    lot_size: int = 1,
) -> dict:
    """
    Returns a dict with:
        trades: list of trade dicts
        equity_curve: pd.Series indexed by timestamp
    """
    if cost_model is None:
        cost_model = CostModel()

    df = df.reset_index(drop=True)
    trades = []
    equity = capital
    equity_curve = []
    position: Optional[Position] = None

    session_end_str = params.session_end_time

    for i in range(len(df) - 1):
        row = df.iloc[i]
        next_row = df.iloc[i + 1]
        equity_curve.append((row["date"], equity))

        # ---- Manage open position first ----
        if position is not None:
            still_same_session = next_row["session_id"] == row["session_id"]
            hit_time_stop = row["time"] >= session_end_str

            exit_price = None
            exit_reason = None

            # Stop-loss check (intrabar using high/low of current bar)
            if position.direction == 1 and row["low"] <= position.stop_price:
                exit_price = position.stop_price
                exit_reason = "stop_loss"
            elif position.direction == -1 and row["high"] >= position.stop_price:
                exit_price = position.stop_price
                exit_reason = "stop_loss"

            # Partial take-profit
            partial_target = (
                position.entry_price
                + position.direction * params.partial_r_multiple * position.initial_risk_per_unit
            )
            if exit_price is None and not position.partial_taken:
                hit_partial = (
                    (position.direction == 1 and row["high"] >= partial_target)
                    or (position.direction == -1 and row["low"] <= partial_target)
                )
                if hit_partial:
                    partial_qty = int(position.qty * params.partial_fraction)
                    if partial_qty > 0:
                        fill = _apply_slippage(partial_target, position.direction, False, cost_model)
                        pnl = position.direction * (fill - position.entry_price) * partial_qty
                        pnl -= fill * partial_qty * (cost_model.cost_pct_per_trade / 100)
                        equity += pnl
                        position.remaining_qty -= partial_qty
                        position.partial_taken = True
                        trades.append({
                            "entry_time": position.entry_time,
                            "exit_time": row["date"],
                            "direction": position.direction,
                            "qty": partial_qty,
                            "entry_price": position.entry_price,
                            "exit_price": fill,
                            "pnl": pnl,
                            "reason": "partial_tp",
                        })
                    # move stop to breakeven on the remainder
                    position.stop_price = position.entry_price

            # Supertrend-flip exit for the remainder (only after partial taken, or as full trail)
            if exit_price is None and position.remaining_qty > 0:
                st_dir = row["supertrend_dir"]
                if position.direction == 1 and st_dir == -1:
                    exit_price = row["close"]
                    exit_reason = "supertrend_flip"
                elif position.direction == -1 and st_dir == 1:
                    exit_price = row["close"]
                    exit_reason = "supertrend_flip"

            # Hard time-stop / session end -> force flat
            if exit_price is None and (hit_time_stop or not still_same_session):
                exit_price = row["close"]
                exit_reason = "time_stop"

            if exit_price is not None and position.remaining_qty > 0:
                fill = _apply_slippage(exit_price, position.direction, False, cost_model)
                pnl = position.direction * (fill - position.entry_price) * position.remaining_qty
                pnl -= fill * position.remaining_qty * (cost_model.cost_pct_per_trade / 100)
                equity += pnl
                trades.append({
                    "entry_time": position.entry_time,
                    "exit_time": row["date"],
                    "direction": position.direction,
                    "qty": position.remaining_qty,
                    "entry_price": position.entry_price,
                    "exit_price": fill,
                    "pnl": pnl,
                    "reason": exit_reason,
                })
                position = None

        # ---- Check for new entry (only if flat) ----
        if position is None and row["entry_signal"] != 0 and row["time"] < session_end_str:
            direction = int(row["entry_signal"])
            raw_entry_price = next_row["open"]  # enter at NEXT bar's open, no lookahead
            entry_price = _apply_slippage(raw_entry_price, direction, True, cost_model)

            # Stop = tighter of OR-based stop and ATR-based stop
            if direction == 1:
                or_stop = row["or_low"]
                atr_stop = entry_price - params.atr_stop_mult * row["atr"]
                stop_price = max(or_stop, atr_stop)
            else:
                or_stop = row["or_high"]
                atr_stop = entry_price + params.atr_stop_mult * row["atr"]
                stop_price = min(or_stop, atr_stop)

            risk_per_unit = abs(entry_price - stop_price)
            if risk_per_unit <= 0 or np.isnan(risk_per_unit):
                continue  # can't size a trade with zero/invalid risk

            risk_capital = equity * (params.risk_per_trade_pct / 100)
            qty = int(risk_capital / risk_per_unit)
            qty = (qty // lot_size) * lot_size  # round down to lot size
            if qty <= 0:
                continue

            position = Position(
                direction=direction,
                entry_time=next_row["date"],
                entry_price=entry_price,
                stop_price=stop_price,
                initial_risk_per_unit=risk_per_unit,
                qty=qty,
            )

    equity_curve.append((df.iloc[-1]["date"], equity))
    equity_series = pd.Series(
        [e for _, e in equity_curve], index=[t for t, _ in equity_curve]
    )
    equity_series = equity_series[~equity_series.index.duplicated(keep="last")]

    return {"trades": trades, "equity_curve": equity_series, "final_equity": equity}
