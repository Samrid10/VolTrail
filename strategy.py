"""
strategy.py

Implements the specific rules-based algorithm discussed:

Regime filter:
    - ADX(14) on the aggregated data > adx_threshold  (skip session if not trending)
    - (VIX filter is optional/left as a hook — plug in India VIX series if you have it)

Entry (once per session, first valid breakout only):
    - Opening range = high/low of first `or_minutes` of the session
    - Long: 5-min candle closes above OR high, close > VWAP, volume > vol_mult * rolling_avg_volume
    - Short: mirror conditions below OR low and below VWAP

Stop-loss:
    - max(OR low for longs / OR high for shorts, entry - atr_mult * ATR)  [tighter of the two]

Take-profit / trailing:
    - Partial exit at partial_r_multiple (default 1.5R)
    - Remainder trails on Supertrend; exits when Supertrend flips against the position
    - Hard time-stop: force flat at `session_end_time`
"""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from indicators import atr, supertrend, vwap_session, adx, rolling_volume_avg


@dataclass
class StrategyParams:
    or_minutes: int = 30            # opening range window in minutes
    candle_minutes: int = 5         # candle size (must match data)
    vol_mult: float = 1.5           # breakout volume vs rolling average
    vol_avg_period: int = 20
    adx_threshold: float = 20.0
    adx_period: int = 14
    atr_period: int = 14
    atr_stop_mult: float = 1.2
    supertrend_period: int = 10
    supertrend_mult: float = 2.0
    partial_r_multiple: float = 1.5  # take partial profit at this multiple of initial risk
    partial_fraction: float = 0.5    # fraction of position closed at partial target
    session_end_time: str = "15:20"  # force-flat time (IST, "HH:MM")
    risk_per_trade_pct: float = 0.75  # % of capital risked per trade
    one_trade_per_session: bool = True


def _assign_session_id(df: pd.DataFrame) -> pd.Series:
    return df["date"].dt.date.astype(str)


def compute_indicators(df: pd.DataFrame, p: StrategyParams) -> pd.DataFrame:
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    df["session_id"] = _assign_session_id(df)
    df["time"] = df["date"].dt.strftime("%H:%M")

    df["vwap"] = vwap_session(df, "session_id")
    df["atr"] = atr(df, p.atr_period)
    st_line, st_dir = supertrend(df, p.supertrend_period, p.supertrend_mult)
    df["supertrend"] = st_line
    df["supertrend_dir"] = st_dir
    df["adx"] = adx(df, p.adx_period)
    df["vol_avg"] = rolling_volume_avg(df, p.vol_avg_period)

    # Opening range per session: high/low of the first `or_minutes` of candles
    candles_in_or = max(1, p.or_minutes // p.candle_minutes)

    def _or_high_low(group):
        or_slice = group.iloc[:candles_in_or]
        group["or_high"] = or_slice["high"].max()
        group["or_low"] = or_slice["low"].min()
        return group

    df = df.groupby("session_id", group_keys=False)[df.columns.tolist()].apply(_or_high_low)
    # mark bars that are still inside the OR-forming window (not tradeable yet)
    df["bar_index_in_session"] = df.groupby("session_id").cumcount()
    df["or_forming"] = df["bar_index_in_session"] < candles_in_or

    return df


def compute_entry_signals(df: pd.DataFrame, p: StrategyParams) -> pd.DataFrame:
    """
    Adds an `entry_signal` column: 1 for long entry, -1 for short entry, 0 none.
    Only the FIRST valid signal per session is kept if one_trade_per_session=True.
    """
    df = df.copy()

    long_cond = (
        (~df["or_forming"])
        & (df["close"] > df["or_high"])
        & (df["close"] > df["vwap"])
        & (df["volume"] > p.vol_mult * df["vol_avg"])
        & (df["adx"] > p.adx_threshold)
    )
    short_cond = (
        (~df["or_forming"])
        & (df["close"] < df["or_low"])
        & (df["close"] < df["vwap"])
        & (df["volume"] > p.vol_mult * df["vol_avg"])
        & (df["adx"] > p.adx_threshold)
    )

    df["entry_signal"] = 0
    df.loc[long_cond, "entry_signal"] = 1
    df.loc[short_cond, "entry_signal"] = -1

    if p.one_trade_per_session:
        # keep only the first non-zero signal per session
        def _first_only(group):
            nz = group[group["entry_signal"] != 0]
            if nz.empty:
                return group
            first_idx = nz.index[0]
            mask = group.index != first_idx
            group.loc[mask, "entry_signal"] = 0
            return group

        df = df.groupby("session_id", group_keys=False)[df.columns.tolist()].apply(_first_only)

    return df
