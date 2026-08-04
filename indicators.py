"""
indicators.py

Pure pandas/numpy implementations of the indicators used by the strategy.
No TA-Lib dependency (avoids a painful C-extension install).

Expects a DataFrame with columns: date, open, high, low, close, volume
Assumes data is already sorted ascending by date and is a SINGLE trading
session's worth of intraday candles when computing VWAP/opening range
(session-based reset is handled in strategy.py, not here).
"""

import numpy as np
import pandas as pd


def true_range(df: pd.DataFrame) -> pd.Series:
    prev_close = df["close"].shift(1)
    tr1 = df["high"] - df["low"]
    tr2 = (df["high"] - prev_close).abs()
    tr3 = (df["low"] - prev_close).abs()
    return pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    tr = true_range(df)
    return tr.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()


def supertrend(df: pd.DataFrame, period: int = 10, multiplier: float = 2.0):
    """
    Returns (supertrend_line, direction) where direction is +1 (uptrend,
    price above the line) or -1 (downtrend, price below the line).
    """
    hl2 = (df["high"] + df["low"]) / 2
    atr_val = atr(df, period)

    upper_band = hl2 + multiplier * atr_val
    lower_band = hl2 - multiplier * atr_val

    final_upper = upper_band.copy()
    final_lower = lower_band.copy()
    supertrend_line = pd.Series(index=df.index, dtype=float)
    direction = pd.Series(index=df.index, dtype=int)

    for i in range(len(df)):
        if i == 0:
            supertrend_line.iloc[i] = upper_band.iloc[i]
            direction.iloc[i] = 1
            continue

        # Band "sticking" logic
        if df["close"].iloc[i - 1] > final_upper.iloc[i - 1]:
            final_upper.iloc[i] = upper_band.iloc[i]
        else:
            final_upper.iloc[i] = min(upper_band.iloc[i], final_upper.iloc[i - 1])

        if df["close"].iloc[i - 1] < final_lower.iloc[i - 1]:
            final_lower.iloc[i] = lower_band.iloc[i]
        else:
            final_lower.iloc[i] = max(lower_band.iloc[i], final_lower.iloc[i - 1])

        prev_dir = direction.iloc[i - 1]
        close = df["close"].iloc[i]

        if prev_dir == 1:
            if close < final_lower.iloc[i]:
                direction.iloc[i] = -1
            else:
                direction.iloc[i] = 1
        else:
            if close > final_upper.iloc[i]:
                direction.iloc[i] = 1
            else:
                direction.iloc[i] = -1

        supertrend_line.iloc[i] = final_lower.iloc[i] if direction.iloc[i] == 1 else final_upper.iloc[i]

    return supertrend_line, direction


def vwap_session(df: pd.DataFrame, session_col: str = "session_id") -> pd.Series:
    """
    Volume-weighted average price, reset at the start of each session
    (session_id groups candles belonging to the same trading day).
    """
    typical_price = (df["high"] + df["low"] + df["close"]) / 3
    pv = typical_price * df["volume"]

    grouped_pv_cumsum = pv.groupby(df[session_col]).cumsum()
    grouped_vol_cumsum = df["volume"].groupby(df[session_col]).cumsum()

    return grouped_pv_cumsum / grouped_vol_cumsum.replace(0, np.nan)


def adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    up_move = df["high"].diff()
    down_move = -df["low"].diff()

    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    tr = true_range(df)
    atr_smooth = tr.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()

    plus_di = 100 * pd.Series(plus_dm, index=df.index).ewm(
        alpha=1 / period, adjust=False, min_periods=period
    ).mean() / atr_smooth
    minus_di = 100 * pd.Series(minus_dm, index=df.index).ewm(
        alpha=1 / period, adjust=False, min_periods=period
    ).mean() / atr_smooth

    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    adx_val = dx.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    return adx_val


def rolling_volume_avg(df: pd.DataFrame, period: int = 20) -> pd.Series:
    return df["volume"].rolling(window=period, min_periods=1).mean()
