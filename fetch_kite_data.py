"""
fetch_kite_data.py

Pulls historical 5-minute OHLCV candles from Zerodha Kite Connect and
caches them to CSV for backtesting.

NOTE: This script must be run on YOUR machine (or any environment with
network access to api.kite.trade). It will not run inside this sandbox.

Setup:
    pip install kiteconnect

Auth flow (Kite Connect requires a fresh access_token daily):
    1. Set KITE_API_KEY and KITE_API_SECRET as environment variables.
    2. Run this script. It will print a login URL if no valid token is cached.
    3. Log in, get the `request_token` from the redirect URL, paste it when prompted.
    4. access_token is cached in kite_session.json for the rest of the day.

Usage:
    python fetch_kite_data.py --symbol NIFTY --exchange NSE --from 2024-01-01 --to 2024-12-31
    python fetch_kite_data.py --symbol BANKNIFTY --exchange NSE --from 2024-01-01 --to 2024-12-31

Historical data API constraints (as of Kite Connect v3):
    - 5-minute candles: max ~100 days per single request (script chunks automatically)
    - Historical data is rate-limited; script sleeps between chunks
    - Kite's historical API needs an `instrument_token`, not just a symbol.
      This script resolves the token from the instruments dump automatically.
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime, timedelta

import pandas as pd

SESSION_FILE = "kite_session.json"
DATA_DIR = "data"
CHUNK_DAYS = 90  # safe chunk size for 5-minute candles


def get_kite_client():
    from kiteconnect import KiteConnect

    api_key = os.environ.get("KITE_API_KEY")
    api_secret = os.environ.get("KITE_API_SECRET")
    if not api_key or not api_secret:
        sys.exit(
            "ERROR: Set KITE_API_KEY and KITE_API_SECRET environment variables first.\n"
            "  export KITE_API_KEY=your_key\n"
            "  export KITE_API_SECRET=your_secret"
        )

    kite = KiteConnect(api_key=api_key)

    # Reuse cached access_token if present and from today
    if os.path.exists(SESSION_FILE):
        with open(SESSION_FILE) as f:
            session = json.load(f)
        if session.get("date") == datetime.now().strftime("%Y-%m-%d"):
            kite.set_access_token(session["access_token"])
            return kite

    # Otherwise do the login flow
    print("No valid cached session found. Login required.")
    print(f"Login URL: {kite.login_url()}")
    request_token = input(
        "Paste the `request_token` param from the redirect URL after login: "
    ).strip()

    data = kite.generate_session(request_token, api_secret=api_secret)
    kite.set_access_token(data["access_token"])

    with open(SESSION_FILE, "w") as f:
        json.dump(
            {"access_token": data["access_token"], "date": datetime.now().strftime("%Y-%m-%d")},
            f,
        )
    return kite


def resolve_instrument_token(kite, symbol: str, exchange: str) -> int:
    """
    Resolves the instrument_token for an index/underlying from Kite's
    instrument dump. For NIFTY/BANKNIFTY spot indices, exchange is typically
    'NSE' and the tradingsymbol is 'NIFTY 50' / 'NIFTY BANK'.
    For futures, you'd want the specific contract's tradingsymbol instead.
    """
    symbol_map = {
        "NIFTY": "NIFTY 50",
        "BANKNIFTY": "NIFTY BANK",
    }
    lookup_symbol = symbol_map.get(symbol.upper(), symbol.upper())

    print(f"Fetching instrument list for {exchange}... (this can take a few seconds)")
    instruments = kite.instruments(exchange)
    df = pd.DataFrame(instruments)

    match = df[df["tradingsymbol"] == lookup_symbol]
    if match.empty:
        # fallback: partial match, useful for futures contracts like NIFTY24DECFUT
        match = df[df["tradingsymbol"].str.contains(symbol.upper(), na=False)]

    if match.empty:
        sys.exit(f"Could not resolve instrument_token for symbol '{symbol}' on {exchange}.")

    if len(match) > 1:
        print("Multiple matches found, using the first. Full list:")
        print(match[["tradingsymbol", "instrument_token", "expiry"]].to_string(index=False))

    token = int(match.iloc[0]["instrument_token"])
    print(f"Resolved '{symbol}' -> instrument_token={token} ({match.iloc[0]['tradingsymbol']})")
    return token


def fetch_historical(kite, instrument_token: int, from_date: str, to_date: str, interval: str):
    """
    Fetches historical candles in CHUNK_DAYS windows (Kite rate-limits and
    caps the range per request for intraday intervals).
    """
    start = datetime.strptime(from_date, "%Y-%m-%d")
    end = datetime.strptime(to_date, "%Y-%m-%d")

    all_rows = []
    cursor = start
    while cursor < end:
        chunk_end = min(cursor + timedelta(days=CHUNK_DAYS), end)
        print(f"  Fetching {cursor.date()} -> {chunk_end.date()} ...")
        try:
            candles = kite.historical_data(
                instrument_token,
                cursor,
                chunk_end,
                interval,
                continuous=False,
                oi=False,
            )
            all_rows.extend(candles)
        except Exception as e:
            print(f"  WARNING: chunk failed ({e}), retrying once after 5s...")
            time.sleep(5)
            try:
                candles = kite.historical_data(
                    instrument_token, cursor, chunk_end, interval, continuous=False, oi=False
                )
                all_rows.extend(candles)
            except Exception as e2:
                print(f"  ERROR: chunk permanently failed, skipping: {e2}")

        cursor = chunk_end + timedelta(days=1)
        time.sleep(0.5)  # be polite to the rate limiter

    df = pd.DataFrame(all_rows)
    if df.empty:
        sys.exit("No data returned. Check date range, instrument_token, and market hours.")

    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    return df[["date", "open", "high", "low", "close", "volume"]]


def main():
    parser = argparse.ArgumentParser(description="Fetch historical OHLCV from Kite Connect")
    parser.add_argument("--symbol", required=True, help="e.g. NIFTY, BANKNIFTY, or a futures tradingsymbol")
    parser.add_argument("--exchange", default="NSE", help="NSE for index/equity, NFO for futures/options")
    parser.add_argument("--from", dest="from_date", required=True, help="YYYY-MM-DD")
    parser.add_argument("--to", dest="to_date", required=True, help="YYYY-MM-DD")
    parser.add_argument("--interval", default="5minute", help="minute, 3minute, 5minute, 15minute, day, etc.")
    args = parser.parse_args()

    os.makedirs(DATA_DIR, exist_ok=True)

    kite = get_kite_client()
    token = resolve_instrument_token(kite, args.symbol, args.exchange)
    df = fetch_historical(kite, token, args.from_date, args.to_date, args.interval)

    out_path = os.path.join(
        DATA_DIR, f"{args.symbol}_{args.interval}_{args.from_date}_{args.to_date}.csv"
    )
    df.to_csv(out_path, index=False)
    print(f"\nSaved {len(df)} candles to {out_path}")
    print(df.head())
    print("...")
    print(df.tail())


if __name__ == "__main__":
    main()
