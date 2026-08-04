# Nifty/Bank Nifty Intraday Backtester (ORB + VWAP + Supertrend)

Implements and backtests the specific rules-based algorithm discussed:

- **Regime filter:** ADX(14) > threshold (default 20) — skips choppy days
- **Entry:** Opening Range Breakout (first 30 min) + VWAP confirmation + volume filter, once per session
- **Stop-loss:** tighter of OR-opposite-edge or ATR(14) × 1.2
- **Exit:** partial profit at 1.5R, remainder trails on Supertrend(10, 2), hard time-stop at 15:20 IST
- **Sizing:** fixed % risk of capital per trade

## Files

| File | Purpose |
|---|---|
| `fetch_kite_data.py` | Pulls 5-min historical candles from Kite Connect (run on your machine, not in this sandbox) |
| `indicators.py` | VWAP, ATR, Supertrend, ADX — pure pandas, no TA-Lib needed |
| `strategy.py` | Signal generation: opening range, entry conditions, regime filter |
| `backtest_engine.py` | Bar-by-bar trade simulator: sizing, stops, partial TP, trailing, costs |
| `metrics.py` | Sharpe, Sortino, max drawdown, win rate, profit factor, CAGR |
| `main.py` | CLI entry point: load CSV → run backtest → print report → save trade log |
| `test_with_synthetic_data.py` | Sanity-checks the whole pipeline on fake data — **already run and passing** |

## Setup (run this on your own machine — needs internet access to Kite's API)

```bash
pip install -r requirements.txt

export KITE_API_KEY=your_key
export KITE_API_SECRET=your_secret
```

## Step 1 — Fetch real data

```bash
python fetch_kite_data.py --symbol NIFTY --exchange NSE --from 2019-01-01 --to 2024-12-31
python fetch_kite_data.py --symbol BANKNIFTY --exchange NSE --from 2019-01-01 --to 2024-12-31
```

First run opens a login URL — log in, copy the `request_token` from the redirect URL, paste it in.
The access token is cached for the day in `kite_session.json`.

Notes:
- For the **spot index** (NIFTY 50 / NIFTY BANK), you get continuous history but can't literally trade the index — you'd trade futures/options against these signals. For a closer-to-tradable backtest, fetch the **futures contract** instead by passing its exact `tradingsymbol` (e.g. `NIFTY24DECFUT`) and `--exchange NFO`, and stitch contracts together yourself for continuous data (`fetch_kite_data.py` resolves by trading symbol, so you'll run it once per expiry and concatenate CSVs).
- Kite's historical data for indices typically goes back to 2015-2016 in practice for 5-min bars — earlier than that, gaps are common. Don't expect a clean 30-year 5-min dataset; that granularity of history doesn't really exist in accessible form for Indian markets.

## Step 2 — Run the backtest

```bash
python main.py --csv data/NIFTY_5minute_2019-01-01_2024-12-31.csv --capital 1000000 --lot-size 25
```

Check current NSE lot sizes before setting `--lot-size` — they change periodically (e.g. Nifty lot size has been revised more than once).

Key flags:
```
--adx-threshold 20      # raise to trade only stronger trends, lower to trade more often
--or-minutes 30         # opening range window
--risk-pct 0.75         # % of capital risked per trade
--slippage-points 1.0   # realistic for Nifty futures; use higher for Bank Nifty options
--cost-pct 0.03         # one-way cost as % of trade value (brokerage+STT+exchange+GST combined estimate — verify against your actual Zerodha cost sheet)
```

Outputs:
- Console report: total trades, win rate, profit factor, CAGR, Sharpe, Sortino, max drawdown, exit-reason breakdown
- `trade_log.csv` — every trade with entry/exit time, price, P&L, exit reason
- `equity_curve.csv` — daily equity series for plotting

## Important: before you trust any result

1. **Walk-forward it.** Don't optimize `adx_threshold`, `atr_stop_mult`, etc. on the full dataset and call the result a "backtest" — split into build (e.g. 2019–2022) and verify (2023–2024) periods, tune only on the build period.
2. **Check regime breakdown.** Slice `trade_log.csv` by year/month and compare performance in trending vs. choppy stretches (2020 vs. 2018 vs. 2022) — a strategy that only works in one regime isn't done yet.
3. **Cost sensitivity.** Re-run with `--cost-pct` and `--slippage-points` doubled. If the edge disappears, it wasn't a real edge — it was a backtest artifact.
4. **Sample size.** Fewer than ~100 trades makes Sharpe/win-rate estimates unreliable — extend the date range if needed.
