"""Alias for binance_fetch.py — last 24h of BTCUSDT candles and trades.

Writes:
  btc_spot_candles.csv
  btc_futures_candles.csv
  btc_spot_trades.csv
  btc_futures_trades.csv
"""

from binance_fetch import main

if __name__ == "__main__":
    raise SystemExit(main())
