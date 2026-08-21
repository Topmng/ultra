"""Alias for binance_fetch.py — 48h+ of BTCUSDT payload streams into database/.

Resumes after each CSV's last timestamp. Writes candles, trades, book ticker,
depth snapshots, depth updates, and futures bookDepth.
"""

from binance_fetch import main

if __name__ == "__main__":
    raise SystemExit(main())
