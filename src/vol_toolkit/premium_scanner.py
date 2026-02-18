"""Volatility risk premium scanner for identifying selling opportunities."""

from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import yfinance as yf

from vol_toolkit.iv_tracker import _hv_series, _try_get_atm_iv


def scan_vol_premium(tickers: list[str], lookback: int = 252) -> pd.DataFrame:
    """Scan tickers for vol risk premium (IV - RV spread).

    For each ticker computes:
    - Current IV (from options chain, or HV percentile proxy)
    - 20-day realized vol
    - Premium = IV - RV (positive means selling edge)
    - Premium percentile over the lookback period

    Args:
        tickers: List of ticker symbols to scan.
        lookback: Number of trading days for percentile calculation.

    Returns:
        DataFrame sorted by premium descending (best selling opportunities first).
        Columns: ticker, current_iv, realized_vol_20d, premium, premium_pct,
                 iv_percentile, signal.
    """
    end = datetime.now()
    start = end - timedelta(days=int(lookback * 1.6))

    rows = []
    for ticker in tickers:
        try:
            row = _scan_single(ticker, start, end, lookback)
            if row is not None:
                rows.append(row)
        except Exception:
            continue

    if not rows:
        return pd.DataFrame(
            columns=[
                "ticker",
                "current_iv",
                "realized_vol_20d",
                "premium",
                "premium_pct",
                "iv_percentile",
                "signal",
            ]
        )

    df = pd.DataFrame(rows)
    df = df.sort_values("premium", ascending=False).reset_index(drop=True)
    return df


def _scan_single(
    ticker: str, start: datetime, end: datetime, lookback: int
) -> dict | None:
    """Scan a single ticker for vol premium data."""
    data = yf.download(
        ticker,
        start=start.strftime("%Y-%m-%d"),
        end=end.strftime("%Y-%m-%d"),
        progress=False,
    )
    if data.empty or len(data) < 30:
        return None

    close = data["Close"].squeeze().dropna()

    # Current IV: try options chain, fall back to HV
    current_iv = _try_get_atm_iv(ticker)

    hv = _hv_series(close, window=20).dropna()
    if len(hv) < 20:
        return None

    if current_iv is None:
        current_iv = float(hv.iloc[-1])

    rv_20d = float(hv.iloc[-1])
    premium = current_iv - rv_20d

    # Premium percentile: how does current premium compare historically?
    # Use trailing HV at different points as proxy for historical IV
    hv_fast = _hv_series(close, window=10).dropna()
    hv_slow = _hv_series(close, window=30).dropna()
    min_len = min(len(hv_fast), len(hv_slow))
    if min_len < 20:
        premium_pct = 50.0
    else:
        hist_premium = hv_fast.iloc[-min_len:].values - hv_slow.iloc[-min_len:].values
        premium_pct = float(np.sum(hist_premium <= premium) / len(hist_premium) * 100)

    # IV percentile
    iv_percentile = float(np.sum(hv.values <= current_iv) / len(hv) * 100)

    if iv_percentile >= 75 and premium > 0:
        signal = "SELL"
    elif iv_percentile <= 25:
        signal = "WAIT"
    else:
        signal = "NEUTRAL"

    return {
        "ticker": ticker,
        "current_iv": round(current_iv, 4),
        "realized_vol_20d": round(rv_20d, 4),
        "premium": round(premium, 4),
        "premium_pct": round(premium_pct, 1),
        "iv_percentile": round(iv_percentile, 1),
        "signal": signal,
    }
