"""Implied volatility percentile tracking."""

from __future__ import annotations

import math
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import yfinance as yf


def _hv_series(prices: pd.Series, window: int = 20) -> pd.Series:
    """Compute rolling historical volatility series."""
    log_ret = np.log(prices / prices.shift(1))
    return log_ret.rolling(window).std() * math.sqrt(252)


def get_iv_percentile(ticker: str, lookback_days: int = 252) -> dict:
    """Calculate current IV percentile rank over lookback period.

    Uses yfinance options data when available. Falls back to historical
    volatility as a proxy when options data is unavailable.

    Args:
        ticker: Stock/ETF ticker symbol.
        lookback_days: Number of trading days for percentile calculation.

    Returns:
        Dictionary with ticker, current_iv, iv_percentile, iv_rank,
        high_iv, low_iv, and signal.
    """
    end = datetime.now()
    start = end - timedelta(days=int(lookback_days * 1.6))  # calendar days buffer

    data = yf.download(ticker, start=start.strftime("%Y-%m-%d"), progress=False)
    if data.empty:
        raise ValueError(f"No price data available for {ticker}")

    close = data["Close"].squeeze().dropna()

    # Try to get ATM IV from nearest expiry options chain
    current_iv = _try_get_atm_iv(ticker)

    # Build HV series as proxy for historical IV distribution
    hv = _hv_series(close, window=20).dropna()
    if len(hv) < 20:
        raise ValueError(f"Insufficient data for {ticker}")

    if current_iv is None:
        # Fall back to current HV as proxy
        current_iv = float(hv.iloc[-1])

    hv_values = hv.values
    iv_percentile = float(np.sum(hv_values <= current_iv) / len(hv_values) * 100)
    high_iv = float(np.max(hv_values))
    low_iv = float(np.min(hv_values))

    iv_range = high_iv - low_iv
    iv_rank = float((current_iv - low_iv) / iv_range * 100) if iv_range > 0 else 50.0

    if iv_percentile >= 75:
        signal = "SELL"
    elif iv_percentile <= 25:
        signal = "WAIT"
    else:
        signal = "NEUTRAL"

    return {
        "ticker": ticker,
        "current_iv": round(current_iv, 4),
        "iv_percentile": round(iv_percentile, 1),
        "iv_rank": round(iv_rank, 1),
        "high_iv": round(high_iv, 4),
        "low_iv": round(low_iv, 4),
        "signal": signal,
    }


def _try_get_atm_iv(ticker: str) -> float | None:
    """Try to extract ATM implied volatility from yfinance options chain."""
    try:
        tk = yf.Ticker(ticker)
        expirations = tk.options
        if not expirations:
            return None

        # Use nearest expiration
        chain = tk.option_chain(expirations[0])
        calls = chain.calls

        if calls.empty or "impliedVolatility" not in calls.columns:
            return None

        # Find ATM: closest strike to current price
        current_price = tk.fast_info.get("lastPrice") or tk.fast_info.get("previousClose")
        if current_price is None:
            return None

        calls = calls.copy()
        calls["distance"] = abs(calls["strike"] - current_price)
        atm_row = calls.loc[calls["distance"].idxmin()]
        iv = float(atm_row["impliedVolatility"])

        # Sanity check: IV should be between 0 and 10 (1000%)
        if 0 < iv < 10:
            return iv
        return None
    except Exception:
        return None


def get_iv_term_structure(ticker: str) -> list[dict]:
    """Get IV across different expirations for term structure analysis.

    Args:
        ticker: Stock/ETF ticker symbol.

    Returns:
        List of dicts with expiration, days_to_expiry, atm_iv for each expiry.
    """
    tk = yf.Ticker(ticker)
    expirations = tk.options
    if not expirations:
        return []

    current_price = tk.fast_info.get("lastPrice") or tk.fast_info.get("previousClose")
    if current_price is None:
        return []

    today = datetime.now().date()
    results = []

    for exp_str in expirations[:8]:  # Limit to first 8 expirations
        try:
            chain = tk.option_chain(exp_str)
            calls = chain.calls
            if calls.empty or "impliedVolatility" not in calls.columns:
                continue

            calls = calls.copy()
            calls["distance"] = abs(calls["strike"] - current_price)
            atm_row = calls.loc[calls["distance"].idxmin()]
            iv = float(atm_row["impliedVolatility"])

            exp_date = datetime.strptime(exp_str, "%Y-%m-%d").date()
            dte = (exp_date - today).days

            if 0 < iv < 10 and dte > 0:
                results.append({
                    "expiration": exp_str,
                    "days_to_expiry": dte,
                    "atm_iv": round(iv, 4),
                })
        except Exception:
            continue

    return results
