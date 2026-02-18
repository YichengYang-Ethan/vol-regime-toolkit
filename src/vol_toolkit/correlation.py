"""Correlation regime monitoring for portfolio diversification analysis."""

from __future__ import annotations

import numpy as np
import pandas as pd
import yfinance as yf


def rolling_correlation(
    tickers: list[str], window: int = 60, period: str = "1y"
) -> pd.DataFrame:
    """Calculate rolling pairwise correlation matrix using most recent window.

    Args:
        tickers: List of ticker symbols.
        window: Rolling window in trading days.
        period: Data period to download (e.g. "1y", "2y").

    Returns:
        DataFrame correlation matrix of the most recent window.
    """
    if len(tickers) < 2:
        raise ValueError("Need at least 2 tickers for correlation")

    data = yf.download(tickers, period=period, progress=False)
    if data.empty:
        raise ValueError("No data available for given tickers")

    close = data["Close"]
    if isinstance(close, pd.Series):
        raise ValueError("Need at least 2 tickers with data")

    close = close.dropna()
    if len(close) < window:
        raise ValueError(f"Need at least {window} data points, got {len(close)}")

    returns = np.log(close / close.shift(1)).dropna()
    recent = returns.iloc[-window:]
    return recent.corr()


def detect_correlation_regime(
    tickers: list[str], threshold: float = 0.85
) -> list[dict]:
    """Detect when correlations spike above threshold.

    Compares current rolling correlation to longer-term average to identify
    periods of diversification breakdown.

    Args:
        tickers: List of ticker symbols.
        threshold: Correlation level considered "high".

    Returns:
        List of dicts with pair, current_corr, avg_corr, regime for each pair.
    """
    if len(tickers) < 2:
        raise ValueError("Need at least 2 tickers")

    data = yf.download(tickers, period="2y", progress=False)
    if data.empty:
        raise ValueError("No data available for given tickers")

    close = data["Close"]
    if isinstance(close, pd.Series):
        raise ValueError("Need at least 2 tickers with data")

    close = close.dropna()
    returns = np.log(close / close.shift(1)).dropna()

    if len(returns) < 120:
        raise ValueError("Need at least 120 data points for regime detection")

    # Current: 30-day rolling correlation
    current_corr = returns.iloc[-30:].corr()
    # Average: full-period correlation
    avg_corr = returns.corr()

    results = []
    seen = set()
    for i, t1 in enumerate(tickers):
        for j, t2 in enumerate(tickers):
            if i >= j:
                continue
            pair_key = (t1, t2)
            if pair_key in seen:
                continue
            seen.add(pair_key)

            if t1 not in current_corr.columns or t2 not in current_corr.columns:
                continue

            curr = float(current_corr.loc[t1, t2])
            avg = float(avg_corr.loc[t1, t2])
            regime = "HIGH" if curr >= threshold else "NORMAL"

            results.append({
                "pair": f"{t1}/{t2}",
                "current_corr": round(curr, 4),
                "avg_corr": round(avg, 4),
                "regime": regime,
            })

    return results
