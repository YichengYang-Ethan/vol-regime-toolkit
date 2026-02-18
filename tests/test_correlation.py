"""Tests for correlation regime detection (synthetic data, no API calls)."""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from vol_toolkit.correlation import detect_correlation_regime, rolling_correlation


def _make_correlated_prices(
    tickers: list[str],
    n: int = 300,
    correlation: float = 0.90,
    seed: int = 42,
) -> pd.DataFrame:
    """Create synthetic correlated price data mimicking yfinance multi-ticker output."""
    rng = np.random.default_rng(seed)
    k = len(tickers)

    # Build correlation matrix
    cov = np.full((k, k), correlation)
    np.fill_diagonal(cov, 1.0)

    # Cholesky decomposition for correlated normals
    L = np.linalg.cholesky(cov)
    raw = rng.standard_normal((n, k))
    correlated = raw @ L.T

    # Convert to prices
    prices = 100 * np.exp(np.cumsum(correlated * 0.01, axis=0))
    dates = pd.date_range(end="2026-02-15", periods=prices.shape[0], freq="B")

    # Build MultiIndex columns like yfinance multi-ticker download
    cols = pd.MultiIndex.from_product([["Close"], tickers])
    df = pd.DataFrame(prices, index=dates, columns=cols)
    return df


class TestRollingCorrelation:
    @patch("vol_toolkit.correlation.yf.download")
    def test_high_correlation(self, mock_download):
        mock_download.return_value = _make_correlated_prices(
            ["TQQQ", "SOXL"], correlation=0.95
        )
        corr = rolling_correlation(["TQQQ", "SOXL"], window=60)

        assert corr.shape == (2, 2)
        assert corr.loc["TQQQ", "SOXL"] > 0.80  # Should detect high correlation

    @patch("vol_toolkit.correlation.yf.download")
    def test_low_correlation(self, mock_download):
        mock_download.return_value = _make_correlated_prices(
            ["A", "B"], correlation=0.10, seed=99
        )
        corr = rolling_correlation(["A", "B"], window=60)

        assert abs(corr.loc["A", "B"]) < 0.50  # Should be low

    def test_single_ticker_raises(self):
        with pytest.raises(ValueError, match="at least 2"):
            rolling_correlation(["AAPL"])


class TestDetectCorrelationRegime:
    @patch("vol_toolkit.correlation.yf.download")
    def test_high_regime_detected(self, mock_download):
        mock_download.return_value = _make_correlated_prices(
            ["TQQQ", "SOXL"], n=300, correlation=0.95
        )
        results = detect_correlation_regime(["TQQQ", "SOXL"], threshold=0.85)

        assert len(results) == 1
        assert results[0]["regime"] == "HIGH"
        assert results[0]["pair"] == "TQQQ/SOXL"

    @patch("vol_toolkit.correlation.yf.download")
    def test_normal_regime(self, mock_download):
        mock_download.return_value = _make_correlated_prices(
            ["A", "B"], n=300, correlation=0.30, seed=123
        )
        results = detect_correlation_regime(["A", "B"], threshold=0.85)

        assert len(results) == 1
        assert results[0]["regime"] == "NORMAL"

    @patch("vol_toolkit.correlation.yf.download")
    def test_multiple_pairs(self, mock_download):
        mock_download.return_value = _make_correlated_prices(
            ["A", "B", "C"], n=300, correlation=0.70
        )
        results = detect_correlation_regime(["A", "B", "C"])

        # 3 tickers = 3 pairs
        assert len(results) == 3
        for r in results:
            assert "pair" in r
            assert "current_corr" in r
            assert "regime" in r
