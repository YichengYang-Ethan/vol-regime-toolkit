"""Tests for IV percentile tracking (all mocked, no live API calls)."""

from __future__ import annotations

import math
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from vol_toolkit.iv_tracker import get_iv_percentile


def _make_mock_prices(n: int = 300, sigma: float = 0.20, seed: int = 42) -> pd.DataFrame:
    """Create mock yfinance download output."""
    rng = np.random.default_rng(seed)
    dt = 1 / 252
    log_ret = (0.0 - 0.5 * sigma**2) * dt + sigma * math.sqrt(dt) * rng.standard_normal(n)
    close = 100 * np.exp(np.cumsum(log_ret))
    dates = pd.bdate_range(start="2025-01-01", periods=len(close))
    return pd.DataFrame({"Close": close}, index=dates)


class TestGetIvPercentile:
    @patch("vol_toolkit.iv_tracker._try_get_atm_iv", return_value=None)
    @patch("vol_toolkit.iv_tracker.yf.download")
    def test_basic_percentile(self, mock_download, mock_atm):
        mock_download.return_value = _make_mock_prices(300, sigma=0.20)

        result = get_iv_percentile("AAPL", lookback_days=252)

        assert result["ticker"] == "AAPL"
        assert 0 <= result["iv_percentile"] <= 100
        assert 0 <= result["iv_rank"] <= 100
        assert result["current_iv"] > 0
        assert result["high_iv"] >= result["low_iv"]
        assert result["signal"] in ("SELL", "WAIT", "NEUTRAL")

    @patch("vol_toolkit.iv_tracker._try_get_atm_iv", return_value=0.80)
    @patch("vol_toolkit.iv_tracker.yf.download")
    def test_high_iv_gives_sell(self, mock_download, mock_atm):
        # Use low-vol prices so that an IV of 0.80 is very high percentile
        mock_download.return_value = _make_mock_prices(300, sigma=0.10)

        result = get_iv_percentile("TQQQ")

        assert result["signal"] == "SELL"
        assert result["iv_percentile"] >= 75

    @patch("vol_toolkit.iv_tracker._try_get_atm_iv", return_value=0.02)
    @patch("vol_toolkit.iv_tracker.yf.download")
    def test_low_iv_gives_wait(self, mock_download, mock_atm):
        mock_download.return_value = _make_mock_prices(300, sigma=0.30)

        result = get_iv_percentile("AAPL")

        assert result["signal"] == "WAIT"
        assert result["iv_percentile"] <= 25

    @patch("vol_toolkit.iv_tracker._try_get_atm_iv", return_value=None)
    @patch("vol_toolkit.iv_tracker.yf.download")
    def test_empty_data_raises(self, mock_download, mock_atm):
        mock_download.return_value = pd.DataFrame()

        with pytest.raises(ValueError, match="No price data"):
            get_iv_percentile("FAKE")
