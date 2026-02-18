"""Tests for vol premium scanner (all mocked, no live API calls)."""

from __future__ import annotations

import math
from unittest.mock import patch

import numpy as np
import pandas as pd

from vol_toolkit.premium_scanner import scan_vol_premium


def _make_mock_prices(n: int = 300, sigma: float = 0.20, seed: int = 42) -> pd.DataFrame:
    """Create mock yfinance download output."""
    rng = np.random.default_rng(seed)
    dt = 1 / 252
    log_ret = (0.0 - 0.5 * sigma**2) * dt + sigma * math.sqrt(dt) * rng.standard_normal(n)
    close = 100 * np.exp(np.cumsum(log_ret))
    dates = pd.bdate_range(end="2026-02-15", periods=n)
    return pd.DataFrame({"Close": close}, index=dates)


class TestScanVolPremium:
    @patch("vol_toolkit.premium_scanner._try_get_atm_iv", return_value=None)
    @patch("vol_toolkit.premium_scanner.yf.download")
    def test_basic_scan(self, mock_download, mock_atm):
        mock_download.return_value = _make_mock_prices(300, sigma=0.25)

        df = scan_vol_premium(["AAPL", "NVDA"])

        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0
        expected_cols = {
            "ticker", "current_iv", "realized_vol_20d",
            "premium", "premium_pct", "iv_percentile", "signal",
        }
        assert expected_cols.issubset(set(df.columns))

    @patch("vol_toolkit.premium_scanner._try_get_atm_iv", return_value=None)
    @patch("vol_toolkit.premium_scanner.yf.download")
    def test_sorted_by_premium(self, mock_download, mock_atm):
        mock_download.return_value = _make_mock_prices(300, sigma=0.30)

        df = scan_vol_premium(["A", "B", "C"])

        if len(df) > 1:
            premiums = df["premium"].tolist()
            assert premiums == sorted(premiums, reverse=True)

    @patch("vol_toolkit.premium_scanner._try_get_atm_iv", return_value=None)
    @patch("vol_toolkit.premium_scanner.yf.download")
    def test_empty_on_no_data(self, mock_download, mock_atm):
        mock_download.return_value = pd.DataFrame()

        df = scan_vol_premium(["FAKE"])

        assert df.empty

    @patch("vol_toolkit.premium_scanner._try_get_atm_iv", return_value=0.50)
    @patch("vol_toolkit.premium_scanner.yf.download")
    def test_with_options_iv(self, mock_download, mock_atm):
        # When ATM IV is available and higher than RV, premium should be positive
        mock_download.return_value = _make_mock_prices(300, sigma=0.15, seed=99)

        df = scan_vol_premium(["TQQQ"])

        assert len(df) == 1
        assert df.iloc[0]["premium"] > 0  # IV=0.50 >> RV~0.15
