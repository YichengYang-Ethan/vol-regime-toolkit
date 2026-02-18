"""Tests for realized volatility calculations."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from vol_toolkit.realized_vol import (
    calculate_parkinson_vol,
    calculate_realized_vol,
    forecast_garch,
)


def _make_gbm_prices(
    n: int = 300, mu: float = 0.0, sigma: float = 0.20, seed: int = 42
) -> pd.Series:
    """Generate synthetic GBM prices with known volatility."""
    rng = np.random.default_rng(seed)
    dt = 1 / 252
    log_returns = (mu - 0.5 * sigma**2) * dt + sigma * math.sqrt(dt) * rng.standard_normal(n)
    prices = 100 * np.exp(np.cumsum(log_returns))
    return pd.Series(prices, name="Close")


class TestCalculateRealizedVol:
    def test_basic_calculation(self):
        prices = _make_gbm_prices(n=300, sigma=0.20, seed=1)
        vol = calculate_realized_vol(prices, window=252, annualize=True)
        # Should be in the neighborhood of 0.20
        assert 0.10 < vol < 0.35

    def test_higher_vol(self):
        low_vol = calculate_realized_vol(_make_gbm_prices(sigma=0.15, seed=10), window=100)
        high_vol = calculate_realized_vol(_make_gbm_prices(sigma=0.50, seed=10), window=100)
        assert high_vol > low_vol

    def test_not_annualized(self):
        prices = _make_gbm_prices(sigma=0.20)
        ann = calculate_realized_vol(prices, window=20, annualize=True)
        raw = calculate_realized_vol(prices, window=20, annualize=False)
        assert ann == pytest.approx(raw * math.sqrt(252), rel=1e-10)

    def test_insufficient_data_raises(self):
        prices = pd.Series([100.0, 101.0, 102.0])
        with pytest.raises(ValueError, match="Need at least"):
            calculate_realized_vol(prices, window=20)


class TestParkinsonVol:
    def test_basic(self):
        rng = np.random.default_rng(42)
        n = 100
        close = 100 + np.cumsum(rng.standard_normal(n) * 0.5)
        high = close + abs(rng.standard_normal(n)) * 1.0
        low = close - abs(rng.standard_normal(n)) * 1.0

        vol = calculate_parkinson_vol(pd.Series(high), pd.Series(low), window=20)
        assert 0.0 < vol < 2.0  # Reasonable annualized vol

    def test_zero_range_gives_zero(self):
        flat = pd.Series([100.0] * 30)
        vol = calculate_parkinson_vol(flat, flat, window=20)
        assert vol == 0.0

    def test_insufficient_data(self):
        with pytest.raises(ValueError):
            calculate_parkinson_vol(pd.Series([1.0]), pd.Series([0.9]), window=20)


class TestForecastGarch:
    def test_basic_forecast(self):
        prices = _make_gbm_prices(n=500, sigma=0.25, seed=7)
        returns = np.log(prices / prices.shift(1)).dropna()
        result = forecast_garch(returns, horizon=5)

        assert "current_vol" in result
        assert "forecast_vol" in result
        assert "persistence" in result
        assert "half_life" in result
        assert result["current_vol"] > 0
        assert result["forecast_vol"] > 0
        assert 0 < result["persistence"] < 1.5  # Should be near 1

    def test_insufficient_data_raises(self):
        returns = pd.Series(np.random.default_rng(1).standard_normal(50) * 0.01)
        with pytest.raises(ValueError, match="at least 100"):
            forecast_garch(returns)
