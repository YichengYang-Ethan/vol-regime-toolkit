"""Realized volatility estimators and GARCH forecasting."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

TRADING_DAYS_PER_YEAR = 252


def calculate_realized_vol(
    prices: pd.Series, window: int = 20, annualize: bool = True
) -> float:
    """Close-to-close realized volatility.

    Args:
        prices: Series of closing prices.
        window: Rolling window in trading days.
        annualize: Whether to annualize the result.

    Returns:
        Realized volatility as a decimal (e.g. 0.25 = 25%).
    """
    log_returns = np.log(prices / prices.shift(1)).dropna()
    if len(log_returns) < window:
        raise ValueError(f"Need at least {window} returns, got {len(log_returns)}")
    vol = log_returns.iloc[-window:].std()
    if annualize:
        vol *= math.sqrt(TRADING_DAYS_PER_YEAR)
    return float(vol)


def calculate_parkinson_vol(
    high: pd.Series, low: pd.Series, window: int = 20
) -> float:
    """Parkinson (high-low) volatility estimator.

    More efficient than close-to-close as it uses intraday range information.

    Args:
        high: Series of daily high prices.
        low: Series of daily low prices.
        window: Rolling window in trading days.

    Returns:
        Annualized Parkinson volatility as a decimal.
    """
    if len(high) < window or len(low) < window:
        raise ValueError(f"Need at least {window} data points")
    log_hl = np.log(high / low)
    recent = log_hl.iloc[-window:]
    factor = 1.0 / (4.0 * math.log(2))
    variance = factor * (recent**2).mean()
    return float(math.sqrt(variance * TRADING_DAYS_PER_YEAR))


def forecast_garch(returns: pd.Series, horizon: int = 5) -> dict:
    """GARCH(1,1) volatility forecast.

    Args:
        returns: Series of log returns (not percentage).
        horizon: Forecast horizon in trading days.

    Returns:
        Dictionary with current_vol, forecast_vol, persistence, half_life.
    """
    from arch import arch_model

    # Scale to percentage returns for numerical stability
    scaled = returns.dropna() * 100
    if len(scaled) < 100:
        raise ValueError("Need at least 100 returns for GARCH estimation")

    model = arch_model(scaled, vol="GARCH", p=1, q=1, mean="Zero", rescale=False)
    result = model.fit(disp="off", show_warning=False)

    omega = result.params["omega"]
    alpha = result.params["alpha[1]"]
    beta = result.params["beta[1]"]
    persistence = alpha + beta

    # Current conditional variance (last fitted value, in pct^2)
    cond_vol = pd.Series(result.conditional_volatility)
    current_var = float(cond_vol.iloc[-1] ** 2)

    # Unconditional (long-run) variance
    if persistence < 1.0:
        long_run_var = omega / (1.0 - persistence)
    else:
        long_run_var = current_var

    # Multi-step forecast: h-step variance
    forecast_var = long_run_var + (persistence**horizon) * (current_var - long_run_var)

    # Convert back from pct to decimal and annualize
    current_vol = math.sqrt(current_var) / 100.0 * math.sqrt(TRADING_DAYS_PER_YEAR)
    forecast_vol = math.sqrt(max(forecast_var, 0)) / 100.0 * math.sqrt(TRADING_DAYS_PER_YEAR)

    # Half-life of vol shocks
    if 0 < persistence < 1.0:
        half_life = math.log(2) / (-math.log(persistence))
    else:
        half_life = float("inf")

    return {
        "current_vol": round(current_vol, 4),
        "forecast_vol": round(forecast_vol, 4),
        "persistence": round(persistence, 4),
        "half_life": round(half_life, 1),
    }
