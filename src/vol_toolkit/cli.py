"""Click-based CLI for vol-regime-toolkit."""

from __future__ import annotations

import click
import numpy as np
import yfinance as yf


@click.group()
@click.version_option(package_name="vol-regime-toolkit")
def main() -> None:
    """Vol Regime Trading Toolkit - volatility analysis for options selling."""


@main.command()
@click.argument("tickers", nargs=-1, required=True)
@click.option("--lookback", "-l", default=252, help="Lookback period in trading days.")
def scan(tickers: tuple[str, ...], lookback: int) -> None:
    """Scan tickers for vol risk premium (best selling opportunities)."""
    from vol_toolkit.premium_scanner import scan_vol_premium

    click.echo(f"Scanning {len(tickers)} tickers for vol premium...\n")
    df = scan_vol_premium(list(tickers), lookback=lookback)

    if df.empty:
        click.echo("No data available for the given tickers.")
        return

    # Format output table
    click.echo(
        f"{'Ticker':<8} {'IV':>8} {'RV 20d':>8} {'Premium':>8}"
        f" {'Prem%':>7} {'IV%':>6} {'Signal':>8}"
    )
    click.echo("-" * 60)
    for _, row in df.iterrows():
        click.echo(
            f"{row['ticker']:<8} "
            f"{row['current_iv']:>7.1%} "
            f"{row['realized_vol_20d']:>7.1%} "
            f"{row['premium']:>+7.1%} "
            f"{row['premium_pct']:>6.1f} "
            f"{row['iv_percentile']:>5.1f} "
            f"{row['signal']:>8}"
        )


@main.command()
@click.argument("ticker")
@click.option("--lookback", "-l", default=252, help="Lookback period in trading days.")
def iv(ticker: str, lookback: int) -> None:
    """Show IV percentile details for a ticker."""
    from vol_toolkit.iv_tracker import get_iv_percentile, get_iv_term_structure

    click.echo(f"IV analysis for {ticker}...\n")

    result = get_iv_percentile(ticker, lookback_days=lookback)

    click.echo(f"  Ticker:        {result['ticker']}")
    click.echo(f"  Current IV:    {result['current_iv']:.1%}")
    click.echo(f"  IV Percentile: {result['iv_percentile']:.1f}")
    click.echo(f"  IV Rank:       {result['iv_rank']:.1f}")
    click.echo(f"  52w High IV:   {result['high_iv']:.1%}")
    click.echo(f"  52w Low IV:    {result['low_iv']:.1%}")
    click.echo(f"  Signal:        {result['signal']}")

    click.echo("\nTerm Structure:")
    ts = get_iv_term_structure(ticker)
    if ts:
        click.echo(f"  {'Expiry':<12} {'DTE':>5} {'ATM IV':>8}")
        click.echo("  " + "-" * 28)
        for entry in ts:
            click.echo(
                f"  {entry['expiration']:<12} "
                f"{entry['days_to_expiry']:>5} "
                f"{entry['atm_iv']:>7.1%}"
            )
    else:
        click.echo("  No options data available.")


@main.command()
@click.argument("ticker")
@click.option("--horizon", "-h", default=5, help="Forecast horizon in trading days.")
def forecast(ticker: str, horizon: int) -> None:
    """GARCH volatility forecast for a ticker."""
    from vol_toolkit.realized_vol import forecast_garch

    click.echo(f"GARCH(1,1) forecast for {ticker}...\n")

    data = yf.download(ticker, period="2y", progress=False)
    if data.empty:
        click.echo(f"No data available for {ticker}.")
        return

    close = data["Close"].squeeze().dropna()
    returns = np.log(close / close.shift(1)).dropna()

    result = forecast_garch(returns, horizon=horizon)

    click.echo(f"  Current Vol:   {result['current_vol']:.1%}")
    click.echo(f"  {horizon}d Forecast:  {result['forecast_vol']:.1%}")
    click.echo(f"  Persistence:   {result['persistence']:.4f}")
    click.echo(f"  Half-life:     {result['half_life']:.1f} days")

    if result["forecast_vol"] > result["current_vol"] * 1.1:
        click.echo("\n  Outlook: Vol expected to INCREASE - consider waiting.")
    elif result["forecast_vol"] < result["current_vol"] * 0.9:
        click.echo("\n  Outlook: Vol expected to DECREASE - good window to sell.")
    else:
        click.echo("\n  Outlook: Vol expected to remain STABLE.")


@main.command()
@click.argument("tickers", nargs=-1, required=True)
@click.option("--window", "-w", default=60, help="Rolling window in trading days.")
@click.option("--threshold", "-t", default=0.85, help="High correlation threshold.")
def corr(tickers: tuple[str, ...], window: int, threshold: float) -> None:
    """Correlation regime analysis for a group of tickers."""
    from vol_toolkit.correlation import detect_correlation_regime, rolling_correlation

    ticker_list = list(tickers)
    if len(ticker_list) < 2:
        click.echo("Need at least 2 tickers for correlation analysis.")
        return

    click.echo(f"Correlation analysis for {', '.join(ticker_list)}...\n")

    # Correlation matrix
    corr_matrix = rolling_correlation(ticker_list, window=window)
    click.echo(f"{window}-day Rolling Correlation Matrix:")
    click.echo(corr_matrix.round(3).to_string())

    # Regime detection
    click.echo(f"\nCorrelation Regime (threshold={threshold}):")
    regimes = detect_correlation_regime(ticker_list, threshold=threshold)

    click.echo(f"  {'Pair':<12} {'Current':>8} {'Average':>8} {'Regime':>8}")
    click.echo("  " + "-" * 40)
    for r in regimes:
        click.echo(
            f"  {r['pair']:<12} "
            f"{r['current_corr']:>8.3f} "
            f"{r['avg_corr']:>8.3f} "
            f"{r['regime']:>8}"
        )

    high_count = sum(1 for r in regimes if r["regime"] == "HIGH")
    if high_count > 0:
        click.echo(
            f"\n  WARNING: {high_count} pair(s) showing HIGH correlation - "
            "diversification may be limited."
        )
