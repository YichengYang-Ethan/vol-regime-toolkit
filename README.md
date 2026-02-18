# Vol Regime Trading Toolkit

[![CI](https://github.com/YichengYang-Ethan/vol-regime-toolkit/actions/workflows/ci.yml/badge.svg)](https://github.com/YichengYang-Ethan/vol-regime-toolkit/actions/workflows/ci.yml)

Volatility regime analysis for systematic options selling decisions.

## Why

Options sellers profit from theta decay, but timing matters. This toolkit identifies optimal selling windows by analyzing volatility regimes, implied vs realized vol spreads, and correlation breakdowns across your portfolio.

## Features

- **IV Percentile Tracking** - Know when implied volatility is historically high (good for selling) or low (wait)
- **Realized Volatility** - Close-to-close and Parkinson estimators with GARCH(1,1) forecasting
- **Vol Risk Premium Scanner** - Find tickers where IV exceeds realized vol (positive selling edge)
- **Correlation Regime Monitor** - Detect when leveraged ETF correlations spike (diversification breakdown)

## Quick Start

```bash
pip install -e ".[dev]"
```

### Scan for selling opportunities

```bash
vol-toolkit scan TQQQ SOXL AAPL NVDA
```

Example output:
```
Scanning 4 tickers for vol premium...

Ticker       IV   RV 20d  Premium  Prem%   IV%   Signal
------------------------------------------------------------
TQQQ      72.3%   58.1%   +14.2%   88.5  82.3     SELL
SOXL      68.1%   61.4%    +6.7%   71.2  76.1     SELL
NVDA      35.2%   31.8%    +3.4%   62.0  55.3  NEUTRAL
AAPL      18.5%   17.1%    +1.4%   54.1  42.8  NEUTRAL
```

### IV analysis for a single ticker

```bash
vol-toolkit iv TQQQ
```

### GARCH volatility forecast

```bash
vol-toolkit forecast TQQQ
```

### Correlation regime check

```bash
vol-toolkit corr TQQQ SOXL SPY QQQ
```

## Development

```bash
pip install -e ".[dev]"
ruff check src/ tests/
mypy src/ --ignore-missing-imports
pytest tests/ -v
```

## License

MIT
