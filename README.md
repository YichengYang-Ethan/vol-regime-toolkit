# Vol Regime Trading Toolkit

[![CI](https://github.com/YichengYang-Ethan/vol-regime-toolkit/actions/workflows/ci.yml/badge.svg)](https://github.com/YichengYang-Ethan/vol-regime-toolkit/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Volatility regime analysis for systematic options selling decisions.

**Features**: IV percentile tracking · realized vol (close-to-close + Parkinson) · GARCH(1,1) forecasting · vol risk premium scanner · correlation regime monitor

## Why

Options sellers profit from theta decay, but timing matters. This toolkit identifies optimal selling windows by analyzing volatility regimes, implied vs realized vol spreads, and correlation breakdowns across your portfolio.

## Quick Start

```bash
pip install -e ".[dev]"
```

### Scan for selling opportunities

```bash
vol-toolkit scan TQQQ SOXL AAPL NVDA
```

```
Scanning 4 tickers for vol premium...

Ticker       IV   RV 20d  Premium  Prem%   IV%   Signal
------------------------------------------------------------
TQQQ      72.3%   58.1%   +14.2%   88.5  82.3     SELL
SOXL      68.1%   61.4%    +6.7%   71.2  76.1     SELL
NVDA      35.2%   31.8%    +3.4%   62.0  55.3  NEUTRAL
AAPL      18.5%   17.1%    +1.4%   54.1  42.8  NEUTRAL
```

### Other commands

```bash
vol-toolkit iv TQQQ          # IV analysis for a single ticker
vol-toolkit forecast TQQQ    # GARCH volatility forecast
vol-toolkit corr TQQQ SOXL SPY QQQ  # Correlation regime check
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
