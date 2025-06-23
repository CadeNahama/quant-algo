# Quantitative Momentum Trading Algorithm

A sophisticated momentum-based trading system designed for stock market analysis and automated trading.

## Features

- **Multi-timeframe momentum analysis** (daily, weekly, monthly)
- **Advanced signal generation** with multiple momentum indicators
- **Risk management** with position sizing and stop-loss
- **Comprehensive backtesting** framework
- **Real-time performance monitoring** dashboard
- **Portfolio optimization** and rebalancing
- **Statistical analysis** and performance metrics

## Installation

```bash
pip install -r requirements.txt
```

## Usage

1. **Data Collection**: Run `python data_collector.py` to fetch historical data
2. **Strategy Backtesting**: Run `python backtest.py` to test the momentum strategy
3. **Live Trading**: Run `python live_trader.py` for real-time trading (requires API keys)
4. **Dashboard**: Run `python dashboard.py` to view performance metrics

## Strategy Overview

The algorithm uses multiple momentum indicators:
- Price momentum (relative strength)
- Volume momentum
- Volatility-adjusted momentum
- Cross-sectional momentum ranking
- Mean reversion signals

## Risk Management

- Dynamic position sizing based on volatility
- Maximum drawdown limits
- Sector diversification
- Correlation-based risk adjustment

## Performance Metrics

- Sharpe ratio, Sortino ratio
- Maximum drawdown
- Alpha and beta analysis
- Information ratio
- Calmar ratio
