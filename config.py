"""
Configuration file for the Momentum Trading System

This file contains all the configurable parameters for the trading system.
Modify these values to customize the strategy behavior.
"""

# Data Collection Settings
DATA_SETTINGS = {
    'default_start_date': '2020-01-01',
    'default_end_date': None,  # None means current date
    'data_source': 'yfinance',
    'cache_data': True,
    'data_directory': 'data'
}

# Trading Universe
TRADING_UNIVERSE = {
    # Technology Sector
    'tech': ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'NVDA', 'META', 'AMZN', 'NFLX', 'ADBE', 'CRM'],
    
    # Financial Sector
    'financial': ['JPM', 'V', 'MA', 'BAC', 'GS', 'WFC', 'C', 'MS', 'BLK', 'AXP'],
    
    # Healthcare Sector
    'healthcare': ['JNJ', 'UNH', 'PFE', 'ABT', 'TMO', 'MRK', 'BMY', 'AMGN', 'GILD', 'CVS'],
    
    # Consumer Sector
    'consumer': ['PG', 'HD', 'DIS', 'KO', 'PEP', 'WMT', 'COST', 'TGT', 'MCD', 'SBUX'],
    
    # Industrial Sector
    'industrial': ['CAT', 'DE', 'BA', 'GE', 'HON', 'UPS', 'RTX', 'LMT', 'NOC', 'EMR'],
    
    # Energy Sector
    'energy': ['XOM', 'CVX', 'COP', 'EOG', 'SLB', 'HAL', 'BKR', 'MPC', 'PSX', 'VLO']
}

# Strategy Parameters
STRATEGY_PARAMS = {
    # Momentum calculation periods
    'lookback_periods': [20, 60, 120],
    
    # Portfolio settings
    'max_positions': 15,
    'min_positions': 5,
    'rebalance_frequency': 20,  # days
    
    # Position sizing methods: 'equal_weight', 'momentum_weight', 'risk_parity', 'kelly_criterion'
    'position_size_method': 'risk_parity',
    
    # Risk management - Updated for more trading activity
    'momentum_threshold': 0.01,  # Reduced from 0.02 to 0.01 for more signals
    'volatility_threshold': 0.8,  # Increased from 0.5 to 0.8 to allow more stocks
    'correlation_threshold': 0.8,  # Increased from 0.7 to 0.8 for less aggressive filtering
    'max_drawdown': 0.30,  # Increased from 0.15 to 0.30 to allow more stocks
    
    # Risk-free rate for calculations
    'risk_free_rate': 0.02
}

# Backtesting Parameters
BACKTEST_PARAMS = {
    'initial_capital': 100000,
    'transaction_cost': 0.001,  # 0.1% per trade
    'slippage': 0.0005,  # 0.05% slippage
    'benchmark_symbol': 'SPY',
    'start_date': '2022-01-01',
    'end_date': '2024-01-01'
}

# Live Trading Parameters
LIVE_TRADING_PARAMS = {
    'paper_trading': True,  # Set to False for real trading
    'max_positions': 15,
    'rebalance_frequency': 30,  # minutes
    'risk_management': True,
    'auto_stop_loss': True,
    'stop_loss_pct': 0.05,  # 5% stop loss
    'take_profit_pct': 0.15,  # 15% take profit
}

# Performance Metrics Thresholds
PERFORMANCE_THRESHOLDS = {
    'min_sharpe_ratio': 1.0,
    'max_drawdown': 0.15,
    'min_win_rate': 0.55,
    'max_volatility': 0.25,
    'min_calmar_ratio': 1.0
}

# Dashboard Settings
DASHBOARD_PARAMS = {
    'refresh_interval': 30,  # seconds
    'port': 8050,
    'debug': True,
    'host': 'localhost'
}

# Logging Configuration
LOGGING_CONFIG = {
    'level': 'INFO',
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'file': 'trading_system.log',
    'max_file_size': 10 * 1024 * 1024,  # 10MB
    'backup_count': 5
}

# API Configuration (for real trading)
API_CONFIG = {
    'broker': 'paper_trading',  # 'paper_trading', 'alpaca', 'interactive_brokers', etc.
    'api_key': None,
    'api_secret': None,
    'base_url': None,
    'paper_trading': True
}

# Technical Indicators Parameters
TECHNICAL_INDICATORS = {
    'rsi_period': 14,
    'macd_fast': 12,
    'macd_slow': 26,
    'macd_signal': 9,
    'bollinger_period': 20,
    'bollinger_std': 2,
    'volume_sma_period': 20
}

# Risk Management Rules
RISK_RULES = {
    'max_sector_exposure': 0.3,  # Maximum 30% in any sector
    'max_single_position': 0.15,  # Maximum 15% in single stock
    'min_correlation_threshold': 0.7,
    'volatility_lookback': 252,  # 1 year for volatility calculation
    'drawdown_lookback': 252,  # 1 year for drawdown calculation
    'momentum_decay_factor': 0.95  # Momentum decay over time
}

# Data Validation Rules
DATA_VALIDATION = {
    'min_data_points': 252,  # Minimum 1 year of data
    'max_missing_pct': 0.05,  # Maximum 5% missing data
    'price_change_threshold': 0.5,  # Flag extreme price changes
    'volume_threshold': 1000,  # Minimum volume threshold
}

def get_all_symbols():
    """Get all symbols from all sectors."""
    all_symbols = []
    for sector_symbols in TRADING_UNIVERSE.values():
        all_symbols.extend(sector_symbols)
    return all_symbols

def get_sector_symbols(sector):
    """Get symbols for a specific sector."""
    return TRADING_UNIVERSE.get(sector, [])

def validate_config():
    """Validate configuration parameters."""
    errors = []
    
    # Check strategy parameters
    if STRATEGY_PARAMS['max_positions'] < STRATEGY_PARAMS['min_positions']:
        errors.append("max_positions must be >= min_positions")
    
    if STRATEGY_PARAMS['momentum_threshold'] <= 0:
        errors.append("momentum_threshold must be positive")
    
    if STRATEGY_PARAMS['volatility_threshold'] <= 0:
        errors.append("volatility_threshold must be positive")
    
    # Check backtest parameters
    if BACKTEST_PARAMS['initial_capital'] <= 0:
        errors.append("initial_capital must be positive")
    
    if BACKTEST_PARAMS['transaction_cost'] < 0:
        errors.append("transaction_cost must be non-negative")
    
    # Check risk rules
    if RISK_RULES['max_sector_exposure'] > 1:
        errors.append("max_sector_exposure must be <= 1")
    
    if RISK_RULES['max_single_position'] > 1:
        errors.append("max_single_position must be <= 1")
    
    if errors:
        raise ValueError(f"Configuration validation failed: {'; '.join(errors)}")
    
    return True

if __name__ == "__main__":
    # Test configuration
    try:
        validate_config()
        print("✅ Configuration validation passed!")
        print(f"Total symbols available: {len(get_all_symbols())}")
        print(f"Strategy parameters: {STRATEGY_PARAMS}")
    except ValueError as e:
        print(f"❌ Configuration error: {e}") 