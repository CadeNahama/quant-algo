#!/usr/bin/env python3
"""
Test script with mock data to demonstrate the momentum trading system.
This script creates synthetic stock data to show how the system works.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

def create_mock_stock_data(symbol: str, start_date: str = "2022-01-01", 
                          end_date: str = "2024-01-01") -> pd.DataFrame:
    """Create realistic mock stock data for testing."""
    
    # Generate date range
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    dates = dates[dates.weekday < 5]  # Only weekdays
    
    # Set random seed for reproducible results
    np.random.seed(hash(symbol) % 1000)
    
    # Generate price data with realistic characteristics
    n_days = len(dates)
    
    # Start with a random price between $50 and $500
    start_price = np.random.uniform(50, 500)
    
    # Generate daily returns with some momentum and mean reversion
    daily_returns = np.random.normal(0.0005, 0.02, n_days)  # 0.05% daily return, 2% volatility
    
    # Add momentum component
    momentum_factor = 0.3
    for i in range(20, n_days):
        # Add momentum based on recent performance
        recent_return = np.mean(daily_returns[i-20:i])
        daily_returns[i] += momentum_factor * recent_return
    
    # Add mean reversion
    reversion_factor = 0.1
    for i in range(60, n_days):
        # Revert to long-term mean
        long_term_return = np.mean(daily_returns[i-60:i])
        daily_returns[i] -= reversion_factor * long_term_return
    
    # Calculate prices
    prices = [start_price]
    for ret in daily_returns[1:]:
        new_price = prices[-1] * (1 + ret)
        prices.append(max(new_price, 1))  # Ensure price doesn't go negative
    
    # Generate volume data
    base_volume = np.random.uniform(1000000, 10000000)
    volume = np.random.lognormal(np.log(base_volume), 0.5, n_days)
    
    # Create DataFrame
    data = pd.DataFrame({
        'Open': [p * np.random.uniform(0.98, 1.02) for p in prices],
        'High': [p * np.random.uniform(1.0, 1.05) for p in prices],
        'Low': [p * np.random.uniform(0.95, 1.0) for p in prices],
        'Close': prices,
        'Volume': volume
    }, index=dates)
    
    # Add technical indicators
    data['returns'] = data['Close'].pct_change()
    data['log_returns'] = np.log(data['Close'] / data['Close'].shift(1))
    
    # Moving averages
    for period in [5, 10, 20, 50, 100, 200]:
        data[f'sma_{period}'] = data['Close'].rolling(window=period).mean()
        data[f'ema_{period}'] = data['Close'].ewm(span=period).mean()
    
    # Momentum indicators
    for period in [5, 10, 20, 50]:
        data[f'momentum_{period}'] = data['Close'] / data['Close'].shift(period) - 1
        data[f'roc_{period}'] = (data['Close'] - data['Close'].shift(period)) / data['Close'].shift(period) * 100
    
    # Volatility indicators
    for period in [10, 20, 50]:
        data[f'volatility_{period}'] = data['returns'].rolling(window=period).std() * np.sqrt(252)
    
    # Volume indicators
    data['volume_sma_20'] = data['Volume'].rolling(window=20).mean()
    data['volume_ratio'] = data['Volume'] / data['volume_sma_20']
    
    # RSI
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    data['rsi'] = 100 - (100 / (1 + rs))
    
    # Bollinger Bands
    data['bb_middle'] = data['Close'].rolling(window=20).mean()
    bb_std = data['Close'].rolling(window=20).std()
    data['bb_upper'] = data['bb_middle'] + (bb_std * 2)
    data['bb_lower'] = data['bb_middle'] - (bb_std * 2)
    data['bb_width'] = (data['bb_upper'] - data['bb_lower']) / data['bb_middle']
    
    # MACD
    exp1 = data['Close'].ewm(span=12).mean()
    exp2 = data['Close'].ewm(span=26).mean()
    data['macd'] = exp1 - exp2
    data['macd_signal'] = data['macd'].ewm(span=9).mean()
    data['macd_histogram'] = data['macd'] - data['macd_signal']
    
    return data

def test_momentum_strategy():
    """Test the momentum strategy with mock data."""
    
    print("🧪 Testing Momentum Trading System with Mock Data")
    print("=" * 60)
    
    # Create mock data for multiple stocks
    symbols = ['MOCK_AAPL', 'MOCK_MSFT', 'MOCK_GOOGL', 'MOCK_TSLA', 'MOCK_NVDA',
               'MOCK_META', 'MOCK_AMZN', 'MOCK_NFLX', 'MOCK_JPM', 'MOCK_V']
    
    print(f"Creating mock data for {len(symbols)} stocks...")
    
    data_dict = {}
    for symbol in symbols:
        data = create_mock_stock_data(symbol)
        data_dict[symbol] = data
        print(f"  {symbol}: {len(data)} days, ${data['Close'].iloc[-1]:.2f}")
    
    # Import strategy components
    from momentum_strategy import MomentumStrategy
    from backtest import Backtester
    
    # Initialize strategy
    strategy = MomentumStrategy(
        lookback_periods=[20, 60, 120],
        max_positions=5,
        position_size_method='risk_parity'
    )
    
    # Calculate momentum signals
    print("\n🎯 Calculating momentum signals...")
    signals_df = strategy.calculate_momentum_signals(data_dict)
    
    if not signals_df.empty:
        print(f"✅ Generated signals for {len(signals_df.columns)} indicators")
        print(f"Signal matrix shape: {signals_df.shape}")
        
        # Show sample signals
        latest_date = signals_df.index[-1]
        latest_signals = signals_df.loc[latest_date]
        
        # Extract composite momentum scores
        momentum_scores = {}
        for col in latest_signals.index:
            if col.endswith('_composite_momentum'):
                symbol = col.replace('_composite_momentum', '')
                momentum_scores[symbol] = latest_signals[col]
        
        print("\nTop 5 Momentum Stocks:")
        top_momentum = sorted(momentum_scores.items(), key=lambda x: x[1], reverse=True)[:5]
        for symbol, score in top_momentum:
            print(f"  {symbol}: {score:.3f}")
        
        # Generate portfolio signals
        print("\n💼 Generating portfolio allocation...")
        weights = strategy.generate_portfolio_signals(signals_df, latest_date)
        
        if weights:
            print(f"✅ Selected {len(weights)} stocks for portfolio")
            print("\nPortfolio Allocation:")
            for symbol, weight in weights.items():
                print(f"  {symbol}: {weight:.1%}")
            
            # Apply risk management
            print("\n🛡️ Applying risk management...")
            adjusted_weights = strategy.apply_risk_management(weights, signals_df, latest_date)
            
            if adjusted_weights != weights:
                print("Risk management adjustments applied:")
                for symbol, weight in adjusted_weights.items():
                    original_weight = weights.get(symbol, 0)
                    if abs(weight - original_weight) > 0.001:
                        print(f"  {symbol}: {original_weight:.1%} → {weight:.1%}")
        
        # Run backtest
        print("\n📊 Running backtest...")
        backtester = Backtester(
            initial_capital=100000,
            transaction_cost=0.001,
            slippage=0.0005
        )
        
        results = backtester.run_backtest(
            data_dict=data_dict,
            strategy=strategy,
            start_date="2022-06-01",
            end_date="2023-12-31"
        )
        
        if results:
            print("✅ Backtest completed successfully!")
            
            # Display performance summary
            print("\n" + "="*60)
            print("📈 BACKTEST RESULTS SUMMARY")
            print("="*60)
            
            print(f"Total Return: {results['total_return']:.2%}")
            print(f"Annualized Return: {results['annualized_return']:.2%}")
            print(f"Volatility: {results['volatility']:.2%}")
            print(f"Sharpe Ratio: {results['sharpe_ratio']:.3f}")
            print(f"Sortino Ratio: {results['sortino_ratio']:.3f}")
            print(f"Maximum Drawdown: {results['max_drawdown']:.2%}")
            print(f"Win Rate: {results['win_rate']:.2%}")
            print(f"Total Trades: {results['total_trades']}")
            
            # Create performance visualization
            print("\n📊 Generating performance charts...")
            create_performance_charts(results)
            
        else:
            print("❌ Backtest failed to produce results.")
    
    else:
        print("❌ No signals generated. Check data and strategy parameters.")

def create_performance_charts(results):
    """Create performance visualization."""
    try:
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Momentum Strategy Performance (Mock Data)', fontsize=16, fontweight='bold')
        
        # Portfolio value over time
        portfolio_df = results['portfolio_values']
        axes[0, 0].plot(portfolio_df.index, portfolio_df['value'], linewidth=2, color='blue')
        axes[0, 0].set_title('Portfolio Value Over Time')
        axes[0, 0].set_ylabel('Portfolio Value ($)')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Cumulative returns
        returns = results['returns']
        cumulative_returns = (1 + returns).cumprod()
        axes[0, 1].plot(cumulative_returns.index, cumulative_returns, linewidth=2, color='green')
        axes[0, 1].set_title('Cumulative Returns')
        axes[0, 1].set_ylabel('Cumulative Return')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Drawdown
        rolling_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - rolling_max) / rolling_max
        axes[1, 0].fill_between(drawdown.index, drawdown, 0, alpha=0.3, color='red')
        axes[1, 0].plot(drawdown.index, drawdown, color='red', linewidth=1)
        axes[1, 0].set_title('Drawdown')
        axes[1, 0].set_ylabel('Drawdown')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Returns distribution
        axes[1, 1].hist(returns, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        axes[1, 1].axvline(returns.mean(), color='red', linestyle='--', label=f'Mean: {returns.mean():.4f}')
        axes[1, 1].set_title('Returns Distribution')
        axes[1, 1].set_xlabel('Daily Returns')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('mock_performance_analysis.png', dpi=300, bbox_inches='tight')
        print("✅ Performance charts saved to 'mock_performance_analysis.png'")
        plt.show()
        
    except Exception as e:
        print(f"❌ Error creating charts: {e}")

def analyze_mock_data():
    """Analyze the characteristics of mock data."""
    print("\n🔍 Analyzing Mock Data Characteristics...")
    
    # Create sample data
    symbol = 'MOCK_AAPL'
    data = create_mock_stock_data(symbol)
    
    print(f"\nData for {symbol}:")
    print(f"  Time period: {data.index[0].date()} to {data.index[-1].date()}")
    print(f"  Total days: {len(data)}")
    print(f"  Price range: ${data['Close'].min():.2f} - ${data['Close'].max():.2f}")
    print(f"  Final price: ${data['Close'].iloc[-1]:.2f}")
    
    # Calculate statistics
    returns = data['returns'].dropna()
    print(f"\nReturn Statistics:")
    print(f"  Mean daily return: {returns.mean():.4f} ({returns.mean()*252:.2%} annualized)")
    print(f"  Volatility: {returns.std():.4f} ({returns.std()*np.sqrt(252):.2%} annualized)")
    print(f"  Sharpe ratio: {(returns.mean()*252) / (returns.std()*np.sqrt(252)):.3f}")
    print(f"  Skewness: {returns.skew():.3f}")
    print(f"  Kurtosis: {returns.kurtosis():.3f}")
    
    # Show momentum characteristics
    print(f"\nMomentum Characteristics:")
    momentum_20 = data['momentum_20'].dropna()
    print(f"  20-day momentum mean: {momentum_20.mean():.4f}")
    print(f"  20-day momentum std: {momentum_20.std():.4f}")
    
    # Show technical indicators
    print(f"\nTechnical Indicators (latest values):")
    print(f"  RSI: {data['rsi'].iloc[-1]:.1f}")
    print(f"  MACD: {data['macd'].iloc[-1]:.3f}")
    print(f"  BB Width: {data['bb_width'].iloc[-1]:.3f}")

if __name__ == "__main__":
    # Analyze mock data first
    analyze_mock_data()
    
    # Test the complete system
    test_momentum_strategy()
    
    print("\n" + "="*60)
    print("🎉 Mock Data Test Completed!")
    print("="*60)
    print("\nThis demonstrates the system functionality with synthetic data.")
    print("For real trading, ensure you have proper data sources configured.") 