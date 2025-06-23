#!/usr/bin/env python3
"""
Test script to verify the backtest fixes work correctly.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

from momentum_strategy import MomentumStrategy
from backtest import Backtester

def create_test_data():
    """Create realistic test data for verification."""
    print("📊 Creating realistic test data...")
    
    # Generate realistic price data
    np.random.seed(42)
    dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
    
    test_data = {}
    symbols = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'NVDA']
    
    for symbol in symbols:
        # Generate realistic price data with proper returns
        initial_price = np.random.uniform(50, 500)
        daily_returns = np.random.normal(0.0005, 0.02, len(dates))  # Realistic daily returns
        
        prices = [initial_price]
        for ret in daily_returns[1:]:
            new_price = prices[-1] * (1 + ret)
            prices.append(max(new_price, 1))  # Ensure price doesn't go below 1
        
        # Generate volume data
        base_volume = np.random.randint(1000000, 10000000)
        volume = np.random.poisson(base_volume, len(dates))
        
        # Create DataFrame
        df = pd.DataFrame({
            'Open': [p * np.random.uniform(0.98, 1.02) for p in prices],
            'High': [p * np.random.uniform(1.0, 1.05) for p in prices],
            'Low': [p * np.random.uniform(0.95, 1.0) for p in prices],
            'Close': prices,
            'Volume': volume,
            'Adj Close': prices
        }, index=dates)
        
        # Calculate returns
        df['returns'] = df['Close'].pct_change()
        
        # Calculate technical indicators
        df['volume_ratio'] = df['Volume'] / df['Volume'].rolling(window=20).mean()
        
        # RSI
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # MACD
        exp1 = df['Close'].ewm(span=12).mean()
        exp2 = df['Close'].ewm(span=26).mean()
        df['macd'] = exp1 - exp2
        df['macd_signal'] = df['macd'].ewm(span=9).mean()
        df['macd_histogram'] = df['macd'] - df['macd_signal']
        
        # Bollinger Bands
        df['sma_20'] = df['Close'].rolling(window=20).mean()
        df['bb_upper'] = df['sma_20'] + (df['Close'].rolling(window=20).std() * 2)
        df['bb_lower'] = df['sma_20'] - (df['Close'].rolling(window=20).std() * 2)
        
        # Additional indicators
        df['sma_50'] = df['Close'].rolling(window=50).mean()
        df['sma_200'] = df['Close'].rolling(window=200).mean()
        df['volatility_20'] = df['returns'].rolling(window=20).std()
        
        # Fill NaN values
        df = df.fillna(method='bfill').fillna(method='ffill')
        
        test_data[symbol] = df
    
    print(f"✅ Created test data for {len(test_data)} symbols")
    return test_data

def test_backtest_fixes():
    """Test the backtest fixes with realistic data."""
    print("\n" + "="*60)
    print("🔧 TESTING BACKTEST FIXES")
    print("="*60)
    
    # Create test data
    data_dict = create_test_data()
    
    # Initialize advanced Kelly Criterion strategy for profitable trading
    strategy = MomentumStrategy(
        lookback_periods=[5, 10, 20, 50],  # Multiple timeframes for better signals
        max_positions=8,  # Optimal number of positions for diversification
        position_size_method='kelly_criterion',  # Use Kelly Criterion for optimal sizing
        rebalance_frequency=5,  # Weekly rebalancing to capture moves
        momentum_threshold=0.1,  # Lower threshold for more opportunities
        volatility_threshold=0.4,  # Higher volatility threshold
        use_market_regime=True,  # Adapt to market conditions
        use_volume_confirmation=True,  # Require volume confirmation
        use_breakout_signals=True,  # Use breakout signals
        use_stop_loss=True,  # Use stop-losses for risk management
        stop_loss_pct=0.05,  # 5% stop-loss
        take_profit_pct=0.15,  # 15% take-profit
        use_kelly_criterion=True,  # Enable Kelly Criterion
        max_kelly_fraction=0.25  # Maximum 25% Kelly fraction
    )
    
    # Initialize aggressive backtester for profitable trading
    backtester = Backtester(
        initial_capital=100000,
        transaction_cost=0.0005,  # Lower transaction costs for more frequent trading
        slippage=0.0002  # Lower slippage for better execution
    )
    
    # Run backtest
    print("\n🔄 Running backtest with fixes...")
    try:
        results = backtester.run_backtest(
            data_dict=data_dict,
            strategy=strategy,
            start_date="2023-06-01",
            end_date="2023-12-31"
        )
        
        if results:
            print("✅ Backtest completed successfully!")
            
            # Check for realistic results
            print("\n📊 RESULTS VALIDATION:")
            print(f"Total Return: {results['total_return']:.2%}")
            print(f"Annualized Return: {results['annualized_return']:.2%}")
            print(f"Volatility: {results['volatility']:.2%}")
            print(f"Sharpe Ratio: {results['sharpe_ratio']:.3f}")
            print(f"Total Trades: {results['total_trades']}")
            print(f"Total Volume: ${results['total_volume']:,.0f}")
            print(f"Average Trade Size: ${results['avg_trade_size']:,.0f}")
            
            # Validate results are realistic
            is_realistic = True
            issues = []
            
            if abs(results['total_return']) > 10:  # More than 1000% return
                issues.append(f"Unrealistic total return: {results['total_return']:.2%}")
                is_realistic = False
            
            if abs(results['annualized_return']) > 5:  # More than 500% annualized
                issues.append(f"Unrealistic annualized return: {results['annualized_return']:.2%}")
                is_realistic = False
            
            if results['volatility'] > 2:  # More than 200% volatility
                issues.append(f"Unrealistic volatility: {results['volatility']:.2%}")
                is_realistic = False
            
            if abs(results['sharpe_ratio']) > 5:  # More than 5 Sharpe ratio
                issues.append(f"Unrealistic Sharpe ratio: {results['sharpe_ratio']:.3f}")
                is_realistic = False
            
            if results['total_volume'] > 1e12:  # More than 1 trillion volume
                issues.append(f"Unrealistic total volume: ${results['total_volume']:,.0f}")
                is_realistic = False
            
            if results['avg_trade_size'] > 1e9:  # More than 1 billion per trade
                issues.append(f"Unrealistic average trade size: ${results['avg_trade_size']:,.0f}")
                is_realistic = False
            
            if is_realistic:
                print("\n✅ ALL RESULTS ARE REALISTIC!")
                print("🎉 Backtest fixes are working correctly!")
            else:
                print("\n❌ UNREALISTIC RESULTS DETECTED:")
                for issue in issues:
                    print(f"   • {issue}")
                print("🔧 Additional fixes may be needed.")
            
            return is_realistic
            
        else:
            print("❌ Backtest failed to produce results")
            return False
            
    except Exception as e:
        print(f"❌ Error during backtest: {e}")
        return False

def main():
    """Main test function."""
    print("🚀 Testing Backtest Fixes")
    print("="*60)
    
    success = test_backtest_fixes()
    
    print("\n" + "="*60)
    print("📋 TEST SUMMARY")
    print("="*60)
    
    if success:
        print("✅ All tests passed! Backtest fixes are working correctly.")
        print("🎯 The system now produces realistic results.")
    else:
        print("❌ Some tests failed. Additional fixes may be needed.")
    
    print("\n" + "="*60)

if __name__ == "__main__":
    main() 