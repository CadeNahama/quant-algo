#!/usr/bin/env python3
"""
Comprehensive Momentum Trading System Example

This script demonstrates the complete momentum trading system including:
1. Data collection from multiple sources
2. Momentum signal generation
3. Portfolio optimization and risk management
4. Backtesting with performance analysis
5. Live trading simulation

Run this script to see the system in action!
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Import our custom modules
from data_collector import DataCollector
from momentum_strategy import MomentumStrategy
from backtest import Backtester

def main():
    """Main function to demonstrate the complete momentum trading system."""
    
    print("🚀 Momentum Trading System Demo")
    print("=" * 60)
    print("This demo will show you how the system works step by step.")
    print("=" * 60)
    
    # Step 1: Initialize Components
    print("\n📊 Step 1: Initializing System Components...")
    collector = DataCollector()
    strategy = MomentumStrategy(
        lookback_periods=[20, 60, 120],
        max_positions=10,
        position_size_method='risk_parity',
        rebalance_frequency=20
    )
    backtester = Backtester(
        initial_capital=100000,
        transaction_cost=0.001,
        slippage=0.0005
    )
    
    # Step 2: Define Trading Universe
    print("\n📈 Step 2: Defining Trading Universe...")
    symbols = [
        'AAPL', 'MSFT', 'GOOGL', 'TSLA', 'NVDA',  # Tech
        'META', 'AMZN', 'NFLX', 'ADBE', 'CRM',    # Tech/Media
        'JPM', 'V', 'MA', 'BAC', 'GS',            # Financial
        'JNJ', 'PG', 'HD', 'DIS', 'KO',           # Consumer
        'TMO', 'ABT', 'UNH', 'PFE', 'MRK'         # Healthcare
    ]
    
    print(f"Trading universe: {len(symbols)} stocks across multiple sectors")
    print("Symbols:", ', '.join(symbols))
    
    # Step 3: Collect Historical Data
    print("\n📥 Step 3: Collecting Historical Data...")
    print("Fetching 2 years of data for momentum analysis...")
    
    start_date = "2022-01-01"
    end_date = "2024-01-01"
    
    data_dict = collector.fetch_multiple_stocks(symbols, start_date, end_date)
    
    print(f"✅ Successfully collected data for {len(data_dict)} stocks")
    
    # Show data summary
    print("\nData Summary:")
    for symbol, data in list(data_dict.items())[:5]:  # Show first 5
        print(f"  {symbol}: {len(data)} days, ${data['Close'].iloc[-1]:.2f}")
    
    # Step 4: Calculate Momentum Signals
    print("\n🎯 Step 4: Calculating Momentum Signals...")
    signals_df = strategy.calculate_momentum_signals(data_dict)
    
    if not signals_df.empty:
        print(f"✅ Generated signals for {len(signals_df.columns)} indicators")
        print(f"Signal matrix shape: {signals_df.shape}")
        
        # Show sample signals
        print("\nSample Momentum Signals (latest date):")
        latest_date = signals_df.index[-1]
        latest_signals = signals_df.loc[latest_date]
        
        # Extract composite momentum scores
        momentum_scores = {}
        for col in latest_signals.index:
            if col.endswith('_composite_momentum'):
                symbol = col.replace('_composite_momentum', '')
                momentum_scores[symbol] = latest_signals[col]
        
        # Show top 5 momentum stocks
        top_momentum = sorted(momentum_scores.items(), key=lambda x: x[1], reverse=True)[:5]
        print("Top 5 Momentum Stocks:")
        for symbol, score in top_momentum:
            print(f"  {symbol}: {score:.3f}")
    else:
        print("❌ No signals generated. Check data and strategy parameters.")
        return
    
    # Step 5: Generate Portfolio Signals
    print("\n💼 Step 5: Generating Portfolio Allocation...")
    latest_date = pd.Timestamp(signals_df.index[-1])
    weights = strategy.generate_portfolio_signals(signals_df, latest_date)
    
    if weights:
        print(f"✅ Selected {len(weights)} stocks for portfolio")
        print("\nPortfolio Allocation:")
        for symbol, weight in weights.items():
            print(f"  {symbol}: {weight:.1%}")
        
        # Apply risk management
        print("\n🛡️ Applying Risk Management...")
        adjusted_weights = strategy.apply_risk_management(weights, signals_df, latest_date)
        
        if adjusted_weights != weights:
            print("Risk management adjustments applied:")
            for symbol, weight in adjusted_weights.items():
                original_weight = weights.get(symbol, 0)
                if abs(weight - original_weight) > 0.001:
                    print(f"  {symbol}: {original_weight:.1%} → {weight:.1%}")
    else:
        print("❌ No portfolio signals generated.")
        return
    
    # Step 6: Run Backtest
    print("\n📊 Step 6: Running Backtest...")
    print("Testing strategy performance over historical data...")
    
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
        
        if 'alpha' in results:
            print(f"Alpha: {results['alpha']:.2%}")
            print(f"Beta: {results['beta']:.3f}")
            print(f"Information Ratio: {results['information_ratio']:.3f}")
        
        # Create performance visualization
        print("\n📊 Generating Performance Charts...")
        create_performance_charts(results)
        
    else:
        print("❌ Backtest failed to produce results.")
    
    # Step 7: Strategy Analysis
    print("\n🔍 Step 7: Strategy Analysis...")
    analyze_strategy_performance(results, data_dict, strategy)
    
    print("\n" + "="*60)
    print("🎉 Demo Completed Successfully!")
    print("="*60)
    print("\nNext Steps:")
    print("1. Run 'python live_trader.py' for live trading simulation")
    print("2. Run 'python dashboard.py' for real-time monitoring")
    print("3. Modify strategy parameters in momentum_strategy.py")
    print("4. Add more stocks to the trading universe")
    print("5. Integrate with your preferred broker for real trading")

def create_performance_charts(results):
    """Create comprehensive performance visualization."""
    try:
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Momentum Strategy Performance Analysis', fontsize=16, fontweight='bold')
        
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
        plt.savefig('performance_analysis.png', dpi=300, bbox_inches='tight')
        print("✅ Performance charts saved to 'performance_analysis.png'")
        plt.show()
        
    except Exception as e:
        print(f"❌ Error creating charts: {e}")

def analyze_strategy_performance(results, data_dict, strategy):
    """Analyze strategy performance and provide insights."""
    try:
        print("\nStrategy Performance Analysis:")
        print("-" * 40)
        
        # Risk-adjusted returns
        sharpe = results['sharpe_ratio']
        sortino = results['sortino_ratio']
        calmar = results['calmar_ratio']
        
        print(f"Risk-Adjusted Performance:")
        print(f"  Sharpe Ratio: {sharpe:.3f} {'(Good)' if sharpe > 1.0 else '(Needs Improvement)'}")
        print(f"  Sortino Ratio: {sortino:.3f} {'(Good)' if sortino > 1.0 else '(Needs Improvement)'}")
        print(f"  Calmar Ratio: {calmar:.3f} {'(Good)' if calmar > 1.0 else '(Needs Improvement)'}")
        
        # Risk metrics
        max_dd = results['max_drawdown']
        volatility = results['volatility']
        
        print(f"\nRisk Metrics:")
        print(f"  Maximum Drawdown: {max_dd:.2%} {'(Acceptable)' if abs(max_dd) < 0.15 else '(High Risk)'}")
        print(f"  Volatility: {volatility:.2%} {'(Low)' if volatility < 0.15 else '(Moderate)' if volatility < 0.25 else '(High)'}")
        
        # Trading statistics
        win_rate = results['win_rate']
        total_trades = results['total_trades']
        
        print(f"\nTrading Statistics:")
        print(f"  Win Rate: {win_rate:.2%} {'(Good)' if win_rate > 0.55 else '(Needs Improvement)'}")
        print(f"  Total Trades: {total_trades} {'(Active)' if total_trades > 100 else '(Conservative)'}")
        
        # Recommendations
        print(f"\nRecommendations:")
        if sharpe < 1.0:
            print("  • Consider adjusting momentum thresholds for better risk-adjusted returns")
        if abs(max_dd) > 0.15:
            print("  • Implement stricter risk management rules")
        if win_rate < 0.55:
            print("  • Review signal generation logic for better accuracy")
        if total_trades < 50:
            print("  • Consider more frequent rebalancing for better diversification")
        
        print("  • Monitor correlation between positions to avoid concentration risk")
        print("  • Regularly review and update the trading universe")
        
    except Exception as e:
        print(f"❌ Error analyzing performance: {e}")

if __name__ == "__main__":
    main() 