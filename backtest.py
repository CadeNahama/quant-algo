import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
import logging
from datetime import datetime, timedelta
import os
import warnings
warnings.filterwarnings('ignore')

from data_collector import DataCollector
from momentum_strategy import MomentumStrategy

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Backtester:
    """
    Comprehensive backtesting framework for momentum trading strategies.
    Handles portfolio simulation, performance analysis, and risk management.
    """
    
    def __init__(self, 
                 initial_capital: float = 100000,
                 transaction_cost: float = 0.001,
                 slippage: float = 0.0005,
                 benchmark_symbol: str = 'SPY'):
        
        self.initial_capital = initial_capital
        self.transaction_cost = transaction_cost
        self.slippage = slippage
        self.benchmark_symbol = benchmark_symbol
        
        # Performance tracking
        self.portfolio_values = []
        self.positions = {}
        self.trades = []
        self.daily_returns = []
        
    def run_backtest(self, 
                    data_dict: Dict[str, pd.DataFrame],
                    strategy: MomentumStrategy,
                    start_date: Optional[str] = None,
                    end_date: Optional[str] = None) -> Dict:
        """
        Run comprehensive backtest of the momentum strategy.
        
        Args:
            data_dict: Dictionary of stock data
            strategy: Momentum strategy instance
            start_date: Backtest start date
            end_date: Backtest end date
            
        Returns:
            Dictionary containing backtest results
        """
        logger.info("Starting backtest...")
        
        # Initialize tracking variables
        self.portfolio_values = []
        self.positions = {}
        self.trades = []
        self.daily_returns = []
        
        # Calculate momentum signals
        signals_df = strategy.calculate_momentum_signals(data_dict)
        
        if signals_df.empty:
            logger.error("No signals generated. Check data and strategy parameters.")
            return {}
        
        # Determine date range
        if start_date is None:
            start_date = str(signals_df.index[0])
        if end_date is None:
            end_date = str(signals_df.index[-1])
        
        # Filter date range
        signals_df = signals_df.loc[start_date:end_date]
        
        # Get benchmark data
        benchmark_data = self._get_benchmark_data(start_date, end_date)
        
        # Initialize portfolio
        current_capital = self.initial_capital
        current_positions = {}
        
        # Main backtest loop
        for i, date in enumerate(signals_df.index):
            try:
                # Generate portfolio signals
                weights = strategy.generate_portfolio_signals(signals_df, date)
                
                # Apply risk management
                weights = strategy.apply_risk_management(weights, signals_df, date)
                
                # Calculate portfolio value
                portfolio_value = self._calculate_portfolio_value(
                    current_positions, data_dict, date, current_capital
                )
                
                # Rebalance portfolio
                new_positions, trade_cost = self._rebalance_portfolio(
                    current_positions, weights, data_dict, date, portfolio_value
                )
                
                # Update tracking
                current_positions = new_positions
                current_capital = portfolio_value - trade_cost
                
                self.portfolio_values.append({
                    'date': date,
                    'value': current_capital,
                    'positions': len(current_positions)
                })
                
                # Calculate daily return
                if i > 0:
                    prev_value = self.portfolio_values[-2]['value']
                    daily_return = (current_capital - prev_value) / prev_value
                    self.daily_returns.append(daily_return)
                else:
                    self.daily_returns.append(0.0)
                
            except Exception as e:
                logger.error(f"Error in backtest loop for {date}: {e}")
                continue
        
        # Calculate performance metrics
        results = self._calculate_performance_metrics(benchmark_data)
        
        logger.info("Backtest completed successfully!")
        return results
    
    def _get_benchmark_data(self, start_date: str, end_date: str) -> pd.DataFrame:
        """Fetch benchmark data for comparison."""
        try:
            collector = DataCollector()
            benchmark_data = collector.fetch_stock_data(
                self.benchmark_symbol, start_date, end_date
            )
            return benchmark_data
        except Exception as e:
            logger.warning(f"Could not fetch benchmark data: {e}")
            return pd.DataFrame()
    
    def _calculate_portfolio_value(self, 
                                 positions: Dict[str, float],
                                 data_dict: Dict[str, pd.DataFrame],
                                 date: pd.Timestamp,
                                 cash: float) -> float:
        """Calculate current portfolio value."""
        portfolio_value = cash
        
        for symbol, shares in positions.items():
            if symbol in data_dict and date in data_dict[symbol].index:
                price = data_dict[symbol].loc[date, 'Close']
                portfolio_value += shares * price
        
        return portfolio_value
    
    def _rebalance_portfolio(self,
                           current_positions: Dict[str, float],
                           target_weights: Dict[str, float],
                           data_dict: Dict[str, pd.DataFrame],
                           date: pd.Timestamp,
                           portfolio_value: float) -> Tuple[Dict[str, float], float]:
        """Rebalance portfolio to target weights."""
        
        new_positions = {}
        total_cost = 0
        
        # Calculate target positions
        target_positions = {}
        for symbol, weight in target_weights.items():
            if symbol in data_dict and date in data_dict[symbol].index:
                price = data_dict[symbol].loc[date, 'Close']
                target_shares = (portfolio_value * weight) / price
                target_positions[symbol] = target_shares
        
        # Calculate trades and costs
        for symbol in set(current_positions.keys()) | set(target_positions.keys()):
            current_shares = current_positions.get(symbol, 0)
            target_shares = target_positions.get(symbol, 0)
            
            if symbol in data_dict and date in data_dict[symbol].index:
                price = data_dict[symbol].loc[date, 'Close']
                
                # Calculate trade
                trade_shares = target_shares - current_shares
                
                if abs(trade_shares) > 0.01:  # Minimum trade size
                    # Calculate transaction costs
                    trade_value = abs(trade_shares * price)
                    transaction_cost = trade_value * self.transaction_cost
                    slippage_cost = trade_value * self.slippage
                    total_cost += transaction_cost + slippage_cost
                    
                    # Record trade
                    self.trades.append({
                        'date': date,
                        'symbol': symbol,
                        'shares': trade_shares,
                        'price': price,
                        'value': trade_value,
                        'cost': transaction_cost + slippage_cost
                    })
                
                new_positions[symbol] = target_shares
        
        return new_positions, total_cost
    
    def _calculate_performance_metrics(self, benchmark_data: pd.DataFrame) -> Dict:
        """Calculate comprehensive performance metrics."""
        
        if not self.portfolio_values:
            return {}
        
        # Create portfolio returns series
        portfolio_df = pd.DataFrame(self.portfolio_values)
        portfolio_df.set_index('date', inplace=True)
        
        # Calculate returns - fix the index alignment issue
        if len(self.daily_returns) == len(portfolio_df.index):
            portfolio_returns = pd.Series(self.daily_returns, index=portfolio_df.index)
        else:
            # If there's a mismatch, use the shorter length
            min_length = min(len(self.daily_returns), len(portfolio_df.index))
            portfolio_returns = pd.Series(self.daily_returns[:min_length], 
                                        index=portfolio_df.index[:min_length])
        
        # Basic metrics
        total_return = (portfolio_df['value'].iloc[-1] / self.initial_capital) - 1
        annualized_return = (1 + total_return) ** (252 / len(portfolio_returns)) - 1
        volatility = portfolio_returns.std() * np.sqrt(252)
        sharpe_ratio = (annualized_return - 0.02) / volatility if volatility > 0 else 0
        
        # Risk metrics
        cumulative_returns = (1 + portfolio_returns).cumprod()
        rolling_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - rolling_max) / rolling_max
        max_drawdown = drawdown.min()
        
        # Sortino ratio
        downside_returns = portfolio_returns[portfolio_returns < 0]
        downside_deviation = downside_returns.std() * np.sqrt(252)
        sortino_ratio = (annualized_return - 0.02) / downside_deviation if downside_deviation > 0 else 0
        
        # Calmar ratio
        calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0
        
        # Win rate
        win_rate = (portfolio_returns > 0).mean()
        
        # Benchmark comparison
        benchmark_metrics = {}
        if not benchmark_data.empty:
            # Align benchmark data with portfolio returns
            common_dates = portfolio_returns.index.intersection(benchmark_data.index)
            if len(common_dates) > 0:
                benchmark_returns = benchmark_data['returns'].loc[common_dates]
                portfolio_returns_aligned = portfolio_returns.loc[common_dates]
                
                benchmark_total_return = (1 + benchmark_returns).prod() - 1
                benchmark_annualized = (1 + benchmark_total_return) ** (252 / len(benchmark_returns)) - 1
                benchmark_volatility = benchmark_returns.std() * np.sqrt(252)
                
                # Alpha and Beta
                covariance = np.cov(portfolio_returns_aligned, benchmark_returns)[0, 1]
                benchmark_variance = np.var(benchmark_returns)
                beta = covariance / benchmark_variance if benchmark_variance > 0 else 0
                alpha = annualized_return - (0.02 + beta * (benchmark_annualized - 0.02))
                
                benchmark_metrics = {
                    'benchmark_total_return': benchmark_total_return,
                    'benchmark_annualized_return': benchmark_annualized,
                    'benchmark_volatility': benchmark_volatility,
                    'alpha': alpha,
                    'beta': beta,
                    'information_ratio': (annualized_return - benchmark_annualized) / volatility if volatility > 0 else 0
                }
        
        # Trading statistics
        if self.trades:
            trade_df = pd.DataFrame(self.trades)
            total_trades = len(trade_df)
            total_volume = trade_df['value'].sum()
            avg_trade_size = trade_df['value'].mean()
            total_costs = trade_df['cost'].sum()
        else:
            total_trades = 0
            total_volume = 0
            avg_trade_size = 0
            total_costs = 0
        
        results = {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'calmar_ratio': calmar_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'total_trades': total_trades,
            'total_volume': total_volume,
            'avg_trade_size': avg_trade_size,
            'total_costs': total_costs,
            'portfolio_values': portfolio_df,
            'returns': portfolio_returns,
            'trades': self.trades,
            **benchmark_metrics
        }
        
        return results
    
    def plot_results(self, results: Dict, save_path: Optional[str] = None):
        """Create comprehensive performance visualization."""
        
        if not results:
            logger.warning("No results to plot")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Momentum Strategy Backtest Results', fontsize=16, fontweight='bold')
        
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
        
        if save_path is not None:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Performance plot saved to {save_path}")
        
        plt.show()
    
    def print_summary(self, results: Dict):
        """Print comprehensive backtest summary."""
        
        if not results:
            print("No results to display")
            return
        
        print("\n" + "="*60)
        print("MOMENTUM STRATEGY BACKTEST SUMMARY")
        print("="*60)
        
        print(f"\nPERFORMANCE METRICS:")
        print(f"Total Return: {results['total_return']:.2%}")
        print(f"Annualized Return: {results['annualized_return']:.2%}")
        print(f"Volatility: {results['volatility']:.2%}")
        print(f"Sharpe Ratio: {results['sharpe_ratio']:.3f}")
        print(f"Sortino Ratio: {results['sortino_ratio']:.3f}")
        print(f"Calmar Ratio: {results['calmar_ratio']:.3f}")
        print(f"Maximum Drawdown: {results['max_drawdown']:.2%}")
        print(f"Win Rate: {results['win_rate']:.2%}")
        
        if 'alpha' in results:
            print(f"\nBENCHMARK COMPARISON:")
            print(f"Alpha: {results['alpha']:.2%}")
            print(f"Beta: {results['beta']:.3f}")
            print(f"Information Ratio: {results['information_ratio']:.3f}")
            print(f"Benchmark Return: {results['benchmark_annualized_return']:.2%}")
        
        print(f"\nTRADING STATISTICS:")
        print(f"Total Trades: {results['total_trades']}")
        print(f"Total Volume: ${results['total_volume']:,.0f}")
        print(f"Average Trade Size: ${results['avg_trade_size']:,.0f}")
        print(f"Total Costs: ${results['total_costs']:,.0f}")
        
        print("\n" + "="*60)

def main():
    """Run comprehensive backtest demonstration."""
    
    # Initialize components
    collector = DataCollector()
    strategy = MomentumStrategy(
        lookback_periods=[20, 60, 120],
        max_positions=15,
        position_size_method='risk_parity',
        rebalance_frequency=20
    )
    backtester = Backtester(
        initial_capital=100000,
        transaction_cost=0.001,
        slippage=0.0005
    )
    
    # Fetch data for backtest
    symbols = [
        'AAPL', 'MSFT', 'GOOGL', 'TSLA', 'NVDA', 'META', 'AMZN', 'NFLX',
        'JPM', 'V', 'JNJ', 'PG', 'HD', 'MA', 'DIS', 'PYPL', 'BAC',
        'ADBE', 'CRM', 'KO', 'PEP', 'TMO', 'ABT', 'AVGO', 'WMT'
    ]
    
    print("Fetching data for backtest...")
    data_dict = collector.fetch_multiple_stocks(symbols, start_date="2021-01-01")
    
    print(f"Running backtest with {len(data_dict)} stocks...")
    results = backtester.run_backtest(
        data_dict=data_dict,
        strategy=strategy,
        start_date="2021-06-01",
        end_date="2023-12-31"
    )
    
    if results:
        backtester.print_summary(results)
        backtester.plot_results(results, save_path="backtest_results.png")
    else:
        print("Backtest failed to produce results")

if __name__ == "__main__":
    main() 