import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
from scipy import stats
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class MomentumStrategy:
    """
    Advanced momentum trading strategy with multiple signal generation methods.
    Implements cross-sectional and time-series momentum with risk management.
    """
    
    def __init__(self, 
                 lookback_periods: List[int] = [20, 60, 120],
                 rebalance_frequency: int = 20,
                 max_positions: int = 20,
                 position_size_method: str = 'equal_weight',
                 risk_free_rate: float = 0.02,
                 max_drawdown: float = 0.30):
        
        self.lookback_periods = lookback_periods
        self.rebalance_frequency = rebalance_frequency
        self.max_positions = max_positions
        self.position_size_method = position_size_method
        self.risk_free_rate = risk_free_rate
        self.max_drawdown = max_drawdown
        
        # Strategy parameters
        self.momentum_threshold = 0.01
        self.volatility_threshold = 0.8
        self.correlation_threshold = 0.8
        
    def calculate_momentum_signals(self, data_dict: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Calculate comprehensive momentum signals for all stocks.
        
        Args:
            data_dict: Dictionary of stock data
            
        Returns:
            DataFrame with momentum signals for each stock
        """
        signals = []
        
        for symbol, data in data_dict.items():
            if len(data) < max(self.lookback_periods):
                continue
                
            signal_data = self._calculate_stock_signals(data, symbol)
            if signal_data is not None:
                signals.append(signal_data)
        
        if not signals:
            return pd.DataFrame()
        
        signals_df = pd.concat(signals, axis=1)
        return signals_df
    
    def _calculate_stock_signals(self, data: pd.DataFrame, symbol: str) -> Optional[pd.DataFrame]:
        """Calculate momentum signals for a single stock."""
        try:
            # Ensure we have enough data
            if len(data) < max(self.lookback_periods):
                return None
            
            # Calculate multiple momentum indicators
            signals = {}
            
            # 1. Price momentum (relative strength)
            for period in self.lookback_periods:
                signals[f'{symbol}_price_momentum_{period}'] = data['Close'].pct_change(period)
            
            # 2. Volatility-adjusted momentum
            for period in self.lookback_periods:
                returns = data['returns'].rolling(window=period).mean()
                volatility = data['returns'].rolling(window=period).std()
                signals[f'{symbol}_vol_adj_momentum_{period}'] = returns / (volatility + 1e-8)
            
            # 3. Volume-weighted momentum
            for period in self.lookback_periods:
                volume_weight = data['volume_ratio'].rolling(window=period).mean()
                price_momentum = data['Close'].pct_change(period)
                signals[f'{symbol}_volume_momentum_{period}'] = price_momentum * volume_weight
            
            # 4. RSI-based momentum
            signals[f'{symbol}_rsi_momentum'] = (data['rsi'] - 50) / 50
            
            # 5. MACD momentum
            signals[f'{symbol}_macd_momentum'] = data['macd_histogram'] / data['Close']
            
            # 6. Bollinger Band momentum
            bb_position = (data['Close'] - data['bb_lower']) / (data['bb_upper'] - data['bb_lower'])
            signals[f'{symbol}_bb_momentum'] = bb_position - 0.5
            
            # 7. Moving average momentum
            for period in [20, 50, 200]:
                ma_ratio = data['Close'] / data[f'sma_{period}'] - 1
                signals[f'{symbol}_ma_momentum_{period}'] = ma_ratio
            
            # 8. Composite momentum score
            momentum_indicators = [
                f'{symbol}_price_momentum_20',
                f'{symbol}_vol_adj_momentum_20',
                f'{symbol}_volume_momentum_20',
                f'{symbol}_rsi_momentum',
                f'{symbol}_macd_momentum',
                f'{symbol}_bb_momentum',
                f'{symbol}_ma_momentum_20'
            ]
            
            # Create composite score (z-score weighted average)
            composite_signals = []
            for indicator in momentum_indicators:
                if indicator in signals:
                    signal_series = pd.Series(signals[indicator], index=data.index)
                    # Z-score normalization
                    z_score = (signal_series - signal_series.rolling(252).mean()) / (signal_series.rolling(252).std() + 1e-8)
                    composite_signals.append(z_score)
            
            if composite_signals:
                signals[f'{symbol}_composite_momentum'] = pd.concat(composite_signals, axis=1).mean(axis=1)
            
            # 9. Risk metrics
            signals[f'{symbol}_volatility'] = data['volatility_20']
            close_series = pd.Series(data['Close'].values, index=data.index)
            returns_series = pd.Series(data['returns'].values, index=data.index)
            signals[f'{symbol}_max_drawdown'] = self._calculate_max_drawdown(close_series)
            signals[f'{symbol}_sharpe_ratio'] = self._calculate_sharpe_ratio(returns_series)
            
            return pd.DataFrame(signals, index=data.index)
            
        except Exception as e:
            logger.error(f"Error calculating signals for {symbol}: {e}")
            return None
    
    def _calculate_max_drawdown(self, prices: pd.Series) -> pd.Series:
        """Calculate rolling maximum drawdown."""
        rolling_max = prices.expanding().max()
        drawdown = (prices - rolling_max) / rolling_max
        return drawdown.rolling(window=252).min()
    
    def _calculate_sharpe_ratio(self, returns: pd.Series) -> pd.Series:
        """Calculate rolling Sharpe ratio."""
        excess_returns = returns - self.risk_free_rate / 252
        rolling_mean = excess_returns.rolling(window=252).mean()
        rolling_std = excess_returns.rolling(window=252).std()
        return rolling_mean / (rolling_std + 1e-8)
    
    def generate_portfolio_signals(self, signals_df: pd.DataFrame, 
                                 date: pd.Timestamp) -> Dict[str, float]:
        """
        Generate portfolio allocation signals based on momentum rankings.
        
        Args:
            signals_df: DataFrame with momentum signals
            date: Current date for signal generation
            
        Returns:
            Dictionary mapping symbols to position weights
        """
        try:
            # Get signals for the current date
            current_signals = signals_df.loc[date]
            
            # Extract composite momentum scores
            momentum_scores = {}
            for col in current_signals.index:
                if col.endswith('_composite_momentum'):
                    symbol = col.replace('_composite_momentum', '')
                    momentum_scores[symbol] = current_signals[col]
            
            if not momentum_scores:
                return {}
            
            # Filter stocks based on momentum threshold
            qualified_stocks = {
                symbol: score for symbol, score in momentum_scores.items()
                if score > self.momentum_threshold and not np.isnan(score)
            }
            
            if not qualified_stocks:
                return {}
            
            # Rank stocks by momentum score
            ranked_stocks = sorted(qualified_stocks.items(), 
                                 key=lambda x: x[1], reverse=True)
            
            # Select top stocks
            selected_stocks = ranked_stocks[:self.max_positions]
            
            # Calculate position weights
            weights = self._calculate_position_weights(selected_stocks, signals_df, date)
            
            return weights
            
        except Exception as e:
            logger.error(f"Error generating portfolio signals: {e}")
            return {}
    
    def _calculate_position_weights(self, selected_stocks: List[Tuple[str, float]], 
                                  signals_df: pd.DataFrame, 
                                  date: pd.Timestamp) -> Dict[str, float]:
        """Calculate position weights using various methods."""
        
        if self.position_size_method == 'equal_weight':
            weight = 1.0 / len(selected_stocks)
            return {symbol: weight for symbol, _ in selected_stocks}
        
        elif self.position_size_method == 'momentum_weight':
            # Weight by momentum score
            total_momentum = sum(score for _, score in selected_stocks)
            weights = {}
            for symbol, score in selected_stocks:
                weights[symbol] = score / total_momentum
            return weights
        
        elif self.position_size_method == 'risk_parity':
            # Risk parity weighting based on volatility
            weights = {}
            total_risk = 0
            
            for symbol, _ in selected_stocks:
                vol_col = f'{symbol}_volatility'
                if vol_col in signals_df.columns:
                    volatility = signals_df.loc[date, vol_col]
                    if not np.isnan(volatility) and volatility > 0:
                        risk_weight = 1 / volatility
                        weights[symbol] = risk_weight
                        total_risk += risk_weight
            
            if total_risk > 0:
                weights = {symbol: weight / total_risk for symbol, weight in weights.items()}
            
            return weights
        
        elif self.position_size_method == 'kelly_criterion':
            # Kelly criterion for position sizing
            weights = {}
            for symbol, momentum_score in selected_stocks:
                # Estimate win rate and average return from momentum score
                win_rate = 0.5 + (momentum_score * 0.3)  # Momentum as win rate proxy
                avg_return = momentum_score * 0.1  # Momentum as return proxy
                
                if avg_return > 0:
                    kelly_fraction = (win_rate * avg_return - (1 - win_rate) * avg_return) / avg_return
                    weights[symbol] = max(0, min(kelly_fraction, 0.25))  # Cap at 25%
                else:
                    weights[symbol] = 0
            
            # Normalize weights
            total_weight = sum(weights.values())
            if total_weight > 0:
                weights = {symbol: weight / total_weight for symbol, weight in weights.items()}
            
            return weights
        
        else:
            # Default to equal weight
            weight = 1.0 / len(selected_stocks)
            return {symbol: weight for symbol, _ in selected_stocks}
    
    def apply_risk_management(self, weights: Dict[str, float], 
                            signals_df: pd.DataFrame, 
                            date: pd.Timestamp) -> Dict[str, float]:
        """
        Apply risk management rules to position weights.
        
        Args:
            weights: Current position weights
            signals_df: DataFrame with risk metrics
            date: Current date
            
        Returns:
            Adjusted position weights
        """
        adjusted_weights = weights.copy()
        
        # 1. Volatility filtering
        for symbol in list(adjusted_weights.keys()):
            vol_col = f'{symbol}_volatility'
            if vol_col in signals_df.columns:
                volatility = signals_df.loc[date, vol_col]
                if not np.isnan(volatility) and volatility > self.volatility_threshold:
                    logger.info(f"Removing {symbol} due to high volatility: {volatility:.3f}")
                    del adjusted_weights[symbol]
        
        # 2. Maximum drawdown filtering
        for symbol in list(adjusted_weights.keys()):
            dd_col = f'{symbol}_max_drawdown'
            if dd_col in signals_df.columns:
                max_dd = signals_df.loc[date, dd_col]
                if not np.isnan(max_dd) and max_dd < -self.max_drawdown:
                    logger.info(f"Removing {symbol} due to high drawdown: {max_dd:.3f}")
                    del adjusted_weights[symbol]
        
        # 3. Correlation-based diversification
        if len(adjusted_weights) > 1:
            # Calculate correlation matrix for selected stocks
            returns_data = {}
            for symbol in adjusted_weights.keys():
                returns_col = f'{symbol}_returns'
                if returns_col in signals_df.columns:
                    returns_data[symbol] = signals_df[returns_col].rolling(60).mean()
            
            if len(returns_data) > 1:
                returns_df = pd.DataFrame(returns_data)
                corr_matrix = returns_df.corr()
                
                # Remove highly correlated positions
                high_corr_pairs = []
                for i in range(len(corr_matrix.columns)):
                    for j in range(i+1, len(corr_matrix.columns)):
                        if abs(corr_matrix.iloc[i, j]) > self.correlation_threshold:
                            high_corr_pairs.append((corr_matrix.columns[i], corr_matrix.columns[j]))
                
                # Keep the stock with higher momentum in correlated pairs
                for stock1, stock2 in high_corr_pairs:
                    if stock1 in adjusted_weights and stock2 in adjusted_weights:
                        momentum1 = adjusted_weights.get(f'{stock1}_composite_momentum', 0)
                        momentum2 = adjusted_weights.get(f'{stock2}_composite_momentum', 0)
                        
                        if momentum1 < momentum2:
                            del adjusted_weights[stock1]
                        else:
                            del adjusted_weights[stock2]
        
        # 4. Re-normalize weights
        total_weight = sum(adjusted_weights.values())
        if total_weight > 0:
            adjusted_weights = {symbol: weight / total_weight for symbol, weight in adjusted_weights.items()}
        
        return adjusted_weights
    
    def calculate_performance_metrics(self, portfolio_returns: pd.Series) -> Dict[str, float]:
        """Calculate comprehensive performance metrics."""
        metrics = {}
        
        # Basic metrics
        metrics['total_return'] = (1 + portfolio_returns).prod() - 1
        metrics['annualized_return'] = (1 + metrics['total_return']) ** (252 / len(portfolio_returns)) - 1
        metrics['volatility'] = portfolio_returns.std() * np.sqrt(252)
        metrics['sharpe_ratio'] = (metrics['annualized_return'] - self.risk_free_rate) / metrics['volatility']
        
        # Risk metrics
        cumulative_returns = (1 + portfolio_returns).cumprod()
        rolling_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - rolling_max) / rolling_max
        metrics['max_drawdown'] = drawdown.min()
        
        # Sortino ratio
        downside_returns = portfolio_returns[portfolio_returns < 0]
        downside_deviation = downside_returns.std() * np.sqrt(252)
        metrics['sortino_ratio'] = (metrics['annualized_return'] - self.risk_free_rate) / downside_deviation
        
        # Calmar ratio
        metrics['calmar_ratio'] = metrics['annualized_return'] / abs(metrics['max_drawdown'])
        
        # Win rate
        metrics['win_rate'] = (portfolio_returns > 0).mean()
        
        # Average win/loss
        wins = portfolio_returns[portfolio_returns > 0]
        losses = portfolio_returns[portfolio_returns < 0]
        metrics['avg_win'] = wins.mean() if len(wins) > 0 else 0
        metrics['avg_loss'] = losses.mean() if len(losses) > 0 else 0
        metrics['profit_factor'] = abs(metrics['avg_win'] / metrics['avg_loss']) if metrics['avg_loss'] != 0 else float('inf')
        
        return metrics

def main():
    """Demonstrate the momentum strategy."""
    from data_collector import DataCollector
    
    # Initialize components
    collector = DataCollector()
    strategy = MomentumStrategy(
        lookback_periods=[20, 60, 120],
        max_positions=10,
        position_size_method='risk_parity'
    )
    
    # Fetch sample data
    symbols = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'NVDA', 'META', 'AMZN', 'NFLX']
    data_dict = collector.fetch_multiple_stocks(symbols, start_date="2022-01-01")
    
    # Calculate momentum signals
    signals_df = strategy.calculate_momentum_signals(data_dict)
    
    if not signals_df.empty:
        print("Momentum signals calculated successfully!")
        print(f"Signal matrix shape: {signals_df.shape}")
        
        # Generate portfolio signals for a sample date
        sample_date = pd.Timestamp(signals_df.index[-1])
        weights = strategy.generate_portfolio_signals(signals_df, sample_date)
        
        print(f"\nPortfolio weights for {sample_date}:")
        for symbol, weight in weights.items():
            print(f"{symbol}: {weight:.3f}")

if __name__ == "__main__":
    main() 