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
                 lookback_periods: List[int] = [5, 10, 20, 50],
                 rebalance_frequency: int = 5,
                 max_positions: int = 8,
                 position_size_method: str = 'kelly_criterion',
                 risk_free_rate: float = 0.02,
                 max_drawdown: float = 0.20,
                 momentum_threshold: float = 0.05,
                 volatility_threshold: float = 0.40,
                 correlation_threshold: float = 0.7,
                 use_market_regime: bool = True,
                 use_volume_confirmation: bool = True,
                 use_breakout_signals: bool = True,
                 use_stop_loss: bool = True,
                 stop_loss_pct: float = 0.05,
                 take_profit_pct: float = 0.15,
                 use_kelly_criterion: bool = True,
                 max_kelly_fraction: float = 0.25):
        """
        Advanced momentum strategy with Kelly Criterion, stop-losses, and market regime detection.
        
        Args:
            lookback_periods: Multiple timeframes for momentum calculation
            rebalance_frequency: How often to rebalance (days)
            max_positions: Maximum number of positions
            position_size_method: How to size positions (kelly_criterion, momentum_weight, equal_weight)
            risk_free_rate: Risk-free rate for calculations
            max_drawdown: Maximum allowed drawdown
            momentum_threshold: Minimum momentum to enter position
            volatility_threshold: Maximum volatility for position entry
            correlation_threshold: Maximum correlation between positions
            use_market_regime: Whether to detect and adapt to market regimes
            use_volume_confirmation: Whether to require volume confirmation
            use_breakout_signals: Whether to use breakout signals
            use_stop_loss: Whether to use stop-losses
            stop_loss_pct: Stop-loss percentage
            take_profit_pct: Take-profit percentage
            use_kelly_criterion: Whether to use Kelly Criterion for position sizing
            max_kelly_fraction: Maximum Kelly fraction (risk management)
        """
        self.lookback_periods = lookback_periods
        self.rebalance_frequency = rebalance_frequency
        self.max_positions = max_positions
        self.position_size_method = position_size_method
        self.risk_free_rate = risk_free_rate
        self.max_drawdown = max_drawdown
        
        # Strategy parameters
        self.momentum_threshold = momentum_threshold
        self.volatility_threshold = volatility_threshold
        self.correlation_threshold = correlation_threshold
        self.use_market_regime = use_market_regime
        self.use_volume_confirmation = use_volume_confirmation
        self.use_breakout_signals = use_breakout_signals
        self.use_stop_loss = use_stop_loss
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        self.use_kelly_criterion = use_kelly_criterion
        self.max_kelly_fraction = max_kelly_fraction
        
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
            
            # Create composite score with much more conservative normalization
            composite_signals = []
            for indicator in momentum_indicators:
                if indicator in signals:
                    signal_series = pd.Series(signals[indicator], index=data.index)
                    # IMPROVED: Much more conservative z-score normalization
                    z_score = (signal_series - signal_series.rolling(252).mean()) / (signal_series.rolling(252).std() + 1e-8)
                    # IMPROVED: Much tighter capping to prevent extreme values
                    z_score = np.clip(z_score, -2, 2)  # Cap at 2 standard deviations (much tighter)
                    composite_signals.append(z_score)
            
            if composite_signals:
                composite_momentum = pd.concat(composite_signals, axis=1).mean(axis=1)
                # IMPROVED: Much more conservative normalization
                composite_momentum = np.tanh(composite_momentum / 4)  # Use tanh with smaller scaling factor
                # IMPROVED: Additional capping to ensure reasonable values
                composite_momentum = np.clip(composite_momentum, -0.5, 0.5)  # Cap between -0.5 and 0.5
                signals[f'{symbol}_composite_momentum'] = composite_momentum
            
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
    
    def _calculate_kelly_criterion(self, returns: pd.Series, risk_free_rate: float = 0.02) -> float:
        """Calculate Kelly Criterion for optimal position sizing."""
        try:
            if len(returns) < 20:
                return 0.0
            
            # Calculate mean and variance of returns
            mean_return = returns.mean()
            variance = returns.var()
            
            if variance == 0:
                return 0.0
            
            # Kelly Criterion formula: f = (μ - r) / σ²
            # where μ = mean return, r = risk-free rate, σ² = variance
            kelly_fraction = (mean_return - risk_free_rate) / variance
            
            # Cap Kelly fraction for risk management
            kelly_fraction = max(0, min(kelly_fraction, self.max_kelly_fraction))
            
            return kelly_fraction
        except:
            return 0.0
    
    def _calculate_position_weights(self, selected_stocks: List[Tuple[str, float]], 
                                  signals_df: pd.DataFrame, 
                                  date: pd.Timestamp) -> Dict[str, float]:
        """Calculate position weights using various methods with Kelly Criterion."""
        
        if not selected_stocks:
            return {}
        
        if self.position_size_method == 'equal_weight':
            weight = 1.0 / len(selected_stocks)
            # Cap individual position weights
            max_weight = 0.15  # Max 15% per position
            weight = min(weight, max_weight)
            return {symbol: weight for symbol, _ in selected_stocks}
        
        elif self.position_size_method == 'momentum_weight':
            # Weight by momentum score with safety checks
            total_momentum = sum(max(0, score) for _, score in selected_stocks)  # Only positive momentum
            if total_momentum > 0:
                weights = {}
                for symbol, score in selected_stocks:
                    if score > 0:
                        weight = score / total_momentum
                        # Cap individual position weights
                        max_weight = 0.20  # Max 20% per position
                        weight = min(weight, max_weight)
                        weights[symbol] = weight
                
                # Renormalize after capping
                total_weight = sum(weights.values())
                if total_weight > 0:
                    for symbol in weights:
                        weights[symbol] /= total_weight
                
                return weights
            else:
                return {symbol: 1.0/len(selected_stocks) for symbol, _ in selected_stocks}
        
        elif self.position_size_method == 'kelly_criterion':
            # Use Kelly Criterion for optimal position sizing
            weights = {}
            total_kelly_fraction = 0
            
            for symbol, _ in selected_stocks:
                try:
                    # Get historical returns for Kelly calculation
                    returns_col = f'{symbol}_returns'
                    if returns_col in signals_df.columns:
                        # Use last 252 days of returns for Kelly calculation
                        returns = pd.Series(signals_df[returns_col].tail(252))
                        kelly_fraction = self._calculate_kelly_criterion(returns, self.risk_free_rate)
                        
                        if kelly_fraction > 0:
                            weights[symbol] = kelly_fraction
                            total_kelly_fraction += kelly_fraction
                    else:
                        # Fallback to momentum-based weight
                        momentum_col = f'{symbol}_composite_momentum'
                        if momentum_col in signals_df.columns:
                            momentum = signals_df.loc[date, momentum_col]
                            if not np.isnan(momentum) and momentum > 0:
                                weights[symbol] = momentum
                                total_kelly_fraction += momentum
                except:
                    continue
            
            if total_kelly_fraction > 0:
                # Normalize weights
                for symbol in weights:
                    weight = weights[symbol] / total_kelly_fraction
                    # Cap individual position weights
                    max_weight = 0.15  # Max 15% per position
                    weight = min(weight, max_weight)
                    weights[symbol] = weight
                
                # Renormalize after capping
                total_weight = sum(weights.values())
                if total_weight > 0:
                    for symbol in weights:
                        weights[symbol] /= total_weight
                else:
                    # Fallback to equal weight
                    return {symbol: 1.0/len(selected_stocks) for symbol, _ in selected_stocks}
            else:
                # Fallback to equal weight
                return {symbol: 1.0/len(selected_stocks) for symbol, _ in selected_stocks}
            
            return weights
        
        else:
            # Default to equal weight with caps
            weight = 1.0 / len(selected_stocks)
            max_weight = 0.15  # Max 15% per position
            weight = min(weight, max_weight)
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

    def select_stocks(self, signals_df: pd.DataFrame, date: pd.Timestamp) -> List[Tuple[str, float]]:
        """Select stocks using enhanced signals with fundamental data and market regime detection."""
        # DEBUG: Print available dates and value lookup for a few symbols
        print(f"DEBUG: select_stocks called for date: {date}")
        print(f"DEBUG: signals_df.index sample: {list(signals_df.index[:5])} ... {list(signals_df.index[-5:])}")
        symbols = [col.split('_')[0] for col in signals_df.columns if '_composite_momentum' in col]
        for debug_symbol in symbols[:3]:
            debug_col = f'{debug_symbol}_composite_momentum'
            if debug_col in signals_df.columns:
                try:
                    debug_val = signals_df.loc[date, debug_col]
                    print(f"DEBUG: signals_df.loc[{date}, {debug_col}] = {debug_val}")
                except Exception as e:
                    print(f"DEBUG: Error accessing signals_df.loc[{date}, {debug_col}]: {e}")
        
        available_stocks = []
        
        # Get all symbols from signals
        symbols = [col.split('_')[0] for col in signals_df.columns if '_composite_momentum' in col]
        
        for symbol in symbols:
            # Always process every symbol, regardless of missing columns or values
            momentum_col = f'{symbol}_composite_momentum'
            volatility_col = f'{symbol}_volatility'
            volume_spike_col = f'{symbol}_volume_spike'
            breakout_up_col = f'{symbol}_breakout_up'
            adx_col = f'{symbol}_adx'
            earnings_growth_col = f'{symbol}_earnings_growth'
            revenue_momentum_col = f'{symbol}_revenue_momentum'
            profit_margin_col = f'{symbol}_profit_margin_trend'
            market_trend_col = f'{symbol}_market_trend'
            market_timing_col = f'{symbol}_market_timing'
            sector_momentum_col = f'{symbol}_sector_momentum'
            mean_reversion_col = f'{symbol}_mean_reversion'

            # Use .get with default 0 for all signals
            momentum = signals_df.get(momentum_col, pd.Series(0, index=[date])).get(date, 0)
            volatility = signals_df.get(volatility_col, pd.Series(0, index=[date])).get(date, 0)
            volume_spike = signals_df.get(volume_spike_col, pd.Series(0, index=[date])).get(date, 0)
            breakout_up = signals_df.get(breakout_up_col, pd.Series(0, index=[date])).get(date, 0)
            adx = signals_df.get(adx_col, pd.Series(0, index=[date])).get(date, 0)
            earnings_growth = signals_df.get(earnings_growth_col, pd.Series(0, index=[date])).get(date, 0)
            revenue_momentum = signals_df.get(revenue_momentum_col, pd.Series(0, index=[date])).get(date, 0)
            profit_margin = signals_df.get(profit_margin_col, pd.Series(0, index=[date])).get(date, 0)
            market_trend = signals_df.get(market_trend_col, pd.Series(0, index=[date])).get(date, 0)
            market_timing = signals_df.get(market_timing_col, pd.Series(0, index=[date])).get(date, 0)
            sector_momentum = signals_df.get(sector_momentum_col, pd.Series(0, index=[date])).get(date, 0)
            mean_reversion = signals_df.get(mean_reversion_col, pd.Series(0, index=[date])).get(date, 0)

            # Print/log the actual value of all factors before score calculation
            def safe(x, name):
                if pd.isna(x):
                    print(f"DEBUG: {symbol} {name}=NaN, replaced with 0")
                    return 0
                return x
            momentum = safe(momentum, 'momentum')
            volatility = safe(volatility, 'volatility')
            volume_spike = safe(volume_spike, 'volume_spike')
            breakout_up = safe(breakout_up, 'breakout_up')
            adx = safe(adx, 'adx')
            earnings_growth = safe(earnings_growth, 'earnings_growth')
            revenue_momentum = safe(revenue_momentum, 'revenue_momentum')
            profit_margin = safe(profit_margin, 'profit_margin')
            market_trend = safe(market_trend, 'market_trend')
            market_timing = safe(market_timing, 'market_timing')
            sector_momentum = safe(sector_momentum, 'sector_momentum')
            mean_reversion = safe(mean_reversion, 'mean_reversion')
            # DEBUG: Print all values
            print(f"DEBUG: {symbol} momentum={momentum}, volatility={volatility}, volume_spike={volume_spike}, breakout_up={breakout_up}, adx={adx}, earnings_growth={earnings_growth}, revenue_momentum={revenue_momentum}, profit_margin={profit_margin}, market_trend={market_trend}, market_timing={market_timing}, sector_momentum={sector_momentum}, mean_reversion={mean_reversion}")
            # Score calculation as before
            score = momentum * 0.3
            if earnings_growth > 0:
                score += earnings_growth * 0.1
            if revenue_momentum > 0:
                score += revenue_momentum * 0.05
            if profit_margin > 0:
                score += profit_margin * 0.05
            if sector_momentum > 0:
                score += sector_momentum * 0.05
            if mean_reversion > 0.01:
                score += mean_reversion * 0.05
            if market_trend > 0:
                score += market_trend * 0.05
            if market_timing > 0.3:
                score += market_timing * 0.05
            if self.use_volume_confirmation and volume_spike:
                score += 0.05
            if self.use_breakout_signals and breakout_up:
                score += 0.05
            if self.use_market_regime and adx > 20:
                score += 0.05
            print(f"DEBUG: {symbol} score={score}")
            available_stocks.append((symbol, score))
        
        # Enhanced sorting and selection
        if available_stocks:
            # Sort by enhanced multi-factor score (descending)
            available_stocks.sort(key=lambda x: x[1], reverse=True)
            
            # More aggressive position sizing - take more positions
            max_positions = min(self.max_positions, 20)  # Increased to 20 positions
            selected_stocks = available_stocks[:max_positions]
            
            # Much lower minimum quality threshold
            selected_stocks = [(symbol, score) for symbol, score in selected_stocks if score > 0.02]
            
            return selected_stocks
        
        return []

    def _calculate_enhanced_signals(self, data: pd.DataFrame, symbol: str) -> Dict[str, List[float]]:
        """Calculate enhanced momentum signals with fundamental data and market regime detection."""
        
        signals = {}
        
        try:
            # 1. Enhanced Price Momentum (multiple timeframes)
            for period in self.lookback_periods:
                if len(data) > period:
                    # Price momentum
                    price_momentum = (data['Close'].iloc[-1] - data['Close'].iloc[-period-1]) / data['Close'].iloc[-period-1]
                    signals[f'{symbol}_price_momentum_{period}'] = [price_momentum] * len(data)
                    
                    # Volume-adjusted momentum
                    if 'Volume' in data.columns:
                        avg_volume = data['Volume'].rolling(period).mean().iloc[-1]
                        current_volume = data['Volume'].iloc[-1]
                        volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1
                        vol_adj_momentum = price_momentum * volume_ratio
                        signals[f'{symbol}_vol_adj_momentum_{period}'] = [vol_adj_momentum] * len(data)
                    
                    # Volume momentum
                    if 'Volume' in data.columns:
                        volume_momentum = (data['Volume'].iloc[-1] - data['Volume'].rolling(period).mean().iloc[-1]) / data['Volume'].rolling(period).mean().iloc[-1]
                        signals[f'{symbol}_volume_momentum_{period}'] = [volume_momentum] * len(data)
            
            # 2. Enhanced RSI with multiple timeframes
            for period in [14, 21]:
                if len(data) > period:
                    delta = data['Close'].diff()
                    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
                    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
                    rs = gain / loss
                    rsi = 100 - (100 / (1 + rs))
                    signals[f'{symbol}_rsi_{period}'] = rsi.fillna(50).tolist()
            
            # 3. Enhanced MACD
            if len(data) > 26:
                exp1 = data['Close'].ewm(span=12).mean()
                exp2 = data['Close'].ewm(span=26).mean()
                macd = exp1 - exp2
                signal = macd.ewm(span=9).mean()
                histogram = macd - signal
                
                signals[f'{symbol}_macd'] = macd.fillna(0).tolist()
                signals[f'{symbol}_macd_signal'] = signal.fillna(0).tolist()
                signals[f'{symbol}_macd_histogram'] = histogram.fillna(0).tolist()
                
                # MACD momentum
                macd_momentum = (macd.iloc[-1] - macd.iloc[-5]) / abs(macd.iloc[-5]) if abs(macd.iloc[-5]) > 0 else 0
                signals[f'{symbol}_macd_momentum'] = [macd_momentum] * len(data)
            
            # 4. Enhanced Bollinger Bands
            if len(data) > 20:
                sma = data['Close'].rolling(window=20).mean()
                std = data['Close'].rolling(window=20).std()
                bb_upper = sma + (std * 2)
                bb_lower = sma - (std * 2)
                bb_position = (data['Close'] - bb_lower) / (bb_upper - bb_lower)
                
                signals[f'{symbol}_bb_position'] = bb_position.fillna(0.5).tolist()
                
                # Breakout signals
                if self.use_breakout_signals:
                    breakout_up = (data['Close'] > bb_upper).astype(int)
                    breakout_down = (data['Close'] < bb_lower).astype(int)
                    signals[f'{symbol}_breakout_up'] = breakout_up.tolist()
                    signals[f'{symbol}_breakout_down'] = breakout_down.tolist()
            
            # 5. Moving Average Signals
            for period in [20, 50, 200]:
                if len(data) > period:
                    ma = data['Close'].rolling(window=period).mean()
                    ma_momentum = (data['Close'] - ma) / ma
                    signals[f'{symbol}_ma_momentum_{period}'] = ma_momentum.fillna(0).tolist()
            
            # 6. Volume Confirmation
            if self.use_volume_confirmation and 'Volume' in data.columns:
                # Volume spike detection
                volume_sma = data['Volume'].rolling(20).mean()
                volume_ratio = data['Volume'] / volume_sma
                volume_spike = (volume_ratio > 1.5).astype(int)
                signals[f'{symbol}_volume_spike'] = volume_spike.tolist()
                
                # Price-volume trend
                pvt = ((data['Close'] - data['Close'].shift(1)) / data['Close'].shift(1) * data['Volume']).cumsum()
                signals[f'{symbol}_pvt'] = pvt.fillna(0).tolist()
            
            # 7. Market Regime Detection
            if self.use_market_regime:
                # Trend strength
                adx = self._calculate_adx(data)
                signals[f'{symbol}_adx'] = adx
                
                # Volatility regime
                volatility = data['returns'].rolling(20).std()
                high_vol = (volatility > volatility.quantile(0.8)).astype(int)
                signals[f'{symbol}_high_volatility'] = high_vol.tolist()
                
                # Market trend detection
                market_trend = self._detect_market_trend(data)
                signals[f'{symbol}_market_trend'] = market_trend
            
            # 8. NEW: Fundamental Signals (simulated)
            # In real implementation, you'd use actual fundamental data
            earnings_growth = self._simulate_earnings_growth(data)
            revenue_momentum = self._simulate_revenue_momentum(data)
            profit_margin_trend = self._simulate_profit_margin_trend(data)
            
            signals[f'{symbol}_earnings_growth'] = earnings_growth
            signals[f'{symbol}_revenue_momentum'] = revenue_momentum
            signals[f'{symbol}_profit_margin_trend'] = profit_margin_trend
            
            # 9. NEW: Earnings Momentum (simulated)
            earnings_momentum = self._simulate_earnings_momentum(data)
            signals[f'{symbol}_earnings_momentum'] = earnings_momentum
            
            # 10. NEW: Sector Rotation Signals
            sector_momentum = self._calculate_sector_momentum(symbol, data)
            signals[f'{symbol}_sector_momentum'] = sector_momentum
            
            # 11. NEW: Mean Reversion Signals
            mean_reversion = self._calculate_mean_reversion_signals(data)
            signals[f'{symbol}_mean_reversion'] = mean_reversion
            
            # 12. NEW: Market Timing Signals
            market_timing = self._calculate_market_timing_signals(data)
            signals[f'{symbol}_market_timing'] = market_timing
            
            # 13. Enhanced Composite Momentum Score
            momentum_indicators = [key for key in signals.keys() if 'momentum' in key.lower() and not key.endswith('_momentum')]
            
            if momentum_indicators:
                composite_signals = []
                for indicator in momentum_indicators:
                    try:
                        signal_series = pd.Series(signals[indicator], index=data.index)
                        # Skip if all values are NaN
                        if signal_series.isna().all():
                            continue
                        # Simple normalization instead of z-score
                        signal_series = signal_series.fillna(0)
                        # Clip extreme values
                        signal_series = np.clip(signal_series, -0.5, 0.5)
                        composite_signals.append(signal_series)
                    except Exception as e:
                        logger.warning(f"Error processing indicator {indicator}: {e}")
                        continue
                
                if composite_signals:
                    composite_momentum = pd.concat(composite_signals, axis=1).mean(axis=1)
                    # Simple normalization
                    composite_momentum = np.clip(composite_momentum, -0.6, 0.6)
                    signals[f'{symbol}_composite_momentum'] = composite_momentum.tolist()
                else:
                    # Fallback: use simple price momentum if no composite signals
                    price_momentum = data['Close'].pct_change(20).fillna(0)
                    signals[f'{symbol}_composite_momentum'] = price_momentum.tolist()
            else:
                # Fallback: use simple price momentum if no momentum indicators
                price_momentum = data['Close'].pct_change(20).fillna(0)
                signals[f'{symbol}_composite_momentum'] = price_momentum.tolist()
            
            # 14. Risk metrics
            signals[f'{symbol}_volatility'] = data['volatility_20']
            signals[f'{symbol}_max_drawdown'] = self._calculate_max_drawdown(data['Close'])
            signals[f'{symbol}_sharpe_ratio'] = self._calculate_sharpe_ratio(data['returns'])
            
            return signals
            
        except Exception as e:
            logger.error(f"Error calculating enhanced signals for {symbol}: {e}")
            return {}
    
    def _simulate_earnings_growth(self, data: pd.DataFrame) -> List[float]:
        """Simulate earnings growth based on price action and volume."""
        try:
            # Simulate earnings growth using price momentum and volume
            price_momentum = data['Close'].pct_change(60)  # 3-month momentum
            volume_trend = data['Volume'].pct_change(60) if 'Volume' in data.columns else pd.Series(0, index=data.index)
            
            # Combine price and volume trends
            earnings_growth = price_momentum * 0.8 + volume_trend * 0.2
            earnings_growth = earnings_growth.fillna(0)
            
            # Add quarterly earnings cycle simulation
            np.random.seed(42)  # For reproducibility
            quarterly_cycle = np.random.normal(0, 0.05, len(data))
            earnings_growth = earnings_growth + quarterly_cycle
            
            return earnings_growth.tolist()
        except:
            return [0] * len(data)
    
    def _simulate_revenue_momentum(self, data: pd.DataFrame) -> List[float]:
        """Simulate revenue momentum based on price action."""
        try:
            # Simulate revenue momentum using price action
            price_momentum = data['Close'].pct_change(40)  # 2-month momentum
            volume_momentum = data['Volume'].pct_change(40) if 'Volume' in data.columns else pd.Series(0, index=data.index)
            
            # Revenue momentum is more stable than earnings
            revenue_momentum = price_momentum * 0.6 + volume_momentum * 0.4
            revenue_momentum = revenue_momentum.fillna(0)
            
            # Add some stability
            revenue_momentum = revenue_momentum.rolling(10).mean()
            
            return revenue_momentum.fillna(0).tolist()
        except:
            return [0] * len(data)
    
    def _simulate_profit_margin_trend(self, data: pd.DataFrame) -> List[float]:
        """Simulate profit margin trends."""
        try:
            # Simulate profit margin trends using price vs volume relationship
            price_trend = data['Close'].pct_change(30)
            volume_trend = data['Volume'].pct_change(30) if 'Volume' in data.columns else pd.Series(0, index=data.index)
            
            # Profit margins improve when price increases more than volume
            profit_margin_trend = price_trend - volume_trend * 0.3
            profit_margin_trend = profit_margin_trend.fillna(0)
            
            # Smooth the trend
            profit_margin_trend = profit_margin_trend.rolling(15).mean()
            
            return profit_margin_trend.fillna(0).tolist()
        except:
            return [0] * len(data)
    
    def _detect_market_trend(self, data: pd.DataFrame) -> List[float]:
        """Detect overall market trend."""
        try:
            # Use multiple moving averages to detect trend
            sma_20 = data['Close'].rolling(20).mean()
            sma_50 = data['Close'].rolling(50).mean()
            sma_200 = data['Close'].rolling(200).mean()
            
            # Trend strength
            trend_20_50 = (sma_20 - sma_50) / sma_50
            trend_50_200 = (sma_50 - sma_200) / sma_200
            
            # Combined trend signal
            market_trend = trend_20_50 * 0.6 + trend_50_200 * 0.4
            market_trend = market_trend.fillna(0)
            
            return market_trend.tolist()
        except:
            return [0] * len(data)
    
    def _calculate_market_timing_signals(self, data: pd.DataFrame) -> List[float]:
        """Calculate market timing signals."""
        try:
            # Market timing based on multiple factors
            price = data['Close']
            
            # 1. Moving average crossover
            sma_20 = price.rolling(20).mean()
            sma_50 = price.rolling(50).mean()
            ma_crossover = (sma_20 > sma_50).astype(int)
            
            # 2. Price vs moving average
            price_vs_ma = (price > sma_20).astype(int)
            
            # 3. Momentum vs mean reversion
            momentum = price.pct_change(10)
            mean_reversion = -price.pct_change(5)  # Inverse of short-term momentum
            
            # Combine signals
            market_timing = ma_crossover * 0.4 + price_vs_ma * 0.3 + (momentum > 0).astype(int) * 0.3
            market_timing = market_timing.fillna(0)
            
            return market_timing.tolist()
        except:
            return [0] * len(data)
    
    def _simulate_earnings_momentum(self, data: pd.DataFrame) -> List[float]:
        """Simulate earnings momentum based on price action and volume."""
        try:
            # Simulate earnings momentum using price action and volume
            price_momentum = data['Close'].pct_change(20)
            volume_momentum = data['Volume'].pct_change(20) if 'Volume' in data.columns else pd.Series(0, index=data.index)
            
            # Combine price and volume momentum
            earnings_momentum = price_momentum * 0.7 + volume_momentum * 0.3
            earnings_momentum = earnings_momentum.fillna(0)
            
            # Add some randomness to simulate earnings surprises
            np.random.seed(42)  # For reproducibility
            earnings_surprise = np.random.normal(0, 0.02, len(data))
            earnings_momentum = earnings_momentum + earnings_surprise
            
            return earnings_momentum.tolist()
        except:
            return [0] * len(data)
    
    def _calculate_sector_momentum(self, symbol: str, data: pd.DataFrame) -> List[float]:
        """Calculate sector rotation momentum."""
        try:
            # Define sector groups (simplified)
            tech_stocks = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'NVDA', 'META', 'AMZN', 'NFLX', 'ADBE', 'CRM', 'CSCO', 'IBM', 'INTC', 'QCOM', 'TXN', 'AMD', 'MU', 'INTU', 'ORCL', 'SAP']
            financial_stocks = ['JPM', 'V', 'MA', 'BAC', 'GS', 'WFC', 'C', 'MS', 'BLK', 'AXP']
            healthcare_stocks = ['JNJ', 'UNH', 'PFE', 'ABT', 'TMO', 'MRK', 'BMY', 'AMGN', 'GILD', 'CVS']
            consumer_stocks = ['PG', 'HD', 'DIS', 'KO', 'PEP', 'WMT', 'COST', 'TGT', 'MCD', 'SBUX']
            
            # Determine sector
            sector = None
            if symbol in tech_stocks:
                sector = 'tech'
            elif symbol in financial_stocks:
                sector = 'financial'
            elif symbol in healthcare_stocks:
                sector = 'healthcare'
            elif symbol in consumer_stocks:
                sector = 'consumer'
            else:
                sector = 'other'
            
            # Calculate sector momentum (simplified - in real implementation, use sector ETFs)
            if sector == 'tech':
                # Tech sector momentum (higher volatility)
                sector_momentum = data['Close'].pct_change(10) * 1.2
            elif sector == 'financial':
                # Financial sector momentum (lower volatility)
                sector_momentum = data['Close'].pct_change(15) * 0.8
            elif sector == 'healthcare':
                # Healthcare sector momentum (defensive)
                sector_momentum = data['Close'].pct_change(20) * 0.6
            elif sector == 'consumer':
                # Consumer sector momentum (moderate)
                sector_momentum = data['Close'].pct_change(12) * 1.0
            else:
                # Other sectors
                sector_momentum = data['Close'].pct_change(14) * 0.9
            
            sector_momentum = sector_momentum.fillna(0)
            return sector_momentum.tolist()
        except:
            return [0] * len(data)
    
    def _calculate_mean_reversion_signals(self, data: pd.DataFrame) -> List[float]:
        """Calculate mean reversion signals."""
        try:
            # Calculate Bollinger Band position
            sma = data['Close'].rolling(window=20).mean()
            std = data['Close'].rolling(window=20).std()
            bb_position = (data['Close'] - sma) / (std * 2)
            
            # Mean reversion signal (negative when overbought, positive when oversold)
            mean_reversion = -bb_position  # Inverse of BB position
            
            # Add RSI mean reversion
            delta = data['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            
            # RSI mean reversion (positive when oversold, negative when overbought)
            rsi_reversion = (50 - rsi) / 50
            
            # Combine signals
            combined_reversion = mean_reversion * 0.6 + rsi_reversion * 0.4
            combined_reversion = combined_reversion.fillna(0)
            
            return combined_reversion.tolist()
        except:
            return [0] * len(data)
    
    def _calculate_adx(self, data: pd.DataFrame, period: int = 14) -> List[float]:
        """Calculate Average Directional Index (ADX) for trend strength."""
        try:
            high = data['High']
            low = data['Low']
            close = data['Close']
            
            # Calculate True Range
            tr1 = high - low
            tr2 = abs(high - close.shift(1))
            tr3 = abs(low - close.shift(1))
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            
            # Calculate Directional Movement
            up_move = high - high.shift(1)
            down_move = low.shift(1) - low
            
            plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
            minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
            
            # Smooth the values
            tr_smooth = tr.rolling(period).mean()
            plus_di = pd.Series(plus_dm).rolling(period).mean() / tr_smooth * 100
            minus_di = pd.Series(minus_dm).rolling(period).mean() / tr_smooth * 100
            
            # Calculate ADX
            dx = abs(plus_di - minus_di) / (plus_di + minus_di) * 100
            adx = dx.rolling(period).mean()
            
            return adx.fillna(0).tolist()
        except:
            return [0] * len(data)

def main():
    """Demonstrate the momentum strategy."""
    from data_collector import DataCollector
    
    # Initialize components
    collector = DataCollector()
    strategy = MomentumStrategy(
        lookback_periods=[5, 10, 20, 50],
        max_positions=8,
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
<<<<<<< HEAD
    main() 
=======
    main() 
>>>>>>> b6f4879 (Add debug output for index alignment and value lookup in stock selection)
