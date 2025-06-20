import pandas as pd
import numpy as np
import yfinance as yf
import time
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import json
import os
from threading import Thread, Event
import schedule

from data_collector import DataCollector
from momentum_strategy import MomentumStrategy

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class LiveTrader:
    """
    Live trading system for momentum strategy with real-time monitoring.
    Note: This is a simulation framework. For actual trading, integrate with broker APIs.
    """
    
    def __init__(self,
                 symbols: List[str],
                 initial_capital: float = 100000,
                 max_positions: int = 15,
                 rebalance_frequency: int = 20,
                 risk_management: bool = True,
                 paper_trading: bool = True):
        
        self.symbols = symbols
        self.initial_capital = initial_capital
        self.max_positions = max_positions
        self.rebalance_frequency = rebalance_frequency
        self.risk_management = risk_management
        self.paper_trading = paper_trading
        
        # Initialize components
        self.collector = DataCollector()
        self.strategy = MomentumStrategy(
            lookback_periods=[20, 60, 120],
            max_positions=max_positions,
            position_size_method='risk_parity'
        )
        
        # Trading state
        self.current_positions = {}
        self.portfolio_value = initial_capital
        self.cash = initial_capital
        self.trade_history = []
        self.performance_history = []
        
        # Real-time data
        self.current_prices = {}
        self.signals_df = pd.DataFrame()
        
        # Control flags
        self.is_running = False
        self.stop_event = Event()
        
        # Create data directory
        os.makedirs("trading_data", exist_ok=True)
        
    def start_trading(self):
        """Start the live trading system."""
        logger.info("Starting live trading system...")
        self.is_running = True
        self.stop_event.clear()
        
        # Schedule trading tasks
        schedule.every().day.at("09:30").do(self.market_open)
        schedule.every().day.at("16:00").do(self.market_close)
        schedule.every(self.rebalance_frequency).minutes.do(self.rebalance_portfolio)
        schedule.every(5).minutes.do(self.update_prices)
        schedule.every().hour.do(self.save_performance)
        
        # Start monitoring thread
        monitor_thread = Thread(target=self._monitor_loop)
        monitor_thread.daemon = True
        monitor_thread.start()
        
        logger.info("Live trading system started successfully!")
        
    def stop_trading(self):
        """Stop the live trading system."""
        logger.info("Stopping live trading system...")
        self.is_running = False
        self.stop_event.set()
        
        # Close all positions if needed
        if not self.paper_trading:
            self._close_all_positions()
        
        self.save_performance()
        logger.info("Live trading system stopped.")
    
    def _monitor_loop(self):
        """Main monitoring loop for the trading system."""
        while self.is_running and not self.stop_event.is_set():
            try:
                schedule.run_pending()
                time.sleep(60)  # Check every minute
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(60)
    
    def market_open(self):
        """Handle market open procedures."""
        logger.info("Market opened - initializing trading session...")
        
        # Fetch latest data
        self._update_market_data()
        
        # Generate initial signals
        self._generate_signals()
        
        # Check for trading opportunities
        self._check_trading_signals()
        
        logger.info("Market open procedures completed.")
    
    def market_close(self):
        """Handle market close procedures."""
        logger.info("Market closed - finalizing trading session...")
        
        # Update final prices
        self._update_prices()
        
        # Calculate end-of-day performance
        self._calculate_daily_performance()
        
        # Save trading data
        self.save_performance()
        
        logger.info("Market close procedures completed.")
    
    def rebalance_portfolio(self):
        """Rebalance portfolio based on current signals."""
        if not self.is_running:
            return
        
        try:
            logger.info("Rebalancing portfolio...")
            
            # Update market data
            self._update_market_data()
            
            # Generate new signals
            self._generate_signals()
            
            # Execute rebalancing
            self._execute_rebalancing()
            
            logger.info("Portfolio rebalancing completed.")
            
        except Exception as e:
            logger.error(f"Error during portfolio rebalancing: {e}")
    
    def update_prices(self):
        """Update current market prices."""
        try:
            self._update_prices()
        except Exception as e:
            logger.error(f"Error updating prices: {e}")
    
    def _update_market_data(self):
        """Fetch latest market data for all symbols."""
        try:
            # Fetch data for the last 6 months to ensure enough history
            end_date = datetime.now()
            start_date = end_date - timedelta(days=180)
            
            data_dict = self.collector.fetch_multiple_stocks(
                self.symbols, 
                start_date=start_date.strftime("%Y-%m-%d"),
                end_date=end_date.strftime("%Y-%m-%d")
            )
            
            # Calculate momentum signals
            self.signals_df = self.strategy.calculate_momentum_signals(data_dict)
            
            # Update current prices
            self._update_prices()
            
        except Exception as e:
            logger.error(f"Error updating market data: {e}")
    
    def _update_prices(self):
        """Update current prices for all symbols."""
        try:
            for symbol in self.symbols:
                ticker = yf.Ticker(symbol)
                current_price = ticker.history(period="1d")['Close'].iloc[-1]
                self.current_prices[symbol] = current_price
                
        except Exception as e:
            logger.error(f"Error updating prices: {e}")
    
    def _generate_signals(self):
        """Generate trading signals based on current data."""
        if self.signals_df.empty:
            logger.warning("No signals data available")
            return
        
        try:
            # Get latest signals
            latest_date = self.signals_df.index[-1]
            current_signals = self.signals_df.loc[latest_date]
            
            # Extract momentum scores
            momentum_scores = {}
            for col in current_signals.index:
                if col.endswith('_composite_momentum'):
                    symbol = col.replace('_composite_momentum', '')
                    momentum_scores[symbol] = current_signals[col]
            
            # Rank stocks by momentum
            ranked_stocks = sorted(
                momentum_scores.items(), 
                key=lambda x: x[1], 
                reverse=True
            )
            
            # Select top stocks
            self.selected_stocks = ranked_stocks[:self.max_positions]
            
            logger.info(f"Generated signals for {len(self.selected_stocks)} stocks")
            
        except Exception as e:
            logger.error(f"Error generating signals: {e}")
    
    def _check_trading_signals(self):
        """Check for new trading opportunities."""
        if not hasattr(self, 'selected_stocks'):
            return
        
        try:
            # Calculate target weights
            target_weights = self.strategy.generate_portfolio_signals(
                self.signals_df, 
                self.signals_df.index[-1]
            )
            
            # Apply risk management
            if self.risk_management:
                target_weights = self.strategy.apply_risk_management(
                    target_weights, 
                    self.signals_df, 
                    self.signals_df.index[-1]
                )
            
            # Execute trades
            self._execute_trades(target_weights)
            
        except Exception as e:
            logger.error(f"Error checking trading signals: {e}")
    
    def _execute_rebalancing(self):
        """Execute portfolio rebalancing."""
        if not hasattr(self, 'selected_stocks'):
            return
        
        try:
            # Calculate target weights
            target_weights = self.strategy.generate_portfolio_signals(
                self.signals_df, 
                self.signals_df.index[-1]
            )
            
            # Apply risk management
            if self.risk_management:
                target_weights = self.strategy.apply_risk_management(
                    target_weights, 
                    self.signals_df, 
                    self.signals_df.index[-1]
                )
            
            # Execute rebalancing trades
            self._execute_trades(target_weights, rebalancing=True)
            
        except Exception as e:
            logger.error(f"Error executing rebalancing: {e}")
    
    def _execute_trades(self, target_weights: Dict[str, float], rebalancing: bool = False):
        """Execute trades to achieve target weights."""
        try:
            # Calculate current portfolio value
            current_value = self._calculate_portfolio_value()
            
            # Calculate target positions
            target_positions = {}
            for symbol, weight in target_weights.items():
                if symbol in self.current_prices:
                    price = self.current_prices[symbol]
                    target_shares = (current_value * weight) / price
                    target_positions[symbol] = target_shares
            
            # Calculate required trades
            trades = []
            for symbol in set(self.current_positions.keys()) | set(target_positions.keys()):
                current_shares = self.current_positions.get(symbol, 0)
                target_shares = target_positions.get(symbol, 0)
                
                if abs(target_shares - current_shares) > 0.01:  # Minimum trade size
                    trade_shares = target_shares - current_shares
                    price = self.current_prices.get(symbol, 0)
                    
                    if price > 0:
                        trade = {
                            'timestamp': datetime.now(),
                            'symbol': symbol,
                            'shares': trade_shares,
                            'price': price,
                            'value': abs(trade_shares * price),
                            'type': 'rebalance' if rebalancing else 'signal',
                            'paper_trading': self.paper_trading
                        }
                        trades.append(trade)
            
            # Execute trades
            for trade in trades:
                self._execute_single_trade(trade)
            
            if trades:
                logger.info(f"Executed {len(trades)} trades")
            
        except Exception as e:
            logger.error(f"Error executing trades: {e}")
    
    def _execute_single_trade(self, trade: Dict):
        """Execute a single trade."""
        try:
            symbol = trade['symbol']
            shares = trade['shares']
            price = trade['price']
            
            if self.paper_trading:
                # Paper trading - just update positions
                current_shares = self.current_positions.get(symbol, 0)
                new_shares = current_shares + shares
                
                if new_shares > 0:
                    self.current_positions[symbol] = new_shares
                elif symbol in self.current_positions:
                    del self.current_positions[symbol]
                
                # Update cash
                trade_value = shares * price
                self.cash -= trade_value
                
                logger.info(f"PAPER TRADE: {symbol} {'BUY' if shares > 0 else 'SELL'} {abs(shares):.2f} shares @ ${price:.2f}")
                
            else:
                # Real trading - integrate with broker API here
                logger.info(f"REAL TRADE: {symbol} {'BUY' if shares > 0 else 'SELL'} {abs(shares):.2f} shares @ ${price:.2f}")
                # TODO: Implement actual broker integration
                pass
            
            # Record trade
            self.trade_history.append(trade)
            
        except Exception as e:
            logger.error(f"Error executing trade for {trade['symbol']}: {e}")
    
    def _calculate_portfolio_value(self) -> float:
        """Calculate current portfolio value."""
        portfolio_value = self.cash
        
        for symbol, shares in self.current_positions.items():
            if symbol in self.current_prices:
                price = self.current_prices[symbol]
                portfolio_value += shares * price
        
        return portfolio_value
    
    def _calculate_daily_performance(self):
        """Calculate end-of-day performance metrics."""
        try:
            current_value = self._calculate_portfolio_value()
            
            performance = {
                'date': datetime.now().date(),
                'portfolio_value': current_value,
                'cash': self.cash,
                'positions': len(self.current_positions),
                'total_return': (current_value / self.initial_capital) - 1
            }
            
            self.performance_history.append(performance)
            
            logger.info(f"Daily performance: Portfolio value: ${current_value:,.2f}, Return: {performance['total_return']:.2%}")
            
        except Exception as e:
            logger.error(f"Error calculating daily performance: {e}")
    
    def _close_all_positions(self):
        """Close all open positions."""
        logger.info("Closing all positions...")
        
        for symbol in list(self.current_positions.keys()):
            shares = self.current_positions[symbol]
            if shares != 0:
                trade = {
                    'timestamp': datetime.now(),
                    'symbol': symbol,
                    'shares': -shares,  # Sell all shares
                    'price': self.current_prices.get(symbol, 0),
                    'value': abs(shares * self.current_prices.get(symbol, 0)),
                    'type': 'close',
                    'paper_trading': self.paper_trading
                }
                self._execute_single_trade(trade)
    
    def save_performance(self):
        """Save performance data to file."""
        try:
            # Save trade history
            trade_df = pd.DataFrame(self.trade_history)
            if not trade_df.empty:
                trade_df.to_csv("trading_data/trade_history.csv", index=False)
            
            # Save performance history
            perf_df = pd.DataFrame(self.performance_history)
            if not perf_df.empty:
                perf_df.to_csv("trading_data/performance_history.csv", index=False)
            
            # Save current state
            state = {
                'current_positions': self.current_positions,
                'portfolio_value': self._calculate_portfolio_value(),
                'cash': self.cash,
                'last_updated': datetime.now().isoformat()
            }
            
            with open("trading_data/current_state.json", 'w') as f:
                json.dump(state, f, indent=2, default=str)
            
            logger.info("Performance data saved successfully")
            
        except Exception as e:
            logger.error(f"Error saving performance data: {e}")
    
    def get_current_status(self) -> Dict:
        """Get current trading status."""
        try:
            current_value = self._calculate_portfolio_value()
            
            status = {
                'is_running': self.is_running,
                'portfolio_value': current_value,
                'cash': self.cash,
                'total_return': (current_value / self.initial_capital) - 1,
                'positions': len(self.current_positions),
                'current_positions': self.current_positions.copy(),
                'last_update': datetime.now().isoformat()
            }
            
            return status
            
        except Exception as e:
            logger.error(f"Error getting status: {e}")
            return {}

def main():
    """Demonstrate live trading system."""
    
    # Define trading symbols
    symbols = [
        'AAPL', 'MSFT', 'GOOGL', 'TSLA', 'NVDA', 'META', 'AMZN', 'NFLX',
        'JPM', 'V', 'JNJ', 'PG', 'HD', 'MA', 'DIS'
    ]
    
    # Initialize live trader
    trader = LiveTrader(
        symbols=symbols,
        initial_capital=100000,
        max_positions=10,
        rebalance_frequency=30,  # Rebalance every 30 minutes
        paper_trading=True  # Use paper trading for safety
    )
    
    print("Live Trading System Demo")
    print("=" * 50)
    print("This is a simulation. Set paper_trading=False for real trading.")
    print("=" * 50)
    
    try:
        # Start trading
        trader.start_trading()
        
        # Run for a few minutes to demonstrate
        print("Running live trading simulation for 5 minutes...")
        time.sleep(300)  # Run for 5 minutes
        
        # Stop trading
        trader.stop_trading()
        
        # Show final status
        status = trader.get_current_status()
        print("\nFinal Trading Status:")
        print(f"Portfolio Value: ${status['portfolio_value']:,.2f}")
        print(f"Total Return: {status['total_return']:.2%}")
        print(f"Number of Positions: {status['positions']}")
        
    except KeyboardInterrupt:
        print("\nStopping trading system...")
        trader.stop_trading()
    except Exception as e:
        logger.error(f"Error in main trading loop: {e}")
        trader.stop_trading()

if __name__ == "__main__":
    main() 