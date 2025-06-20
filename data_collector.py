import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import logging
from typing import List, Dict, Optional
import time

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DataCollector:
    """
    Comprehensive data collector for quantitative trading algorithms.
    Handles multiple data sources, error handling, and data validation.
    """
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = data_dir
        self.ensure_data_directory()
        
        # S&P 500 stocks for momentum analysis
        self.sp500_symbols = [
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA', 'META', 'BRK-B',
            'UNH', 'JNJ', 'JPM', 'V', 'PG', 'HD', 'MA', 'DIS', 'PYPL', 'BAC',
            'ADBE', 'CRM', 'NFLX', 'KO', 'PEP', 'TMO', 'ABT', 'AVGO', 'WMT',
            'MRK', 'QCOM', 'TXN', 'ACN', 'HON', 'ORCL', 'LLY', 'UNP', 'LOW',
            'UPS', 'IBM', 'MS', 'RTX', 'SPGI', 'CAT', 'GS', 'AMGN', 'DE',
            'T', 'PM', 'ISRG', 'VRTX', 'INTU', 'SCHW', 'GILD', 'ADI', 'MDLZ',
            'REGN', 'CME', 'TMUS', 'ADP', 'NEE', 'PLD', 'DUK', 'SO', 'D',
            'AON', 'BDX', 'TJX', 'ITW', 'CMI', 'EOG', 'SLB', 'USB', 'PFE',
            'COST', 'TGT', 'MMC', 'CI', 'ETN', 'AIG', 'GE', 'F', 'GM', 'XOM',
            'CVX', 'COP', 'EOG', 'SLB', 'HAL', 'BKR', 'MPC', 'PSX', 'VLO',
            'DVN', 'PXD', 'OXY', 'HES', 'APA', 'MRO', 'NBL', 'CHK', 'RRC'
        ]
        
    def ensure_data_directory(self):
        """Create data directory if it doesn't exist."""
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)
            logger.info(f"Created data directory: {self.data_dir}")
    
    def fetch_stock_data(self, symbol: str, start_date: str = "2020-01-01", 
                        end_date: Optional[str] = None) -> pd.DataFrame:
        """
        Fetch stock data with comprehensive error handling and validation.
        
        Args:
            symbol: Stock symbol
            start_date: Start date for data collection
            end_date: End date (defaults to today)
            
        Returns:
            DataFrame with OHLCV data and additional features
        """
        try:
            if end_date is None:
                end_date = datetime.now().strftime("%Y-%m-%d")
            
            logger.info(f"Fetching data for {symbol} from {start_date} to {end_date}")
            
            # Fetch data from Yahoo Finance
            ticker = yf.Ticker(symbol)
            data = ticker.history(start=start_date, end=end_date, interval="1d")
            
            if data.empty:
                logger.warning(f"No data found for {symbol}")
                return pd.DataFrame()
            
            # Add technical indicators
            data = self.add_technical_indicators(data)
            
            # Add fundamental data if available
            try:
                info = ticker.info
                data['market_cap'] = info.get('marketCap', np.nan)
                data['pe_ratio'] = info.get('trailingPE', np.nan)
                data['dividend_yield'] = info.get('dividendYield', np.nan)
            except Exception as e:
                logger.warning(f"Could not fetch fundamental data for {symbol}: {e}")
                data['market_cap'] = np.nan
                data['pe_ratio'] = np.nan
                data['dividend_yield'] = np.nan
            
            # Save to file
            self.save_data(data, symbol)
            
            return data
            
        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {e}")
            return pd.DataFrame()
    
    def add_technical_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add comprehensive technical indicators to the dataset."""
        
        # Price-based indicators
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
    
    def save_data(self, data: pd.DataFrame, symbol: str):
        """Save data to CSV file."""
        filename = os.path.join(self.data_dir, f"{symbol}_data.csv")
        data.to_csv(filename)
        logger.info(f"Saved data for {symbol} to {filename}")
    
    def fetch_multiple_stocks(self, symbols: List[str], start_date: str = "2020-01-01",
                            end_date: Optional[str] = None) -> Dict[str, pd.DataFrame]:
        """
        Fetch data for multiple stocks with rate limiting.
        
        Args:
            symbols: List of stock symbols
            start_date: Start date for data collection
            end_date: End date (defaults to today)
            
        Returns:
            Dictionary mapping symbols to DataFrames
        """
        results = {}
        
        for i, symbol in enumerate(symbols):
            logger.info(f"Processing {symbol} ({i+1}/{len(symbols)})")
            
            data = self.fetch_stock_data(symbol, start_date, end_date)
            if not data.empty:
                results[symbol] = data
            
            # Rate limiting to avoid API restrictions
            if i < len(symbols) - 1:
                time.sleep(0.1)
        
        logger.info(f"Successfully fetched data for {len(results)} stocks")
        return results
    
    def get_sp500_data(self, start_date: str = "2020-01-01", 
                      end_date: Optional[str] = None) -> Dict[str, pd.DataFrame]:
        """Fetch data for S&P 500 stocks."""
        return self.fetch_multiple_stocks(self.sp500_symbols, start_date, end_date)
    
    def validate_data(self, data: pd.DataFrame, symbol: str) -> bool:
        """
        Validate data quality and completeness.
        
        Args:
            data: DataFrame to validate
            symbol: Stock symbol for logging
            
        Returns:
            True if data is valid, False otherwise
        """
        if data.empty:
            logger.warning(f"Empty dataset for {symbol}")
            return False
        
        # Check for missing values
        missing_pct = data.isnull().sum() / len(data) * 100
        if missing_pct['Close'] > 5:
            logger.warning(f"High percentage of missing close prices for {symbol}: {missing_pct['Close']:.2f}%")
            return False
        
        # Check for zero or negative prices
        if (data['Close'] <= 0).any():
            logger.warning(f"Found zero or negative prices for {symbol}")
            return False
        
        # Check for reasonable price movements
        returns = data['Close'].pct_change().dropna()
        if (returns.abs() > 0.5).any():
            logger.warning(f"Found extreme price movements for {symbol}")
        
        return True

    def fetch_stock_data_alternative(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Alternative method to fetch stock data using a different approach.
        This method tries multiple data sources and fallback options.
        """
        logger.info(f"Fetching data for {symbol} using alternative method from {start_date} to {end_date}")
        
        try:
            # Try using yfinance with different parameters
            import yfinance as yf
            
            # Try different ticker formats
            ticker_formats = [symbol, f"{symbol}.TO", f"{symbol}.L"]
            
            for ticker_format in ticker_formats:
                try:
                    ticker = yf.Ticker(ticker_format)
                    
                    # Try different intervals
                    for interval in ['1d', '1wk']:
                        try:
                            data = ticker.history(
                                start=start_date,
                                end=end_date,
                                interval=interval,
                                auto_adjust=True,
                                prepost=False
                            )
                            
                            if not data.empty and len(data) > 10:
                                # Process the data
                                data = self.add_technical_indicators(data)
                                logger.info(f"Successfully fetched {symbol} using {ticker_format} with {interval} interval")
                                return data
                                
                        except Exception as e:
                            logger.debug(f"Failed to fetch {ticker_format} with {interval} interval: {e}")
                            continue
                            
                except Exception as e:
                    logger.debug(f"Failed to create ticker for {ticker_format}: {e}")
                    continue
            
            # If all attempts fail, return empty DataFrame
            logger.warning(f"All alternative methods failed for {symbol}")
            return pd.DataFrame()
            
        except Exception as e:
            logger.error(f"Alternative data collection failed for {symbol}: {e}")
            return pd.DataFrame()
    
    def fetch_stock_data_with_retry(self, symbol: str, start_date: str, end_date: str, max_retries: int = 3) -> pd.DataFrame:
        """
        Fetch stock data with retry logic and multiple fallback methods.
        """
        logger.info(f"Fetching data for {symbol} with retry logic from {start_date} to {end_date}")
        
        for attempt in range(max_retries):
            try:
                # Try primary method
                data = self.fetch_stock_data(symbol, start_date, end_date)
                if not data.empty and len(data) > 10:
                    return data
                
                # Try alternative method
                data = self.fetch_stock_data_alternative(symbol, start_date, end_date)
                if not data.empty and len(data) > 10:
                    return data
                
                # Wait before retry
                if attempt < max_retries - 1:
                    import time
                    time.sleep(2 ** attempt)  # Exponential backoff
                    
            except Exception as e:
                logger.warning(f"Attempt {attempt + 1} failed for {symbol}: {e}")
                if attempt < max_retries - 1:
                    import time
                    time.sleep(2 ** attempt)
        
        logger.warning(f"All retry attempts failed for {symbol}")
        return pd.DataFrame()

def main():
    """Main function to demonstrate data collection."""
    collector = DataCollector()
    
    # Fetch data for a few sample stocks
    sample_symbols = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'NVDA']
    
    print("Starting data collection...")
    data_dict = collector.fetch_multiple_stocks(sample_symbols, start_date="2022-01-01")
    
    print(f"\nCollected data for {len(data_dict)} stocks:")
    for symbol, data in data_dict.items():
        print(f"{symbol}: {len(data)} days of data")
        if collector.validate_data(data, symbol):
            print(f"  ✓ Data validation passed")
        else:
            print(f"  ✗ Data validation failed")

if __name__ == "__main__":
    main() 