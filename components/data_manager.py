import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from utils.constants import NIFTY_50_STOCKS

class DataManager:
    def __init__(self):
        self.stocks = NIFTY_50_STOCKS
        self.cache_duration = 300  # 5 minutes cache
        self._initialize_cache()
    
    def _initialize_cache(self):
        """Initialize data cache"""
        if 'stock_cache' not in st.session_state:
            st.session_state.stock_cache = {}
        if 'last_update' not in st.session_state:
            st.session_state.last_update = {}
    
    @st.cache_data(ttl=300)
    def fetch_live_data(_self, symbols=None):
        """Fetch live stock data with caching"""
        if symbols is None:
            symbols = _self.stocks
        
        try:
            # Fetch data for the last 6 months for analysis
            end_date = datetime.now()
            start_date = end_date - timedelta(days=180)
            
            data = yf.download(
                symbols, 
                start=start_date.strftime('%Y-%m-%d'),
                end=end_date.strftime('%Y-%m-%d'),
                progress=False
            )
            
            if data is None or data.empty:
                raise ValueError("No data received from API")
            
            # Store in cache
            st.session_state.stock_cache = data
            st.session_state.last_update = datetime.now()
            
            return data
            
        except Exception as e:
            st.error(f"Error fetching live data: {str(e)}")
            return _self._get_fallback_data()
    
    def _get_fallback_data(self):
        """Generate fallback data when APIs fail"""
        st.warning("Using fallback data - APIs temporarily unavailable")
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=180)
        dates = pd.date_range(start=start_date, end=end_date, freq='D')
        
        # Create realistic stock price movements
        data = {}
        for column in ['Open', 'High', 'Low', 'Close', 'Volume']:
            if column == 'Volume':
                base_values = np.random.uniform(1000000, 10000000, len(self.stocks))
            else:
                base_values = np.random.uniform(100, 2000, len(self.stocks))
            
            stock_data = {}
            for i, stock in enumerate(self.stocks):
                if column == 'Volume':
                    # Generate volume data
                    volumes = []
                    current_volume = base_values[i]
                    for _ in dates:
                        change = np.random.normal(0, 0.1)
                        current_volume *= (1 + change)
                        volumes.append(max(current_volume, 100000))
                    stock_data[stock] = volumes
                else:
                    # Generate price data with realistic movements
                    prices = []
                    current_price = base_values[i]
                    for _ in dates:
                        change = np.random.normal(0, 0.02)  # 2% daily volatility
                        current_price *= (1 + change)
                        prices.append(max(current_price, 10))
                    stock_data[stock] = prices
            
            data[column] = pd.DataFrame(stock_data, index=dates)
        
        return pd.concat(data, axis=1)
    
    def get_latest_prices(self):
        """Get latest stock prices"""
        try:
            data = self.fetch_live_data()
            if 'Close' in data.columns:
                return data['Close'].iloc[-1].to_dict()
            return {}
        except Exception as e:
            st.error(f"Error getting latest prices: {str(e)}")
            return {}
    
    def get_stock_history(self, symbol, period_days=30):
        """Get historical data for a specific stock"""
        try:
            data = self.fetch_live_data([symbol])
            if data.empty:
                return pd.DataFrame()
            
            # Get the last period_days of data
            end_idx = len(data)
            start_idx = max(0, end_idx - period_days)
            
            return data.iloc[start_idx:end_idx]
            
        except Exception as e:
            st.error(f"Error getting stock history: {str(e)}")
            return pd.DataFrame()
    
    def get_market_summary(self):
        """Get overall market summary"""
        try:
            latest_prices = self.get_latest_prices()
            if not latest_prices:
                return {}
            
            # Calculate market metrics
            data = self.fetch_live_data()
            if 'Close' in data.columns:
                close_prices = data['Close']
                prev_close = close_prices.iloc[-2]
                current_close = close_prices.iloc[-1]
                
                changes = ((current_close - prev_close) / prev_close * 100).fillna(0)
                
                return {
                    'total_stocks': len(self.stocks),
                    'gainers': (changes > 0).sum(),
                    'losers': (changes < 0).sum(),
                    'average_change': changes.mean(),
                    'top_gainer': changes.idxmax(),
                    'top_loser': changes.idxmin(),
                    'max_gain': changes.max(),
                    'max_loss': changes.min()
                }
        except Exception as e:
            st.error(f"Error calculating market summary: {str(e)}")
            
        return {}
    
    def get_sector_performance(self):
        """Get sector-wise performance (simulated for now)"""
        sectors = {
            'Banking': ['HDFCBANK.NS', 'ICICIBANK.NS', 'AXISBANK.NS', 'SBIN.NS', 'KOTAKBANK.NS'],
            'IT': ['TCS.NS', 'INFY.NS', 'HCLTECH.NS', 'WIPRO.NS', 'TECHM.NS'],
            'Auto': ['MARUTI.NS', 'TATAMOTORS.NS', 'BAJAJ-AUTO.NS', 'M&M.NS', 'EICHERMOT.NS'],
            'Pharma': ['SUNPHARMA.NS', 'DRREDDY.NS', 'CIPLA.NS', 'DIVISLAB.NS'],
            'Energy': ['RELIANCE.NS', 'ONGC.NS', 'BPCL.NS', 'NTPC.NS', 'POWERGRID.NS']
        }
        
        sector_performance = {}
        latest_prices = self.get_latest_prices()
        
        for sector, stocks in sectors.items():
            sector_stocks = [s for s in stocks if s in latest_prices]
            if sector_stocks:
                # Simulate performance
                performance = np.random.uniform(-3, 3)
                sector_performance[sector] = {
                    'change': performance,
                    'stocks_count': len(sector_stocks)
                }
        
        return sector_performance
    
    def is_market_open(self):
        """Check if market is currently open (Indian market hours)"""
        now = datetime.now()
        # Indian market: 9:15 AM to 3:30 PM IST (Mon-Fri)
        if now.weekday() >= 5:  # Weekend
            return False
        
        market_start = now.replace(hour=9, minute=15, second=0)
        market_end = now.replace(hour=15, minute=30, second=0)
        
        return market_start <= now <= market_end
