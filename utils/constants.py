# NIFTY 50 stock symbols
NIFTY_50_STOCKS = [
    'ADANIENT.NS', 'ADANIPORTS.NS', 'APOLLOHOSP.NS', 'ASIANPAINT.NS', 'AXISBANK.NS',
    'BAJAJ-AUTO.NS', 'BAJAJFINSV.NS', 'BAJFINANCE.NS', 'BHARTIARTL.NS', 'BPCL.NS',
    'BRITANNIA.NS', 'CIPLA.NS', 'COALINDIA.NS', 'DIVISLAB.NS', 'DRREDDY.NS',
    'EICHERMOT.NS', 'GRASIM.NS', 'HCLTECH.NS', 'HDFCBANK.NS', 'HDFCLIFE.NS',
    'HEROMOTOCO.NS', 'HINDALCO.NS', 'HINDUNILVR.NS', 'ICICIBANK.NS', 'INDUSINDBK.NS',
    'INFY.NS', 'ITC.NS', 'JSWSTEEL.NS', 'KOTAKBANK.NS', 'LT.NS',
    'LTIM.NS', 'M&M.NS', 'MARUTI.NS', 'NESTLEIND.NS', 'NTPC.NS',
    'ONGC.NS', 'POWERGRID.NS', 'RELIANCE.NS', 'SBIN.NS', 'SHRIRAMFIN.NS',
    'SUNPHARMA.NS', 'TATACONSUM.NS', 'TATAMOTORS.NS', 'TATASTEEL.NS', 'TCS.NS',
    'TECHM.NS', 'TITAN.NS', 'ULTRACEMCO.NS', 'UPL.NS', 'WIPRO.NS'
]

# Sector mapping for NIFTY 50 stocks
SECTOR_MAPPING = {
    # Banking & Financial Services
    'AXISBANK.NS': 'Banking',
    'HDFCBANK.NS': 'Banking',
    'ICICIBANK.NS': 'Banking',
    'INDUSINDBK.NS': 'Banking',
    'KOTAKBANK.NS': 'Banking',
    'SBIN.NS': 'Banking',
    'BAJAJFINSV.NS': 'Financial Services',
    'BAJFINANCE.NS': 'Financial Services',
    'HDFCLIFE.NS': 'Financial Services',
    'SHRIRAMFIN.NS': 'Financial Services',
    
    # Information Technology
    'HCLTECH.NS': 'Information Technology',
    'INFY.NS': 'Information Technology',
    'LTIM.NS': 'Information Technology',
    'TCS.NS': 'Information Technology',
    'TECHM.NS': 'Information Technology',
    'WIPRO.NS': 'Information Technology',
    
    # Oil & Gas
    'BPCL.NS': 'Oil & Gas',
    'NTPC.NS': 'Oil & Gas',
    'ONGC.NS': 'Oil & Gas',
    'POWERGRID.NS': 'Oil & Gas',
    'RELIANCE.NS': 'Oil & Gas',
    
    # Consumer Goods
    'ASIANPAINT.NS': 'Consumer Goods',
    'BRITANNIA.NS': 'Consumer Goods',
    'HINDUNILVR.NS': 'Consumer Goods',
    'ITC.NS': 'Consumer Goods',
    'NESTLEIND.NS': 'Consumer Goods',
    'TATACONSUM.NS': 'Consumer Goods',
    'TITAN.NS': 'Consumer Goods',
    
    # Automobile
    'BAJAJ-AUTO.NS': 'Automobile',
    'EICHERMOT.NS': 'Automobile',
    'HEROMOTOCO.NS': 'Automobile',
    'M&M.NS': 'Automobile',
    'MARUTI.NS': 'Automobile',
    'TATAMOTORS.NS': 'Automobile',
    
    # Pharmaceuticals
    'APOLLOHOSP.NS': 'Pharmaceuticals',
    'CIPLA.NS': 'Pharmaceuticals',
    'DIVISLAB.NS': 'Pharmaceuticals',
    'DRREDDY.NS': 'Pharmaceuticals',
    'SUNPHARMA.NS': 'Pharmaceuticals',
    
    # Metals & Mining
    'COALINDIA.NS': 'Metals & Mining',
    'HINDALCO.NS': 'Metals & Mining',
    'JSWSTEEL.NS': 'Metals & Mining',
    'TATASTEEL.NS': 'Metals & Mining',
    
    # Infrastructure & Construction
    'ADANIPORTS.NS': 'Infrastructure',
    'GRASIM.NS': 'Infrastructure',
    'LT.NS': 'Infrastructure',
    'ULTRACEMCO.NS': 'Infrastructure',
    
    # Diversified
    'ADANIENT.NS': 'Diversified',
    'BHARTIARTL.NS': 'Telecommunications',
    'UPL.NS': 'Chemicals'
}

# Color schemes for different themes
COLOR_SCHEMES = {
    'neon': {
        'primary': '#00ffff',
        'secondary': '#00ff88',
        'accent': '#ff4757',
        'warning': '#ffa502',
        'background': '#0e1117',
        'surface': '#262730'
    },
    'financial': {
        'bullish': '#00ff88',
        'bearish': '#ff4757',
        'neutral': '#ffa502',
        'info': '#00ffff',
        'success': '#2ed573',
        'danger': '#ff3838'
    }
}

# Application configuration
APP_CONFIG = {
    'name': 'FinVM',
    'version': '1.0.0',
    'description': 'AI-Powered Stock Analysis & Trading Platform',
    'initial_game_balance': 100000,  # ₹1,00,000
    'max_portfolio_stocks': 20,
    'min_trade_amount': 1000,
    'cache_duration': 300,  # 5 minutes
    'update_frequency': 60  # 1 minute
}

# ML Model parameters
ML_CONFIG = {
    'n_clusters': 5,
    'prediction_days': 10,
    'lookback_period': 126,  # ~6 months
    'min_data_points': 50,
    'confidence_threshold': 0.6,
    'volatility_window': 30,
    'rsi_period': 14,
    'bollinger_period': 20,
    'volume_window': 20
}

# Achievement definitions
ACHIEVEMENTS = [
    {
        'id': 'first_trade',
        'name': 'First Trade',
        'description': 'Complete your first trade',
        'icon': '🎯',
        'requirement': {'type': 'trades', 'value': 1}
    },
    {
        'id': 'day_trader',
        'name': 'Day Trader',
        'description': 'Complete 10 trades',
        'icon': '📈',
        'requirement': {'type': 'trades', 'value': 10}
    },
    {
        'id': 'profit_maker',
        'name': 'Profit Maker',
        'description': 'Achieve positive total P&L',
        'icon': '💰',
        'requirement': {'type': 'profit', 'value': 0}
    },
    {
        'id': 'ten_percent_club',
        'name': '10% Club',
        'description': 'Achieve 10% profit',
        'icon': '🏆',
        'requirement': {'type': 'profit_percent', 'value': 10}
    },
    {
        'id': 'hot_streak',
        'name': 'Hot Streak',
        'description': '5 profitable trades in a row',
        'icon': '🔥',
        'requirement': {'type': 'streak', 'value': 5}
    },
    {
        'id': 'portfolio_builder',
        'name': 'Portfolio Builder',
        'description': 'Hold 5 different stocks',
        'icon': '🏗️',
        'requirement': {'type': 'portfolio_size', 'value': 5}
    },
    {
        'id': 'diversified_investor',
        'name': 'Diversified Investor',
        'description': 'Hold stocks from 3 different sectors',
        'icon': '🌟',
        'requirement': {'type': 'sector_diversity', 'value': 3}
    },
    {
        'id': 'diamond_hands',
        'name': 'Diamond Hands',
        'description': 'Hold a stock for 30 days',
        'icon': '💎',
        'requirement': {'type': 'holding_period', 'value': 30}
    },
    {
        'id': 'risk_taker',
        'name': 'Risk Taker',
        'description': 'Trade a high volatility stock',
        'icon': '⚡',
        'requirement': {'type': 'volatility_trade', 'value': 0.3}
    },
    {
        'id': 'market_timer',
        'name': 'Market Timer',
        'description': 'Make a profitable trade on prediction',
        'icon': '⏰',
        'requirement': {'type': 'prediction_profit', 'value': 1}
    }
]

# News sources for sentiment analysis
NEWS_SOURCES = {
    'moneycontrol': {
        'base_url': 'https://www.moneycontrol.com',
        'news_path': '/news/tags/{symbol}.html',
        'selector': 'h2, h3, h4'
    },
    'economic_times': {
        'base_url': 'https://economictimes.indiatimes.com',
        'news_path': '/markets/stocks/news',
        'selector': '.eachStory h3'
    },
    'business_standard': {
        'base_url': 'https://www.business-standard.com',
        'news_path': '/markets',
        'selector': '.headline'
    }
}

# Market hours (IST)
MARKET_HOURS = {
    'start_time': '09:15',
    'end_time': '15:30',
    'weekdays_only': True,
    'holidays': []  # Can be populated with market holidays
}

# Risk levels and their thresholds
RISK_LEVELS = {
    'low': {'min': 0.0, 'max': 0.3, 'color': '#00ff88'},
    'medium': {'min': 0.3, 'max': 0.7, 'color': '#ffa502'},
    'high': {'min': 0.7, 'max': 1.0, 'color': '#ff4757'}
}

# Factor weights for AI predictions
FACTOR_WEIGHTS = {
    'momentum': 0.25,
    'volatility': 0.20,
    'rsi': 0.15,
    'bollinger_position': 0.10,
    'volume_profile': 0.10,
    'sentiment': 0.20
}

# Default values for fallback data
FALLBACK_VALUES = {
    'stock_price_range': (100, 2000),
    'daily_change_range': (-5, 5),
    'volume_range': (100000, 10000000),
    'sentiment_range': (-0.5, 0.5),
    'volatility_range': (0.1, 0.8),
    'rsi_range': (30, 70)
}

# API configuration
API_CONFIG = {
    'yfinance': {
        'timeout': 10,
        'retry_attempts': 3,
        'retry_delay': 1
    },
    'news_scraping': {
        'timeout': 10,
        'user_agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'max_headlines': 15
    }
}

# UI Constants
UI_CONSTANTS = {
    'max_table_rows': 50,
    'chart_height': 400,
    'animation_duration': 500,
    'refresh_interval': 60000,  # milliseconds
    'toast_duration': 3000
}

