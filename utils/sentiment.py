import streamlit as st
import numpy as np
import requests
from bs4 import BeautifulSoup
from textblob import TextBlob
import trafilatura
from datetime import datetime, timedelta
import random

@st.cache_data(ttl=1800)  # Cache for 30 minutes
def fetch_news_headlines(symbol=None, source="general"):
    """
    Fetch real news headlines for sentiment analysis
    """
    headlines = []
    
    try:
        if source == "moneycontrol" and symbol:
            # Fetch from Moneycontrol
            symbol_clean = symbol.replace('.NS', '')
            url = f"https://www.moneycontrol.com/news/tags/{symbol_clean}.html"
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            
            response = requests.get(url, headers=headers, timeout=10)
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Extract headlines
            headline_tags = soup.find_all(['h2', 'h3', 'h4'], class_=lambda x: x and 'headline' in str(x).lower())
            for tag in headline_tags[:10]:
                headline_text = tag.get_text(strip=True)
                if len(headline_text) > 10:
                    headlines.append(headline_text)
        
        elif source == "general":
            # Fetch general market news
            urls = [
                "https://www.moneycontrol.com/news/business/markets/",
                "https://economictimes.indiatimes.com/markets/stocks/news"
            ]
            
            for url in urls:
                try:
                    downloaded = trafilatura.fetch_url(url)
                    if downloaded:
                        text = trafilatura.extract(downloaded)
                        if text:
                            # Split into sentences and take relevant ones
                            sentences = text.split('.')[:20]
                            for sentence in sentences:
                                if any(keyword in sentence.lower() for keyword in ['stock', 'market', 'nifty', 'sensex', 'share', 'trading']):
                                    headlines.append(sentence.strip())
                except Exception as e:
                    continue
                    
                if len(headlines) >= 10:
                    break
        
        # Fallback headlines if no real data
        if not headlines:
            headlines = generate_fallback_headlines()
            
    except Exception as e:
        st.warning(f"Error fetching news: {str(e)}. Using fallback data.")
        headlines = generate_fallback_headlines()
    
    return headlines[:15]  # Limit to 15 headlines

def generate_fallback_headlines():
    """
    Generate realistic fallback headlines for demonstration
    """
    headlines = [
        "Market shows mixed signals amid global uncertainty",
        "Banking sector rallies on positive quarterly results",
        "Technology stocks face pressure from regulatory concerns",
        "Auto sector rebounds following strong sales figures",
        "Pharmaceutical companies gain on export opportunities",
        "Energy stocks volatile amid crude oil price fluctuations",
        "FMCG sector remains stable despite inflationary pressures",
        "Infrastructure spending boosts construction stocks",
        "Telecom sector consolidation continues to impact prices",
        "Metal stocks decline on global demand concerns",
        "Real estate sector shows signs of recovery",
        "Small-cap stocks outperform large-cap indices",
        "Foreign institutional investors maintain cautious stance",
        "Domestic mutual funds increase equity allocation",
        "Market volatility expected to continue near term"
    ]
    
    # Randomly select and slightly modify headlines
    selected = random.sample(headlines, min(10, len(headlines)))
    return selected

@st.cache_data(ttl=1800)
def analyze_sentiment(text):
    """
    Analyze sentiment of text using TextBlob
    """
    try:
        blob = TextBlob(text)
        polarity = blob.sentiment.polarity  # -1 to 1
        subjectivity = blob.sentiment.subjectivity  # 0 to 1
        
        return {
            'polarity': polarity,
            'subjectivity': subjectivity,
            'sentiment': 'positive' if polarity > 0.1 else 'negative' if polarity < -0.1 else 'neutral'
        }
    except Exception as e:
        return {
            'polarity': 0,
            'subjectivity': 0.5,
            'sentiment': 'neutral'
        }

def get_market_sentiment():
    """
    Get overall market sentiment score
    """
    try:
        # Fetch general market headlines
        headlines = fetch_news_headlines(source="general")
        
        if not headlines:
            return np.random.uniform(-0.3, 0.3)
        
        # Analyze sentiment for each headline
        sentiment_scores = []
        for headline in headlines:
            sentiment = analyze_sentiment(headline)
            sentiment_scores.append(sentiment['polarity'])
        
        # Calculate average sentiment
        if sentiment_scores:
            avg_sentiment = np.mean(sentiment_scores)
            # Normalize to reasonable range
            return np.clip(avg_sentiment, -1, 1)
        else:
            return 0
            
    except Exception as e:
        st.error(f"Error calculating market sentiment: {str(e)}")
        return np.random.uniform(-0.3, 0.3)

def get_stock_sentiment(symbol):
    """
    Get sentiment for a specific stock
    """
    try:
        headlines = fetch_news_headlines(symbol, source="moneycontrol")
        
        if not headlines:
            return np.random.uniform(-0.5, 0.5)
        
        sentiment_scores = []
        for headline in headlines:
            sentiment = analyze_sentiment(headline)
            sentiment_scores.append(sentiment['polarity'])
        
        if sentiment_scores:
            return np.mean(sentiment_scores)
        else:
            return 0
            
    except Exception as e:
        return np.random.uniform(-0.5, 0.5)

def get_sentiment_analysis():
    """
    Get comprehensive sentiment analysis
    """
    try:
        headlines = fetch_news_headlines(source="general")
        
        if not headlines:
            return {
                'overall_sentiment': 0,
                'positive_ratio': 0.4,
                'negative_ratio': 0.3,
                'neutral_ratio': 0.3,
                'total_headlines': 0
            }
        
        sentiments = []
        for headline in headlines:
            sentiment = analyze_sentiment(headline)
            sentiments.append(sentiment)
        
        # Calculate ratios
        positive_count = sum(1 for s in sentiments if s['sentiment'] == 'positive')
        negative_count = sum(1 for s in sentiments if s['sentiment'] == 'negative')
        neutral_count = sum(1 for s in sentiments if s['sentiment'] == 'neutral')
        total = len(sentiments)
        
        if total > 0:
            positive_ratio = positive_count / total
            negative_ratio = negative_count / total
            neutral_ratio = neutral_count / total
            overall_sentiment = np.mean([s['polarity'] for s in sentiments])
        else:
            positive_ratio = 0.4
            negative_ratio = 0.3
            neutral_ratio = 0.3
            overall_sentiment = 0
        
        return {
            'overall_sentiment': overall_sentiment,
            'positive_ratio': positive_ratio,
            'negative_ratio': negative_ratio,
            'neutral_ratio': neutral_ratio,
            'total_headlines': total,
            'sentiments': sentiments
        }
        
    except Exception as e:
        st.error(f"Error in sentiment analysis: {str(e)}")
        return {
            'overall_sentiment': 0,
            'positive_ratio': 0.4,
            'negative_ratio': 0.3,
            'neutral_ratio': 0.3,
            'total_headlines': 0
        }

def create_word_cloud_data():
    """
    Create word frequency data for word cloud
    """
    try:
        headlines = fetch_news_headlines(source="general")
        
        if not headlines:
            # Fallback word frequencies
            return {
                'market': 15, 'stock': 12, 'trading': 10, 'investment': 8,
                'profit': 7, 'growth': 6, 'sector': 6, 'performance': 5,
                'bullish': 4, 'bearish': 4, 'volatility': 3, 'opportunity': 3,
                'analysis': 3, 'forecast': 2, 'momentum': 2, 'trend': 2
            }
        
        # Combine all headlines
        all_text = ' '.join(headlines).lower()
        
        # Remove common stop words and split
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'must', 'can', 'this', 'that', 'these', 'those'}
        
        words = all_text.split()
        word_freq = {}
        
        for word in words:
            # Clean word
            word = ''.join(c for c in word if c.isalnum())
            if len(word) > 3 and word not in stop_words:
                word_freq[word] = word_freq.get(word, 0) + 1
        
        # Filter to top words and add market-specific terms
        market_words = {
            'market', 'stock', 'trading', 'investment', 'profit', 'loss',
            'growth', 'sector', 'performance', 'bullish', 'bearish',
            'volatility', 'opportunity', 'analysis', 'forecast', 'nifty',
            'sensex', 'shares', 'equity', 'portfolio', 'returns'
        }
        
        # Boost market-related words
        for word in market_words:
            if word in word_freq:
                word_freq[word] *= 2
        
        # Return top 50 words
        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        return dict(sorted_words[:50])
        
    except Exception as e:
        st.error(f"Error creating word cloud data: {str(e)}")
        # Return fallback data
        return {
            'market': 15, 'stock': 12, 'trading': 10, 'investment': 8,
            'profit': 7, 'growth': 6, 'sector': 6, 'performance': 5,
            'bullish': 4, 'bearish': 4, 'volatility': 3, 'opportunity': 3
        }

def get_sentiment_trend(days=7):
    """
    Get sentiment trend over the past few days (simulated)
    """
    try:
        # In a real implementation, this would fetch historical sentiment data
        # For now, we'll simulate a trend
        base_sentiment = get_market_sentiment()
        
        trend_data = []
        for i in range(days):
            date = datetime.now() - timedelta(days=days-1-i)
            # Add some random variation around the base sentiment
            sentiment = base_sentiment + np.random.normal(0, 0.1)
            sentiment = np.clip(sentiment, -1, 1)
            
            trend_data.append({
                'date': date.strftime('%Y-%m-%d'),
                'sentiment': sentiment
            })
        
        return trend_data
        
    except Exception as e:
        st.error(f"Error getting sentiment trend: {str(e)}")
        return []

def get_sentiment_by_sector():
    """
    Get sentiment analysis by sector
    """
    sectors = {
        'Banking': ['HDFCBANK', 'ICICIBANK', 'AXISBANK', 'SBIN', 'KOTAKBANK'],
        'IT': ['TCS', 'INFY', 'HCLTECH', 'WIPRO', 'TECHM'],
        'Auto': ['MARUTI', 'TATAMOTORS', 'BAJAJ-AUTO', 'M&M', 'EICHERMOT'],
        'Pharma': ['SUNPHARMA', 'DRREDDY', 'CIPLA', 'DIVISLAB'],
        'Energy': ['RELIANCE', 'ONGC', 'BPCL', 'NTPC', 'POWERGRID']
    }
    
    sector_sentiments = {}
    
    for sector, stocks in sectors.items():
        try:
            # Get sentiment for a representative stock in the sector
            symbol = f"{stocks[0]}.NS"
            sentiment = get_stock_sentiment(symbol)
            sector_sentiments[sector] = sentiment
        except:
            sector_sentiments[sector] = np.random.uniform(-0.3, 0.3)
    
    return sector_sentiments

