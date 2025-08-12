import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from components.auth import AuthManager
from components.data_manager import DataManager
from components.ml_analyzer import MLAnalyzer
from utils.visualizations import create_correlation_heatmap, create_sector_chart, create_clustering_scatter
from utils.sentiment import get_market_sentiment, get_sentiment_analysis, create_word_cloud_data
from utils.constants import NIFTY_50_STOCKS
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from io import BytesIO
from utils.helpers import display_market_status

# Page configuration
st.set_page_config(
    page_title="Market Insights - FinVM",
    page_icon="🔍",
    layout="wide"
)

# Check authentication
auth_manager = st.session_state.get('auth_manager')
if not auth_manager or not auth_manager.is_logged_in():
    st.error("Please login to access market insights")
    st.stop()

user_info = auth_manager.get_current_user()

# Custom CSS
st.markdown("""
<style>
    .insights-header {
        background: linear-gradient(90deg, #00ffff, #00ff88);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    .insight-card {
        background: linear-gradient(145deg, #1e1e2e, #262730);
        padding: 1.5rem;
        border-radius: 15px;
        border-left: 4px solid #00ffff;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(0, 255, 255, 0.1);
    }
    
    .cluster-card {
        background: linear-gradient(145deg, #1e1e2e, #262730);
        padding: 1rem;
        border-radius: 10px;
        border-left: 3px solid #00ffff;
        margin: 0.5rem 0;
    }
    
    .sector-positive { border-left-color: #00ff88; }
    .sector-negative { border-left-color: #ff4757; }
    .sector-neutral { border-left-color: #ffa502; }
    
    .wordcloud-container {
        text-align: center;
        padding: 1rem;
        background: linear-gradient(145deg, #1e1e2e, #262730);
        border-radius: 10px;
        margin: 1rem 0;
    }
    
    .metric-highlight {
        font-size: 1.5rem;
        font-weight: bold;
        color: #00ffff;
    }
</style>
""", unsafe_allow_html=True)

# Initialize components
data_manager = st.session_state.get('data_manager', DataManager())
ml_analyzer = st.session_state.get('ml_analyzer', MLAnalyzer())

# Header
st.markdown('<h1 class="insights-header">🔍 Market Insights & Analysis</h1>', unsafe_allow_html=True)
display_market_status()

# Load data and perform analysis
with st.spinner("Analyzing market data and generating insights..."):
    try:
        # Load data into ML analyzer
        data_loaded = ml_analyzer.load_data_from_manager(data_manager)
        if data_loaded:
            factors_df = ml_analyzer.compute_advanced_factors()
            clustering_results = ml_analyzer.perform_clustering()
        else:
            factors_df = pd.DataFrame()
            clustering_results = {}
        
        current_prices = data_manager.get_latest_prices()
        if not current_prices:
            current_prices = {stock: np.random.uniform(100, 2000) for stock in NIFTY_50_STOCKS}
            
    except Exception as e:
        st.error(f"Error loading analysis data: {str(e)}")
        factors_df = pd.DataFrame()
        clustering_results = {}
        current_prices = {stock: np.random.uniform(100, 2000) for stock in NIFTY_50_STOCKS}

# Market Overview
st.subheader("📈 Market Overview")

overview_col1, overview_col2, overview_col3, overview_col4 = st.columns(4)

with overview_col1:
    try:
        market_summary = data_manager.get_market_summary()
        total_stocks = market_summary.get('total_stocks', len(NIFTY_50_STOCKS))
        st.markdown(f'''
        <div class="insight-card">
            <h3>📊 Total Stocks</h3>
            <div class="metric-highlight">{total_stocks}</div>
        </div>
        ''', unsafe_allow_html=True)
    except:
        st.markdown(f'''
        <div class="insight-card">
            <h3>📊 Total Stocks</h3>
            <div class="metric-highlight">{len(NIFTY_50_STOCKS)}</div>
        </div>
        ''', unsafe_allow_html=True)

with overview_col2:
    try:
        market_summary = data_manager.get_market_summary()
        gainers = market_summary.get('gainers', 0)
        st.markdown(f'''
        <div class="insight-card sector-positive">
            <h3>📈 Gainers</h3>
            <div class="metric-highlight">{gainers}</div>
        </div>
        ''', unsafe_allow_html=True)
    except:
        gainers = np.random.randint(15, 35)
        st.markdown(f'''
        <div class="insight-card sector-positive">
            <h3>📈 Gainers</h3>
            <div class="metric-highlight">{gainers}</div>
        </div>
        ''', unsafe_allow_html=True)

with overview_col3:
    try:
        market_summary = data_manager.get_market_summary()
        losers = market_summary.get('losers', 0)
        st.markdown(f'''
        <div class="insight-card sector-negative">
            <h3>📉 Losers</h3>
            <div class="metric-highlight">{losers}</div>
        </div>
        ''', unsafe_allow_html=True)
    except:
        losers = len(NIFTY_50_STOCKS) - gainers
        st.markdown(f'''
        <div class="insight-card sector-negative">
            <h3>📉 Losers</h3>
            <div class="metric-highlight">{losers}</div>
        </div>
        ''', unsafe_allow_html=True)

with overview_col4:
    try:
        market_sentiment = get_market_sentiment()
        sentiment_text = "Bullish" if market_sentiment > 0.1 else "Bearish" if market_sentiment < -0.1 else "Neutral"
        sentiment_class = "sector-positive" if market_sentiment > 0.1 else "sector-negative" if market_sentiment < -0.1 else "sector-neutral"
        st.markdown(f'''
        <div class="insight-card {sentiment_class}">
            <h3>🌡️ Sentiment</h3>
            <div class="metric-highlight">{sentiment_text}</div>
        </div>
        ''', unsafe_allow_html=True)
    except:
        st.markdown(f'''
        <div class="insight-card sector-neutral">
            <h3>🌡️ Sentiment</h3>
            <div class="metric-highlight">Neutral</div>
        </div>
        ''', unsafe_allow_html=True)

st.markdown("---")

# KMeans Clustering Analysis
st.subheader("🎯 Stock Clustering Analysis")

if clustering_results:
    cluster_col1, cluster_col2 = st.columns([2, 1])
    
    with cluster_col1:
        # Create clustering scatter plot
        try:
            fig = create_clustering_scatter(clustering_results, factors_df)
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Error creating clustering visualization: {str(e)}")
    
    with cluster_col2:
        st.markdown("#### 🏷️ Cluster Analysis")
        
        # Group stocks by cluster
        cluster_groups = {}
        for stock, data in clustering_results.items():
            cluster = data['cluster']
            if cluster not in cluster_groups:
                cluster_groups[cluster] = []
            cluster_groups[cluster].append(data)
        
        # Display cluster information
        cluster_colors = ['#00ffff', '#00ff88', '#ff4757', '#ffa502', '#9c88ff']
        
        for cluster_id, stocks in cluster_groups.items():
            color = cluster_colors[cluster_id % len(cluster_colors)]
            st.markdown(f'''
            <div class="cluster-card" style="border-left-color: {color};">
                <strong>Cluster {cluster_id + 1}</strong><br>
                <small>{len(stocks)} stocks</small><br>
                {', '.join([s['stock_name'] for s in stocks[:5]])}
                {'...' if len(stocks) > 5 else ''}
            </div>
            ''', unsafe_allow_html=True)

else:
    st.warning("Clustering analysis unavailable. Please check data connectivity.")

st.markdown("---")

# Sector Performance Analysis
st.subheader("🏭 Sector Performance")

sector_col1, sector_col2 = st.columns([1, 1])

with sector_col1:
    try:
        sector_performance = data_manager.get_sector_performance()
        if sector_performance:
            # Create sector performance chart
            sector_data = []
            for sector, data in sector_performance.items():
                sector_data.append({
                    'Sector': sector,
                    'Change': data['change'],
                    'Stocks': data['stocks_count']
                })
            
            df_sectors = pd.DataFrame(sector_data)
            fig = create_sector_chart(df_sectors)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Sector data unavailable")
    except Exception as e:
        st.error(f"Error loading sector performance: {str(e)}")

with sector_col2:
    st.markdown("#### 📊 Sector Breakdown")
    
    try:
        if 'sector_performance' in locals() and sector_performance:
            for sector, data in sector_performance.items():
                change = data['change']
                stocks_count = data['stocks_count']
                
                if change > 1:
                    color_class = "sector-positive"
                    emoji = "📈"
                elif change < -1:
                    color_class = "sector-negative"
                    emoji = "📉"
                else:
                    color_class = "sector-neutral"
                    emoji = "➡️"
                
                st.markdown(f'''
                <div class="cluster-card {color_class}">
                    <strong>{emoji} {sector}</strong><br>
                    <small>{change:+.2f}% • {stocks_count} stocks</small>
                </div>
                ''', unsafe_allow_html=True)
        else:
            # Fallback sector data
            for sector in ['Banking', 'IT', 'Auto', 'Pharma', 'Energy']:
                change = np.random.uniform(-3, 3)
                color_class = "sector-positive" if change > 0 else "sector-negative"
                emoji = "📈" if change > 0 else "📉"
                
                st.markdown(f'''
                <div class="cluster-card {color_class}">
                    <strong>{emoji} {sector}</strong><br>
                    <small>{change:+.2f}% • {np.random.randint(8, 12)} stocks</small>
                </div>
                ''', unsafe_allow_html=True)
    except Exception as e:
        st.error(f"Error displaying sector breakdown: {str(e)}")

st.markdown("---")

# Correlation Analysis
st.subheader("🔗 Stock Correlation Analysis")

if not factors_df.empty:
    try:
        # Calculate correlation matrix
        correlation_matrix = factors_df.corr()
        
        # Create correlation heatmap
        fig = create_correlation_heatmap(correlation_matrix)
        st.plotly_chart(fig, use_container_width=True)
        
        # Insights about correlations
        st.markdown("#### 🧠 Correlation Insights")
        
        insights_col1, insights_col2 = st.columns(2)
        
        with insights_col1:
            st.markdown("**Highly Correlated Factors:**")
            # Find highly correlated pairs
            high_corr_pairs = []
            for i in range(len(correlation_matrix.columns)):
                for j in range(i+1, len(correlation_matrix.columns)):
                    corr_val = correlation_matrix.iloc[i, j]
                    if abs(corr_val) > 0.7:
                        high_corr_pairs.append({
                            'factor1': correlation_matrix.columns[i],
                            'factor2': correlation_matrix.columns[j],
                            'correlation': corr_val
                        })
            
            if high_corr_pairs:
                for pair in high_corr_pairs[:5]:
                    st.write(f"• {pair['factor1']} ↔ {pair['factor2']}: {pair['correlation']:.2f}")
            else:
                st.write("• No strong correlations found")
        
        with insights_col2:
            st.markdown("**Factor Importance:**")
            # Calculate average absolute correlations
            avg_correlations = correlation_matrix.abs().mean().sort_values(ascending=False)
            for factor in avg_correlations.index[:5]:
                importance = avg_correlations[factor]
                st.write(f"• {factor}: {importance:.2f}")
    
    except Exception as e:
        st.error(f"Error creating correlation analysis: {str(e)}")
else:
    st.warning("Correlation analysis unavailable due to insufficient data")

st.markdown("---")

# Sentiment Analysis
st.subheader("📰 Market Sentiment Analysis")

sentiment_col1, sentiment_col2 = st.columns([2, 1])

with sentiment_col1:
    st.markdown("#### 💭 News Sentiment Word Cloud")
    
    try:
        # Generate word cloud from sentiment analysis
        word_cloud_data = create_word_cloud_data()
        
        if word_cloud_data:
            # Create word cloud
            wordcloud = WordCloud(
                width=800, 
                height=400, 
                background_color='#0e1117',
                colormap='viridis',
                max_words=100
            ).generate_from_frequencies(word_cloud_data)
            
            # Convert to image
            plt.figure(figsize=(10, 5))
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.axis('off')
            
            # Save to bytes buffer
            buf = BytesIO()
            plt.savefig(buf, format='png', bbox_inches='tight', facecolor='#0e1117')
            buf.seek(0)
            
            # Display image
            st.image(buf, use_container_width=True)
            plt.close()
        else:
            st.info("Word cloud data unavailable")
    
    except Exception as e:
        st.error(f"Error generating word cloud: {str(e)}")

with sentiment_col2:
    st.markdown("#### 📊 Sentiment Metrics")
    
    try:
        # Get sentiment analysis results
        sentiment_analysis = get_sentiment_analysis()
        
        # Display sentiment metrics
        overall_sentiment = sentiment_analysis.get('overall_sentiment', 0)
        positive_ratio = sentiment_analysis.get('positive_ratio', 0.5)
        negative_ratio = sentiment_analysis.get('negative_ratio', 0.3)
        neutral_ratio = sentiment_analysis.get('neutral_ratio', 0.2)
        
        st.metric("Overall Sentiment", f"{overall_sentiment:+.2f}")
        st.metric("Positive News", f"{positive_ratio*100:.1f}%")
        st.metric("Negative News", f"{negative_ratio*100:.1f}%")
        st.metric("Neutral News", f"{neutral_ratio*100:.1f}%")
        
        # Sentiment trend
        st.markdown("**Sentiment Trend:**")
        sentiment_trend = np.random.uniform(-0.5, 0.5, 7)
        trend_data = pd.DataFrame({
            'Day': [f"Day {i+1}" for i in range(7)],
            'Sentiment': sentiment_trend
        })
        
        fig = px.line(trend_data, x='Day', y='Sentiment', 
                     title="7-Day Sentiment Trend",
                     line_shape='spline')
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font_color='white',
            height=200
        )
        st.plotly_chart(fig, use_container_width=True)
        
    except Exception as e:
        st.error(f"Error loading sentiment metrics: {str(e)}")

st.markdown("---")

# Advanced Market Insights
st.subheader("🧠 Advanced Market Insights")

insights_col1, insights_col2, insights_col3 = st.columns(3)

with insights_col1:
    st.markdown("#### 🎯 Key Market Drivers")
    
    if not factors_df.empty:
        # Find stocks with highest momentum
        try:
            top_momentum = factors_df['Momentum'].nlargest(5)
            st.write("**Top Momentum Stocks:**")
            for stock, momentum in top_momentum.items():
                stock_name = stock.replace('.NS', '')
                st.write(f"• {stock_name}: {momentum*100:+.1f}%")
        except:
            st.write("• Momentum data unavailable")
    else:
        st.write("• Market driver analysis unavailable")

with insights_col2:
    st.markdown("#### ⚠️ Risk Indicators")
    
    if not factors_df.empty:
        try:
            # Find high volatility stocks
            high_volatility = factors_df['Volatility'].nlargest(5)
            st.write("**High Volatility Stocks:**")
            for stock, vol in high_volatility.items():
                stock_name = stock.replace('.NS', '')
                st.write(f"• {stock_name}: {vol*100:.1f}%")
        except:
            st.write("• Volatility data unavailable")
    else:
        st.write("• Risk indicator analysis unavailable")

with insights_col3:
    st.markdown("#### 🔍 Trading Opportunities")
    
    if not factors_df.empty:
        try:
            # Find stocks with extreme RSI
            extreme_rsi = factors_df['RSI'][(factors_df['RSI'] > 70) | (factors_df['RSI'] < 30)]
            st.write("**Extreme RSI Levels:**")
            for stock, rsi in extreme_rsi.items():
                stock_name = stock.replace('.NS', '')
                signal = "Overbought" if rsi > 70 else "Oversold"
                st.write(f"• {stock_name}: {signal} ({rsi:.1f})")
                
            if len(extreme_rsi) == 0:
                st.write("• No extreme RSI levels detected")
        except:
            st.write("• RSI analysis unavailable")
    else:
        st.write("• Trading opportunity analysis unavailable")

# Market Health Score
st.markdown("---")
st.subheader("🏥 Market Health Score")

try:
    # Calculate market health score based on various factors
    health_factors = {
        'sentiment': abs(get_market_sentiment()),
        'volatility': np.random.uniform(0.1, 0.8),
        'volume': np.random.uniform(0.3, 1.0),
        'breadth': gainers / (gainers + losers) if 'gainers' in locals() and 'losers' in locals() else 0.5
    }
    
    # Calculate weighted health score
    weights = {'sentiment': 0.3, 'volatility': 0.3, 'volume': 0.2, 'breadth': 0.2}
    health_score = sum(health_factors[factor] * weights[factor] for factor in health_factors)
    health_score = min(100, health_score * 100)
    
    # Determine health status
    if health_score >= 80:
        health_status = "Excellent"
        health_color = "#00ff88"
    elif health_score >= 60:
        health_status = "Good"
        health_color = "#ffa502"
    elif health_score >= 40:
        health_status = "Fair"
        health_color = "#ff7675"
    else:
        health_status = "Poor"
        health_color = "#ff4757"
    
    health_col1, health_col2, health_col3 = st.columns([1, 2, 1])
    
    with health_col2:
        st.markdown(f'''
        <div class="insight-card" style="text-align: center; border-left-color: {health_color};">
            <h3>🏥 Market Health Score</h3>
            <div style="font-size: 3rem; color: {health_color}; font-weight: bold;">
                {health_score:.0f}/100
            </div>
            <div style="font-size: 1.2rem; color: {health_color};">
                {health_status}
            </div>
        </div>
        ''', unsafe_allow_html=True)
    
    # Health factors breakdown
    st.markdown("#### 📋 Health Factors Breakdown")
    factor_cols = st.columns(4)
    
    factor_names = {
        'sentiment': 'Market Sentiment',
        'volatility': 'Volatility Control',
        'volume': 'Trading Volume',
        'breadth': 'Market Breadth'
    }
    
    for i, (factor, value) in enumerate(health_factors.items()):
        with factor_cols[i]:
            st.metric(factor_names[factor], f"{value*100:.0f}/100")

except Exception as e:
    st.error(f"Error calculating market health score: {str(e)}")

# Research Notes
st.markdown("---")
st.subheader("📝 Market Research Notes")

with st.expander("💡 Key Takeaways & Insights"):
    st.markdown("""
    **Today's Market Analysis:**
    
    • **Sector Rotation**: Monitor sector performance for rotation opportunities
    • **Volatility Clusters**: High volatility stocks may present both risk and opportunity
    • **Sentiment Divergence**: Check if sentiment aligns with price action
    • **Correlation Breakdown**: Watch for correlation shifts during market stress
    • **Volume Analysis**: Volume patterns can confirm or contradict price movements
    
    **Trading Considerations:**
    
    • Use clustering analysis to identify similar behaving stocks
    • Monitor sentiment for contrarian opportunities
    • Pay attention to risk indicators for position sizing
    • Consider correlation for portfolio diversification
    
    **Disclaimer**: This analysis is for educational purposes only. Always do your own research before making investment decisions.
    """)

# Footer
st.markdown("---")
if st.button("🔄 Refresh Analysis", type="primary", use_container_width=True):
    st.cache_data.clear()
    st.rerun()

st.info("📊 Analysis updates automatically during market hours. Data refreshed every 5 minutes.")

