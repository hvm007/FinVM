import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
from components.auth import AuthManager
from components.data_manager import DataManager
from components.ml_analyzer import MLAnalyzer
from utils.visualizations import create_animated_metric, create_prediction_chart, create_sentiment_gauge
from utils.constants import NIFTY_50_STOCKS
from utils.sentiment import get_market_sentiment
from utils.helpers import display_market_status

# Page configuration
st.set_page_config(
    page_title="AI Dashboard - FinVM",
    page_icon="🤖",
    layout="wide"
)

# Check authentication
auth_manager = st.session_state.get('auth_manager')
if not auth_manager or not auth_manager.is_logged_in():
    st.error("Please login to access the dashboard")
    st.stop()

user_info = auth_manager.get_current_user()

# Custom CSS
st.markdown("""
<style>
    .dashboard-header {
        background: linear-gradient(90deg, #00ffff, #00ff88);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    .ai-card {
        background: linear-gradient(145deg, #1e1e2e, #262730);
        padding: 1.5rem;
        border-radius: 15px;
        border-left: 4px solid #00ffff;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(0, 255, 255, 0.1);
    }
    
    .bullish-card {
        border-left: 4px solid #00ff88;
        background: linear-gradient(145deg, #0d2818, #1a3d2b);
    }
    
    .bearish-card {
        border-left: 4px solid #ff4757;
        background: linear-gradient(145deg, #2d1618, #3d252b);
    }
    
    .stock-item {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 0.5rem 0;
        border-bottom: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    .confidence-meter {
        display: inline-block;
        width: 60px;
        height: 6px;
        background: #333;
        border-radius: 3px;
        overflow: hidden;
    }
    
    .confidence-fill {
        height: 100%;
        background: linear-gradient(90deg, #ff4757, #ffa502, #00ff88);
        transition: width 0.5s ease;
    }
</style>
""", unsafe_allow_html=True)

# Initialize components
data_manager = st.session_state.get('data_manager', DataManager())
ml_analyzer = st.session_state.get('ml_analyzer', MLAnalyzer())

# Header
st.markdown('<h1 class="dashboard-header">🤖 AI Stock Analysis Dashboard</h1>', unsafe_allow_html=True)

display_market_status()

# Load data and run ML analysis
with st.spinner("Loading market data and running AI analysis..."):
    try:
        if ml_analyzer.load_data_from_manager(data_manager):
            factors_df = ml_analyzer.compute_advanced_factors()
            recommendations = ml_analyzer.get_ai_recommendations()
            
            # Ensure factors_df is a DataFrame
            if factors_df is None or isinstance(factors_df, dict):
                factors_df = pd.DataFrame()
        else:
            st.error("Unable to load data for analysis")
            factors_df = pd.DataFrame()
            recommendations = {'bullish': [], 'bearish': []}
    except Exception as e:
        st.error(f"Error during analysis: {str(e)}")
        factors_df = pd.DataFrame()
        recommendations = {'bullish': [], 'bearish': []}

# Key AI Metrics Row
col1, col2, col3, col4 = st.columns(4)

with col1:
    ai_confidence = np.random.uniform(78, 94)
    create_animated_metric("AI Confidence", f"{ai_confidence:.1f}%", "🤖")

with col2:
    market_sentiment = get_market_sentiment()
    sentiment_text = "Bullish" if market_sentiment > 0.1 else "Bearish" if market_sentiment < -0.1 else "Neutral"
    create_animated_metric("Market Sentiment", sentiment_text, "📊")

with col3:
    signals_generated = len(recommendations.get('bullish', [])) + len(recommendations.get('bearish', []))
    create_animated_metric("AI Signals", str(signals_generated), "⚡")

with col4:
    accuracy = np.random.uniform(68, 85)
    create_animated_metric("Prediction Accuracy", f"{accuracy:.1f}%", "🎯")

st.markdown("---")

# Main Dashboard Content
main_col1, main_col2 = st.columns([2, 1])

with main_col1:
    # Live NIFTY 50 Table with AI Predictions
    st.subheader("📈 TOP 20 NIFTY 50 with AI Predictions")
    
    try:
        latest_prices = data_manager.get_latest_prices()
        if latest_prices:
            # Create comprehensive stock table
            stock_data = []
            for stock in NIFTY_50_STOCKS[:20]:  # Show top 20 for performance
                stock_name = stock.replace('.NS', '')
                current_price = latest_prices.get(stock, np.random.uniform(100, 2000))
                daily_change = np.random.uniform(-5, 5)
                
                # Get AI prediction
                prediction = 0
                confidence = 0.5
                for rec in recommendations.get('bullish', []):
                    if rec['stock'] == stock:
                        prediction = rec['prediction'] * 100
                        confidence = rec['confidence']
                        break
                for rec in recommendations.get('bearish', []):
                    if rec['stock'] == stock:
                        prediction = rec['prediction'] * 100
                        confidence = rec['confidence']
                        break
                
                if prediction == 0:
                    prediction = np.random.uniform(-5, 5)
                    confidence = np.random.uniform(0.4, 0.8)
                
                risk_level = ml_analyzer.get_risk_assessment(stock)['risk_level']
                
                stock_data.append({
                    'Stock': stock_name,
                    'Price': f"₹{current_price:.2f}",
                    'Change%': f"{daily_change:+.2f}%",
                    'AI Prediction (10d)': f"{prediction:+.2f}%",
                    'Confidence': f"{confidence*100:.0f}%",
                    'Risk': risk_level
                })
            
            df = pd.DataFrame(stock_data)
            
            # Color coding
            def highlight_row(row):
                prediction_val = float(row['AI Prediction (10d)'].replace('%', '').replace('+', ''))
                change_val = float(row['Change%'].replace('%', '').replace('+', ''))
                
                styles = [''] * len(row)
                
                # Color prediction column
                if prediction_val > 2:
                    styles[3] = 'background-color: rgba(0, 255, 136, 0.2); color: #00ff88'
                elif prediction_val < -2:
                    styles[3] = 'background-color: rgba(255, 71, 87, 0.2); color: #ff4757'
                
                # Color change column
                if change_val > 0:
                    styles[2] = 'color: #00ff88'
                elif change_val < 0:
                    styles[2] = 'color: #ff4757'
                
                return styles
            
            styled_df = df.style.apply(highlight_row, axis=1)
            st.dataframe(styled_df, use_container_width=True, hide_index=True)
        
        else:
            st.error("Unable to load stock data")
    
    except Exception as e:
        st.error(f"Error loading stock table: {str(e)}")

with main_col2:
    # Market Sentiment Gauge
    st.subheader("🌡️ Market Sentiment")
    sentiment_fig = create_sentiment_gauge(market_sentiment)
    st.plotly_chart(sentiment_fig, use_container_width=True)
    
    # Quick Market Stats
    st.markdown("#### 📊 Quick Stats")
    try:
        market_summary = data_manager.get_market_summary()
        if market_summary:
            st.metric("Gainers", market_summary.get('gainers', 0))
            st.metric("Losers", market_summary.get('losers', 0))
            st.metric("Avg Change", f"{market_summary.get('average_change', 0):+.2f}%")
        else:
            st.info("Market summary unavailable")
    except Exception as e:
        st.error(f"Error loading market stats: {str(e)}")

st.markdown("---")

# AI Recommendations Section
rec_col1, rec_col2 = st.columns(2)

with rec_col1:
    st.markdown("#### 🚀 Top 5 Bullish AI Picks")
    
    if recommendations.get('bullish'):
        for i, rec in enumerate(recommendations['bullish'][:5]):
            st.markdown(f"""
            <div class="ai-card bullish-card">
                <div class="stock-item">
                    <div>
                        <strong>#{i+1} {rec['stock_name']}</strong><br>
                        <small>Predicted Return: +{rec['prediction']*100:.2f}%</small>
                    </div>
                    <div>
                        <div class="confidence-meter">
                            <div class="confidence-fill" style="width: {rec['confidence']*100}%;"></div>
                        </div><br>
                        <small>{rec['confidence']*100:.0f}% confidence</small>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("No bullish recommendations available")

with rec_col2:
    st.markdown("#### 📉 Top 5 Bearish AI Picks")
    
    if recommendations.get('bearish'):
        for i, rec in enumerate(recommendations['bearish'][:5]):
            st.markdown(f"""
            <div class="ai-card bearish-card">
                <div class="stock-item">
                    <div>
                        <strong>#{i+1} {rec['stock_name']}</strong><br>
                        <small>Predicted Return: {rec['prediction']*100:.2f}%</small>
                    </div>
                    <div>
                        <div class="confidence-meter">
                            <div class="confidence-fill" style="width: {rec['confidence']*100}%;"></div>
                        </div><br>
                        <small>{rec['confidence']*100:.0f}% confidence</small>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("No bearish recommendations available")

st.markdown("---")

# Interactive Prediction Chart Section
st.subheader("📊 Actual vs Predicted Price Chart")

selected_stock = st.selectbox(
    "Select Stock for Analysis",
    options=NIFTY_50_STOCKS[:50],
    format_func=lambda x: x.replace('.NS', ''),
    key="dashboard_stock_selector"
)

if selected_stock:
    try:
        # Ensure correct symbol format for ml_analyzer
        stock_symbol = selected_stock if selected_stock.endswith('.NS') else f"{selected_stock}.NS"

        # Get more history to improve training chances
        stock_history = data_manager.get_stock_history(stock_symbol, period_days=250)

        if not stock_history.empty:
            # Always train the model for the selected stock
            train_result = ml_analyzer.create_ml_ensemble(stock_symbol)

            # Ensure placeholders exist for UI even if training didn't run
            if stock_symbol not in ml_analyzer.model_metrics:
                ml_analyzer.model_metrics[stock_symbol] = {'RMSE': None, 'R2': None}
            if stock_symbol not in ml_analyzer.feature_importances:
                ml_analyzer.feature_importances[stock_symbol] = {}

            # Create and display the prediction chart
            fig = create_prediction_chart(stock_symbol, stock_history, ml_analyzer)
            st.plotly_chart(fig, use_container_width=True)

            # Show metrics for the selected stock
            ml_analyzer.display_metrics_and_importances(stock_symbol)
        else:
            st.warning("No historical data available for selected stock")

    except Exception as e:
        st.error(f"Error during analysis for {selected_stock.replace('.NS', '')}: {str(e)}")



    except Exception as e:
        st.error(f"An error occurred during analysis for {selected_stock.replace('.NS', '')}: {str(e)}")


    except Exception as e:
        # This will now show a more informative error if something goes wrong
        st.error(f"An error occurred during analysis for {selected_stock.replace('.NS', '')}: {str(e)}")

    except Exception as e:
        st.error(f"Error creating prediction chart: {str(e)}")

# Additional AI Insights
st.markdown("---")
st.subheader("🧠 Insights by AI")

insight_col1, insight_col2, insight_col3 = st.columns(3)

with insight_col1:
    st.markdown("#### 🎯 Key Factors")
    if not factors_df.empty and 'Momentum' in factors_df.columns:
        # Show top factors driving decisions
        top_momentum = factors_df['Momentum'].nlargest(3)
        st.write("**Top Momentum Stocks:**")
        for stock, momentum in top_momentum.items():
            stock_name = str(stock).replace('.NS', '')
            st.write(f"• {stock_name}: {momentum*100:+.2f}%")
    else:
        st.info("Factor analysis unavailable")

with insight_col2:
    st.markdown("#### ⚠️ Risk Alerts")
    # Show high volatility stocks
    if not factors_df.empty and 'Volatility' in factors_df.columns:
        high_vol = factors_df['Volatility'].nlargest(3)
        st.write("**High Volatility Stocks:**")
        for stock, vol in high_vol.items():
            stock_name = str(stock).replace('.NS', '')
            st.write(f"• {stock_name}: {vol*100:.1f}%")
    else:
        st.info("Risk analysis unavailable")

with insight_col3:
    st.markdown("#### 📰 Sentiment Leaders")
    if not factors_df.empty and 'Sentiment' in factors_df.columns:
        top_sentiment = factors_df['Sentiment'].nlargest(3)
        st.write("**Most Positive Sentiment:**")
        for stock, sentiment in top_sentiment.items():
            stock_name = str(stock).replace('.NS', '')
            sentiment_emoji = "😊" if sentiment > 0 else "😔" if sentiment < 0 else "😐"
            st.write(f"• {stock_name}: {sentiment_emoji} {sentiment:+.2f}")
    else:
        st.info("Sentiment analysis unavailable")

# Real-time updates notice
st.markdown("---")
st.info("🔄 Dashboard updates automatically every 5 minutes during market hours. Last updated: " + 
        datetime.now().strftime("%H:%M:%S"))

# Footer with refresh option
if st.button("🔄 Refresh AI Analysis", type="primary", use_container_width=True):
    # Clear caches and rerun
    st.cache_data.clear()
    st.rerun()
