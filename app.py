import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timezone
import os
from components.auth import AuthManager
from components.data_manager import DataManager
from components.ml_analyzer import MLAnalyzer
from components.portfolio import PortfolioManager
from components.trading_game import TradingGame
from utils.visualizations import create_animated_metric, create_stock_heatmap
from utils.constants import NIFTY_50_STOCKS
from components.database import get_connection
from streamlit.runtime.scriptrunner import get_script_run_ctx
from utils.helpers import display_market_status


# Page configuration
st.set_page_config(
    page_title="FinVM - AI Stock Analysis Platform",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="auto"
)

# Custom CSS for styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #00ffff, #00ff88);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(145deg, #1e1e2e, #262730);
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #00ffff;
        margin: 0.5rem 0;
    }
    .bullish { color: #00ff88; }
    .bearish { color: #ff4757; }
    .neutral { color: #fafafa; }
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #0e1117, #262730);
    }
    .stAlert > div {
        background-color: #262730;
        border-radius: 10px;
    }
    div[data-testid="stSidebarNav"] {
        display: none;
    }
</style>
""", unsafe_allow_html=True)


# --- Helper Functions for Dynamic Content ---


def format_time_ago(timestamp_str):
    """Formats a timestamp string into 'time ago' format."""
    if not timestamp_str:
        return "never"

    # The timestamp from Supabase is already timezone-aware
    dt_obj = datetime.fromisoformat(timestamp_str)

    # 2. CORRECTED: Get the current time in a timezone-aware format (UTC)
    now_aware = datetime.now(timezone.utc)

    # Now you are correctly subtracting two timezone-aware datetimes
    diff = now_aware - dt_obj

    seconds = diff.total_seconds()
    if seconds < 60:
        return f"{int(seconds)}s ago"
    minutes = seconds / 60
    if minutes < 60:
        return f"{int(minutes)}m ago"
    hours = minutes / 60
    if hours < 24:
        return f"{int(hours)}h ago"
    days = hours / 24
    return f"{int(days)}d ago"

def get_recent_activity(username):
    """Fetches and combines recent trades and achievements for a user via Supabase."""
    sb = st.session_state.trading_game.sb
    activities = []

    # --- Fetch recent trades ---
    trades = sb.table("game_trades") \
        .select("trade_type, stock_symbol, quantity, timestamp") \
        .eq("username", username) \
        .order("timestamp", desc=True) \
        .limit(3) \
        .execute().data

    for trade in trades:
        action = "Bought" if trade['trade_type'] == 'buy' else "Sold"
        stock_name = trade['stock_symbol'].replace('.NS', '')
        activities.append({
            "icon": "📈" if trade['trade_type'] == 'buy' else "📉",
            "action": "Trade Executed",
            "details": f"{action} {trade['quantity']} shares of {stock_name}",
            "time": format_time_ago(trade['timestamp']),
            "timestamp": datetime.fromisoformat(trade['timestamp'])
        })

    # --- Fetch recent achievements ---
    achievements = sb.table("game_achievements") \
        .select("achievement_name, icon, earned_date") \
        .eq("username", username) \
        .order("earned_date", desc=True) \
        .limit(3) \
        .execute().data

    for ach in achievements:
        activities.append({
            "icon": ach['icon'],
            "action": "Achievement Unlocked!",
            "details": f"You earned the '{ach['achievement_name']}' achievement",
            "time": format_time_ago(ach['earned_date']),
            "timestamp": datetime.fromisoformat(ach['earned_date'])
        })

    # --- Sort and limit ---
    activities.sort(key=lambda x: x['timestamp'], reverse=True)
    return activities[:5]
def generate_ai_alerts(username, ml_analyzer, portfolio_manager):
    """Generates a list of personalized AI alerts for the user."""
    alerts = []
    portfolio = portfolio_manager.get_user_portfolio(username)
    portfolio_stocks = set(portfolio.keys())
    recommendations = ml_analyzer.get_ai_recommendations(top_n=1)

    bullish_rec = recommendations.get('bullish', [])
    if bullish_rec and bullish_rec[0]['stock'] not in portfolio_stocks:
        top_pick = bullish_rec[0]
        alerts.append({
            "type": "info", "icon": "💡", "title": "Opportunity Alert",
            "details": f"AI identifies **{top_pick['stock_name']}** as a top bullish pick with a predicted return of **+{top_pick['prediction'] * 100:.1f}%**."
        })

    bearish_rec = recommendations.get('bearish', [])
    if bearish_rec and bearish_rec[0]['stock'] in portfolio_stocks:
        top_risk = bearish_rec[0]
        alerts.append({
            "type": "warning", "icon": "⚠️", "title": "Portfolio Review Alert",
            "details": f"Your holding **{top_risk['stock_name']}** is showing bearish signals. AI predicts a potential decline of **{top_risk['prediction'] * 100:.1f}%**."
        })

    for stock in portfolio_stocks:
        risk_assessment = ml_analyzer.get_risk_assessment(stock)
        if risk_assessment['risk_level'] == 'High':
            alerts.append({
                "type": "warning", "icon": "🔥", "title": "High-Risk Alert",
                "details": f"Your holding **{stock.replace('.NS', '')}** is currently marked as high-risk. Consider reviewing your position."
            })
            break
    return alerts


# --- Main Application Logic ---

# Initialize session state for all components
if 'auth_manager' not in st.session_state:
    st.session_state.auth_manager = AuthManager()
if 'data_manager' not in st.session_state:
    st.session_state.data_manager = DataManager()
if 'ml_analyzer' not in st.session_state:
    st.session_state.ml_analyzer = MLAnalyzer()
if 'portfolio_manager' not in st.session_state:
    st.session_state.portfolio_manager = PortfolioManager()
if 'trading_game' not in st.session_state:
    st.session_state.trading_game = TradingGame()

# Authentication check
auth_manager = st.session_state.auth_manager
if not auth_manager.is_logged_in():
    # --- Login/Signup Page ---
    # This CSS hides the sidebar ONLY when the user is not logged in.
    st.markdown("""
        <style>
            section[data-testid="stSidebar"] {
                display: none !important;
            }
        </style>
    """, unsafe_allow_html=True)

    st.markdown('<h1 class="main-header">🚀 Welcome to FinVM</h1>', unsafe_allow_html=True)
    st.markdown(
        "<h3 style='text-align: center;'>AI-Powered Stock Analysis Platform</h3>",
        unsafe_allow_html=True
    )

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        auth_container = st.container()
        with auth_container:
            tab1, tab2 = st.tabs(["Login", "Sign Up"])
            with tab1:
                with st.form("login_form"):
                    username = st.text_input("Username")
                    password = st.text_input("Password", type="password")
                    if st.form_submit_button("Login", type="primary"):
                        if auth_manager.login(username, password):
                            st.rerun()
            with tab2:
                with st.form("signup_form"):
                    new_username = st.text_input("Username", key="signup_username")
                    new_password = st.text_input("Password", type="password", key="signup_password")
                    confirm_password = st.text_input("Confirm Password", type="password", key="confirm_password")
                    avatar_options = ["👤", "👨‍💼", "👩‍💼", "🧑‍💻", "👨‍🚀", "👩‍🚀", "🦸‍♂️", "🦸‍♀️"]
                    selected_avatar = st.selectbox("Choose Avatar", avatar_options)
                    if st.form_submit_button("Sign Up", type="primary"):
                        if new_password == confirm_password:
                            auth_manager.signup(new_username, new_password, selected_avatar)
                        else:
                            st.error("Passwords don't match")
else:
    # --- Main App for Logged-In Users ---
    user_info = auth_manager.get_current_user()
    data_manager = st.session_state.data_manager
    ml_analyzer = st.session_state.ml_analyzer

    # --- Master Sidebar (Applied to all pages) ---
    with st.sidebar:
        st.markdown(f"### {user_info['avatar']} Welcome, {user_info['username']}!")
        display_market_status()
        st.markdown("---")

        st.markdown("#### 📊 Market Summary")
        try:
            summary = data_manager.get_market_summary()
            if summary:
                nifty_change = summary.get('average_change', 0)
                color = "bullish" if nifty_change > 0 else "bearish"
                st.markdown(f'<p class="{color}">NIFTY 50 Avg: {nifty_change:+.2f}%</p>', unsafe_allow_html=True)

                gainer_name = summary.get('top_gainer', 'N/A').replace('.NS', '')
                gainer_val = summary.get('max_gain', 0)
                st.markdown(f'<small class="bullish">🚀 {gainer_name}: {gainer_val:+.2f}%</small>',
                            unsafe_allow_html=True)

                loser_name = summary.get('top_loser', 'N/A').replace('.NS', '')
                loser_val = summary.get('max_loss', 0)
                st.markdown(f'<small class="bearish">📉 {loser_name}: {loser_val:+.2f}%</small>', unsafe_allow_html=True)
        except Exception:
            st.error("Error loading market summary.")

        st.markdown("---")

        # Custom Navigation Buttons
        pages = {
            "Home": "app.py",
            "Dashboard": "1_Dashboard.py",
            "Portfolio": "2_Portfolio.py",
            "Trading Game": "3_Trading_Game.py",
            "Market Insights": "4_Market_Insights.py",
            "Go Pro FinVM": "5_Go_Pro_FinVM.py",
        }

        ctx = get_script_run_ctx()
        current_page_script = os.path.basename(ctx.main_script_path) if ctx else "app.py"

        for page_name, page_script_name in pages.items():
            is_active = (current_page_script == page_script_name)
            button_type = "primary" if is_active else "secondary"

            icon = ""
            if page_name == "Home":
                icon = "🏠"
            elif page_name == "Dashboard":
                icon = "📊"
            elif page_name == "Portfolio":
                icon = "💼"
            elif page_name == "Trading Game":
                icon = "🎮"
            elif page_name == "Market Insights":
                icon = "🔍"
            elif page_name == "Go Pro FinVM":
                icon = "👑"

            if st.button(f"{icon} {page_name}", use_container_width=True, type=button_type, key=f"nav_{page_name}"):
                page_path = page_script_name if page_script_name == "app.py" else f"pages/{page_script_name}"
                st.switch_page(page_path)

        st.markdown("---")
        st.button("🚪 Logout", use_container_width=True, on_click=auth_manager.logout)

    # --- Main Page Content ---
    st.markdown('<h1 class="main-header">FinVM Dashboard</h1>', unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        portfolio_value = st.session_state.portfolio_manager.get_total_value(user_info['username'])
        create_animated_metric("Portfolio Value", f"₹{portfolio_value:,.0f}", "📈")
    with col2:
        game_balance = st.session_state.trading_game.get_balance(user_info['username'])
        create_animated_metric("Game Balance", f"₹{game_balance:,.0f}", "🎮")
    with col3:
        confidence = np.random.uniform(75, 95)
        create_animated_metric("AI Confidence", f"{confidence:.1f}%", "🤖")
    with col4:
        sentiment = np.random.uniform(-1, 1)
        sentiment_text = "Bullish" if sentiment > 0.2 else "Bearish" if sentiment < -0.2 else "Neutral"
        sentiment_color = "bullish" if sentiment > 0.2 else "bearish" if sentiment < -0.2 else "neutral"
        st.markdown(
            f'<div class="metric-card"><h3>Market Sentiment</h3><p class="{sentiment_color}">{sentiment_text}</p></div>',
            unsafe_allow_html=True)

    st.markdown("---")

    st.subheader("🤖 AI-Powered Alerts")
    with st.spinner("Scanning for personalized alerts..."):
        ml_analyzer.load_data_from_manager(data_manager)
        alerts = generate_ai_alerts(user_info['username'], ml_analyzer, st.session_state.portfolio_manager)
    if not alerts:
        st.success("✅ No critical alerts for you right now. Your portfolio looks stable.")
    else:
        for alert in alerts:
            st.warning(f"**{alert['title']}** — {alert['details']}", icon=alert['icon'])

    st.markdown("---")
    st.subheader("📈 Market Overview")
    try:
        latest_prices = data_manager.get_latest_prices()
        if latest_prices:
            changes = np.random.uniform(-5, 5, len(NIFTY_50_STOCKS))
            heatmap_data = [{'Stock': s.replace('.NS', ''), 'Change': c, 'Price': latest_prices.get(s, 100)} for s, c in
                            zip(NIFTY_50_STOCKS, changes)]
            df_heatmap = pd.DataFrame(heatmap_data)
            fig = create_stock_heatmap(df_heatmap)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Market data temporarily unavailable.")
    except Exception as e:
        st.error(f"Unable to load market overview: {str(e)}")

    st.markdown("---")
    st.subheader("📋 Recent Activity")
    activities = get_recent_activity(user_info['username'])
    if not activities:
        st.info("No recent activity to display. Start trading to see your actions here!")
    else:
        for activity in activities:
            st.markdown(f"""
            <div class="metric-card">
                <strong>{activity['icon']} {activity['action']}</strong> - <small>{activity['time']}</small><br>
                <small>{activity['details']}</small>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 💡 Pro Tips")
    tips = [
        "Check the AI Dashboard daily for top stock picks",
        "Diversify your portfolio across different sectors",
        "Use the trading game to practice without real money",
        "Monitor sentiment analysis for market trends"
    ]
    tip_cols = st.columns(2)
    for i, tip in enumerate(tips):
        with tip_cols[i % 2]:
            st.info(f"💡 {tip}")
