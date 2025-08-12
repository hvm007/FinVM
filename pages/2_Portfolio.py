import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
from components.auth import AuthManager
from components.data_manager import DataManager
from components.portfolio import PortfolioManager
from components.ml_analyzer import MLAnalyzer
from utils.visualizations import create_portfolio_pie_chart, create_performance_chart
from utils.constants import NIFTY_50_STOCKS
from utils.helpers import display_market_status

# Page configuration
st.set_page_config(
    page_title="Portfolio Manager - FinVM",
    page_icon="💼",
    layout="wide"
)

# Check authentication
auth_manager = st.session_state.get('auth_manager')
if not auth_manager or not auth_manager.is_logged_in():
    st.error("Please login to access your portfolio")
    st.stop()

user_info = auth_manager.get_current_user()
username = user_info['username']

# Custom CSS
st.markdown("""
<style>
    .portfolio-header {
        background: linear-gradient(90deg, #00ffff, #00ff88);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    .metric-card {
        background: linear-gradient(145deg, #1e1e2e, #262730);
        padding: 1.5rem;
        border-radius: 15px;
        border-left: 4px solid #00ffff;
        margin: 0.5rem 0;
        text-align: center;
    }
    
    .profit-positive { 
        border-left-color: #00ff88; 
        background: linear-gradient(145deg, #0d2818, #1a3d2b);
    }
    
    .profit-negative { 
        border-left-color: #ff4757; 
        background: linear-gradient(145deg, #2d1618, #3d252b);
    }
    
    .stock-card {
        background: linear-gradient(145deg, #1e1e2e, #262730);
        padding: 1rem;
        border-radius: 10px;
        border-left: 3px solid #00ffff;
        margin: 0.5rem 0;
    }
    
    .add-stock-form {
        background: rgba(0, 255, 255, 0.05);
        padding: 1.5rem;
        border-radius: 10px;
        border: 1px solid rgba(0, 255, 255, 0.2);
    }
    
</style>
""", unsafe_allow_html=True)

# Initialize components
data_manager = st.session_state.get('data_manager', DataManager())
portfolio_manager = st.session_state.get('portfolio_manager', PortfolioManager())
ml_analyzer = st.session_state.get('ml_analyzer', MLAnalyzer())

# Header
st.markdown(f'<h1 class="portfolio-header">💼 Portfolio Manager</h1>', unsafe_allow_html=True)
st.markdown(f"### Welcome back, {user_info['username']} {user_info['avatar']}")

display_market_status()

# Get current prices
with st.spinner("Loading portfolio data..."):
    try:
        current_prices = data_manager.get_latest_prices()
        if not current_prices:
            # Fallback prices for demo
            current_prices = {stock: np.random.uniform(100, 2000) for stock in NIFTY_50_STOCKS}
    except Exception as e:
        st.error(f"Error loading market data: {str(e)}")
        current_prices = {stock: np.random.uniform(100, 2000) for stock in NIFTY_50_STOCKS}

# Portfolio Overview Section
st.subheader("📊 Portfolio Overview")

portfolio_metrics = portfolio_manager.get_portfolio_metrics(username, current_prices)
portfolio_value_data = portfolio_manager.get_portfolio_value(username, current_prices)

# Main metrics row
if portfolio_metrics:
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_value = portfolio_metrics['total_value']
        st.markdown(f'''
        <div class="metric-card">
            <h3>Total Value</h3>
            <h2>₹{total_value:,.0f}</h2>
        </div>
        ''', unsafe_allow_html=True)
    
    with col2:
        total_pl = portfolio_metrics['total_profit_loss']
        pl_class = "profit-positive" if total_pl >= 0 else "profit-negative"
        st.markdown(f'''
        <div class="metric-card {pl_class}">
            <h3>Total P&L</h3>
            <h2>₹{total_pl:+,.0f}</h2>
        </div>
        ''', unsafe_allow_html=True)
    
    with col3:
        pl_percent = portfolio_metrics['total_profit_loss_percent']
        pl_class = "profit-positive" if pl_percent >= 0 else "profit-negative"
        st.markdown(f'''
        <div class="metric-card {pl_class}">
            <h3>Return %</h3>
            <h2>{pl_percent:+.2f}%</h2>
        </div>
        ''', unsafe_allow_html=True)
    
    with col4:
        total_stocks = portfolio_metrics['total_stocks']
        st.markdown(f'''
        <div class="metric-card">
            <h3>Holdings</h3>
            <h2>{total_stocks} stocks</h2>
        </div>
        ''', unsafe_allow_html=True)
else:
    st.info("Your portfolio is empty. Add some stocks to get started!")

# Charts Section
if portfolio_metrics and portfolio_metrics['total_stocks'] > 0:
    chart_col1, chart_col2 = st.columns(2)
    
    with chart_col1:
        # Portfolio allocation pie chart
        st.subheader("🥧 Portfolio Allocation")
        portfolio_summary = portfolio_manager.get_portfolio_summary(username, current_prices)
        if portfolio_summary:
            fig = create_portfolio_pie_chart(portfolio_summary)
            st.plotly_chart(fig, use_container_width=True)
    
    with chart_col2:
        # Performance chart (simulated historical data)
        st.subheader("📈 Portfolio Performance")
        try:
            fig = create_performance_chart(portfolio_summary, days=30)
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Error creating performance chart: {str(e)}")

st.markdown("---")

# Portfolio Holdings Table
st.subheader("📋 Your Holdings")

portfolio_summary = portfolio_manager.get_portfolio_summary(username, current_prices)

if portfolio_summary:
    # Create holdings DataFrame
    holdings_data = []
    for holding in portfolio_summary:
        holdings_data.append({
            'Stock': holding['stock_name'],
            'Quantity': holding['quantity'],
            'Avg Price': f"₹{holding['average_price']:.2f}",
            'Current Price': f"₹{holding['current_price']:.2f}",
            'Invested': f"₹{holding['invested_value']:,.0f}",
            'Current Value': f"₹{holding['current_value']:,.0f}",
            'P&L': f"₹{holding['profit_loss']:+,.0f}",
            'P&L %': f"{holding['profit_loss_percent']:+.2f}%",
            'Weight': f"{holding['weight']:.1f}%"
        })
    
    df = pd.DataFrame(holdings_data)
    
    # Style the dataframe
    def highlight_pl(row):
        styles = [''] * len(row)
        pl_val = float(row['P&L'].replace('₹', '').replace(',', '').replace('+', ''))
        pl_percent_val = float(row['P&L %'].replace('%', '').replace('+', ''))
        
        if pl_val > 0:
            styles[6] = 'color: #00ff88; font-weight: bold'
            styles[7] = 'color: #00ff88; font-weight: bold'
        elif pl_val < 0:
            styles[6] = 'color: #ff4757; font-weight: bold'
            styles[7] = 'color: #ff4757; font-weight: bold'
        
        return styles
    
    styled_df = df.style.apply(highlight_pl, axis=1)
    st.dataframe(styled_df, use_container_width=True, hide_index=True)
    
    # Quick actions for each stock
    st.subheader("⚡ Quick Actions")
    action_cols = st.columns(min(len(portfolio_summary), 4))
    
    for i, holding in enumerate(portfolio_summary[:4]):
        with action_cols[i]:
            stock = holding['stock']
            stock_name = holding['stock_name']
            
            st.markdown(f"**{stock_name}**")
            
            # Sell button
            if st.button(f"Sell {stock_name}", key=f"sell_{stock}", use_container_width=True):
                st.session_state[f"show_sell_form_{stock}"] = True
            
            # Show sell form if button was clicked
            if st.session_state.get(f"show_sell_form_{stock}", False):
                with st.form(f"sell_form_{stock}"):
                    max_qty = holding['quantity']
                    sell_qty = st.number_input(f"Quantity to sell (Max: {max_qty})", 
                                             min_value=1, max_value=max_qty, value=1)
                    sell_price = st.number_input("Sell Price", value=holding['current_price'])
                    
                    if st.form_submit_button("Confirm Sale"):
                        success, message = portfolio_manager.remove_from_portfolio(
                            username, stock, sell_qty
                        )
                        if success:
                            st.success(message)
                            st.session_state[f"show_sell_form_{stock}"] = False
                            st.rerun()
                        else:
                            st.error(message)

else:
    st.info("No holdings found. Start by adding stocks to your portfolio!")

st.markdown("---")

# Add Stock to Portfolio Section
st.subheader("➕ Add Stock to Portfolio")

with st.form("add_stock_form", clear_on_submit=True):
    st.markdown('<div class="add-stock-form">', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        selected_stock = st.selectbox(
            "Select Stock",
            options=NIFTY_50_STOCKS,
            format_func=lambda x: x.replace('.NS', ''),
            key="add_stock_selector"
        )
    
    with col2:
        quantity = st.number_input("Quantity", min_value=1, value=1)
    
    with col3:
        current_price = current_prices.get(selected_stock, 100)
        price = st.number_input("Price per share", value=current_price, step=0.01)
    
    total_cost = quantity * price
    st.write(f"**Total Cost: ₹{total_cost:,.2f}**")
    
    if st.form_submit_button("Add to Portfolio", type="primary"):
        success, message = portfolio_manager.add_to_portfolio(
            username, selected_stock, quantity, price, 'buy'
        )
        if success:
            st.success(message)
            st.rerun()
        else:
            st.error(message)
    
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown("---")

# Watchlist Section
st.subheader("👁️ Watchlist")

watchlist = portfolio_manager.get_watchlist(username)

if watchlist:
    watchlist_data = []
    for stock in watchlist:
        stock_name = stock.replace('.NS', '')
        current_price = current_prices.get(stock, 0)
        daily_change = np.random.uniform(-5, 5)  # Simulated change
        
        watchlist_data.append({
            'Stock': stock_name,
            'Price': f"₹{current_price:.2f}",
            'Change': f"{daily_change:+.2f}%"
        })
    
    df_watchlist = pd.DataFrame(watchlist_data)
    
    # Color code changes
    def highlight_change(row):
        styles = [''] * len(row)
        change_val = float(row['Change'].replace('%', '').replace('+', ''))
        if change_val > 0:
            styles[2] = 'color: #00ff88; font-weight: bold'
        elif change_val < 0:
            styles[2] = 'color: #ff4757; font-weight: bold'
        return styles
    
    styled_watchlist = df_watchlist.style.apply(highlight_change, axis=1)
    st.dataframe(styled_watchlist, use_container_width=True, hide_index=True)
    
    # Remove from watchlist
    col1, col2 = st.columns([3, 1])
    with col1:
        stock_to_remove = st.selectbox(
            "Remove from watchlist",
            options=watchlist,
            format_func=lambda x: x.replace('.NS', ''),
            key="remove_watchlist"
        )
    with col2:
        if st.button("Remove", type="secondary"):
            success, message = portfolio_manager.remove_from_watchlist(username, stock_to_remove)
            if success:
                st.success(message)
                st.rerun()

else:
    st.info("Your watchlist is empty")

# Add to watchlist
with st.expander("➕ Add to Watchlist"):
    col1, col2 = st.columns([3, 1])
    with col1:
        stock_to_add = st.selectbox(
            "Select stock to add",
            options=[s for s in NIFTY_50_STOCKS if s not in watchlist],
            format_func=lambda x: x.replace('.NS', ''),
            key="add_watchlist"
        )
    with col2:
        if st.button("Add to Watchlist", type="primary"):
            success, message = portfolio_manager.add_to_watchlist(username, stock_to_add)
            if success:
                st.success(message)
                st.rerun()
            else:
                st.error(message)

st.markdown("---")

# AI Suggestions Section
st.subheader("🤖 AI Portfolio Suggestions")

try:
    ml_analyzer.load_data_from_manager(data_manager)
    ai_suggestions = portfolio_manager.get_ai_suggestions(username, ml_analyzer, current_prices)
    
    if ai_suggestions:
        sug_col1, sug_col2 = st.columns(2)
        
        buy_suggestions = [s for s in ai_suggestions if s['type'] == 'buy']
        sell_suggestions = [s for s in ai_suggestions if s['type'] == 'sell']
        
        with sug_col1:
            st.markdown("#### 📈 Suggested Buys")
            for suggestion in buy_suggestions:
                confidence_color = "#00ff88" if suggestion['confidence'] > 0.7 else "#ffa502"
                st.markdown(f"""
                <div class="stock-card">
                    <strong>{suggestion['stock_name']}</strong><br>
                    <small style="color: {confidence_color};">
                        {suggestion['reason']}
                    </small>
                </div>
                """, unsafe_allow_html=True)
        
        with sug_col2:
            st.markdown("#### 📉 Suggested Sells")
            for suggestion in sell_suggestions:
                confidence_color = "#ff4757" if suggestion['confidence'] > 0.7 else "#ffa502"
                st.markdown(f"""
                <div class="stock-card">
                    <strong>{suggestion['stock_name']}</strong><br>
                    <small style="color: {confidence_color};">
                        {suggestion['reason']}
                    </small>
                </div>
                """, unsafe_allow_html=True)
    else:
        st.info("No AI suggestions available at this time")

except Exception as e:
    st.error(f"Error loading AI suggestions: {str(e)}")

# Risk Analysis Section
st.subheader("⚠️ Portfolio Risk Analysis")

if portfolio_summary:
    try:
        # Calculate portfolio risk metrics
        risk_data = []
        for holding in portfolio_summary:
            stock = holding['stock']
            risk_assessment = ml_analyzer.get_risk_assessment(stock)
            risk_data.append({
                'Stock': holding['stock_name'],
                'Weight': f"{holding['weight']:.1f}%",
                'Risk Level': risk_assessment['risk_level'],
                'Risk Score': f"{risk_assessment['risk_score']:.2f}"
            })
        
        df_risk = pd.DataFrame(risk_data)
        
        # Color code risk levels
        def highlight_risk(row):
            styles = [''] * len(row)
            risk_level = row['Risk Level']
            if risk_level == 'High':
                styles[2] = 'color: #ff4757; font-weight: bold'
            elif risk_level == 'Medium':
                styles[2] = 'color: #ffa502; font-weight: bold'
            else:
                styles[2] = 'color: #00ff88; font-weight: bold'
            return styles
        
        styled_risk = df_risk.style.apply(highlight_risk, axis=1)
        st.dataframe(styled_risk, use_container_width=True, hide_index=True)
        
        # Risk summary
        high_risk_count = sum(1 for item in risk_data if item['Risk Level'] == 'High')
        if high_risk_count > 0:
            st.warning(f"⚠️ You have {high_risk_count} high-risk holdings. Consider diversifying.")
        else:
            st.success("✅ Your portfolio risk is well balanced!")
        
    except Exception as e:
        st.error(f"Error calculating risk analysis: {str(e)}")

# Export Portfolio
st.markdown("---")
if st.button("📄 Export Portfolio", use_container_width=True):
    try:
        export_df = portfolio_manager.export_portfolio(username)
        if export_df is not None:
            csv = export_df.to_csv(index=False)
            st.download_button(
                label="Download Portfolio CSV",
                data=csv,
                file_name=f"portfolio_{username}_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )
        else:
            st.info("No portfolio data to export")
    except Exception as e:
        st.error(f"Error exporting portfolio: {str(e)}")
