import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
from components.auth import AuthManager
from components.data_manager import DataManager
from components.trading_game import TradingGame
from utils.constants import NIFTY_50_STOCKS
import streamlit as st

if "confirm_reset" not in st.session_state:
    st.session_state.confirm_reset = False

# Page configuration
st.set_page_config(
    page_title="Trading Game - FinVM",
    page_icon="🎮",
    layout="wide"
)

# Check authentication
auth_manager = st.session_state.get('auth_manager')
if not auth_manager or not auth_manager.is_logged_in():
    st.error("Please login to access the trading game")
    st.stop()

user_info = auth_manager.get_current_user()
username = user_info['username']

# Custom CSS
st.markdown("""
<style>
    .game-header {
        background: linear-gradient(90deg, #00ffff, #00ff88);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    .game-card {
        background: linear-gradient(145deg, #1e1e2e, #262730);
        padding: 1.5rem;
        border-radius: 15px;
        border-left: 4px solid #00ffff;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(0, 255, 255, 0.1);
    }
    
    .profit-card {
        border-left-color: #00ff88;
        background: linear-gradient(145deg, #0d2818, #1a3d2b);
    }
    
    .loss-card {
        border-left-color: #ff4757;
        background: linear-gradient(145deg, #2d1618, #3d252b);
    }
    
    .achievement-badge {
        display: inline-block;
        background: linear-gradient(45deg, #00ffff, #00ff88);
        color: #000;
        padding: 0.3rem 0.8rem;
        border-radius: 15px;
        font-weight: bold;
        margin: 0.2rem;
        font-size: 0.8rem;
    }
    
    .trade-form {
        background: rgba(0, 255, 255, 0.05);
        padding: 1.5rem;
        border-radius: 10px;
        border: 1px solid rgba(0, 255, 255, 0.2);
    }
    
    .leaderboard-item {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 0.8rem;
        margin: 0.3rem 0;
        background: rgba(0, 255, 255, 0.05);
        border-radius: 8px;
        border-left: 3px solid #00ffff;
    }
    
    .rank-1 { border-left-color: #ffd700; }
    .rank-2 { border-left-color: #c0c0c0; }
    .rank-3 { border-left-color: #cd7f32; }
</style>
""", unsafe_allow_html=True)

# Initialize components
data_manager = st.session_state.get('data_manager', DataManager())
trading_game = st.session_state.get('trading_game', TradingGame())

# Ensure user has a trading account
trading_game.create_account(username)

# Header
st.markdown('<h1 class="game-header">🎮 Virtual Trading Game</h1>', unsafe_allow_html=True)
st.markdown(f"### Welcome to the arena, {user_info['username']} {user_info['avatar']}")

# Get current prices
with st.spinner("Loading market data..."):
    try:
        current_prices = data_manager.get_latest_prices()
        if not current_prices:
            current_prices = {stock: np.random.uniform(100, 2000) for stock in NIFTY_50_STOCKS}
    except Exception as e:
        st.error(f"Error loading market data: {str(e)}")
        current_prices = {stock: np.random.uniform(100, 2000) for stock in NIFTY_50_STOCKS}

# Game Overview Section
st.subheader("🎯 Game Overview")

account_data = trading_game.get_account(username)
portfolio_value_data = trading_game.get_portfolio_value(username, current_prices)
game_stats = trading_game.get_game_statistics(username)

# Main metrics
col1, col2, col3, col4 = st.columns(4)

with col1:
    cash_balance = portfolio_value_data['cash_balance']
    st.markdown(f'''
    <div class="game-card">
        <h3>💰 Cash Balance</h3>
        <h2>₹{cash_balance:,.0f}</h2>
    </div>
    ''', unsafe_allow_html=True)

with col2:
    portfolio_value = portfolio_value_data['portfolio_value']
    st.markdown(f'''
    <div class="game-card">
        <h3>📈 Portfolio Value</h3>
        <h2>₹{portfolio_value:,.0f}</h2>
    </div>
    ''', unsafe_allow_html=True)

with col3:
    total_value = portfolio_value_data['total_value']
    profit_loss = total_value - 100000  # Initial amount
    card_class = "profit-card" if profit_loss >= 0 else "loss-card"
    st.markdown(f'''
    <div class="game-card {card_class}">
        <h3>🎯 Total P&L</h3>
        <h2>₹{profit_loss:+,.0f}</h2>
    </div>
    ''', unsafe_allow_html=True)

with col4:
    win_rate = game_stats['win_rate']
    st.markdown(f'''
    <div class="game-card">
        <h3>🏆 Win Rate</h3>
        <h2>{win_rate:.1f}%</h2>
    </div>
    ''', unsafe_allow_html=True)

# Game Statistics
st.markdown("---")
st.subheader("📊 Your Trading Statistics")

stats_col1, stats_col2, stats_col3 = st.columns(3)

with stats_col1:
    st.metric("Total Trades", game_stats['total_trades'])
    st.metric("Profitable Trades", game_stats['profitable_trades'])
    st.metric("Current Streak", f"{game_stats['current_streak']} trades")

with stats_col2:
    st.metric("Portfolio Stocks", game_stats['portfolio_stocks'])
    st.metric("Max Streak", f"{game_stats['max_streak']} trades")
    st.metric("Achievements", game_stats['achievements_earned'])

with stats_col3:
    st.metric("Realized P&L", f"₹{game_stats['realized_profit_loss']:,.0f}")
    st.metric("Trades/Day", f"{game_stats['trades_per_day']:.1f}")
    
    # Portfolio return percentage
    portfolio_return = ((total_value - 100000) / 100000) * 100
    st.metric("Total Return", f"{portfolio_return:+.2f}%")

# Trading Interface
st.markdown("---")
st.subheader("💹 Trading Interface")

trade_col1, trade_col2 = st.columns(2)

with trade_col1:
    # Buy Section
    st.markdown("#### 📈 Buy Stocks")
    
    with st.form("buy_form", clear_on_submit=True):
        st.markdown('<div class="trade-form">', unsafe_allow_html=True)
        
        buy_stock = st.selectbox(
            "Select Stock to Buy",
            options=NIFTY_50_STOCKS,
            format_func=lambda x: x.replace('.NS', ''),
            key="buy_stock_selector"
        )
        
        buy_quantity = st.number_input("Quantity", min_value=1, value=1, key="buy_quantity")
        
        current_price = current_prices.get(buy_stock, 100)
        buy_price = st.number_input("Price per share", value=current_price, step=0.01, key="buy_price")
        
        total_cost = buy_quantity * buy_price
        st.write(f"**Total Cost: ₹{total_cost:,.2f}**")
        st.write(f"**Available Balance: ₹{cash_balance:,.2f}**")
        
        if total_cost > cash_balance:
            st.error("Insufficient balance!")
        
        if st.form_submit_button("🛒 Buy Stock", type="primary", disabled=(total_cost > cash_balance)):
            success, message = trading_game.buy_stock(username, buy_stock, buy_quantity, buy_price)
            if success:
                st.success(message)
                st.rerun()
            else:
                st.error(message)
        
        st.markdown('</div>', unsafe_allow_html=True)

with trade_col2:
    # Sell Section
    st.markdown("#### 📉 Sell Stocks")
    
    portfolio_summary = trading_game.get_portfolio_summary(username, current_prices)
    
    if portfolio_summary:
        with st.form("sell_form", clear_on_submit=True):
            st.markdown('<div class="trade-form">', unsafe_allow_html=True)
            
            # Get stocks in portfolio
            portfolio_stocks = [item['stock'] for item in portfolio_summary]
            
            sell_stock = st.selectbox(
                "Select Stock to Sell",
                options=portfolio_stocks,
                format_func=lambda x: x.replace('.NS', ''),
                key="sell_stock_selector"
            )
            
            # Get max quantity for selected stock
            max_quantity = 0
            for item in portfolio_summary:
                if item['stock'] == sell_stock:
                    max_quantity = item['quantity']
                    break
            
            sell_quantity = st.number_input(
                f"Quantity (Max: {max_quantity})",
                min_value=1,
                max_value=max_quantity,
                value=1,
                key="sell_quantity"
            )
            
            current_price = current_prices.get(sell_stock, 100)
            sell_price = st.number_input("Price per share", value=current_price, step=0.01, key="sell_price")
            
            total_proceeds = sell_quantity * sell_price
            st.write(f"**Total Proceeds: ₹{total_proceeds:,.2f}**")
            
            if st.form_submit_button("💰 Sell Stock", type="secondary"):
                success, message = trading_game.sell_stock(username, sell_stock, sell_quantity, sell_price)
                if success:
                    st.success(message)
                    st.rerun()
                else:
                    st.error(message)
            
            st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.info("No stocks in portfolio to sell")

# Current Portfolio
st.markdown("---")
st.subheader("📋 Current Portfolio")

if portfolio_summary:
    portfolio_data = []
    for item in portfolio_summary:
        portfolio_data.append({
            'Stock': item['stock_name'],
            'Quantity': item['quantity'],
            'Avg Price': f"₹{item['average_price']:.2f}",
            'Current Price': f"₹{item['current_price']:.2f}",
            'Investment': f"₹{item['invested_value']:,.0f}",
            'Current Value': f"₹{item['current_value']:,.0f}",
            'P&L': f"₹{item['profit_loss']:+,.0f}",
            'P&L %': f"{item['profit_loss_percent']:+.2f}%"
        })
    
    df_portfolio = pd.DataFrame(portfolio_data)
    
    # Style the dataframe
    def highlight_portfolio_pl(row):
        styles = [''] * len(row)
        pl_val = float(row['P&L'].replace('₹', '').replace(',', '').replace('+', ''))
        
        if pl_val > 0:
            styles[6] = 'color: #00ff88; font-weight: bold'
            styles[7] = 'color: #00ff88; font-weight: bold'
        elif pl_val < 0:
            styles[6] = 'color: #ff4757; font-weight: bold'
            styles[7] = 'color: #ff4757; font-weight: bold'
        
        return styles
    
    styled_portfolio = df_portfolio.style.apply(highlight_portfolio_pl, axis=1)
    st.dataframe(styled_portfolio, use_container_width=True, hide_index=True)

else:
    st.info("Your portfolio is empty. Start trading to build your portfolio!")

# Transaction History
st.markdown("---")
st.subheader("📜 Recent Transactions")

transaction_history = trading_game.get_transaction_history(username, limit=10)

if transaction_history:
    transaction_data = []
    for txn in transaction_history:
        # Convert the timestamp string from the DB into a datetime object first
        transaction_time = datetime.fromisoformat(txn['timestamp'])

        transaction_data.append({
            'Date': transaction_time.strftime('%Y-%m-%d %H:%M'),
            'Type': txn['trade_type'].upper(),
            'Stock': txn['stock_name'].replace('.NS', ''),
            'Quantity': txn['quantity'],
            'Price': f"₹{txn['price']:.2f}",
            'Total': f"₹{txn['total']:,.2f}",
            'P&L': f"₹{txn.get('profit_loss', 0):+,.2f}" if txn['trade_type'] == 'sell' else '-'
        })
    
    df_transactions = pd.DataFrame(transaction_data)
    
    # Style transactions
    def highlight_transaction_type(row):
        styles = [''] * len(row)
        if row['Type'] == 'BUY':
            styles[1] = 'color: #00ff88; font-weight: bold'
        elif row['Type'] == 'SELL':
            styles[1] = 'color: #ff4757; font-weight: bold'
        
        # Color P&L if it's a sell transaction
        if row['P&L'] != '-':
            pl_val = float(row['P&L'].replace('₹', '').replace(',', '').replace('+', ''))
            if pl_val > 0:
                styles[6] = 'color: #00ff88; font-weight: bold'
            elif pl_val < 0:
                styles[6] = 'color: #ff4757; font-weight: bold'
        
        return styles
    
    styled_transactions = df_transactions.style.apply(highlight_transaction_type, axis=1)
    st.dataframe(styled_transactions, use_container_width=True, hide_index=True)

else:
    st.info("No transactions yet. Start trading to see your history!")

# Daily Challenge
st.markdown("---")
st.subheader("🎯 Daily Prediction Challenge")

daily_challenge = trading_game.create_daily_challenge()
challenge_stocks = daily_challenge['stocks']

st.write(f"**Challenge Date:** {daily_challenge['date']}")
st.write("**Challenge:** Predict the direction of these 3 stocks for tomorrow!")

# Check if the user has already submitted predictions for today
if trading_game.has_user_predicted_today(username):
    st.success("✅ You have already submitted your predictions for today! Check back tomorrow for results.")

else:
    # Display the prediction form if the user has not participated
    with st.form("daily_challenge_form"):
        st.write("**Stocks to predict:**")
        predictions = {}

        for stock in challenge_stocks:
            stock_name = stock.replace('.NS', '')
            current_price = current_prices.get(stock, 100)

            col1, col2 = st.columns([2, 1])
            with col1:
                st.write(f"**{stock_name}** - Current: ₹{current_price:.2f}")
            with col2:
                prediction = st.selectbox(
                    f"Prediction for {stock_name}",
                    options=['up', 'down', 'same'],
                    format_func=lambda x: {'up': '📈 Up', 'down': '📉 Down', 'same': '➡️ Same'}[x],
                    key=f"prediction_{stock}"
                )
                predictions[stock] = prediction

        if st.form_submit_button("Submit Predictions", type="primary"):
            success, message = trading_game.submit_prediction(username, predictions)
            if success:
                st.success(message)
                st.rerun()
            else:
                st.error(message)

# Achievements
st.markdown("---")
st.subheader("🏆 Achievements")

achievements = trading_game.get_achievements(username)

if achievements:
    st.write("**Your Achievements:**")
    achievement_cols = st.columns(min(len(achievements), 4))

    for i, achievement in enumerate(achievements):
        col_idx = i % 4
        with achievement_cols[col_idx]:
            st.markdown(f'''
            <div class="achievement-badge">
                {achievement['icon']} {achievement['achievement_name']}
            </div>
            ''', unsafe_allow_html=True)
            st.caption(achievement['description'])

    # Progress towards next achievements
    st.markdown("#### 🎯 Progress Towards Next Achievements")

    # Create a set of earned achievement names for efficient lookup
    earned_achievement_names = {a['achievement_name'] for a in achievements}

    progress_items = [
        {
            'name': 'Day Trader',
            'description': 'Complete 10 trades',
            'current': game_stats['total_trades'],
            'target': 10,
            'achieved': 'Day Trader' in earned_achievement_names
        },
        {
            'name': '10% Club',
            'description': 'Achieve 10% profit',
            'current': max(0, profit_loss),
            'target': 10000,
            'achieved': '10% Club' in earned_achievement_names
        },
        {
            'name': 'Hot Streak',
            'description': '5 profitable trades in a row',
            'current': game_stats['current_streak'],
            'target': 5,
            'achieved': 'Hot Streak' in earned_achievement_names
        }
    ]

    for item in progress_items:
        if not item['achieved']:
            progress = min(100, (item['current'] / item['target']) * 100)
            st.progress(progress / 100, text=f"{item['name']}: {item['current']}/{item['target']}")

else:
    st.info("No achievements yet. Start trading to earn your first achievement!")
# Leaderboard
st.markdown("---")
st.subheader("🏆 Leaderboard")

try:
    leaderboard = trading_game.get_leaderboard(current_prices)

    if leaderboard:
        st.write("**Top Traders:**")

        for i, player in enumerate(leaderboard[:10]):
            rank = i + 1
            rank_class = f"rank-{rank}" if rank <= 3 else ""

            # Use the new method to get the correct avatar for each player
            avatar = auth_manager.get_user_avatar(player['username'])

            is_current_user = player['username'] == username
            highlight = "background: rgba(0, 255, 255, 0.1);" if is_current_user else ""

            st.markdown(f'''
            <div class="leaderboard-item {rank_class}" style="{highlight}">
                <div>
                    <strong>#{rank} {avatar} {player['username']}</strong>
                    {' (You)' if is_current_user else ''}<br>
                    <small>{player['total_trades']} trades • {player['win_rate']:.1f}% win rate</small>
                </div>
                <div style="text-align: right;">
                    <strong>₹{player['total_value']:,.0f}</strong><br>
                    <small style="color: {'#00ff88' if player['profit_loss'] >= 0 else '#ff4757'};">
                        {player['profit_loss_percent']:+.2f}%
                    </small>
                </div>
            </div>
            ''', unsafe_allow_html=True)

    else:
        st.info("Leaderboard is empty. Start trading to claim your spot!")

except Exception as e:
    st.error(f"Error loading leaderboard: {str(e)}")

# Game Controls
st.markdown("---")
st.subheader("⚙️ Game Controls")

control_col1, control_col2, control_col3 = st.columns(3)

with control_col1:
    if st.button("🔄 Refresh Data", use_container_width=True):
        # Clearing cache and rerunning is a good way to refresh
        st.cache_data.clear()
        st.rerun()

with control_col2:
    with st.expander("📊 View Detailed Stats", expanded=False):
        st.json({
            'Account Summary': {
                'Total Value': f"₹{total_value:,.0f}",
                'Cash Balance': f"₹{cash_balance:,.0f}",
                'Portfolio Value': f"₹{portfolio_value:,.0f}",
                'Total P&L': f"₹{profit_loss:+,.0f}",
                'Return %': f"{portfolio_return:+.2f}%"
            },
            'Trading Stats': {
                'Total Trades': game_stats['total_trades'],
                'Profitable Trades': game_stats['profitable_trades'],
                'Win Rate': f"{game_stats['win_rate']:.2f}%",
                'Current Streak': game_stats['current_streak'],
                'Max Streak': game_stats['max_streak'],
                'Avg Trades/Day': f"{game_stats['trades_per_day']:.2f}"
            }
        })

with control_col3:
    if st.button("⚠️ Reset Account", use_container_width=True, type="secondary"):
        st.session_state.confirm_reset = not st.session_state.confirm_reset

if st.session_state.confirm_reset:
    st.warning("**Are you sure?** This action is irreversible and will delete all your trading data.", icon="🔥")

    confirm_col1, confirm_col2 = st.columns(2)

    with confirm_col1:
        if st.button("✅ Yes, Reset My Account", use_container_width=True, type="primary"):
            if trading_game.reset_account(username):
                st.success("Account reset successfully!")
                # Turn off the confirmation dialog and rerun
                st.session_state.confirm_reset = False
                st.rerun()
            else:
                st.error("Error resetting account")

    with confirm_col2:
        if st.button("❌ No, Cancel", use_container_width=True):
            # Just turn off the confirmation dialog and rerun
            st.session_state.confirm_reset = False
            st.rerun()

# Footer
st.markdown("---")
st.info(
    "💡 **Pro Tip:** This is a virtual trading game with fake money. Use it to practice your trading skills without any real financial risk!")