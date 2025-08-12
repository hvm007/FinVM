import streamlit as st
import pandas as pd
from datetime import datetime, date
from components.database import get_connection
from utils.constants import NIFTY_50_STOCKS
import random

class TradingGame:
    def __init__(self):
        self.sb = get_connection()
        self.initial_balance = 100000

    # --- Account Management ---
    def create_account(self, username):
        """Creates a trading account if it doesn't exist."""
        res = self.sb.table("game_balances").select("username").eq("username", username).execute()
        if not res.data:
            self.sb.table("game_balances").insert({
                "username": username,
                "balance": self.initial_balance,
                "total_trades": 0,
                "profitable_trades": 0,
                "realized_pl": 0,
                "current_streak": 0,
                "max_streak": 0
            }).execute()

    def get_account(self, username):
        """Retrieve game account for a user, safely returning defaults if none exists."""
        try:
            res = (
                self.sb.table("game_balances")  # <-- Changed from game_portfolio
                .select("*")
                .eq("username", username)
                .execute()
            )

            if res and getattr(res, "data", None) and len(res.data) > 0:
                return res.data[0]  # Found account

        except Exception as e:
            print(f"[get_account] Error fetching account: {e}")

        # Return default structure if not found
        return {
            "username": username,
            "balance": self.initial_balance,  # <-- Added default
            "total_trades": 0,
            "profitable_trades": 0,
            "realized_pl": 0,
            "current_streak": 0,
            "max_streak": 0
        }

    def get_balance(self, username):
        """Return the user's account balance."""
        account = self.get_account(username)
        return account["balance"] if account else 0
    # --- Core Trading Logic ---
    def buy_stock(self, username, stock, quantity, price):
        """Processes a 'buy' transaction."""
        account = self.get_account(username)
        if not account: return False, "Account not found."

        total_cost = quantity * price
        if account['balance'] < total_cost: return False, "Insufficient balance"

        self.sb.table("game_balances").update({
            "balance": account['balance'] - total_cost,
            "total_trades": account['total_trades'] + 1
        }).eq("username", username).execute()

        self.sb.table("game_trades").insert({
            "username": username, "stock_symbol": stock, "trade_type": 'buy',
            "quantity": quantity, "price": price, "total": total_cost
        }).execute()

        self._check_achievements(username)
        return True, "Stock purchased successfully"

    def sell_stock(self, username, stock, quantity, price):
        """Processes a 'sell' transaction."""
        portfolio = self._calculate_portfolio_from_db(username)
        if stock not in portfolio or portfolio[stock]['quantity'] < quantity:
            return False, "Insufficient stock quantity"

        account = self.get_account(username)
        avg_price = portfolio[stock]['average_price']
        profit_loss = (price - avg_price) * quantity
        total_received = quantity * price
        new_streak = account['current_streak'] + 1 if profit_loss > 0 else 0

        self.sb.table("game_balances").update({
            "balance": account['balance'] + total_received,
            "realized_pl": account['realized_pl'] + profit_loss,
            "total_trades": account['total_trades'] + 1,
            "profitable_trades": account['profitable_trades'] + (1 if profit_loss > 0 else 0),
            "current_streak": new_streak,
            "max_streak": max(account['max_streak'], new_streak)
        }).eq("username", username).execute()

        self.sb.table("game_trades").insert({
            "username": username, "stock_symbol": stock, "trade_type": 'sell',
            "quantity": quantity, "price": price, "total": total_received, "profit_loss": profit_loss
        }).execute()

        self._check_achievements(username)
        return True, f"Stock sold. P&L: ₹{profit_loss:,.2f}"

    # --- Data Retrieval & Calculation ---
    def get_transaction_history(self, username, limit=10):
        """Retrieves the user's trade history."""
        res = self.sb.table("game_trades").select("*").eq("username", username).order("timestamp", desc=True).limit(
            limit).execute()
        history = res.data
        for item in history:
            item['stock_name'] = item['stock_symbol'].replace('.NS', '')
        return history

    def get_portfolio_summary(self, username, current_prices):
        """Generates a detailed portfolio summary from trade history."""
        portfolio = self._calculate_portfolio_from_db(username)
        summary = []
        for stock, data in portfolio.items():
            current_price = current_prices.get(stock, data['average_price'])
            invested_value = data['quantity'] * data['average_price']
            current_value = data['quantity'] * current_price
            profit_loss = current_value - invested_value
            profit_loss_percent = (profit_loss / invested_value * 100) if invested_value > 0 else 0

            summary.append({
                'stock': stock,
                'stock_name': stock.replace('.NS', ''),
                'quantity': data['quantity'],
                'average_price': data['average_price'],
                'current_price': current_price,
                'invested_value': invested_value,
                'current_value': current_value,
                'profit_loss': profit_loss,
                'profit_loss_percent': profit_loss_percent
            })
        return summary

    def get_portfolio_value(self, username, current_prices):
        """Calculates the current total value of a user's account."""
        account = self.get_account(username)
        portfolio_summary = self.get_portfolio_summary(username, current_prices)

        portfolio_value = sum(item['current_value'] for item in portfolio_summary)
        cash_balance = account['balance'] if account is not None else self.initial_balance

        return {
            'cash_balance': cash_balance,
            'portfolio_value': portfolio_value,
            'total_value': cash_balance + portfolio_value
        }

    def _calculate_portfolio_from_db(self, username):
        """Helper to calculate current holdings from the trade log."""
        res = self.sb.table("game_trades").select("stock_symbol, trade_type, quantity, price").eq("username",
                                                                                                  username).execute()
        trades_df = pd.DataFrame(res.data)
        if trades_df.empty:
            return {}
        holdings = {}
        for trade in trades_df.itertuples():
            stock = trade.stock_symbol
            if stock not in holdings:
                holdings[stock] = {'quantity': 0, 'total_invested': 0.0}

            if trade.trade_type == 'buy':
                holdings[stock]['quantity'] += trade.quantity
                holdings[stock]['total_invested'] += trade.quantity * trade.price
            elif trade.trade_type == 'sell':
                if holdings[stock]['quantity'] > 0:
                    avg_price = holdings[stock]['total_invested'] / holdings[stock]['quantity']
                    holdings[stock]['total_invested'] -= trade.quantity * avg_price
                holdings[stock]['quantity'] -= trade.quantity

        final_portfolio = {}
        for stock, data in holdings.items():
            if data['quantity'] > 0.001 and data['quantity'] > 0:
                final_portfolio[stock] = {
                    'quantity': data['quantity'],
                    'average_price': data['total_invested'] / data['quantity']
                }
        return final_portfolio

    # --- Achievements, Stats & Leaderboard ---
    def get_achievements(self, username):
        """Retrieves all achievements earned by a user."""
        res = self.sb.table("game_achievements").select("*").eq("username", username).execute()
        return res.data

    def _check_achievements(self, username):
        """Checks for and grants new achievements."""
        account = self.get_account(username)
        if not account: return

        portfolio = self._calculate_portfolio_from_db(username)
        earned_achievements = {a['achievement_name'] for a in self.get_achievements(username)}

        achievement_criteria = [
            {'name': 'First Trade', 'desc': 'Complete your first trade', 'icon': '🎯',
             'cond': account['total_trades'] >= 1},
            {'name': 'Day Trader', 'desc': 'Complete 10 trades', 'icon': '📈', 'cond': account['total_trades'] >= 10},
            {'name': 'Profit Maker', 'desc': 'Achieve positive total P&L', 'icon': '💰',
             'cond': account['realized_pl'] > 0},
            {'name': 'Hot Streak', 'desc': '5 profitable trades in a row', 'icon': '🔥',
             'cond': account['max_streak'] >= 5},
            {'name': 'Portfolio Builder', 'desc': 'Hold 5 different stocks', 'icon': '🏗️', 'cond': len(portfolio) >= 5},
        ]

        achievements_to_add = [
            {"username": username, "achievement_name": ach['name'], "description": ach['desc'], "icon": ach['icon']}
            for ach in achievement_criteria if ach['cond'] and ach['name'] not in earned_achievements
        ]

        if achievements_to_add:
            self.sb.table("game_achievements").upsert(achievements_to_add,
                                                      on_conflict='username,achievement_name').execute()

    def get_leaderboard(self, current_prices):
        """Generates the leaderboard."""
        res = self.sb.table("game_balances").select("*").execute()
        all_accounts = res.data
        leaderboard = []

        for account in all_accounts:
            user = account['username']
            portfolio_data = self.get_portfolio_value(user, current_prices)
            total_value = portfolio_data['total_value']
            profit_loss = total_value - self.initial_balance

            # Ensure total_trades exists, defaulting to 0 if not
            total_trades = account.get('total_trades', 0)
            profitable_trades = account.get('profitable_trades', 0)

            # Calculate win rate and profit loss percentage safely
            win_rate = (profitable_trades / total_trades * 100) if total_trades > 0 else 0
            profit_loss_percent = (profit_loss / self.initial_balance * 100) if self.initial_balance > 0 else 0

            leaderboard.append({
                'username': user,
                'total_value': total_value,
                'profit_loss': profit_loss,
                'total_trades': total_trades,  # <-- ADD THIS LINE
                'win_rate': win_rate,
                'profit_loss_percent': profit_loss_percent  # <-- AND THIS LINE
            })

        return sorted(leaderboard, key=lambda x: x['total_value'], reverse=True)

    def get_game_statistics(self, username):
        account = self.get_account(username) or {}

        # Ensure default values for keys
        account.setdefault("total_trades", 0)
        account.setdefault("profitable_trades", 0)
        account.setdefault("current_streak", 0)
        account.setdefault("max_streak", 0)
        account.setdefault("realized_pl", 0)

        # Get first trade date
        first_trade_res = (
            self.sb.table("game_trades")
            .select("timestamp")
            .eq("username", username)
            .order("timestamp")
            .limit(1)
            .execute()
        )

        first_trade_date = (
            datetime.fromisoformat(first_trade_res.data[0]["timestamp"])
            if first_trade_res and getattr(first_trade_res, "data", None)
            else None
        )

        trades_per_day = 0
        if first_trade_date:
            days_trading = (datetime.now(first_trade_date.tzinfo) - first_trade_date).days + 1
            trades_per_day = account["total_trades"] / days_trading if days_trading > 0 else 0

        portfolio = self._calculate_portfolio_from_db(username)
        portfolio_stocks = len(portfolio)  # number of distinct stocks currently held

        return {
            "total_trades": account["total_trades"],
            "profitable_trades": account["profitable_trades"],
            "current_streak": account["current_streak"],
            "portfolio_stocks": portfolio_stocks,
            "win_rate": (
                account["profitable_trades"] / account["total_trades"] * 100
                if account["total_trades"] > 0
                else 0
            ),
            "realized_profit_loss": account["realized_pl"],
            "max_streak": account["max_streak"],
            "trades_per_day": trades_per_day,
            "achievements_earned": len(self.get_achievements(username)),
        }

    # --- Daily Challenge ---
    def create_daily_challenge(self):
        """Selects 3 random stocks for the daily challenge UI."""
        random.seed(date.today().toordinal())
        challenge_stocks = random.sample(NIFTY_50_STOCKS, 3)
        random.seed()
        return {'date': date.today(), 'stocks': challenge_stocks}

    def has_user_predicted_today(self, username):
        """Checks if the user has already submitted predictions for today."""
        today = date.today().isoformat()
        res = self.sb.table("game_daily_predictions").select("id", count='exact').match(
            {"username": username, "challenge_date": today}).execute()
        return res.count > 0

    def submit_prediction(self, username, predictions):
        """Saves a user's daily predictions to the database."""
        today = date.today().isoformat()
        records_to_insert = [
            {"username": username, "challenge_date": today, "stock_symbol": stock, "prediction": pred}
            for stock, pred in predictions.items()
        ]
        try:
            self.sb.table("game_daily_predictions").upsert(records_to_insert,
                                                           on_conflict='username,challenge_date,stock_symbol').execute()
            return True, "Predictions submitted!"
        except Exception as e:
            return False, f"Error submitting predictions: {e}"

    # --- Game Controls ---
    def reset_account(self, username):
        """Resets a user's game account to its initial state."""
        try:
            self.sb.table("game_trades").delete().eq("username", username).execute()
            self.sb.table("game_achievements").delete().eq("username", username).execute()
            self.sb.table("game_daily_predictions").delete().eq("username", username).execute()
            self.sb.table("game_balances").upsert({
                "username": username, "balance": self.initial_balance, "total_trades": 0,
                "profitable_trades": 0, "realized_pl": 0, "current_streak": 0, "max_streak": 0
            }).execute()
            return True
        except Exception as e:
            st.error(f"Error resetting account: {e}")
            return False