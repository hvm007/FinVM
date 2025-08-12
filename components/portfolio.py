import streamlit as st
import pandas as pd
import numpy as np
from components.database import get_connection
from utils.constants import NIFTY_50_STOCKS

class PortfolioManager:
    def __init__(self):
        self.sb=get_connection()
        self._initialize_portfolio()
        self._initialize_watchlists()

    def _initialize_portfolio(self):
        if 'portfolios' not in st.session_state:
            st.session_state.portfolios = {}

    def _initialize_watchlists(self):
        if 'watchlists' not in st.session_state:
            st.session_state.watchlists = {}

    def _load_portfolio_from_db(self, username):
        try:
            # Corrected: Use Supabase select
            res = self.sb.table("portfolio").select("stock_symbol, quantity, avg_price").eq("username", username).execute()

            if username not in st.session_state.portfolios:
                st.session_state.portfolios[username] = {}

            # Corrected: Iterate over response data
            for row in res.data:
                st.session_state.portfolios[username][row['stock_symbol']] = {
                    'quantity': row['quantity'],
                    'average_price': row['avg_price']
                }
        except Exception as e:
            st.warning(f"Could not load portfolio from DB for {username}: {e}")

    def _save_stock_to_db(self, username, stock_symbol, quantity, avg_price):
        try:
            self.sb.table("portfolio").upsert({
                "username": username,
                "stock_symbol": stock_symbol,
                "quantity": quantity,
                "avg_price": avg_price
            }).execute()
        except Exception as e:
            st.warning(f"Could not save portfolio row to DB: {e}")

    def _delete_stock_from_db(self, username, stock_symbol):
        try:
            self.sb.table("portfolio").delete().match({
                "username": username,
                "stock_symbol": stock_symbol
            }).execute()
        except Exception as e:
            st.warning(f"Could not delete portfolio row from DB: {e}")

    def get_user_portfolio(self, username):
        """Get user's portfolio"""
        if username not in st.session_state.portfolios:
            st.session_state.portfolios[username] = {}
            self._load_portfolio_from_db(username)
        return st.session_state.portfolios[username]

    def add_to_portfolio(self, username, stock, quantity, price, transaction_type='buy'):
        portfolio = self.get_user_portfolio(username)
        if stock not in portfolio:
            portfolio[stock] = {'quantity': 0, 'average_price': 0, 'total_invested': 0, 'transactions': []}
        stock_data = portfolio[stock]
        if transaction_type == 'buy':
            new_total_invested = stock_data.get('total_invested', 0) + (quantity * price)
            new_quantity = stock_data['quantity'] + quantity
            if new_quantity > 0:
                stock_data['average_price'] = new_total_invested / new_quantity
            stock_data['quantity'] = new_quantity
            stock_data['total_invested'] = new_total_invested
        elif transaction_type == 'sell':
            if stock_data['quantity'] >= quantity:
                stock_data['quantity'] -= quantity
                if stock_data['quantity'] == 0:
                    stock_data['total_invested'] = 0
                    stock_data['average_price'] = 0
                else:
                    stock_data['total_invested'] -= (quantity * stock_data['average_price'])
            else:
                return False, "Insufficient quantity to sell"
        if stock_data['quantity'] == 0:
            del portfolio[stock]
            self._delete_stock_from_db(username, stock)
        else:
            self._save_stock_to_db(username, stock, stock_data['quantity'], stock_data['average_price'])
        return True, "Transaction completed successfully"

    def remove_from_portfolio(self, username, stock, quantity=None):
        portfolio = self.get_user_portfolio(username)
        if stock not in portfolio:
            return False, "Stock not found in portfolio"
        if quantity is None:
            del portfolio[stock]
            self._delete_stock_from_db(username, stock)
            return True, "Stock removed from portfolio"
        else:
            stock_data = portfolio[stock]
            if stock_data['quantity'] >= quantity:
                stock_data['quantity'] -= quantity
                if stock_data['quantity'] == 0:
                    del portfolio[stock]
                    self._delete_stock_from_db(username, stock)
                else:
                    self._save_stock_to_db(username, stock, stock_data['quantity'], stock_data['average_price'])
                return True, f"Sold {quantity} shares"
            else:
                return False, "Insufficient quantity"
    def get_portfolio_value(self, username, current_prices):
        """Calculate current portfolio value"""
        portfolio = self.get_user_portfolio(username)
        total_value = 0
        total_invested = 0

        for stock, data in portfolio.items():
            quantity = data['quantity']
            avg_price = data['average_price']
            current_price = current_prices.get(stock, avg_price)

            current_value = quantity * current_price
            invested_value = quantity * avg_price

            total_value += current_value
            total_invested += invested_value

        return {
            'current_value': total_value,
            'invested_value': total_invested,
            'profit_loss': total_value - total_invested,
            'profit_loss_percent': ((total_value - total_invested) / total_invested * 100) if total_invested > 0 else 0
        }

    def get_total_value(self, username=None):
        """Get total portfolio value"""
        if username is None:
            # Get current user
            if 'current_user' in st.session_state and st.session_state.current_user:
                username = st.session_state.current_user
            else:
                return 0

        try:
            data_manager = st.session_state.get('data_manager')
            if data_manager:
                current_prices = data_manager.get_latest_prices()
            else:
                current_prices = {}
        except Exception:
            current_prices = {}

        # If no prices, simulate for calculation
        if not current_prices:
            current_prices = {stock: np.random.uniform(100, 2000) for stock in NIFTY_50_STOCKS}

        portfolio_data = self.get_portfolio_value(username, current_prices)
        return portfolio_data['current_value']

    def get_portfolio_summary(self, username, current_prices):
        """Get detailed portfolio summary"""
        portfolio = self.get_user_portfolio(username)
        summary = []

        for stock, data in portfolio.items():
            quantity = data['quantity']
            avg_price = data['average_price']
            current_price = current_prices.get(stock, avg_price)

            current_value = quantity * current_price
            invested_value = quantity * avg_price
            profit_loss = current_value - invested_value
            profit_loss_percent = (profit_loss / invested_value * 100) if invested_value > 0 else 0

            summary.append({
                'stock': stock,
                'stock_name': stock.replace('.NS', ''),
                'quantity': quantity,
                'average_price': avg_price,
                'current_price': current_price,
                'invested_value': invested_value,
                'current_value': current_value,
                'profit_loss': profit_loss,
                'profit_loss_percent': profit_loss_percent,
                'weight': 0  # Will be calculated after all stocks are processed
            })

        # Calculate weights
        total_value = sum(item['current_value'] for item in summary)
        for item in summary:
            item['weight'] = (item['current_value'] / total_value * 100) if total_value > 0 else 0

        return summary

    def get_watchlist(self, username):  # <-- added self here
        if username not in st.session_state.watchlists:
            st.session_state.watchlists[username] = []
        return st.session_state.watchlists[username]

    def add_to_watchlist(self, username, stock):
        """Add stock to watchlist"""
        watchlist = self.get_watchlist(username)
        if stock not in watchlist:
            watchlist.append(stock)
            return True, "Stock added to watchlist"
        return False, "Stock already in watchlist"

    def remove_from_watchlist(self, username, stock):
        """Remove stock from watchlist"""
        watchlist = self.get_watchlist(username)
        if stock in watchlist:
            watchlist.remove(stock)
            return True, "Stock removed from watchlist"
        return False, "Stock not found in watchlist"

    def get_portfolio_metrics(self, username, current_prices):
        """Get advanced portfolio metrics"""
        portfolio = self.get_user_portfolio(username)
        if not portfolio:
            return {}

        summary = self.get_portfolio_summary(username, current_prices)

        # Calculate metrics
        total_value = sum(item['current_value'] for item in summary)
        total_invested = sum(item['invested_value'] for item in summary)

        # Diversification metrics
        weights = [item['weight'] for item in summary]
        diversification_ratio = len(weights) / (1 + np.std(weights)) if len(weights) > 1 else 1

        # Risk metrics (simplified)
        returns = [item['profit_loss_percent'] for item in summary]
        portfolio_return = np.mean(returns) if returns else 0
        portfolio_volatility = np.std(returns) if len(returns) > 1 else 0

        return {
            'total_stocks': len(portfolio),
            'total_value': total_value,
            'total_invested': total_invested,
            'total_profit_loss': total_value - total_invested,
            'total_profit_loss_percent': (
                        (total_value - total_invested) / total_invested * 100) if total_invested > 0 else 0,
            'diversification_ratio': diversification_ratio,
            'portfolio_return': portfolio_return,
            'portfolio_volatility': portfolio_volatility,
            'best_performer': max(summary, key=lambda x: x['profit_loss_percent']) if summary else None,
            'worst_performer': min(summary, key=lambda x: x['profit_loss_percent']) if summary else None
        }

    def get_ai_suggestions(self, username, ml_analyzer, current_prices):
        """Get AI-powered portfolio suggestions"""
        portfolio = self.get_user_portfolio(username)
        suggestions = []

        try:
            # Get AI recommendations
            recommendations = ml_analyzer.get_ai_recommendations()
            portfolio_stocks = set(portfolio.keys())

            # Suggest buys for new opportunities
            for rec in recommendations.get('bullish', [])[:3]:
                stock = rec['stock']
                if stock not in portfolio_stocks:
                    suggestions.append({
                        'type': 'buy',
                        'stock': stock,
                        'stock_name': rec.get('stock_name', stock.replace('.NS', '')),
                        'reason': f"AI predicts {rec.get('prediction', 0) * 100:.1f}% return (Confidence: {rec.get('confidence', 0) * 100:.1f}%)",
                        'confidence': rec.get('confidence', 0)
                    })

            # Suggest sales for poor performers
            for rec in recommendations.get('bearish', [])[:2]:
                stock = rec['stock']
                if stock in portfolio_stocks:
                    suggestions.append({
                        'type': 'sell',
                        'stock': stock,
                        'stock_name': rec.get('stock_name', stock.replace('.NS', '')),
                        'reason': f"AI predicts {rec.get('prediction', 0) * 100:.1f}% decline (Confidence: {rec.get('confidence', 0) * 100:.1f}%)",
                        'confidence': rec.get('confidence', 0)
                    })

            return suggestions

        except Exception as e:
            st.error(f"Error getting AI suggestions: {str(e)}")
            return []

    def calculate_portfolio_correlation(self, username, data_manager):
        """Calculate correlation matrix for portfolio stocks"""
        portfolio = self.get_user_portfolio(username)
        if len(portfolio) < 2:
            return pd.DataFrame()

        try:
            # Get historical data for portfolio stocks
            stock_list = list(portfolio.keys())
            data = data_manager.fetch_live_data(stock_list)

            if 'Close' not in data.columns:
                return pd.DataFrame()

            prices = data['Close'][stock_list]
            returns = prices.pct_change().dropna()
            correlation_matrix = returns.corr()

            return correlation_matrix

        except Exception as e:
            st.error(f"Error calculating correlation: {str(e)}")
            return pd.DataFrame()

    def export_portfolio(self, username):
        """Export portfolio to CSV format"""
        portfolio = self.get_user_portfolio(username)
        if not portfolio:
            return None

        export_data = []
        for stock, data in portfolio.items():
            export_data.append({
                'Stock': stock.replace('.NS', ''),
                'Quantity': data['quantity'],
                'Average_Price': data['average_price'],
                'Total_Invested': data['total_invested'],
                'Transactions_Count': len(data['transactions'])
            })

        return pd.DataFrame(export_data)
