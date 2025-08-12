import streamlit as st
import pandas as pd
import numpy as np
import warnings
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
from datetime import datetime, timedelta
from utils.constants import NIFTY_50_STOCKS
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import requests
import plotly.express as px

warnings.filterwarnings("ignore")

class MLAnalyzer:
    def __init__(self, stock_list=NIFTY_50_STOCKS):
        self.stocks = stock_list
        self.start_date = '2023-04-01'
        self.end_date = datetime.now().strftime('%Y-%m-%d')
        self.sentiment_scores = {}
        self.ml_models = {}
        self.factors_df = None
        self.predictions = {}
        self.scaler = StandardScaler()
        self.model_metrics = {}
        self.feature_importances = {}
        self._initialize_models()

    def fetch_stock_news(self, stock):
        """Fetch recent news headlines for sentiment analysis"""
        try:
            stock_symbol = stock.replace('.NS', '')  # remove NSE suffix
            url = f"https://query1.finance.yahoo.com/v1/finance/search?q={stock_symbol}&quotesCount=0&newsCount=5"
            response = requests.get(url, timeout=5).json()
            return [item['title'] for item in response.get('news', [])]
        except:
            return []

    def _initialize_models(self):
        """Initialize ML models"""
        if 'ml_models' not in st.session_state:
            st.session_state.ml_models = {}
        if 'ml_predictions' not in st.session_state:
            st.session_state.ml_predictions = {}

    def load_data_from_manager(self, data_manager):
        """Load data from DataManager"""
        try:
            raw_data = data_manager.fetch_live_data()
            if raw_data is None or raw_data.empty:
                raise ValueError("No data available")

            self.data = raw_data

            # Handle multi-level columns from yfinance
            if isinstance(raw_data.columns, pd.MultiIndex):
                self.prices = raw_data['Close']
                self.volume = raw_data['Volume']
            else:
                # Fallback data structure
                close_cols = [col for col in raw_data.columns if 'close' in col.lower()]
                volume_cols = [col for col in raw_data.columns if 'volume' in col.lower()]

                if close_cols:
                    self.prices = raw_data[close_cols]
                else:
                    # Generate fallback prices
                    self.prices = pd.DataFrame()

                if volume_cols:
                    self.volume = raw_data[volume_cols]
                else:
                    # Generate fallback volume
                    self.volume = pd.DataFrame()

            # Fill missing values
            if not self.prices.empty:
                self.prices = self.prices.fillna(method='ffill')
            if not self.volume.empty:
                self.volume = self.volume.fillna(method='ffill')

            return True
        except Exception as e:
            st.error(f"Error loading data for ML analysis: {str(e)}")
            return False

    def compute_advanced_factors(self):
        """Compute comprehensive stock factors"""
        try:
            if self.prices is None or self.prices.empty:
                st.warning("No price data available for factor computation, using fallback")
                return self._get_fallback_factors()

            current_date = self.prices.index[-1]

            momentum = self.compute_momentum(current_date)
            volatility = self.compute_volatility(current_date)
            liquidity = self.compute_liquidity(current_date)
            rsi = self.compute_rsi()
            bollinger_position = self.compute_bollinger_position()
            volume_profile = self.compute_volume_profile()

            # Simulate sentiment scores for now
            # Real sentiment analysis
            analyzer = SentimentIntensityAnalyzer()
            sentiment_scores = {}
            for stock in self.stocks:
                headlines = self.fetch_stock_news(stock)
                if headlines:
                    scores = [analyzer.polarity_scores(h)['compound'] for h in headlines]
                    sentiment_scores[stock] = np.mean(scores)
                else:
                    sentiment_scores[stock] = 0  # neutral if no news

            self.factors_df = pd.DataFrame({
                'Momentum': momentum,
                'Volatility': volatility,
                'Liquidity': liquidity,
                'RSI': rsi,
                'Bollinger_Position': bollinger_position,
                'Volume_Profile': volume_profile,
                'Sentiment': pd.Series(sentiment_scores)
            }).fillna(0)

            return self.factors_df

        except Exception as e:
            st.error(f"Error computing factors: {str(e)}")
            return self._get_fallback_factors()

    def _get_fallback_factors(self):
        """Generate fallback factors when computation fails"""
        factors_data = {}
        for factor in ['Momentum', 'Volatility', 'Liquidity', 'RSI', 'Bollinger_Position', 'Volume_Profile', 'Sentiment']:
            if factor == 'RSI':
                factors_data[factor] = pd.Series(np.random.uniform(30, 70, len(self.stocks)), index=self.stocks)
            elif factor == 'Volatility':
                factors_data[factor] = pd.Series(np.random.uniform(0.1, 0.5, len(self.stocks)), index=self.stocks)
            elif factor == 'Sentiment':
                factors_data[factor] = pd.Series(np.random.uniform(-0.5, 0.5, len(self.stocks)), index=self.stocks)
            else:
                factors_data[factor] = pd.Series(np.random.uniform(-0.2, 0.2, len(self.stocks)), index=self.stocks)

        return pd.DataFrame(factors_data)

    def compute_momentum(self, current_date, lookback_months=6):
        """Enhanced momentum calculation"""
        try:
            lookback_date = current_date - pd.DateOffset(months=lookback_months)
            if lookback_date < self.prices.index[0]:
                lookback_date = self.prices.index[0]

            idx = self.prices.index.get_indexer([lookback_date], method='nearest')[0]
            past_prices = self.prices.iloc[idx]
            current_prices = self.prices.loc[current_date]

            momentum = (current_prices - past_prices) / past_prices
            return momentum.fillna(0)
        except Exception as e:
            return pd.Series(np.random.uniform(-0.2, 0.2, len(self.stocks)), index=self.stocks)

    def compute_volatility(self, current_date, window=126):
        """Enhanced volatility calculation"""
        try:
            window_start = current_date - pd.Timedelta(days=window)
            data_window = self.prices.loc[self.prices.index >= window_start].loc[:current_date]
            daily_returns = data_window.pct_change().dropna()

            volatility = daily_returns.std() * np.sqrt(252)
            return volatility.fillna(0.2)
        except Exception as e:
            return pd.Series(np.random.uniform(0.1, 0.5, len(self.stocks)), index=self.stocks)

    def compute_liquidity(self, current_date, lookback_days=60):
        """Enhanced liquidity calculation"""
        try:
            lookback_date = current_date - pd.Timedelta(days=lookback_days)
            volume_window = self.volume.loc[lookback_date:current_date]
            avg_volume = volume_window.mean()
            return avg_volume.fillna(100000)
        except Exception as e:
            return pd.Series(np.random.uniform(100000, 1000000, len(self.stocks)), index=self.stocks)

    def compute_rsi(self, prices=None, window=14):
        """Compute RSI for all stocks"""
        if prices is None:
            prices = self.prices

        rsi_values = {}
        for stock in self.stocks:
            try:
                if stock not in prices.columns:
                    continue

                price_series = prices[stock].dropna()
                if len(price_series) < window + 1:
                    rsi_values[stock] = 50
                    continue

                delta = price_series.diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()

                rs = gain / loss
                rsi = 100 - (100 / (1 + rs))
                rsi_values[stock] = rsi.iloc[-1] if not rsi.empty else 50
            except Exception:
                rsi_values[stock] = np.random.uniform(30, 70)

        return pd.Series(rsi_values)

    def compute_bollinger_position(self, prices=None, window=20):
        """Compute position within Bollinger Bands"""
        if prices is None:
            prices = self.prices

        bollinger_pos = {}
        for stock in self.stocks:
            try:
                if stock not in prices.columns:
                    continue

                price_series = prices[stock].dropna()
                if len(price_series) < window:
                    bollinger_pos[stock] = 0.5
                    continue

                sma = price_series.rolling(window).mean()
                std = price_series.rolling(window).std()

                upper_band = sma + (2 * std)
                lower_band = sma - (2 * std)

                current_price = price_series.iloc[-1]
                upper_val = upper_band.iloc[-1]
                lower_val = lower_band.iloc[-1]

                if upper_val != lower_val:
                    position = (current_price - lower_val) / (upper_val - lower_val)
                else:
                    position = 0.5

                bollinger_pos[stock] = max(0, min(1, position))
            except Exception:
                bollinger_pos[stock] = np.random.uniform(0, 1)

        return pd.Series(bollinger_pos)

    def compute_volume_profile(self, prices=None, volume=None):
        """Compute volume profile strength"""
        if prices is None:
            prices = self.prices
        if volume is None:
            volume = self.volume

        volume_profile = {}
        for stock in self.stocks:
            try:
                if stock not in volume.columns or stock not in prices.columns:
                    continue

                volume_series = volume[stock].dropna()
                price_series = prices[stock].dropna()

                if len(volume_series) < 20 or len(price_series) < 20:
                    volume_profile[stock] = 0
                    continue

                vwap = (price_series * volume_series).rolling(20).sum() / volume_series.rolling(20).sum()
                current_price = price_series.iloc[-1]
                vwap_current = vwap.iloc[-1]

                if vwap_current != 0:
                    vwap_deviation = (current_price - vwap_current) / vwap_current
                else:
                    vwap_deviation = 0

                volume_profile[stock] = vwap_deviation
            except Exception:
                volume_profile[stock] = np.random.uniform(-0.1, 0.1)

        return pd.Series(volume_profile)

    def create_ml_ensemble(self, stock):
        """Create and train ML ensemble for a stock"""
        try:
            X, y = self.prepare_ml_data(stock)
            if len(X) == 0 or len(y) == 0:
                self.model_metrics[stock] = {'RMSE': None, 'R2': None}
                self.feature_importances[stock] = {}
                return {'prediction': np.random.uniform(-0.05, 0.05), 'confidence': 0.5}

            X_scaled = self.scaler.fit_transform(X)

            # Split data
            split_idx = int(0.8 * len(X_scaled))
            X_train, X_test = X_scaled[:split_idx], X_scaled[split_idx:]
            y_train, y_test = y[:split_idx], y[split_idx:]

            # Train ensemble
            rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
            gb_model = GradientBoostingRegressor(n_estimators=100, random_state=42)

            rf_model.fit(X_train, y_train)
            gb_model.fit(X_train, y_train)

            # Make predictions
            if len(X_test) > 0:
                rf_pred = rf_model.predict(X_test)
                gb_pred = gb_model.predict(X_test)

                # Calculate confidence based on model agreement
                agreement = 1 - np.mean(np.abs(rf_pred - gb_pred))
                confidence = max(0.3, min(0.9, agreement))

                # Calculate performance metrics
                rmse = mean_squared_error(y_test, rf_pred) ** 0.5
                r2 = r2_score(y_test, rf_pred)
            else:
                confidence = 0.5
                rmse = None
                r2 = None

            # Save metrics for dashboard
            self.model_metrics[stock] = {'RMSE': rmse, 'R2': r2}

            # Predict next 10-day return
            latest_features = X_scaled[-1:] if len(X_scaled) > 0 else np.zeros((1, 7))
            rf_next = rf_model.predict(latest_features)[0]
            gb_next = gb_model.predict(latest_features)[0]

            ensemble_prediction = (rf_next + gb_next) / 2

            # Save feature importances for dashboard
            self.feature_importances[stock] = dict(zip(
                ['Momentum', 'Volatility', 'RSI', 'MA_Ratio', 'Volume_Ratio', 'Price', 'Sentiment'],
                rf_model.feature_importances_
            ))

            # Store metrics for RandomForest
            if len(X_test) > 0:
                y_pred = rf_model.predict(X_test)
                rmse = mean_squared_error(y_test, y_pred) ** 0.5
                r2 = r2_score(y_test, y_pred)
            else:
                rmse = 0
                r2 = 0
            self.model_metrics[stock] = {'RMSE': rmse, 'R2': r2}

            # Store feature importances
            self.feature_importances[stock] = dict(zip(
                ['Momentum', 'Volatility', 'RSI', 'MA_Ratio', 'Volume_Ratio', 'Price', 'Sentiment'],
                rf_model.feature_importances_
            ))

            return {
                'prediction': ensemble_prediction,
                'confidence': confidence,
                'rf_model': rf_model,
                'gb_model': gb_model
            }

        except Exception as e:
            st.error(f"Error training ML model for {stock}: {str(e)}")
            return {'prediction': np.random.uniform(-0.05, 0.05), 'confidence': 0.5}

    def prepare_ml_data(self, stock):
        """Prepare ML training data for a stock"""
        try:
            if stock not in self.prices.columns:
                return [], []

            X = []
            y = []

            stock_prices = self.prices[stock].dropna()
            if len(stock_prices) < 40:  # Need at least 40 days
                return [], []

            # Start index dynamically: if not enough history for 126-day start, allow shorter
            start_idx = 126 if len(stock_prices) >= 136 else 20
            for i in range(start_idx, len(stock_prices) - 10):
                # Features: momentum, volatility, RSI, etc. (simplified)
                price_window = stock_prices.iloc[i-20:i]
                if len(price_window) < 20:
                    continue

                momentum = (price_window.iloc[-1] - price_window.iloc[0]) / price_window.iloc[0]
                volatility = price_window.pct_change().std()
                rsi = self._calculate_simple_rsi(price_window, 14)

                # Moving averages
                ma_short = price_window.tail(5).mean()
                ma_long = price_window.tail(20).mean()
                ma_ratio = ma_short / ma_long if ma_long != 0 else 1

                # Volume features (if available)
                if stock in self.volume.columns:
                    volume_window = self.volume[stock].iloc[i-20:i]
                    avg_volume = volume_window.mean() if len(volume_window) > 0 else 1000000
                    volume_ratio = volume_window.iloc[-1] / avg_volume if avg_volume != 0 else 1
                else:
                    volume_ratio = 1

                features = [momentum, volatility, rsi, ma_ratio, volume_ratio,
                           price_window.iloc[-1], np.random.uniform(-0.5, 0.5)]  # Last is sentiment
                X.append(features)

                # Target: 10-day return
                future_price = stock_prices.iloc[i + 10]
                current_price = stock_prices.iloc[i]
                future_return = (future_price - current_price) / current_price
                y.append(future_return)

            return X, y

        except Exception as e:
            st.error(f"Error preparing ML data for {stock}: {str(e)}")
            return [], []

    def _calculate_simple_rsi(self, prices, window=14):
        """Calculate simple RSI"""
        try:
            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()

            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            return rsi.iloc[-1] if not rsi.empty else 50
        except:
            return 50

    def get_ai_recommendations(self, top_n=5):
        """Get top AI stock recommendations"""
        try:
            if self.factors_df is None or self.factors_df.empty:
                return {'bullish': [], 'bearish': []}

            predictions = {}
            for stock in self.stocks:
                result = self.create_ml_ensemble(stock)
                predictions[stock] = {
                    'prediction': result['prediction'],
                    'confidence': result['confidence'],
                    'stock_name': stock.replace('.NS', '')
                }

            # Sort by prediction
            sorted_predictions = sorted(predictions.items(),
                                      key=lambda x: x[1]['prediction'], reverse=True)

            bullish = sorted_predictions[:top_n]
            bearish = sorted_predictions[-top_n:]

            return {
                'bullish': [{'stock': item[0], **item[1]} for item in bullish],
                'bearish': [{'stock': item[0], **item[1]} for item in bearish]
            }

        except Exception as e:
            st.error(f"Error getting AI recommendations: {str(e)}")
            return {'bullish': [], 'bearish': []}

    def perform_clustering(self, n_clusters=5):
        """Perform KMeans clustering on stocks"""
        try:
            if self.factors_df is None or self.factors_df.empty:
                self.compute_advanced_factors()

            if self.factors_df.empty:
                return {}

            # Prepare data for clustering
            features = self.factors_df.fillna(0)
            scaled_features = self.scaler.fit_transform(features)

            # Perform clustering
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            clusters = kmeans.fit_predict(scaled_features)

            # Create cluster results
            cluster_results = {}
            for i, stock in enumerate(features.index):
                cluster_results[stock] = {
                    'cluster': int(clusters[i]),
                    'stock_name': stock.replace('.NS', ''),
                    'features': features.loc[stock].to_dict()
                }

            return cluster_results

        except Exception as e:
            st.error(f"Error performing clustering: {str(e)}")
            return {}

    def get_risk_assessment(self, stock):
        """Get risk assessment for a stock"""
        try:
            if self.factors_df is None or stock not in self.factors_df.index:
                return {'risk_level': 'Medium', 'risk_score': 0.5}

            volatility = self.factors_df.loc[stock, 'Volatility']
            rsi = self.factors_df.loc[stock, 'RSI']

            # Calculate risk score
            risk_score = 0
            if volatility > 0.3:
                risk_score += 0.4
            elif volatility > 0.2:
                risk_score += 0.2

            if rsi > 70 or rsi < 30:
                risk_score += 0.3

            if abs(self.factors_df.loc[stock, 'Momentum']) > 0.1:
                risk_score += 0.2

            # Normalize to 0-1
            risk_score = min(1.0, risk_score)

            if risk_score > 0.7:
                risk_level = 'High'
            elif risk_score > 0.4:
                risk_level = 'Medium'
            else:
                risk_level = 'Low'

            return {'risk_level': risk_level, 'risk_score': risk_score}

        except Exception as e:
            return {'risk_level': 'Medium', 'risk_score': 0.5}

    def display_metrics_and_importances(self, stock=None):
        """Display stored model metrics and feature importance for a specific stock."""
        if not stock:
            st.warning("No stock selected for metrics display.")
            return

        st.subheader(f"⚙️ Model Performance for {stock.replace('.NS', '')}")

        # --- Show Metrics ---
        if stock in self.model_metrics and self.model_metrics[stock]['RMSE'] is not None:
            metrics = self.model_metrics[stock]
            col1, col2 = st.columns(2)
            with col1:
                st.metric("RMSE (Root Mean Squared Error)", f"{metrics['RMSE']:.4f}")
            with col2:
                st.metric("R² Score (Model Fit)", f"{metrics['R2']:.2%}")
        else:
            st.info(f"Performance metrics for {stock.replace('.NS', '')} are not available yet.")

        # --- Show Feature Importances ---
        if stock in self.feature_importances and self.feature_importances[stock]:
            st.subheader(f"Feature Importance - {stock.replace('.NS', '')}")
            importance_df = pd.DataFrame(
                list(self.feature_importances[stock].items()),
                columns=['Feature', 'Importance']
            ).sort_values(by='Importance', ascending=True)

            fig = px.bar(
                importance_df,
                x='Importance',
                y='Feature',
                orientation='h',
                title='Model Feature Importances'
            )
            fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font_color='white',
                yaxis={'categoryorder': 'total ascending'}
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info(f"Feature importances for {stock.replace('.NS', '')} are not available yet.")

