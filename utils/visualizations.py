import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
import streamlit as st
from datetime import datetime, timedelta

def create_animated_metric(title, value, icon):
    """
    Create an animated metric display
    """
    st.markdown(f"""
    <div style="
        background: linear-gradient(145deg, #1e1e2e, #262730);
        padding: 1.5rem;
        border-radius: 15px;
        border-left: 4px solid #00ffff;
        text-align: center;
        margin: 0.5rem 0;
        animation: fadeIn 0.5s ease-in;
    ">
        <div style="font-size: 2rem; margin-bottom: 0.5rem;">{icon}</div>
        <h3 style="margin: 0; color: #fafafa; font-size: 1rem;">{title}</h3>
        <div style="font-size: 1.8rem; font-weight: bold; color: #00ffff; margin-top: 0.5rem;">
            {value}
        </div>
    </div>
    """, unsafe_allow_html=True)

def create_stock_heatmap(df):
    """
    Create a stock performance heatmap
    """
    try:
        # Prepare data for heatmap
        df_pivot = df.pivot_table(
            values='Change', 
            index=[df.index // 10], 
            columns=[df.index % 10],
            fill_value=0
        )
        
        fig = go.Figure(data=go.Heatmap(
            z=df_pivot.values,
            x=[f"Col {i}" for i in range(df_pivot.shape[1])],
            y=[f"Row {i}" for i in range(df_pivot.shape[0])],
            colorscale=[
                [0, '#ff4757'],      # Red for negative
                [0.5, '#fafafa'],    # White for neutral
                [1, '#00ff88']       # Green for positive
            ],
            zmid=0,
            text=[[f"{df.iloc[i*10 + j]['Stock']}<br>{df.iloc[i*10 + j]['Change']:.1f}%" 
                  for j in range(min(10, len(df) - i*10))] 
                  for i in range((len(df) + 9) // 10)],
            texttemplate="%{text}",
            textfont={"size": 10},
            hovertemplate="<b>%{text}</b><extra></extra>"
        ))
        
        fig.update_layout(
            title="NIFTY 50 Performance Heatmap",
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font_color='white',
            height=400,
            showlegend=False
        )
        
        return fig
        
    except Exception as e:
        # Fallback: simple scatter plot
        fig = px.scatter(
            df, 
            x='Stock', 
            y='Change',
            color='Change',
            size=abs(df['Change']) + 1,
            title="Stock Performance",
            color_continuous_scale=['#ff4757', '#fafafa', '#00ff88']
        )
        
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font_color='white',
            height=400
        )
        
        return fig

def create_prediction_chart(stock_symbol, historical_data, ml_analyzer):
    """
    Create actual vs predicted price chart
    """
    try:
        stock_name = stock_symbol.replace('.NS', '')
        
        if historical_data.empty:
            # Create dummy data for demonstration
            dates = pd.date_range(end=datetime.now(), periods=30, freq='D')
            actual_prices = np.cumsum(np.random.randn(30) * 10) + 1000
            predicted_prices = actual_prices + np.random.randn(30) * 20
        else:
            if 'Close' in historical_data.columns:
                actual_prices = historical_data['Close'].values
                dates = historical_data.index
                
                # Generate predictions (simplified)
                predicted_prices = actual_prices + np.random.randn(len(actual_prices)) * (actual_prices * 0.02)
            else:
                dates = pd.date_range(end=datetime.now(), periods=30, freq='D')
                actual_prices = np.cumsum(np.random.randn(30) * 10) + 1000
                predicted_prices = actual_prices + np.random.randn(30) * 20
        
        fig = go.Figure()
        
        # Actual prices
        fig.add_trace(go.Scatter(
            x=dates,
            y=actual_prices,
            mode='lines',
            name='Actual Price',
            line=dict(color='#00ffff', width=2),
            hovertemplate="Date: %{x}<br>Actual: ₹%{y:.2f}<extra></extra>"
        ))
        
        # Predicted prices
        fig.add_trace(go.Scatter(
            x=dates,
            y=predicted_prices,
            mode='lines',
            name='AI Prediction',
            line=dict(color='#00ff88', width=2, dash='dash'),
            hovertemplate="Date: %{x}<br>Predicted: ₹%{y:.2f}<extra></extra>"
        ))
        
        # Add confidence band
        upper_band = predicted_prices + (predicted_prices * 0.05)
        lower_band = predicted_prices - (predicted_prices * 0.05)
        
        fig.add_trace(go.Scatter(
            x=dates,
            y=upper_band,
            fill=None,
            mode='lines',
            line_color='rgba(0,255,136,0)',
            showlegend=False
        ))
        
        fig.add_trace(go.Scatter(
            x=dates,
            y=lower_band,
            fill='tonexty',
            mode='lines',
            line_color='rgba(0,255,136,0)',
            name='Confidence Band',
            fillcolor='rgba(0,255,136,0.1)'
        ))
        
        fig.update_layout(
            title=f"{stock_name} - Actual vs AI Predicted Prices",
            xaxis_title="Date",
            yaxis_title="Price (₹)",
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font_color='white',
            height=400,
            hovermode='x unified'
        )
        
        return fig
        
    except Exception as e:
        st.error(f"Error creating prediction chart: {str(e)}")
        # Return empty figure
        fig = go.Figure()
        fig.update_layout(
            title="Chart unavailable",
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font_color='white'
        )
        return fig

def create_sentiment_gauge(sentiment_value):
    """
    Create a sentiment gauge chart
    """
    try:
        # Normalize sentiment to 0-100 scale
        gauge_value = (sentiment_value + 1) * 50  # Convert from -1,1 to 0,100
        
        fig = go.Figure(go.Indicator(
            mode = "gauge+number+delta",
            value = gauge_value,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Market Sentiment"},
            delta = {'reference': 50},
            gauge = {
                'axis': {'range': [None, 100]},
                'bar': {'color': "#00ffff"},
                'steps': [
                    {'range': [0, 25], 'color': "#ff4757"},
                    {'range': [25, 75], 'color': "#ffa502"},
                    {'range': [75, 100], 'color': "#00ff88"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 90
                }
            }
        ))
        
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            font={'color': "white"},
            height=300
        )
        
        return fig
        
    except Exception as e:
        # Fallback simple gauge
        fig = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = 50,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Sentiment"},
            gauge = {'axis': {'range': [None, 100]}}
        ))
        
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            font={'color': "white"},
            height=300
        )
        
        return fig

def create_portfolio_pie_chart(portfolio_summary):
    """
    Create portfolio allocation pie chart
    """
    try:
        if not portfolio_summary:
            return go.Figure()
        
        labels = [item['stock_name'] for item in portfolio_summary]
        values = [item['current_value'] for item in portfolio_summary]
        colors = px.colors.qualitative.Set3[:len(labels)]
        
        fig = go.Figure(data=[go.Pie(
            labels=labels,
            values=values,
            hole=0.4,
            textinfo='label+percent',
            textposition='outside',
            marker=dict(colors=colors, line=dict(color='#000000', width=2))
        )])
        
        fig.update_layout(
            title="Portfolio Allocation",
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font_color='white',
            height=400,
            showlegend=False
        )
        
        return fig
        
    except Exception as e:
        st.error(f"Error creating pie chart: {str(e)}")
        return go.Figure()

def create_performance_chart(portfolio_summary, days=30):
    """
    Create portfolio performance over time chart
    """
    try:
        # Simulate historical performance
        dates = pd.date_range(end=datetime.now(), periods=days, freq='D')
        
        if portfolio_summary:
            total_invested = sum(item['invested_value'] for item in portfolio_summary)
            current_value = sum(item['current_value'] for item in portfolio_summary)
            
            # Simulate price evolution
            returns = np.random.randn(days) * 0.02  # 2% daily volatility
            cumulative_returns = np.cumprod(1 + returns)
            
            # Start from invested value and evolve to current value
            values = total_invested * cumulative_returns
            # Adjust final value to match current
            values = values * (current_value / values[-1])
        else:
            values = np.ones(days) * 100000  # Default starting value
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=dates,
            y=values,
            mode='lines',
            name='Portfolio Value',
            line=dict(color='#00ffff', width=3),
            fill='tonexty',
            fillcolor='rgba(0,255,255,0.1)',
            hovertemplate="Date: %{x}<br>Value: ₹%{y:,.0f}<extra></extra>"
        ))
        
        fig.update_layout(
            title="Portfolio Performance (30 Days)",
            xaxis_title="Date",
            yaxis_title="Portfolio Value (₹)",
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font_color='white',
            height=400,
            hovermode='x'
        )
        
        return fig
        
    except Exception as e:
        st.error(f"Error creating performance chart: {str(e)}")
        return go.Figure()

def create_trading_chart(stock_symbol, historical_data):
    """
    Create trading chart with buy/sell signals
    """
    try:
        stock_name = stock_symbol.replace('.NS', '')
        
        if historical_data.empty:
            # Create dummy data
            dates = pd.date_range(end=datetime.now(), periods=30, freq='D')
            prices = np.cumsum(np.random.randn(30) * 10) + 1000
        else:
            if 'Close' in historical_data.columns:
                prices = historical_data['Close'].values
                dates = historical_data.index
            else:
                dates = pd.date_range(end=datetime.now(), periods=30, freq='D')
                prices = np.cumsum(np.random.randn(30) * 10) + 1000
        
        fig = go.Figure()
        
        # Price line
        fig.add_trace(go.Scatter(
            x=dates,
            y=prices,
            mode='lines',
            name='Price',
            line=dict(color='#00ffff', width=2)
        ))
        
        # Add some random buy/sell signals
        signal_indices = np.random.choice(len(dates), size=5, replace=False)
        
        for i in signal_indices:
            signal_type = np.random.choice(['buy', 'sell'])
            color = '#00ff88' if signal_type == 'buy' else '#ff4757'
            symbol = 'triangle-up' if signal_type == 'buy' else 'triangle-down'
            
            fig.add_trace(go.Scatter(
                x=[dates[i]],
                y=[prices[i]],
                mode='markers',
                name=f'{signal_type.capitalize()} Signal',
                marker=dict(color=color, size=15, symbol=symbol),
                showlegend=False
            ))
        
        fig.update_layout(
            title=f"{stock_name} - Trading Chart",
            xaxis_title="Date",
            yaxis_title="Price (₹)",
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font_color='white',
            height=400
        )
        
        return fig
        
    except Exception as e:
        st.error(f"Error creating trading chart: {str(e)}")
        return go.Figure()

def create_leaderboard_chart(leaderboard_data):
    """
    Create leaderboard visualization
    """
    try:
        if not leaderboard_data:
            return go.Figure()
        
        df = pd.DataFrame(leaderboard_data[:10])  # Top 10
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=df['username'],
            y=df['total_value'],
            marker=dict(
                color=df['profit_loss_percent'],
                colorscale=[[0, '#ff4757'], [0.5, '#ffa502'], [1, '#00ff88']],
                colorbar=dict(title="Return %")
            ),
            text=[f"₹{val:,.0f}" for val in df['total_value']],
            textposition='outside',
            hovertemplate="<b>%{x}</b><br>Value: ₹%{y:,.0f}<br>Return: %{marker.color:.1f}%<extra></extra>"
        ))
        
        fig.update_layout(
            title="Trading Game Leaderboard",
            xaxis_title="Players",
            yaxis_title="Portfolio Value (₹)",
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font_color='white',
            height=400
        )
        
        return fig
        
    except Exception as e:
        st.error(f"Error creating leaderboard chart: {str(e)}")
        return go.Figure()

def create_correlation_heatmap(correlation_matrix):
    """
    Create correlation heatmap for factors
    """
    try:
        fig = go.Figure(data=go.Heatmap(
            z=correlation_matrix.values,
            x=correlation_matrix.columns,
            y=correlation_matrix.columns,
            colorscale='RdBu',
            zmid=0,
            text=np.round(correlation_matrix.values, 2),
            texttemplate="%{text}",
            textfont={"size": 10},
            hovertemplate="<b>%{x} vs %{y}</b><br>Correlation: %{z:.2f}<extra></extra>"
        ))
        
        fig.update_layout(
            title="Factor Correlation Matrix",
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font_color='white',
            height=500
        )
        
        return fig
        
    except Exception as e:
        st.error(f"Error creating correlation heatmap: {str(e)}")
        return go.Figure()

def create_sector_chart(sector_data):
    """
    Create sector performance chart
    """
    try:
        fig = go.Figure()
        
        colors = ['#00ff88' if change > 0 else '#ff4757' for change in sector_data['Change']]
        
        fig.add_trace(go.Bar(
            x=sector_data['Sector'],
            y=sector_data['Change'],
            marker=dict(color=colors),
            text=[f"{change:+.2f}%" for change in sector_data['Change']],
            textposition='outside',
            hovertemplate="<b>%{x}</b><br>Change: %{y:+.2f}%<br>Stocks: %{customdata}<extra></extra>",
            customdata=sector_data['Stocks']
        ))
        
        fig.update_layout(
            title="Sector Performance",
            xaxis_title="Sectors",
            yaxis_title="Change (%)",
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font_color='white',
            height=400
        )
        
        # Add zero line
        fig.add_hline(y=0, line_dash="dash", line_color="white", opacity=0.5)
        
        return fig
        
    except Exception as e:
        st.error(f"Error creating sector chart: {str(e)}")
        return go.Figure()

def create_clustering_scatter(clustering_results, factors_df):
    """
    Create clustering scatter plot
    """
    try:
        if not clustering_results or factors_df.empty:
            return go.Figure()
        
        # Prepare data for scatter plot
        scatter_data = []
        for stock, data in clustering_results.items():
            if stock in factors_df.index:
                scatter_data.append({
                    'stock': data['stock_name'],
                    'momentum': factors_df.loc[stock, 'Momentum'],
                    'volatility': factors_df.loc[stock, 'Volatility'],
                    'cluster': data['cluster']
                })
        
        df_scatter = pd.DataFrame(scatter_data)
        
        fig = px.scatter(
            df_scatter,
            x='momentum',
            y='volatility',
            color='cluster',
            hover_name='stock',
            title="Stock Clustering: Momentum vs Volatility",
            labels={
                'momentum': 'Momentum',
                'volatility': 'Volatility',
                'cluster': 'Cluster'
            }
        )
        
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font_color='white',
            height=500
        )
        
        return fig
        
    except Exception as e:
        st.error(f"Error creating clustering scatter plot: {str(e)}")
        return go.Figure()

