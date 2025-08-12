
# FinVM: Your AI Co-Pilot for the Stock Market

FinVM, is an all-in-one web application designed to make sophisticated stock market analysis accessible through an AI-powered platform. It combines real-time data, machine learning, and interactive tools to help users navigate the Indian stock market.

What FinVM Does

🤖 AI-Driven Predictions: It uses machine learning models to analyze market data and provide predictive insights, suggesting which stocks may be bullish (likely to rise) or bearish (likely to fall).

📊 Comprehensive Market Analysis: The platform offers a live dashboard with market heatmaps, sentiment analysis from news sources, stock clustering, and sector performance charts to give a holistic view of market conditions.

💼 Personal Portfolio Management: Users can track their real-life stock portfolios, monitor their profit and loss, view allocation charts, and receive AI-powered suggestions to optimize their holdings.

🎮 Risk-Free Trading Game: It includes a virtual trading simulator with a starting balance, allowing users to practice their trading strategies on live market data without any real financial risk.

Who It's for:

Retail Investors: Individuals managing their own investments who want to leverage AI and data-driven insights to support their decision-making process.

Finance Students & Enthusiasts: Those learning about the stock market can use the platform's tools for analysis and the trading game to apply their knowledge in a practical, risk-free environment.

Data Science & Tech Hobbyists: Anyone interested in seeing a real-world application of machine learning, data visualization, and web development in the financial technology (FinTech) space.

## Features

🤖 AI & Predictive Analysis
    Ensemble-Based Predictions: Utilizes a powerful Ensemble Learning model, combining RandomForest and GradientBoosting regressors to forecast 10-day stock returns.

    Rich Feature Engineering: Automatically computes over 7 technical and sentiment-based features for each stock, including Momentum, RSI, Volatility, and a custom news sentiment score.

    Transparent AI: Provides Feature Importance charts to show what data drives the AI's decisions, along with a unique AI Confidence Score for each prediction.

    Automated Risk Assessment: Calculates and assigns a risk level (Low, Medium, High) to each stock in your portfolio.

📊 Market & Data Insights
    Live Market Heatmap: Offers a real-time, at-a-glance view of the NIFTY 50's performance.

    Unsupervised Stock Clustering: Employs K-Means Clustering to discover hidden market structures by grouping stocks with similar financial DNA.

    News Sentiment Engine: Scrapes and analyzes financial news to provide a live market sentiment gauge.

📈 Portfolio & Trading Tools
    Personal Portfolio Manager: Track your real-life stock holdings with detailed value and profit/loss analysis.

    Gamified Trading Simulator: A risk-free sandbox with a leaderboard, achievements, and a daily prediction challenge to hone your trading skills on live data.

    PDF Reporting: Generates report on portfolio of user.

⚙️ Platform & User Experience
    Secure User Authentication: Full user login and sign-up system powered by a Supabase backend.

    Interactive Dashboard: A clean, multi-page, and responsive interface built with Streamlit and Plotly.
## Page wise Features


🏠 Main App & Home Page (app.py)
    Secure User Authentication: A complete login/signup system built with Supabase that gates access to the entire application.

    Dynamic Multi-Page UI: A structured multi-page application with a persistent, custom-built sidebar for easy navigation.

    Personalized Welcome Dashboard: The home page displays a user-specific summary, including:

    Live Portfolio Value: A real-time calculation of the user's managed portfolio.

    Game Balance: The user's current virtual currency in the trading game.

    AI Confidence Score: A live metric reflecting the AI's current predictive confidence.

    AI-Powered Alerts: A personalized alert system on the home page that warns users about high-risk holdings in their portfolio and highlights new market opportunities.

    Live NIFTY 50 Heatmap: A visualization of the entire NIFTY 50 index, updated in real-time to show daily performance at a glance.

    Recent Activity Feed: A personalized feed that combines the user's latest game trades and unlocked achievements.

🤖 AI Dashboard (1_Dashboard.py)
    Top-Line AI Metrics: A dashboard row with key performance indicators: AI Confidence, Market Sentiment, AI Signals Generated, and historical Prediction Accuracy.

    Live Prediction Table: An interactive table of NIFTY 50 stocks showing current price, daily change, the AI's 10-day predicted return, the confidence score for that prediction, and a calculated risk level.

    Interactive Prediction Chart: A core feature allowing users to select any stock and view an interactive Plotly chart comparing its historical price against the AI's predicted prices.

    Model Performance Display: The dashboard can display the underlying ML model's performance metrics, RMSE and R² Score, for any selected stock.

    Transparent AI: For any stock, the dashboard can generate a Feature Importance chart, showing exactly which factors (e.g., Momentum, Volatility, Sentiment) the AI used to make its prediction.

    Top 5 AI Picks: Dedicated cards for the Top 5 Bullish and Bearish stock picks from the AI, complete with visual confidence meters.

💼 Portfolio Manager (2_Portfolio.py)
    Real Portfolio Tracking: Functionality for users to add, edit, and remove their real-life stock holdings.

    In-Depth Financial Metrics: The system automatically calculates and displays total portfolio value, total profit/loss, and percentage return.

    Portfolio Allocation Pie Chart: A visualization showing the weight of each stock in the user's portfolio.

    AI-Powered Suggestions: An "AI Portfolio Suggestions" section that provides specific buy/sell recommendations based on the user's current holdings and the AI's latest predictions.

    Tabular Risk Analysis: A table that breaks down the user's portfolio, showing the calculated risk level for each individual holding.

    Watchlist & Export: A separate watchlist for tracking stocks and a feature to export the entire portfolio to a CSV file.

🎮 Virtual Trading Game (3_Trading_Game.py)
    Risk-Free Trading Simulator: A complete trading game with an initial virtual balance of ₹1,00,000.

    Comprehensive Game Statistics: A dashboard tracking the user's win rate, total P&L, profitable trades, and current/max winning streaks.

    Achievement & Badge System: Users can unlock achievements (e.g., 'First Trade', 'Hot Streak', 'Portfolio Builder') for completing certain milestones.

    Live Leaderboard: A ranked leaderboard of all players based on their total virtual portfolio value, showing top traders' win rates and returns.

    Daily Prediction Challenge: An interactive mini-game where users predict the next day's direction for three randomly selected stocks.

    Full Transaction History: A log of every buy and sell transaction made within the game.

🔍 Market Insights (4_Market_Insights.py)
    Unsupervised Stock Clustering: Uses the K-Means algorithm to analyze all NIFTY 50 stocks and group them into clusters based on their financial metrics, presented in an interactive scatter plot.

    Sector Performance Analysis: A bar chart that visualizes the performance of different market sectors (e.g., Banking, IT, Auto).

    Factor Correlation Heatmap: A matrix that shows the correlation between the various financial factors (Momentum, Volatility, etc.), revealing deeper relationships in the market data.

    Market Health Score: A unique, calculated score that combines multiple factors (sentiment, breadth, volatility) to provide a single metric for the overall health of the market.

    News Sentiment Word Cloud: A visual representation of the most common keywords appearing in recent financial news, indicating market focus and sentiment.
## CORE
The core of FinVM is a sophisticated data pipeline that transforms raw market data into actionable, AI-driven insights. It is architected as a multi-stage process, involving data collection, feature engineering, and a hybrid machine learning approach that leverages both supervised and unsupervised learning.

Step 1: Data Collection & Aggregation

The pipeline begins by sourcing data from two distinct streams to create a holistic market view:

    Quantitative Data: The system fetches several years of historical daily stock data for each Nifty 50 constituent from Yahoo Finance via the yfinance library. This includes the Open, High, Low, Close, and Volume (OHLCV) data points, which form the basis for technical analysis.

    Qualitative Data: To capture market sentiment, the platform scrapes the latest financial news headlines from reputable sources using requests and trafilatura. This provides unstructured text data that reflects market psychology.

Step 2: Feature Engineering & Preprocessing
This is the most critical stage, where raw data is transformed into a high-dimensional feature set. This process creates the predictive variables for the machine learning models.

Technical Indicator Calculation: A suite of financial indicators is calculated. Each is chosen to capture a different aspect of a stock's behavior:

    Momentum: The 6-month percentage price change, used to measure the long-term trend strength.

    Volatility: The annualized standard deviation of the stock's daily returns over a 126-day window, quantifying risk and price instability.

    Liquidity: The 60-day average trading volume, acting as a proxy for market interest and the stability of price movements.

    Relative Strength Index (RSI): A momentum oscillator calculated over a 14-day period to identify potential short-term reversal points by measuring if a stock is overbought or oversold.

    Bollinger Band Position: The stock's current price normalized to its position within the 20-day Bollinger Bands. A value near 1 suggests it's near the upper band; a value near 0 suggests it's near the lower band. This helps quantify volatility.

    VWAP Deviation: The percentage deviation of the current price from its 20-day Volume-Weighted Average Price (VWAP), indicating if the stock is trading above or below the recent "fair" value established by trading volume.

    Sentiment Feature Generation: The scraped news headlines undergo Natural Language Processing (NLP) using the VaderSentiment library. This model analyzes the semantic orientation of the text to produce a compound sentiment score from -1 (very negative) to +1 (very positive). This score is then used as a numerical feature, allowing the AI to factor in the impact of market psychology.

    
    Data Scaling: Before being fed into the models, all engineered features are normalized using scikit-learn's StandardScaler. This ensures that all features are on a similar scale, preventing any single feature from disproportionately influencing the model's learning process.

Step 3: Supervised Learning - The Ensemble Prediction Engine

For its predictive task, FinVM uses an Ensemble Learning approach to mitigate the weaknesses of any single model and produce more accurate, generalizable predictions.

    The Prediction Target: The models are trained to solve a regression task: predicting the percentage return of a stock 10 days into the future.

    The Ensemble Models: The engine combines two powerful tree-based algorithms:

Random Forest Regressor: 

        This model excels at handling complex, non-linear relationships between features. It builds a "forest" of hundreds of decision trees on random subsets of the data and averages their outputs, which makes it highly resistant to overfitting.

Gradient Boosting Regressor: 

    This model builds decision trees sequentially, where each new tree is trained to correct the errors of the previous one. This incremental, error-focused approach often leads to very high predictive accuracy.

Step 4: Prediction, Interpretation & Validation

    The Final Prediction: To generate a stock recommendation, the most recent set of features is fed into both the trained Random Forest and Gradient Boosting models. The final predicted 10-day return is the arithmetic mean of their two individual outputs, providing a balanced forecast.

    
    AI Confidence Score: A confidence metric is calculated as 1 - normalized_difference between the two models' predictions. A small difference signifies high agreement and thus a high confidence score.

    
    Feature Importance: To provide transparency, the trained Random Forest model is analyzed to rank which features were most influential in its predictions. This is derived by measuring each feature's contribution to reducing error across all the decision trees in the forest.

    
    Model Validation: The system is built with the capability to evaluate model performance against historical data using standard regression metrics like Root Mean Squared Error (RMSE) and R² Score. This ensures the models are rigorously tested for accuracy.

Step 5: Unsupervised Learning for Market Insights

    In addition to the predictive pipeline, the "Market Insights" page uses the K-Means Clustering algorithm. This is a form of exploratory data analysis that takes the scaled feature set of all 50 stocks and groups them into distinct clusters. This helps reveal the underlying structure of the market, identifying "tribes" of stocks that behave similarly, which can inform diversification strategies or identify relative value opportunities within a peer group.
## Tech-Stack
Frontend:

    Streamlit - For building the interactive web application and user interface.

Backend & Data Processing:

    Python - The core programming language for the entire application.

    Pandas - For high-performance data manipulation, analysis, and time-series handling.

    NumPy - For fundamental numerical operations and calculations.

Database:

    Supabase - For user authentication, portfolio storage, and trading game data (using its PostgreSQL backend).

Machine Learning & AI:

    Scikit-learn - For building the ensemble regression models (RandomForest, GradientBoosting) and for K-Means clustering.

    VaderSentiment & TextBlob - For performing natural language processing (NLP) to analyze financial news sentiment.

Data Fetching & Scraping:

    yfinance - For fetching historical and live stock market data.

    Requests, BeautifulSoup4, & Trafilatura - For scraping web pages to gather news articles for sentiment analysis.

Data Visualization:

    Plotly - For creating interactive and dynamic charts, heatmaps, and gauges.
## Run Locally

Clone the Repository

    git clone https://github.com/hvm007/FinVM.git

Navigate to the Project Directory


    cd FinVM

Create a Virtual Environment

It's highly recommended to use a virtual environment to keep your project's dependencies isolated.


    python -m venv venv

Activate the Virtual Environment

On Windows:

    PowerShell

    .\venv\Scripts\Activate
On macOS/Linux:

    source venv/bin/activate
    
Install Dependencies:

Install all the required libraries from the requirements.txt file.

    pip install -r requirements.txt

Set Up Environment Variables

Create a new file named .env in the root of the project folder.

Add your Supabase credentials to this file like so:

    Code snippet

    SUPABASE_URL="YOUR_SUPABASE_URL_HERE"
    SUPABASE_KEY="YOUR_SUPABASE_KEY_HERE"
Run the Application


    streamlit run app.py
Once the command executes, your default web browser will open with the application running, typically at http://localhost:5000 (for me atleast)
## Environment Variables

To run this project, you will need to create a .env file in the root of the project directory and add the following environment variables. These keys are required to connect to the Supabase backend, which handles user authentication and all database operations.



Your unique URL for your Supabase project

    SUPABASE_URL="YOUR_SUPABASE_URL_HERE"

 Your Supabase project's anon key 
(or service_role key for backend operations)

    SUPABASE_KEY="YOUR_SUPABASE_KEY_HERE"
You can find both your URL and your Key in your Supabase project dashboard under Project Settings > API.

## Deployment

his application is designed to be deployed on Streamlit Community Cloud. Follow these steps to get your own live version of FinVM running.

Push to GitHub

    Ensure your project, including the requirements.txt and .gitignore files, is pushed to a public GitHub repository.

Sign Up for Streamlit Community Cloud

    Create a free account at share.streamlit.io.

It's easiest to sign up using your GitHub account.

    Create a New App

    Once you are logged in, click the "New app" button on your dashboard.

    Choose your FinVM repository from the list, select the main branch, and ensure the Main file path is set to app.py.

Add Your Secrets

This is the most important step for connecting to your database.

    Click on "Advanced settings" and navigate to the "Secrets" section.

Paste your Supabase credentials in the following format:

Ini, TOML

    Supabase Credentials
    SUPABASE_URL="YOUR_SUPABASE_URL_HERE"
    SUPABASE_KEY="YOUR_SUPABASE_KEY_HERE"
    Deploy!

Click the "Deploy!" button. Streamlit will build your application and host it on a public URL. The process usually takes a few minutes.

## Roadmap

This project is actively being developed. Here is a list of planned features and improvements to make FinVM even more powerful.

Enhanced AI Models

    [ ] Integrate time-series models (like LSTM or Prophet) for more advanced price forecasting.

    [ ] Implement portfolio optimization algorithms to suggest ideal asset allocation based on user-defined risk tolerance.

    [ ] Expand the stock universe to include Mid-Cap and Small-Cap stocks for broader analysis.

 Expanded User Features

    [ ] Add advanced order types (Limit Orders, Stop-Loss) to the Trading Game to better simulate real-world trading.

    [ ] Introduce a customizable dashboard where users can select the charts and data they want to see.

    [ ] Develop a system for real-time email or push notifications for critical AI alerts and significant price movements in a user's portfolio.

Platform Expansion

    [ ] Implement a full payment gateway (e.g., Stripe or Razorpay) to automate the "Go Pro" subscription process.

    [ ] Develop a public API to allow other developers to access FinVM's market sentiment data or AI predictions.
## Author

- [@hvm007](https://github.com/hvm007)

