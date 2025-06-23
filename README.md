# STOCK PRICE PREDICTION USING LSTM AND REINFORCEMENT LEARNING

## Introduction
  In the dynamic world of financial markets, accurate stock price prediction and timely trading decisions are crucial for maximizing returns and minimizing risks. Traditional rule-based systems often fall short in capturing the complex, non-linear patterns of market behavior. This project addresses these challenges by leveraging LSTM and Reinforcement Learning techniques to build a hybrid intelligent system for stock forecasting and trading.
The project focuses on Apple Inc. (AAPL) stock as a case study and integrates two powerful components:
### Long Short-Term Memory (LSTM) – 
  A type of recurrent neural network suited for time-series data, used here to predict future closing prices based on historical indicators like open, high, low prices, and moving averages.
### Proximal Policy Optimization (PPO) –
  A reinforcement learning algorithm that learns optimal trading strategies (Buy/Hold/Sell) through interaction with a simulated trading environment.

The system is deployed via a Streamlit web application, allowing users to input market parameters, receive predicted prices, and get AI-driven trading advice.
This documentation outlines the methodology, model architecture, data pipeline, evaluation metrics, and user interface of the project, demonstrating a complete solution from training to deployment for stock market prediction and autonomous trading.

## Objective
To build a hybrid intelligent system that:
•	Predicts future stock prices using a deep learning model (LSTM).
•	Provides trading recommendations (Buy/Hold/Sell) using a Reinforcement Learning agent (PPO).


## Technologies Used
•	Python
•	Streamlit for UI
•	TensorFlow/Keras for LSTM model
•	Stable-Baselines3 (PPO) for RL agent
•	Gym for environment simulation
•	yFinance for financial data
•	scikit-learn for preprocessing

## Workflow Overview
### Data Collection and Feature Engineering
•	Collected AAPL stock data using yfinance.
•	Generated additional features:
  o	 SMA_10 (Simple Moving Average)
  o	EMA_10 (Exponential Moving Average)

### Data Preprocessing
•	Handled missing values.
•	Normalized input features using MinMaxScaler.
•	Saved the scaler to scaler_x.pkl.

### PPO RL Agent for Trading
•	Created a custom StockTradingEnv using OpenAI Gym:
  o	Observations: normalized price, balance, stock held
  o	Actions: 0 = Sell, 1 = Hold, 2 = Buy
•	Trained PPO using stable_baselines3.
•	Model saved to ppo_stock_trading.zip.


## Streamlit Application
### Features
#### User Input: 
Accepts real-time inputs for:
•	Open, High, Low, SMA_10, EMA_10
#### Prediction:
•	Loads the LSTM model and scaler.
•	Predicts the Close price based on inputs.
#### Trading Advice:
•	Initializes PPO agent using previously trained environment.
•	Simulates environment and provides Buy/Hold/Sell suggestion.
### Logic Flow:
Load models → Get user input → Predict price → Simulate PPO agent → Recommend action

## Evaluation
•	Metrics used:
  o	RMSE (Root Mean Squared Error)
  o	MAE (Mean Absolute Error)
  o	R² Score
•	Visual comparison: predicted vs. actual closing prices

