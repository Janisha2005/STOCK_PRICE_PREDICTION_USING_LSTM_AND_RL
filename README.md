# 📈 Stock Price Prediction Using LSTM and Reinforcement Learning

## 1. Introduction

In the dynamic world of financial markets, accurate stock price prediction and timely trading decisions are crucial for maximizing returns and minimizing risks. Traditional rule-based systems often fall short in capturing the complex, non-linear patterns of market behavior.

This project addresses these challenges by leveraging **LSTM** and **Reinforcement Learning (PPO)** to build a hybrid intelligent system for stock forecasting and trading, using **Apple Inc. (AAPL)** stock as a case study.

- **LSTM (Long Short-Term Memory)**: A type of recurrent neural network ideal for time-series forecasting, used to predict future closing prices.
- **PPO (Proximal Policy Optimization)**: A reinforcement learning algorithm that learns optimal trading strategies (Buy/Hold/Sell) in a simulated environment.

The system is deployed via a **Streamlit** web application, enabling users to input market parameters, view predicted prices, and receive AI-driven trading recommendations.

---

## 2. Objective

- 🔮 Predict future stock prices using a deep learning model (**LSTM**)
- 🤖 Recommend trading actions (**Buy/Hold/Sell**) using a reinforcement learning agent (**PPO**)

---

## 3. Technologies Used

- Python
- Streamlit (for UI)
- TensorFlow / Keras (for LSTM)
- Stable-Baselines3 (for PPO RL agent)
- OpenAI Gym (for environment simulation)
- yFinance (for financial data)
- scikit-learn (for preprocessing)

---

## 4. Workflow Overview

### 4.1. Data Collection and Feature Engineering

- Fetched AAPL stock data using `yfinance`
- Generated technical indicators:
  - `SMA_10`: Simple Moving Average
  - `EMA_10`: Exponential Moving Average

### 4.2. Data Preprocessing

- Handled missing values
- Normalized features using `MinMaxScaler`
- Saved the scaler as `scaler_x.pkl`

### 4.3. PPO RL Agent for Trading

- Created a custom `StockTradingEnv` using **OpenAI Gym**
  - **Observations**: normalized price, balance, stock held
  - **Actions**: `0 = Sell`, `1 = Hold`, `2 = Buy`
- Trained PPO using `stable_baselines3`
- Model saved as `ppo_stock_trading.zip`

---

## 5. Streamlit Application

### 5.1. Features

#### 5.1.1. User Input
- Accepts real-time values for:
  - Open, High, Low, SMA_10, EMA_10

#### 5.1.2. Prediction
- Loads pre-trained LSTM model and scaler
- Predicts the **Close price** based on inputs

#### 5.1.3. Trading Advice
- Initializes PPO agent and simulated trading environment
- Outputs **Buy/Hold/Sell** recommendation

### 5.2. Logic Flow
Load Models → Get User Input → Predict Price → Simulate PPO Agent → Recommend Action

## 6. Evaluation

- 📏 **Metrics Used**:
  - RMSE (Root Mean Squared Error)
  - MAE (Mean Absolute Error)
  - R² Score

- 📊 **Visualization**:
  - Graphical comparison of predicted vs. actual closing prices

## 🚀 Conclusion

This project provides a full pipeline from data collection and model training to deployment, offering a powerful tool for stock prediction and automated trading. By combining deep learning and reinforcement learning, it opens new possibilities for intelligent financial decision-making.
