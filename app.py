import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import joblib
import yfinance as yf
from stable_baselines3 import PPO
import gym
from gym import spaces

class StockTradingEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, prices, initial_balance=10000):
        super(StockTradingEnv, self).__init__()

        self.prices = prices
        self.initial_balance = initial_balance
        self.n_steps = len(prices)

        self.action_space = spaces.Discrete(3)  # Sell, Hold, Buy
        self.max_price = np.max(prices)
        self.max_stock = 1000  # adjust if needed
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(3,), dtype=np.float32)

        self.reset()

    def reset(self):
        self.current_step = 0
        self.balance = self.initial_balance
        self.stock_held = 0
        self.total_value = self.balance
        return self._get_observation()

    def _get_observation(self):
        current_price_raw = self.prices[self.current_step]

        if isinstance(current_price_raw, (np.ndarray, list)):
            current_price_raw = current_price_raw[0] 
        current_price = float(current_price_raw) / float(self.max_price)

        norm_balance = float(self.balance) / float(self.initial_balance)
        norm_stock = float(self.stock_held) / float(self.max_stock)

        obs = np.array([current_price, norm_balance, norm_stock], dtype=np.float32)
        return obs



    def step(self, action):
        current_price = self.prices[self.current_step]

        if action == 0:  # Sell
            self.balance += self.stock_held * current_price
            self.stock_held = 0
        elif action == 2:  # Buy
            shares = self.balance // current_price
            self.stock_held += shares
            self.balance -= shares * current_price

        self.current_step += 1
        done = self.current_step >= self.n_steps - 1
        new_total_value = self.balance + self.stock_held * current_price
        reward = new_total_value - self.total_value

        # Encourage holding when you already own stock
        if action == 1 and self.stock_held > 0:
          reward += 0.1 

        # Penalize trying to sell when nothing is held
        if action == 0 and self.stock_held == 0:
          reward -= 0.5

        # Encourage buying when balance is available
        if action == 2 and self.balance > current_price:
          reward += 0.2

        self.total_value = new_total_value

        return self._get_observation(), reward, done, {}

    def render(self, mode='human'):
        print(f"Step: {self.current_step} | Price: {self.prices[self.current_step]} | "
              f"Balance: {self.balance} | Stock Held: {self.stock_held} | Total Value: {self.total_value}")

@st.cache_resource
def load_lstm_model():
    return load_model("lstm_stock_model.h5")

@st.cache_resource
def load_scaler():
    return joblib.load("scaler_x.pkl")

def predict_close_price(open_val, high_val, low_val, sma_10, ema_10):
    input_data = np.array([[open_val, high_val, low_val, sma_10, ema_10]], dtype=np.float32)
    input_scaled = scaler_x.transform(input_data)
    input_reshaped = input_scaled.reshape((1, 1, input_scaled.shape[1]))
    prediction = lstm_model.predict(input_reshaped, verbose=0)
    return prediction[0][0]

st.set_page_config(page_title="Stock Predictor + RL Trader", layout="centered")
st.title("📈 AAPL Stock Prediction & Trading Advice")

lstm_model = load_lstm_model()
scaler_x = load_scaler()

data = yf.download("AAPL", start="2020-01-01", end="2021-01-01")
prices = data['Close'].values.astype(np.float32)
env = StockTradingEnv(prices)
ppo_model = PPO.load("ppo_stock_trading.zip", env=env)

# User Input Section
st.subheader("Enter Stock Features")
open_val = st.number_input("Open", min_value=0.0, value=150.0, step=10.0)
high_val = st.number_input("High", min_value=0.0, value=152.0, step=10.0)
low_val = st.number_input("Low", min_value=0.0, value=148.0, step=10.0)
sma_10 = st.number_input("SMA_10", min_value=0.0, value=149.0, step=10.0)
ema_10 = st.number_input("EMA_10", min_value=0.0, value=149.5, step=10.0)

# Prediction and Decision Button
if st.button("Predict and Decide"):
    # LSTM prediction
    predicted_price = predict_close_price(open_val, high_val, low_val, sma_10, ema_10)
    st.success(f"📊 Predicted Close Price: ${predicted_price:.2f}")

    # PPO trading decision
    obs = env.reset()
    #st.write("Initial PPO Obs:", obs)

    for _ in range(5):  # simulate steps to generate diverse state
        action, _ = ppo_model.predict(obs, deterministic=True)
        obs, reward, done, _ = env.step(action)
        if done:
            break

    final_action = int(action)
    action_map = {0: "Sell", 1: "Hold", 2: "Buy"}
    st.info(f"🤖 PPO Trading Recommendation: **{action_map.get(final_action, 'Unknown')}**")

    #st.write("Final PPO Obs:", obs)
    #st.write("Action:", final_action)