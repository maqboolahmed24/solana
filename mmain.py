# Import Libraries
import sys
sys.path.append('E:\\Rahemunnisa\\libs\\abides')  # Dynamically include the path to ABIDES
from bayes_opt import BayesianOptimization
from kerastuner.tuners import BayesianOptimization
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, Concatenate, Attention
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split, TimeSeriesSplit
import os
import math
from tensorflow.keras.callbacks import LearningRateScheduler, ModelCheckpoint
import gym
from gym import spaces
from stable_baselines3 import PPO
from river import compose, linear_model, preprocessing  # Updated import
import tkinter as tk
from tkinter import filedialog
import logging
from tqdm import tqdm
import time
import streamlit as st
from mesa import Agent, Model
from mesa.time import RandomActivation
from mesa.datacollection import DataCollector
import random

# Set random seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Set up logging
logging.basicConfig(filename='model_monitoring.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

def log_event(message):
    logging.info(message)

# Function to load data with GUI
def load_data_with_gui(data_type, cryptocurrency):
    root = tk.Tk()
    root.withdraw()  # Use to hide the tkinter root window
    
    # Set up file types
    filetypes = [('CSV files', '*.csv')]
    
    # Open the file dialog
    filename = filedialog.askopenfilename(
        title=f'Select the {data_type} data file for {cryptocurrency}',
        filetypes=filetypes
    )
    
    # Check if a file was selected
    if filename:
        print(f"Loading {data_type} data for {cryptocurrency} from {filename}")
        data = pd.read_csv(filename)
        print(f"Loaded {data.shape[0]} rows and {data.shape[1]} columns.")
        return data
    else:
        print(f"No file selected for {data_type} data for {cryptocurrency}.")
        return None

# Load data using the GUI
train_data_btc = load_data_with_gui('train', 'Bitcoin')
test_data_btc = load_data_with_gui('test', 'Bitcoin')
val_data_btc = load_data_with_gui('validation', 'Bitcoin')

train_data_sol = load_data_with_gui('train', 'Solana')
test_data_sol = load_data_with_gui('test', 'Solana')
val_data_sol = load_data_with_gui('validation', 'Solana')

# Combine train and validation data
full_train_data_btc = pd.concat([train_data_btc, val_data_btc])
full_train_data_sol = pd.concat([train_data_sol, val_data_sol])

# Assume 'Close' is the target variable
scaler_btc = MinMaxScaler(feature_range=(0, 1))
scaled_data_btc = scaler_btc.fit_transform(full_train_data_btc[['Close']].values.reshape(-1,1))

scaler_sol = MinMaxScaler(feature_range=(0, 1))
scaled_data_sol = scaler_sol.fit_transform(full_train_data_sol[['Close']].values.reshape(-1,1))

# Function to create dataset for LSTM
def create_dataset(data, time_step=1):
    X, Y = [], []
    for i in range(len(data) - time_step - 1):
        a = data[i:(i + time_step), 0]
        X.append(a)
        Y.append(data[i + time_step, 0])
    return np.array(X), np.array(Y)

time_step = 100
X_train_btc, y_train_btc = create_dataset(scaled_data_btc, time_step)
X_train_btc = X_train_btc.reshape(X_train_btc.shape[0], X_train_btc.shape[1], 1)  # Reshape for LSTM [samples, time steps, features]

X_train_sol, y_train_sol = create_dataset(scaled_data_sol, time_step)
X_train_sol = X_train_sol.reshape(X_train_sol.shape[0], X_train_sol.shape[1], 1)  # Reshape for LSTM [samples, time steps, features]

# Define the LSTM model with hyperparameters to tune
def build_model(hp):
    model = Sequential()
    model.add(LSTM(units=hp.Int('units', min_value=32, max_value=512, step=32),
                   return_sequences=True,
                   input_shape=(X_train_btc.shape[1], 1)))
    model.add(Dropout(rate=hp.Float('dropout_1', min_value=0.0, max_value=0.5, step=0.1)))
    model.add(LSTM(units=hp.Int('units', min_value=32, max_value=512, step=32), return_sequences=True))
    model.add(Dropout(rate=hp.Float('dropout_2', min_value=0.0, max_value=0.5, step=0.1)))
    model.add(LSTM(units=hp.Int('units', min_value=32, max_value=512, step=32)))
    model.add(Dropout(rate=hp.Float('dropout_3', min_value=0.0, max_value=0.5, step=0.1)))
    model.add(Dense(1))

    model.compile(optimizer=Adam(hp.Float('learning_rate', min_value=1e-4, max_value=1e-2, sampling='LOG')),
                  loss='mean_squared_error')
    return model

# Configure and run the tuner
tuner = BayesianOptimization(
    build_model,
    objective='val_loss',
    max_trials=1,  # Set the number of trials, or experiments, to run.
    executions_per_trial=3,  # Set the number of models that should be built and fit for each trial.
    directory='model_tuning',
    project_name='LSTM_Crypto_Trading'
)

tuner.search(X_train_btc, y_train_btc, epochs=50, validation_data=(X_train_sol, y_train_sol), verbose=1)

# Review and use the best model
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
print('Best number of units in the first LSTM layer:', best_hps.get('units'))
print('Best dropout rate in the first dropout layer:', best_hps.get('dropout_1'))
print('Best learning rate for the optimizer:', best_hps.get('learning_rate'))

# Build the model with the best hyperparameters and train it on the data.
model = tuner.hypermodel.build(best_hps)

# Save the best model checkpoint
checkpoint = ModelCheckpoint('best_model.h5', save_best_only=True, monitor='val_loss', mode='min')

# Training the model with a progress bar
def train_model(model, X_train, y_train, epochs, X_val, y_val):
    for epoch in tqdm(range(epochs), desc="Training Progress"):
        model.fit(X_train, y_train, validation_data=(X_val, y_val), verbose=0)
        log_event(f"Epoch {epoch + 1}/{epochs} completed.")
        # Optionally log additional metrics
        val_loss = model.evaluate(X_val, y_val, verbose=0)
        log_event(f"Validation loss after epoch {epoch + 1}: {val_loss:.4f}")

train_model(model, X_train_btc, y_train_btc, epochs=100, X_val=X_train_sol, y_val=y_train_sol)

# Save the entire model
model.save('my_model.h5')

# Save only the weights
model.save_weights('my_model_weights.h5')

# Save the architecture only
json_string = model.to_json()
with open('model_architecture.json', 'w') as f:
    f.write(json_string)

# Model monitoring dashboard using Streamlit
def run_streamlit_dashboard():
    st.title('Model Monitoring Dashboard')

    # Placeholder for model metrics
    loss_chart = st.line_chart(np.random.randn(10, 2))

    for epoch in range(100):
        # Simulate epoch loss
        new_loss = np.random.randn(1, 2)
        loss_chart.add_rows(new_loss)
        time.sleep(0.1)  # Simulate time delay for epoch completion

if __name__ == "__main__":
    run_streamlit_dashboard()

# Function to update the model if needed
def update_model_if_needed(model, X_update, y_update, threshold=0.05):
    # Simulate checking model performance degradation
    current_performance = model.evaluate(X_update, y_update, verbose=0)
    if current_performance > threshold:
        log_event(f"Performance degradation detected: {current_performance:.4f}. Triggering retraining...")
        model.fit(X_update, y_update, epochs=10)
        log_event("Model retraining completed.")
    else:
        log_event("Model performance is within acceptable limits.")

# Making predictions
def predict_and_scale(model, X, scaler):
    pred = model.predict(X)
    pred = scaler.inverse_transform(pred)  # Reverting scaling
    return pred

# Prepare test data similarly to train data
X_test_btc, y_test_btc = create_dataset(scaler_btc.transform(test_data_btc[['Close']].values.reshape(-1,1)), time_step)
X_test_btc = X_test_btc.reshape(X_test_btc.shape[0], X_test_btc.shape[1], 1)

X_test_sol, y_test_sol = create_dataset(scaler_sol.transform(test_data_sol[['Close']].values.reshape(-1,1)), time_step)
X_test_sol = X_test_sol.reshape(X_test_sol.shape[0], X_test_sol.shape[1], 1)

# Predicting and comparing with the actual values
predicted_prices_btc = predict_and_scale(model, X_test_btc, scaler_btc)
real_prices_btc = scaler_btc.inverse_transform(y_test_btc.reshape(-1, 1))

predicted_prices_sol = predict_and_scale(model, X_test_sol, scaler_sol)
real_prices_sol = scaler_sol.inverse_transform(y_test_sol.reshape(-1, 1))

# Plotting predicted vs real prices
plt.figure(figsize=(10, 6))
plt.plot(real_prices_btc, label='Actual BTC Price')
plt.plot(predicted_prices_btc, label='Predicted BTC Price')
plt.title('Prediction vs Real BTC Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(real_prices_sol, label='Actual SOL Price')
plt.plot(predicted_prices_sol, label='Predicted SOL Price')
plt.title('Prediction vs Real SOL Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.show()

# Fourier Transform Feature
def fourier_features(data, cols, n_periods):
    result = pd.DataFrame(index=data.index)
    for col in cols:
        # Applying Fourier transform to each specified column
        transformed = np.fft.fft(data[col])
        for i in range(1, n_periods + 1):
            # Adding sin and cos components as features
            result[f'{col}_cos_{i}'] = np.cos(np.angle(transformed[i]))
            result[f'{col}_sin_{i}'] = np.sin(np.angle(transformed[i]))
    return pd.concat([data, result], axis=1)

combined_data = fourier_features(combined_data, ['Close_sol', 'Close_btc'], 3)

# Enhanced LSTM with Attention Mechanism
def build_attention_model(input_shape):
    inputs = Input(shape=input_shape)
    x = LSTM(100, return_sequences=True)(inputs)
    x = Dropout(0.3)(x)
    # Attention layer
    query, value = x, x
    attention = Attention()([query, value])
    context_vector = Concatenate(axis=-1)([x, attention])
    x = LSTM(50)(context_vector)
    x = Dropout(0.3)(x)
    outputs = Dense(1)(x)
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Preparing combined data
scaled_combined_data = scaler_btc.fit_transform(full_train_data_btc[['Close']].values.reshape(-1,1))
X_combined, y_combined = create_dataset(scaled_combined_data, time_step)
X_combined = X_combined.reshape(X_combined.shape[0], X_combined.shape[1], 1)  # Reshape for LSTM [samples, time steps, features]

model = build_attention_model((X_combined.shape[1], X_combined.shape[2]))
model.summary()

# Cyclical Learning Rates
def cyclical_lr(rate_min, rate_max, step_size):
    def lr_scheduler(epoch):
        cycle = math.floor(1 + epoch / (2 * step_size))
        x = abs(epoch / step_size - 2 * cycle + 1)
        return rate_min + (rate_max - rate_min) * max(0, (1 - x))
    return LearningRateScheduler(lr_scheduler)

callback = cyclical_lr(0.0001, 0.001, 500)
history = model.fit(X_train_btc, y_train_btc, epochs=200, batch_size=32, callbacks=[callback], validation_data=(X_train_sol, y_train_sol))

# Ensemble Techniques
def ensemble_predictions(models, X):
    predictions = [model.predict(X) for model in models]
    # Example: weighted average, weights could be tuned based on historical performance
    weights = [0.6, 0.4]  # Assuming two models
    weighted_predictions = np.average(predictions, axis=0, weights=weights)
    return weighted_predictions

# Assuming model1 and model2 are already trained and X_test is prepared
final_predictions_btc = ensemble_predictions([model1, model2], X_test_btc)
final_predictions_sol = ensemble_predictions([model1, model2], X_test_sol)

# Deep Reinforcement Learning Environment
class CryptoTradingEnv(gym.Env):
    """A cryptocurrency trading environment for OpenAI gym"""
    metadata = {'render.modes': ['human']}

    def __init__(self, df, initial_balance=1000, lookback_window_size=50):
        super(CryptoTradingEnv, self).__init__()

        # General variables defining the environment
        self.df = df
        self.initial_balance = initial_balance
        self.lookback_window_size = lookback_window_size

        # Action space: 0 = hold, 1 = buy, 2 = sell
        self.action_space = spaces.Discrete(3)

        # Observation space: prices, indicators, balance, etc.
        self.observation_space = spaces.Box(low=0, high=np.inf, shape=(lookback_window_size + 4, df.shape[1]))

        # State
        self.state = self._get_initial_state()

    def _get_initial_state(self):
        return np.random.rand(self.lookback_window_size + 4, self.df.shape[1])  # Placeholder

    def step(self, action):
        # Assuming 'action' is 0 (hold), 1 (buy), or 2 (sell)
        current_price = self.df.loc[self.current_step, 'Close']
        if action == 1:  # Buy
            reward = self._calculate_reward(buy_price=current_price)
        elif action == 2:  # Sell
            reward = self._calculate_reward(sell_price=current_price)
        else:
            reward = 0
        
        # Update the state, etc.
        # Return the next state, reward, done, info
        return next_state, reward, done, {}

    def _calculate_reward(self, buy_price=None, sell_price=None):
        # Custom logic to calculate reward based on trading action
        if buy_price:
            # Example: reward based on future price change perspective
            future_price = self.df.loc[self.current_step + 10, 'Close']  # Future price after 10 steps
            reward = future_price - buy_price
        elif sell_price:
            future_price = self.df.loc[self.current_step + 10, 'Close']
            reward = sell_price - future_price
        return reward

    def reset(self):
        # Reset the state of the environment to an initial state
        return self._get_initial_state()

    def render(self, mode='human', close=False):
        # Render the environment to the screen
        pass

# Reinforcement Learning Agent
env = CryptoTradingEnv(combined_data)
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=10000)

# High-Frequency Trading Feature
def calculate_microprice(data):
    data['micro_price'] = (data['bid_price'] * data['ask_size'] + data['ask_price'] * data['bid_size']) / (data['ask_size'] + data['bid_size'])
    return data

# Continuous Learning
model = compose.Pipeline(
    preprocessing.StandardScaler(),
    linear_model.LogisticRegression()
)

# Example data stream (replace with actual data stream)
X_stream = [{'feature1': 1.0, 'feature2': 2.0}, {'feature1': 1.2, 'feature2': 2.1}]
y_stream = [1, 0]

for x, y in zip(X_stream, y_stream):
    model = model.fit_one(x, y)

# Advanced Neural Network Architectures
from tensorflow.keras.layers import Transformer

def build_transformer_model(input_shape):
    inputs = Input(shape=input_shape)
    transformer_layer = Transformer(num_layers=2, d_model=64, num_heads=8, ff_dim=256, dropout=0.1)(inputs)
    outputs = Dense(1)(transformer_layer)
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

input_shape = (X_train_btc.shape[1], X_train_btc.shape[2])
model = build_transformer_model(input_shape)
model.summary()

# Meta-Learning
from meta_learning import ModelAgnosticMetaLearning

model = build_base_model(input_shape)  # Any base model
maml = ModelAgnosticMetaLearning(model, optimizer=tf.keras.optimizers.Adam(), loss='mean_squared_error')
maml.train(X_train_btc, y_train_btc, epochs=50, learning_rate=0.01)

# Simulation and Backtesting Framework
# Backtesting with Backtrader
class CryptoStrategy(bt.Strategy):
    def __init__(self):
        # Access the parameters here and create any indicators
        self.dataclose = self.datas[0].close

    def next(self):
        if not self.position:  # not in the market
            if self.dataclose[0] < self.dataclose[-1]:
                # current close less than previous close
                if self.dataclose[-1] < self.dataclose[-2]:
                    # previous close less than the previous-previous close
                    self.buy()
        else:
            if len(self) >= (self.bar_executed + 5):
                self.sell()

def run_backtest(strategy, data, cash=10000, commission=.002):
    cerebro = bt.Cerebro()
    cerebro.addstrategy(strategy)
    data_feed = bt.feeds.PandasData(dataname=data)
    cerebro.adddata(data_feed)
    cerebro.broker.setcash(cash)
    cerebro.broker.setcommission(commission=commission)
    cerebro.addsizer(bt.sizers.FixedSize, stake=10)

    print('Starting Portfolio Value: %.2f' % cerebro.broker.getvalue())
    cerebro.run()
    print('Final Portfolio Value: %.2f' % cerebro.broker.getvalue())
    cerebro.plot()

# Assuming 'df' is a DataFrame loaded with historical price data
run_backtest(CryptoStrategy, df)

# Multi-Agent Simulation with Mesa
class TrendFollowerAgent(Agent):
    """ An agent that buys when the price is rising and sells when it's falling. """
    def step(self):
        if len(self.model.price_history) > 2:
            if self.model.price_history[-1] > self.model.price_history[-2]:
                self.wealth += 10  # Assuming this means buying
                self.model.buy_pressure += 1
            else:
                self.wealth -= 10  # Assuming this means selling
                self.model.sell_pressure += 1

class MeanReversionAgent(Agent):
    """ Enhanced to consider market pressures. """
    def step(self):
        average_price = sum(self.model.price_history) / len(self.model.price_history)
        current_price = self.model.price_history[-1]
        threshold = 0.05  # Threshold to react more strongly if market pressures are high

        if current_price < average_price * (1 - threshold):
            self.model.buy_pressure += 1
            self.wealth += 10
        elif current_price > average_price * (1 + threshold):
            self.model.sell_pressure += 1
            self.wealth -= 10

class RandomAgent(Agent):
    """ An agent whose actions are determined randomly. """
    def step(self):
        import random
        if random.random() > 0.5:
            self.wealth += 10
            self.model.buy_pressure += 1
        else:
            self.wealth -= 10
            self.model.sell_pressure += 1

# Data Collection Functions
def get_model_price(model):
    return model.price_history[-1] if model.price_history else None

def get_total_wealth(model):
    return sum([agent.wealth for agent in model.schedule.agents])

class CryptoMarketModel(Model):
    def __init__(self, N):
        self.num_agents = N
        self.schedule = RandomActivation(self)
        self.price_history = [100]  # Starting price
        self.buy_pressure = 0
        self.sell_pressure = 0
        self.datacollector = DataCollector(
            model_reporters={"Price": get_model_price, "TotalWealth": get_total_wealth},
            agent_reporters={"Wealth": "wealth"}
        )

        # Initialize agents with random types
        for i in range(self.num_agents):
            agent_type = random.choice(["trend", "mean", "random"])
            if agent_type == "trend":
                a = TrendFollowerAgent(i, self)
            elif agent_type == "mean":
                a = MeanReversionAgent(i, self)
            else:
                a = RandomAgent(i, self)
            self.schedule.add(a)
        self.datacollector.collect(self)

    def step(self):
        # Reset pressures
        self.buy_pressure = 0
        self.sell_pressure = 0

        # Step for agents which will accumulate buy and sell pressures
        self.schedule.step()

        # Update market price based on net pressure
        net_pressure = self.buy_pressure - self.sell_pressure
        price_change = net_pressure / 1000  # Divisor to moderate the price change
        new_price = self.price_history[-1] * (1 + price_change)
        self.price_history.append(new_price)

        # Data collection for analysis
        self.datacollector.collect(self)

# Example of running the model
model = CryptoMarketModel(50)
for i in range(100):
    model.step()

# Retrieve model-level data
model_data = model.datacollector.get_model_vars_dataframe()
print(model_data.head())

# Retrieve agent-level data
agent_wealth = model.datacollector.get_agent_vars_dataframe()
print(agent_wealth.head())

# Visualization
plt.figure(figsize=(10, 5))
plt.plot(model_data['Price'], label='Market Price')
plt.title('Market Price Over Time')
plt.xlabel('Time Steps')
plt.ylabel('Price')
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(10, 5))
model_data['TotalWealth'].plot()
plt.title('Total Wealth in the Market')
plt.xlabel('Time Steps')
plt.ylabel('Total Wealth')
plt.grid(True)
plt.show()

# Time Series Cross-Validation
tscv = TimeSeriesSplit(n_splits=5)
model = build_model(best_hps)  # Assuming a model-building function with best hyperparameters

for train_index, test_index in tscv.split(X_combined):
    X_train, X_test = X_combined[train_index], X_combined[test_index]
    y_train, y_test = y_combined[train_index], y_combined[test_index]
    model.fit(X_train, y_train)
    print("Test score: ", model.score(X_test, y_test))

# Performance Metrics Evaluation
def calculate_sharpe_ratio(returns, risk_free_rate=0.01):
    excess_returns = returns - risk_free_rate
    sharpe_ratio = np.mean(excess_returns) / np.std(excess_returns)
    return sharpe_ratio

# Assuming 'daily_returns' is a pandas Series of daily returns from the backtest
daily_returns = pd.Series([0.01, 0.02, -0.01, 0.005, 0.003])  # Example data
print("Sharpe Ratio: ", calculate_sharpe_ratio(daily_returns))
