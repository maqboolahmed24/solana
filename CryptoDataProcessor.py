import os
import pandas as pd
import numpy as np
from tqdm.auto import tqdm
from scipy.fft import fft
from pywt import dwt
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import zscore
import logging
from tkinter import Tk
from tkinter.filedialog import askopenfilename

# Set up logging to track progress and errors
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def apply_fft(series):
    if isinstance(series, pd.Series) and not series.empty:
        windowed = series * np.hanning(len(series))
        return np.abs(fft(windowed))
    return np.array([])

def apply_dwt(series):
    if isinstance(series, pd.Series) and not series.empty:
        coeffs = dwt(series, 'db1')
        return coeffs[0]
    return np.array([])

def load_data(file_path):
    try:
        data = pd.read_csv(file_path)
        logging.info("Data loaded successfully.")
        logging.info(f"Column names in the file: {data.columns.tolist()}")
        return data
    except Exception as e:
        logging.error(f"Failed to load data: {e}")
        return None

def convert_dates(data, column_name):
    try:
        if data[column_name].dtype == 'int64' or data[column_name].astype(str).str.isdigit().all():
            data[column_name] = pd.to_datetime(data[column_name], unit='ms')
        else:
            data[column_name] = pd.to_datetime(data[column_name])
        return data
    except ValueError as e:
        logging.error(f"Error converting dates for {column_name}: {e}")
        return data

def preprocess_data(data):
    if data is None:
        logging.error("Preprocessing failed: Data is None")
        return None
    try:
        for column in ['Open Time', 'Close Time']:
            if column in data.columns:
                data = convert_dates(data, column)
            for prefix in ['SOL_', 'BTC_']:
                if f'{prefix}{column}' in data.columns:
                    data = convert_dates(data, f'{prefix}{column}')
        data.dropna(inplace=True)
        logging.info("Preprocessing completed successfully.")
        return data
    except Exception as e:
        logging.error(f"Preprocessing error: {e}")
        return None

def calculate_moving_averages(data):
    data['MA10'] = data['Close'].rolling(window=10).mean()
    data['MA50'] = data['Close'].rolling(window=50).mean()
    return data

def calculate_exponential_moving_averages(data):
    data['EMA10'] = data['Close'].ewm(span=10, adjust=False).mean()
    data['EMA50'] = data['Close'].ewm(span=50, adjust=False).mean()
    return data

def calculate_rsi(data, periods=14):
    close_delta = data['Close'].diff()
    gain = (close_delta.where(close_delta > 0, 0))
    loss = (-close_delta.where(close_delta < 0, 0))
    avg_gain = gain.ewm(com=periods - 1, min_periods=periods).mean()
    avg_loss = loss.ewm(com=periods - 1, min_periods=periods).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    data['RSI'] = rsi
    return data

def calculate_bollinger_bands(data, window=20, num_std=2):
    rolling_mean = data['Close'].rolling(window=window).mean()
    rolling_std = data['Close'].rolling(window=window).std()
    data['Bollinger High'] = rolling_mean + (rolling_std * num_std)
    data['Bollinger Low'] = rolling_mean - (rolling_std * num_std)
    data['Bollinger Mean'] = rolling_mean
    return data

def feature_engineering(data):
    if data is None:
        logging.error("Feature engineering failed: Data is None")
        return None
    try:
        logging.info("Starting feature engineering...")
        prefixes = ['SOL_', 'BTC_'] if 'SOL_Close' in data.columns else ['']
        for prefix in prefixes:
            if any(f'{prefix}Close' in col for col in data.columns):
                logging.info(f"Processing data for prefix: {prefix}")
                data = process_crypto_data(data, prefix)
                data = calculate_moving_averages(data)
                data = calculate_exponential_moving_averages(data)
                data = calculate_rsi(data)
                data = calculate_bollinger_bands(data)
        logging.info("Feature engineering completed.")
        return data
    except Exception as e:
        logging.error(f"Feature engineering error: {e}")
        return None

def process_crypto_data(data, prefix):
    close_col = f'{prefix}Close'
    lags = [1, 2, 3, 6, 12]
    for lag in lags:
        data[f'{prefix}Close_lag_{lag}'] = data[close_col].shift(lag)
    windows = [3, 6, 9, 12]
    for window in windows:
        data[f'{prefix}Rolling_mean_{window}'] = data[close_col].rolling(window=window).mean()
        data[f'{prefix}Rolling_std_{window}'] = data[close_col].rolling(window=window).std()
    data[f'{prefix}EMA_12'] = data[close_col].ewm(span=12, adjust=False).mean()
    data[f'{prefix}ROC'] = data[close_col].pct_change(periods=1)
    logging.info(f"Completed processing for {prefix}")
    return data

def robust_feature_engineering(data):
    if data is None:
        logging.error("Robust feature engineering failed: Data is None")
        return None
    try:
        window_size = 20
        if len(data) < window_size:
            logging.error(f"Insufficient data for rolling calculations: {len(data)} entries, required: {window_size}")
            return None
        prefixes = ['SOL_', 'BTC_'] if 'SOL_Close' in data.columns else ['']
        for prefix in prefixes:
            close_col = f'{prefix}Close'
            if close_col in data.columns:
                data = calculate_rolling_features(data, close_col, prefix, window_size)
        data.replace([np.inf, -np.inf], np.nan, inplace=True)
        data.dropna(inplace=True)
        logging.info("Robust feature engineering completed successfully.")
        return data
    except Exception as e:
        logging.error(f"Robust feature engineering error: {e}")
        return None

def calculate_rolling_features(data, close_col, prefix, window_size):
    if data[close_col].dropna().empty or len(data[close_col].dropna()) < window_size:
        logging.error(f"Not enough valid data points for rolling calculations for {prefix}")
        return data
    try:
        data[f'{prefix}rolling_mean_close'] = data[close_col].rolling(window=window_size, min_periods=1).mean()
        data[f'{prefix}rolling_std_close'] = data[close_col].rolling(window=window_size, min_periods=1).std()
        data[f'{prefix}bollinger_upper'] = data[f'{prefix}rolling_mean_close'] + (data[f'{prefix}rolling_std_close'] * 2)
        data[f'{prefix}bollinger_lower'] = data[f'{prefix}rolling_mean_close'] - (data[f'{prefix}rolling_std_close'] * 2)
        data[f'{prefix}zscore_close'] = data[close_col].rolling(window=window_size, min_periods=1).apply(
            lambda x: zscore(x)[-1] if len(x) >= window_size else np.nan)
        return data
    except Exception as e:
        logging.error(f"Error calculating rolling features for {prefix}: {e}")
        return data
def scale_features(data):
    if data is None:
        return None
    try:
        # Exclude datetime columns explicitly from scaling
        numeric_columns = data.select_dtypes(include=[np.number]).columns.tolist()
        scaler = MinMaxScaler()
        data[numeric_columns] = scaler.fit_transform(data[numeric_columns])
        logging.info("Scaling of features completed.")
        return data
    except Exception as e:
        logging.error(f"Feature scaling error: {e}")
        return None

def save_processed_data(data, directory='processed_data', file_path=None):
    if data is None:
        logging.error("Save failed: Data is None")
        return
    try:
        if file_path is None:
            logging.error("File path not provided.")
            return
        
        # Extract the original filename from the full path
        original_filename = os.path.basename(file_path)
        
        # Create a new filename by prepending 'processed_' to the original filename
        processed_filename = f"processed_{original_filename}"
        
        # Create the full path for the new file to be saved
        full_path = os.path.join(directory, processed_filename)
        
        # Ensure the directory exists
        os.makedirs(directory, exist_ok=True)
        
        # Save the DataFrame to a CSV file
        data.to_csv(full_path, index=False)
        logging.info(f"Data saved successfully at {full_path}")
    except Exception as e:
        logging.error(f"Saving data failed: {e}")

def process_data_pipeline():
    logging.info("Data processing pipeline started")
    root = Tk()
    root.withdraw()  # Hide the Tkinter root window
    file_path = askopenfilename(title="Select a CSV file for processing", filetypes=[("CSV files", "*.csv")])
    if not file_path:
        logging.info("No file selected. Exiting...")
        return

    data = load_data(file_path)
    if data is None:
        return

    data = preprocess_data(data)
    if data is None:
        return

    data = feature_engineering(data)
    if data is None:
        return

    data = robust_feature_engineering(data)
    if data is None:
        return

    data = scale_features(data)
    if data is None:
        return

    save_processed_data(data, directory='processed_data', file_path=file_path)
    logging.info("Data processing pipeline completed")

if __name__ == "__main__":
    logging.info("Starting data processing pipeline...")
    try:
        process_data_pipeline()
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}")
