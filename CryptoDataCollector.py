import os
import sys
import requests
import pandas as pd
from datetime import datetime
from tqdm import tqdm
import logging
import zipfile

# Setup Environment Variables
API_KEY = os.getenv('API_KEY')
SECRET_KEY = os.getenv('SECRET_KEY')

class CryptoDataCollector:
    def __init__(self):
        self.base_url = 'https://api.binance.com/api/v3'
        self.setup_logging()
        self.coins = {
            'SOL': 'SOLUSDT',
            'BTC': 'BTCUSDT',
            'BOTH': ['SOLUSDT', 'BTCUSDT']
        }
        self.set_parameters()

    def setup_logging(self):
        logging.basicConfig(filename='crypto_data_collector.log', level=logging.INFO, 
                            format='%(asctime)s - %(levelname)s - %(message)s')
        logging.info("Initialized logging.")

    def set_parameters(self):
        print("Available coins for data collection: SOL, BTC, BOTH")
        self.coin_selection = input("Select the coin (SOL, BTC, BOTH): ").upper()
        if self.coin_selection not in self.coins:
            raise ValueError("Invalid coin selection. Please choose from 'SOL', 'BTC', 'BOTH'.")

        self.start_date = input("Enter start date (YYYY-MM-DD): ")
        self.end_date = input("Enter end date (YYYY-MM-DD): ")
        self.validate_dates()
        
        print("Available intervals: 1m, 3m, 5m, 15m, 30m, 1h, 1d")
        self.interval = input("Select data interval (e.g., '1m' for one minute, '1h' for one hour): ")
        if self.interval not in ['1m', '3m', '5m', '15m', '30m', '1h', '1d']:
            raise ValueError("Invalid interval. Choose from '1m', '3m', '5m', '15m', '30m', '1h', '1d'.")

    def validate_dates(self):
        try:
            self.start_date = datetime.strptime(self.start_date, '%Y-%m-%d')
            self.end_date = datetime.strptime(self.end_date, '%Y-%m-%d')
            if self.end_date < self.start_date:
                raise ValueError("End date cannot be earlier than start date.")
        except ValueError as e:
            print(f"Error in date input: {e}")
            sys.exit(1)

    def fetch_historical_data(self, symbol, start_date, end_date):
        endpoint = f'{self.base_url}/klines'
        params = {
            'symbol': symbol,
            'interval': self.interval,
            'startTime': int(start_date.timestamp() * 1000),
            'endTime': int(end_date.timestamp() * 1000),
            'limit': 1000
        }
        data = []
        total_data_points = int((end_date - start_date).total_seconds() / 60)
        progress_bar = tqdm(total=total_data_points, desc=f"Fetching {symbol}", unit='min', 
                            bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')

        while True:
            try:
                response = requests.get(endpoint, params=params)
                response.raise_for_status()
                json_response = response.json()
                if not json_response:
                    break
                data.extend(json_response)
                last_entry = json_response[-1][0]
                params['startTime'] = last_entry + 1
                progress_bar.update(len(json_response))
                if params['startTime'] > int(end_date.timestamp() * 1000):
                    break
            except requests.exceptions.RequestException as e:
                progress_bar.close()
                logging.error(f"Error fetching data: {e}")
                return None
        progress_bar.close()
        columns = ['Open Time', 'Open', 'High', 'Low', 'Close', 'Volume', 'Close Time',
                   'Quote Asset Volume', 'Number of Trades', 'Taker Buy Base Asset Volume',
                   'Taker Buy Quote Asset Volume', 'Ignore']
        df = pd.DataFrame(data, columns=columns)
        df['Open Time'] = pd.to_datetime(df['Open Time'], unit='ms')
        df['Close Time'] = pd.to_datetime(df['Close Time'], unit='ms')
        return df

    def save_data_to_file(self, data, formats=['csv']):
        folder_name = input("Enter folder name to save data (default is 'data'): ") or 'data'
        base_directory = f"./{folder_name}"
        os.makedirs(base_directory, exist_ok=True)
        start_date_str = self.start_date.strftime('%Y%m%d')
        end_date_str = self.end_date.strftime('%Y%m%d')
        file_paths = []
        for format in formats:
            file_name = f"{self.coin_selection.lower()}_{start_date_str}_to_{end_date_str}.{format}"
            file_path = os.path.join(base_directory, file_name)
            file_paths.append(file_path)
            try:
                if format == 'csv':
                    data.to_csv(file_path, index=False)
                elif format == 'xlsx':
                    data.to_excel(file_path, index=False)
                elif format == 'json':
                    data.to_json(file_path, orient='records')
                logging.info(f"Data successfully saved in {format.upper()} format to {file_path}")
            except Exception as e:
                logging.error(f"Failed to save data in {format.upper()}. Error: {e}")
        self.compress_files(file_paths, base_directory, start_date_str, end_date_str)

    def compress_files(self, file_paths, directory, start_date_str, end_date_str):
        zip_filename = f"{self.coin_selection.lower()}_{start_date_str}_to_{end_date_str}.zip"
        zip_path = os.path.join(directory, zip_filename)
        with zipfile.ZipFile(zip_path, 'w') as zipf:
            for file in file_paths:
                zipf.write(file, os.path.basename(file))
        logging.info(f"Compressed data files into {zip_path}")

    def aggregate_data(self, sol_data, btc_data):
        sol_columns = {col: f"SOL_{col}" for col in sol_data.columns}
        btc_columns = {col: f"BTC_{col}" for col in btc_data.columns}
        sol_data.rename(columns=sol_columns, inplace=True)
        btc_data.rename(columns=btc_columns, inplace=True)
        combined_data = pd.merge(sol_data, btc_data, left_on='SOL_Open Time', right_on='BTC_Open Time', how='outer')
        combined_data.sort_values(by='SOL_Open Time', inplace=True)
        combined_data.fillna(method='ffill', inplace=True)
        combined_data.fillna(method='bfill', inplace=True)
        return combined_data

    def data_collection_flow(self):
        try:
            # Manage directory setup and file handling
            folder_name = input("Enter folder name to save data (default is 'data'): ") or 'data'
            base_directory = f"./{folder_name}"
            os.makedirs(base_directory, exist_ok=True)

            # Collect and process data based on user's coin selection
            if self.coin_selection == 'BOTH':
                sol_data = self.fetch_historical_data('SOLUSDT', self.start_date, self.end_date)
                btc_data = self.fetch_historical_data('BTCUSDT', self.start_date, self.end_date)
                combined_data = self.aggregate_data(sol_data, btc_data)
                self.save_data_to_file(combined_data, formats=['csv', 'xlsx', 'json'])
            else:
                data = self.fetch_historical_data(self.coins[self.coin_selection], self.start_date, self.end_date)
                self.save_data_to_file(data, formats=['csv', 'xlsx', 'json'])

            print("Data collection completed successfully.")
        except Exception as e:
            logging.error(f"An error occurred during the data collection process: {e}")
            raise  # Re-raise the exception to be caught in the run method

    def run(self):
        print("Starting the data collection process...")
        try:
            self.data_collection_flow()
            print("Data collection completed successfully.")
        except Exception as e:
            logging.error(f"An error occurred during the data collection process: {e}")
            print(f"An error occurred: {e}")
        finally:
            print("Process completed. Check logs and output files for details.")

if __name__ == '__main__':
    collector = CryptoDataCollector()
    collector.run()
