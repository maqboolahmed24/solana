import os
import pandas as pd
import logging
from sklearn.model_selection import train_test_split

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class AdvancedDataSplitter:
    def __init__(self, data, base_directory='processed_data'):
        """
        Initialize the AdvancedDataSplitter class with the dataset and base directory for saving processed files.
        
        Args:
            data (DataFrame): The complete preprocessed dataset.
            base_directory (str): The base directory where dataset splits will be stored.
        """
        self.data = data
        self.base_directory = base_directory
        os.makedirs(self.base_directory, exist_ok=True)
        logging.info(f"Base directory set at: {self.base_directory}")

    def split_data(self, test_size=0.2, val_size=0.1):
        """
        Splits the data into training, validation, and test sets while maintaining the order.
        
        Args:
            test_size (float): Proportion of the dataset to include in the test split.
            val_size (float): Proportion of the dataset to include in the validation split.
        
        Returns:
            tuple: Contains training data, validation data, and test data as DataFrames.
        """
        if self.data is None:
            logging.error("Data splitting failed: Data is None")
            return None, None, None
        
        assert (test_size + val_size) < 1, "Test and validation sizes must sum to less than 1"
        
        test_index = int((1 - test_size) * len(self.data))
        val_index = int((1 - test_size - val_size) * len(self.data))
        
        train_data = self.data.iloc[:val_index]
        val_data = self.data.iloc[val_index:test_index]
        test_data = self.data.iloc[test_index:]
        
        logging.info(f"Data split into train ({len(train_data)} rows), validation ({len(val_data)} rows), and test ({len(test_data)} rows) sets.")
        
        return train_data, val_data, test_data

    def save_data(self, train_data, val_data, test_data, subfolder='default'):
        """
        Saves the split data to separate files in structured subdirectories.
        
        Args:
            train_data, val_data, test_data (DataFrame): Datasets to be saved.
            subfolder (str): Subfolder under the base directory where files will be saved.
        """
        # Define paths for each dataset split
        path = os.path.join(self.base_directory, subfolder)
        os.makedirs(path, exist_ok=True)
        
        train_data.to_csv(os.path.join(path, 'train_data.csv'), index=False)
        val_data.to_csv(os.path.join(path, 'validation_data.csv'), index=False)
        test_data.to_csv(os.path.join(path, 'test_data.csv'), index=False)
        
        logging.info(f"Data saved successfully in '{path}'")

# Example usage
if __name__ == "__main__":
    # This would be integrated with your data processing pipeline
    # Suppose 'data' is your DataFrame after all preprocessing and feature engineering
    data = pd.DataFrame()  # Placeholder for actual data
    splitter = AdvancedDataSplitter(data)
    train, val, test = splitter.split_data()
    splitter.save_data(train, val, test, subfolder='model_version_1')
